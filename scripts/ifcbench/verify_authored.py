#!/usr/bin/env python3
"""Check the authored multi-hop questions against the graphs that should answer them.

Why this exists
---------------
The authored questions (``data/ifcbench/authored/<model_id>.jsonl``) exist to
exercise inter-room relations that IFC-Bench's own clinic questions barely
touch, so their ground truth has to be beyond doubt. Each record carries a
structured ``answer`` and up to two gold queries: ``gold_cypher.base`` for the
plain ETL graph (``null`` when that graph cannot answer, e.g. adjacency needs
geometry) and ``gold_cypher.topology`` for the graph with the topology
enrichment. This script restores each snapshot, runs every gold query and
compares its single result row with ``answer``, so a loader or enrichment
change that silently moves an answer fails loudly instead of corrupting a run.

Comparison: the row's columns must be exactly the answer's keys. A top-level
list is compared as a set (order ignored), except under ``ORDERED_KEYS``;
nested lists (a route, a ``[room, corridor]`` pair, a layer build-up) keep
their order. Numbers match within ``--tol`` (relative, default 0.1%).

The snapshots come from the manifest (``cache/<model_id>.<state>.jsonl.gz``);
a snapshot is restored only if the live graph's fingerprint differs from it.
States run in the order given (default base, then topology) and the graph is
left in ``--leave`` state (default base).

Usage
-----
    python scripts/ifcbench/verify_authored.py
    python scripts/ifcbench/verify_authored.py --ids 9001 9013 --states topology --leave as-is
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase, Query

from src.config import get_settings
from src.etl.graph_cache import fingerprint, graph_summary, read_header, restore_graph
from src.etl.manifest import cache_path, load_manifest, model_entry

DEFAULT_FILE = Path("data/ifcbench/authored/dental_clinic__arc.jsonl")
STATES = ("base", "topology")
REQUIRED = ("id", "question", "ground_truth", "category", "split", "project", "ifc_model",
            "source", "kind", "hops", "needs", "answer", "gold_cypher", "verification")
KINDS = {"door_connectivity", "adjacency", "vertical", "path_route", "exterior", "combined"}
ORDERED_KEYS = {"route"}          # top-level answer lists whose order matters
QUERY_TIMEOUT_S = 60

logger = logging.getLogger("verify_authored")


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

def _sort_key(value: Any) -> str:
    """A stable key for set comparison; floats rounded so near-equal values sort together."""
    def norm(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        return v
    return json.dumps(norm(value), sort_keys=True)


def same(got: Any, want: Any, tol: float, ordered: bool = True) -> bool:
    """Structural equality; numbers within ``tol`` (relative), lists as sets unless ``ordered``."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want, rel_tol=tol, abs_tol=1e-9)
    if isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            return False
        if not ordered:
            got, want = sorted(got, key=_sort_key), sorted(want, key=_sort_key)
        return all(same(g, w, tol) for g, w in zip(got, want))
    if isinstance(want, dict) and isinstance(got, dict):
        return got.keys() == want.keys() and all(same(got[k], want[k], tol) for k in want)
    return got == want


def compare(rows: List[Dict[str, Any]], answer: Dict[str, Any], tol: float) -> Tuple[bool, str]:
    if len(rows) != 1:
        return False, f"{len(rows)} rows, expected 1"
    row = rows[0]
    if set(row) != set(answer):
        return False, f"columns {sorted(row)} != answer keys {sorted(answer)}"
    bad = [k for k in answer if not same(row[k], answer[k], tol, ordered=k in ORDERED_KEYS)]
    if bad:
        return False, "; ".join(f"{k}: got {row[k]!r}, want {answer[k]!r}" for k in bad)
    return True, ""


# -----------------------------------------------------------------------------
# Records and graphs
# -----------------------------------------------------------------------------

def load_records(path: Path, ids: Optional[set]) -> List[Dict[str, Any]]:
    records, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            missing = [k for k in REQUIRED if k not in rec]
            problems = [f"missing {missing}"] if missing else []
            if rec.get("id") in seen:
                problems.append("duplicate id")
            if rec.get("split") != "train":
                problems.append("split is not 'train'")
            if rec.get("kind") not in KINDS:
                problems.append(f"unknown kind {rec.get('kind')!r}")
            if rec.get("needs") not in ("base", "geometry"):
                problems.append(f"needs must be base|geometry, got {rec.get('needs')!r}")
            gold = rec.get("gold_cypher") or {}
            if not gold.get("topology"):
                problems.append("no gold_cypher.topology")
            if rec.get("needs") == "base" and not gold.get("base"):
                problems.append("needs 'base' but gold_cypher.base is null")
            if problems:
                raise ValueError(f"{path}:{n} (id {rec.get('id')}): {', '.join(problems)}")
            seen.add(rec["id"])
            if not ids or rec["id"] in ids:
                records.append(rec)
    return records


def ensure_state(driver, manifest, entry, state: str) -> str:
    """Make the live graph this snapshot (restore only if the fingerprint differs)."""
    snapshot = cache_path(manifest, entry, state)
    if not snapshot.is_file():
        raise RuntimeError(f"no {state} snapshot at {snapshot}; build it with "
                           f"`python -m src.agent.run --model-id {entry['model_id']} --state {state} "
                           f"--snapshots-only`")
    expected = read_header(snapshot)["fingerprint"]
    if fingerprint(graph_summary(driver)) != expected:
        logger.info("Restoring %s", snapshot.name)
        restore_graph(driver, snapshot)
    return expected


def run_query(driver, text: str) -> List[Dict[str, Any]]:
    with driver.session() as session:
        return [r.data() for r in session.run(Query(text, timeout=QUERY_TIMEOUT_S))]


def check_state(driver, records: List[Dict[str, Any]], state: str, tol: float) -> Dict[int, Tuple[str, str]]:
    results: Dict[int, Tuple[str, str]] = {}
    for rec in records:
        text = rec["gold_cypher"].get(state)
        if not text:
            results[rec["id"]] = ("n/a", "")
            continue
        try:
            ok, why = compare(run_query(driver, text), rec["answer"], tol)
        except Exception as exc:  # noqa: BLE001 - report every failing query, not just the first
            results[rec["id"]] = ("ERROR", f"{type(exc).__name__}: {exc}")
            continue
        results[rec["id"]] = ("ok" if ok else "MISMATCH", why)
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--manifest", type=Path, default=settings.ifcbench_manifest_path)
    parser.add_argument("--ids", type=int, nargs="+", help="Only these question ids")
    parser.add_argument("--states", nargs="+", choices=STATES, default=list(STATES))
    parser.add_argument("--leave", choices=STATES + ("as-is",), default="base",
                        help="Graph state to leave loaded at the end (default: base)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Relative tolerance for numbers")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("neo4j").setLevel(logging.ERROR)

    records = load_records(args.file, set(args.ids) if args.ids else None)
    if not records:
        print(f"No questions selected from {args.file}")
        return 1
    model_ids = {f"{r['project']}__{r['ifc_model']}" for r in records}
    if len(model_ids) != 1:
        raise SystemExit(f"{args.file} mixes models {sorted(model_ids)}; one file per model")
    manifest = load_manifest(args.manifest)
    entry = model_entry(manifest, model_ids.pop())

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password_value))
    try:
        results = {}
        for state in args.states:
            fp = ensure_state(driver, manifest, entry, state)
            logger.info("%s %s (%s): %d questions", entry["model_id"], state, fp[:12], len(records))
            results[state] = check_state(driver, records, state, args.tol)
        if args.leave != "as-is" and args.leave != args.states[-1]:
            ensure_state(driver, manifest, entry, args.leave)
    finally:
        driver.close()

    header = f"{'id':>5}  {'kind':<17} {'needs':<8} {'hops':>4}  " + "  ".join(f"{s:<8}" for s in args.states)
    print(header)
    print("-" * len(header))
    failures = []
    for rec in records:
        cells = [results[s][rec["id"]][0] for s in args.states]
        print(f"{rec['id']:>5}  {rec['kind']:<17} {rec['needs']:<8} {rec['hops']:>4}  "
              + "  ".join(f"{c:<8}" for c in cells))
        failures += [(rec["id"], s, *results[s][rec["id"]]) for s in args.states
                     if results[s][rec["id"]][0] in ("MISMATCH", "ERROR")]
    checked = sum(1 for s in args.states for r in records if results[s][r["id"]][0] != "n/a")
    print(f"\n{len(records)} questions, {checked} gold queries run, {len(failures)} failed")
    for qid, state, status, why in failures:
        print(f"  {qid} [{state}] {status}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
