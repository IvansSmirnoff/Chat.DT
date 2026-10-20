"""Run IFC-Bench questions through the graph arms and judge them.

One process, next to Neo4j: for each model in the manifest, make sure its graph
is the one loaded (restore the snapshot if the fingerprint differs), build the
prompt context from the live graph, then run every (question x paradigm) cell
and judge it. Nothing is uploaded anywhere; the context each run used is saved
beside its results.

Results go to ``results/ifcbench/runs/<run>/``:

    config.json              arguments, dataset commit, per-model graph identity
    context/<model_id>.txt   the schema prompt the model saw
    episodes.jsonl           one line per (model, question, paradigm): episode + judgement
    summary.json             accuracy / abstention / cost per paradigm and category

Re-running the same command resumes: cells already in ``episodes.jsonl`` with a
non-error status are skipped, so a dropped Colab session loses nothing.

Usage:
    python -m src.agent.run --model-id dental_clinic__arc --split train \\
        --llm openrouter:z-ai/glm-4.5-air --run clinic_smoke
    python -m src.agent.run --manifest-split test --run full_glm45air ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase

from src.agent import loop
from src.agent.context import build_graph_context
from src.agent.llm import client_from_spec
from src.agent.tools import GraphTools
from src.config import get_settings
from src.etl.graph_cache import fingerprint, graph_summary, read_header, restore_graph
from src.etl.manifest import cache_path, load_manifest, resolve
from src.eval.judge import DEFAULT_JUDGE, judge_answer, make_judge

logger = logging.getLogger("agent.run")

PARADIGMS = {"single_shot": loop.single_shot, "agentic": loop.agentic}
Key = Tuple[str, int, str]


def load_questions(manifest: Dict[str, Any], entry: Dict[str, Any], split: str,
                   ids: Optional[set]) -> List[Dict[str, Any]]:
    with open(resolve(manifest, entry["paths"]["testset"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows
            if (split == "all" or r["split"] == split) and (not ids or r["id"] in ids)]


def done_keys(path: Path, need_judgement: bool) -> set:
    """Cells that need no re-run: no infrastructure error, judged if judging."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["episode"]["status"] == "error":
                continue
            if need_judgement and rec.get("judgement") is None:
                continue
            keys.add((rec["model_id"], rec["question_id"], rec["paradigm"]))
    return keys


def build_snapshot(driver, manifest, entry, snapshot: Path, etl_report_dir: Path) -> str:
    """ETL the model's IFC file and export it: a fresh machine needs no hand-made files.

    Only the ``base`` state can be built this way; enriched states are derived
    from it by their own scripts.
    """
    from src.etl.graph_cache import export_with_meta
    from src.etl.loader import IFCToNeo4jLoader

    ifc = resolve(manifest, entry["paths"]["ifc"])
    if not ifc.is_file():
        raise RuntimeError(f"{entry['model_id']}: {ifc} missing; run scripts/ifcbench/fetch.py")
    logger.info("No snapshot for %s: running the ETL on %s", entry["model_id"], ifc.name)
    etl_report_dir.mkdir(parents=True, exist_ok=True)
    stats = IFCToNeo4jLoader().run(ifc_path=ifc, report_path=etl_report_dir / f"{entry['model_id']}.json")
    if stats["errors"]:
        logger.warning("%s: ETL reported %d element errors", entry["model_id"], stats["errors"])
    header = export_with_meta(driver, snapshot, entry["model_id"], "base", ifc)
    return header["fingerprint"]


def ensure_graph(driver, manifest, entry, state: str, policy: str,
                 etl_report_dir: Path) -> Dict[str, Any]:
    """Make the loaded graph the manifest's snapshot for this model, or refuse.

    ``auto``: restore the snapshot if the live graph differs, and build the
    snapshot first (ETL + export) if there is none. ``never``: only accept a
    graph that already matches.
    """
    snapshot = cache_path(manifest, entry, state)
    if not snapshot.is_file():
        if policy != "auto" or state != "base":
            raise RuntimeError(f"{entry['model_id']}: no {state} snapshot at {snapshot}")
        return {"fingerprint": build_snapshot(driver, manifest, entry, snapshot, etl_report_dir),
                "restored": False, "built": True}
    expected = read_header(snapshot)["fingerprint"]
    live = fingerprint(graph_summary(driver))
    if live == expected:
        return {"fingerprint": live, "restored": False}
    if policy == "never":
        raise RuntimeError(
            f"{entry['model_id']}: loaded graph {live[:12]} is not snapshot "
            f"{expected[:12]} at {snapshot}; restore it or pass --restore auto"
        )
    logger.info("Restoring %s (%s)", entry["model_id"], snapshot.name)
    restore_graph(driver, snapshot)
    return {"fingerprint": expected, "restored": True}


def summarise(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    cells: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("judgement") is not None:
            groups[r["paradigm"]].append(r)
    for paradigm, rs in sorted(groups.items()):
        verdicts = Counter(r["judgement"]["verdict"] for r in rs)
        by_cat: Dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            by_cat[str(r["category"])][r["judgement"]["verdict"]] += 1
        n = len(rs)
        cells[paradigm] = {
            "n": n,
            "correct": verdicts["correct"], "wrong": verdicts["wrong"],
            "abstained": verdicts["abstained"],
            "accuracy": round(verdicts["correct"] / n, 4) if n else None,
            "by_category": {c: {"n": sum(v.values()),
                                "accuracy": round(v["correct"] / sum(v.values()), 4)}
                            for c, v in sorted(by_cat.items())},
            "mean_queries": round(sum(r["episode"]["n_queries"] for r in rs) / n, 2) if n else None,
            "cost_usd": round(sum(r["episode"]["cost_usd"] for r in rs), 4),
            "judge_cost_usd": round(sum(r["judgement"].get("cost_usd") or 0 for r in rs), 4),
            "mean_wall_s": round(sum(r["episode"]["wall_s"] for r in rs) / n, 1) if n else None,
            "step_limit": sum(r["episode"]["status"] == "step_limit" for r in rs),
        }
    return cells


def main(argv: Optional[List[str]] = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--model-id", action="append", help="Manifest model_id (repeatable)")
    sel.add_argument("--manifest-split", choices=["test", "train"],
                     help="Every manifest model with questions in this split")
    p.add_argument("--split", choices=["test", "train", "all"], default="test",
                   help="Which questions to ask (default: test)")
    p.add_argument("--ids", type=int, nargs="*", help="Only these IFC-Bench question ids")
    p.add_argument("--paradigm", choices=sorted(PARADIGMS), action="append",
                   help="Default: both")
    p.add_argument("--llm", required=True, help="e.g. openrouter:z-ai/glm-4.5-air")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help=f"Default: {DEFAULT_JUDGE}")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--run", required=True, help="Run name (results/ifcbench/runs/<run>)")
    p.add_argument("--state", default="base", help="Graph state snapshot to use")
    p.add_argument("--restore", choices=["auto", "never"], default="auto",
                   help="auto: restore the snapshot if needed, building it (ETL) if missing")
    p.add_argument("--max-steps", type=int, default=loop.MAX_STEPS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="At most N questions per model")
    p.add_argument("--manifest", type=Path, default=settings.ifcbench_manifest_path)
    p.add_argument("--out-root", type=Path, default=Path("results/ifcbench/runs"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    manifest = load_manifest(args.manifest)
    if args.model_id:
        wanted = set(args.model_id)
        entries = [m for m in manifest["models"] if m["model_id"] in wanted]
        missing = wanted - {m["model_id"] for m in entries}
        if missing:
            p.error(f"not in manifest: {sorted(missing)}")
    else:
        entries = [m for m in manifest["models"] if m["questions"][args.manifest_split]]
    paradigms = args.paradigm or ["single_shot", "agentic"]

    llm = client_from_spec(args.llm)
    judge = None if args.no_judge else make_judge(args.judge)

    out = args.out_root / args.run
    (out / "context").mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.jsonl"
    done = done_keys(episodes_path, need_judgement=judge is not None)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": manifest["dataset"], "models": {},
    }
    config.update({"llm": args.llm, "judge": None if args.no_judge else args.judge,
                   "paradigms": paradigms, "split": args.split, "state": args.state,
                   "max_steps": args.max_steps})

    driver = GraphDatabase.driver(settings.neo4j_uri,
                                  auth=(settings.neo4j_user, settings.neo4j_password_value))
    tools = GraphTools(driver)
    write_lock = threading.Lock()
    try:
        for entry in entries:
            mid = entry["model_id"]
            questions = load_questions(manifest, entry, args.split,
                                       set(args.ids) if args.ids else None)[: args.limit]
            todo = [(q, par) for q in questions for par in paradigms
                    if (mid, q["id"], par) not in done]
            if not todo:
                logger.info("%s: nothing to do (%d questions)", mid, len(questions))
                continue
            identity = ensure_graph(driver, manifest, entry, args.state, args.restore,
                                    etl_report_dir=out / "etl")
            ctx = build_graph_context(driver, resolve(manifest, entry["paths"]["ifc"]))
            schema = ctx.schema_prompt()
            (out / "context" / f"{mid}.txt").write_text(schema, encoding="utf-8")
            config["models"][mid] = {**ctx.graph, **identity}
            config_path.write_text(json.dumps(config, indent=2))
            logger.info("%s: %d cells (%d questions x %s), schema %d chars",
                        mid, len(todo), len(questions), paradigms, len(schema))

            def cell(q: Dict[str, Any], paradigm: str) -> Dict[str, Any]:
                kwargs = {"max_steps": args.max_steps} if paradigm == "agentic" else {}
                episode = PARADIGMS[paradigm](q["id"], q["question"], schema, llm, tools, **kwargs)
                judgement = None
                if judge is not None and episode.status != "error":
                    judgement = judge_answer(judge, question=q["question"], category=q["category"],
                                             ground_truth=q["ground_truth"],
                                             answer=episode.answer).to_dict()
                return {"model_id": mid, "question_id": q["id"], "paradigm": paradigm,
                        "category": q["category"], "split": q["split"],
                        "question": q["question"], "ground_truth": q["ground_truth"],
                        "episode": episode.to_dict(), "judgement": judgement}

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(cell, q, par): (q["id"], par) for q, par in todo}
                for fut in as_completed(futures):
                    qid, par = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001 - one cell must not kill the run
                        logger.error("%s q%s %s failed: %s", mid, qid, par, exc)
                        continue
                    with write_lock, open(episodes_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    verdict = (rec["judgement"] or {}).get("verdict", rec["episode"]["status"])
                    logger.info("%s q%-4s %-11s %-9s queries=%d cost=$%.4f",
                                mid, qid, par, verdict, rec["episode"]["n_queries"],
                                rec["episode"]["cost_usd"])
    finally:
        driver.close()

    records = [json.loads(line) for line in open(episodes_path, encoding="utf-8")] \
        if episodes_path.exists() else []
    # Latest record per cell wins (a re-run after an error appends a new one).
    latest: Dict[Key, Dict[str, Any]] = {
        (r["model_id"], r["question_id"], r["paradigm"]): r for r in records}
    summary = summarise(latest.values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'paradigm':12s} {'n':>4s} {'acc':>6s} {'correct':>8s} {'wrong':>6s} "
          f"{'abst':>5s} {'queries':>8s} {'cost$':>8s}")
    for par, s in summary.items():
        print(f"{par:12s} {s['n']:4d} {s['accuracy']:6.3f} {s['correct']:8d} {s['wrong']:6d} "
              f"{s['abstained']:5d} {s['mean_queries']:8.2f} {s['cost_usd'] + s['judge_cost_usd']:8.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
