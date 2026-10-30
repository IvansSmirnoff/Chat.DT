"""Run IFC-Bench questions through the graph arms and judge them.

One process, next to Neo4j: for each model in the manifest, make sure its graph
is the one loaded (restore the snapshot if the fingerprint differs), build the
prompt context from the live graph, then run every (question x paradigm) cell
and judge it. Nothing is uploaded anywhere; the context each run used is saved
beside its results.

Results go to ``results/ifcbench/runs/<run>/``:

    config.json              arguments, dataset commit, per-model graph identity
    context/<model_id>.txt   the schema prompt the model saw
    context/<model_id>.<reply>.lark   the reply grammars, with ``--grammar cfg``
    episodes.jsonl           one line per (model, question, paradigm): episode + judgement
    summary.json             accuracy / abstention / cost per paradigm and category

Re-running the same command resumes: a cell whose latest record in
``episodes.jsonl`` has a non-error status is not run again, so a dropped Colab
session loses nothing. A judge failure keeps the paid episode (``judgement:
null`` plus ``judge_error``) and the resume judges it without re-running it;
a ``--no-judge`` run is judged the same way by a resume with the judge on. The
LLM settings (backbone, reasoning, providers, sampling, question file, prompt
variant) are stamped into ``config.json`` and a resume with different ones is
refused: one run directory is one experimental cell. The judge may be added to
an unjudged run, not swapped.

``--prompt-variant ours`` adds our answer-writing addendum to Hellin's prompts
(``prompts.ANSWER_ADDENDUM``); the default ``hellin`` leaves them unchanged.

``--questions`` asks the questions of another JSONL file (an authored set) in
the testsets' row schema instead of each model's IFC-Bench testset; rows go to
the model ``<project>__<ifc_model>``. ``--snapshots-only`` builds the missing
graph snapshots of the selected models and exits, without an LLM.

``--state`` picks the graph: ``base`` (our ETL), ``topology`` (base + the
topology enrichment) or ``generic`` (rung 0: the ECPPM instance graph,
``src/etl/generic_loader.py``). A missing snapshot is built from the IFC; so is
a topology snapshot whose ``topology_rev`` is older than the rules' (the old
file is kept as ``<model_id>.topology-v<rev>.jsonl.gz``). The
generic graph gets its own schema block and prompt frame
(``context.GenericContext``, ``prompts.FRAMES``) and no grammar.

Usage:
    python -m src.agent.run --model-id dental_clinic__arc --split train \\
        --llm openrouter:z-ai/glm-4.5-air --provider z-ai --reasoning off --run clinic_smoke
    python -m src.agent.run --model-id dental_clinic__arc --split train \\
        --llm vllm:Qwen/Qwen3-8B-AWQ --reasoning off --grammar cfg --run clinic_qwen3_8b_cfg
    python -m src.agent.run --questions data/ifcbench/authored/dental_clinic__arc.jsonl \\
        --split all --state topology --llm ... --run clinic_authored_topology
    python -m src.agent.run --manifest-split test --snapshots-only --run tierb_snapshots
    python -m src.agent.run --model-id dental_clinic__arc --state generic --snapshots-only --run prep
    python -m src.agent.run --manifest-split test --run full_glm45air ...
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.agent.context import build_generic_context, build_graph_context
from src.agent.llm import REASONING_MODES, ChatClient, client_from_spec
from src.agent.prompts import PROMPT_VARIANTS
from src.agent.tools import GraphTools
from src.config import get_settings
from src.constraints.cfg_grammar import grammar_from_summary
from src.etl import generic_loader
from src.etl.graph_cache import fingerprint, graph_summary, read_header, restore_graph
from src.etl.manifest import cache_path, load_manifest, resolve
from src.etl.topology import ENRICHMENT, TOPOLOGY_REV, apply_topology, compute_topology
from src.eval.judge import DEFAULT_JUDGE, judge_answer, make_judge

logger = logging.getLogger("agent.run")

PARADIGMS = {"single_shot": loop.single_shot, "agentic": loop.agentic}
GENERIC = generic_loader.STATE
# Graph states a missing snapshot can be built for.
BUILDABLE_STATES = ("base", ENRICHMENT, GENERIC)
# Which reply grammar constrains each paradigm's query-writing calls.
GRAMMAR_REPLY = {"single_shot": "generator", "agentic": "agent"}
GRAMMARS = ("none", "cfg")
# A question row (data/ifcbench/testsets/*.jsonl), and the optional keys of an
# authored set that are carried into its episode records.
QUESTION_KEYS = ("id", "question", "ground_truth", "category", "split", "project", "ifc_model")
EXTRA_KEYS = ("source", "kind", "hops")
# Settings that do not shape the episodes (see resume_settings).
JUDGE_SETTINGS = ("judge", "judge_provider")
Key = Tuple[str, int, str]


def read_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_questions(rows: List[Dict[str, Any]], split: str,
                     ids: Optional[set]) -> List[Dict[str, Any]]:
    return [r for r in rows
            if (split == "all" or r["split"] == split) and (not ids or r["id"] in ids)]


def questions_by_model(rows: List[Dict[str, Any]], model_ids: set) -> Dict[str, List[Dict[str, Any]]]:
    """A ``--questions`` file's rows by model_id (``<project>__<ifc_model>``).

    Refuses a row without a testset field, one naming a model the manifest
    lacks, and an id repeated within a model (cells are keyed by id).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for n, row in enumerate(rows, 1):
        missing = [k for k in QUESTION_KEYS if k not in row]
        if missing:
            raise ValueError(f"row {n}: missing {missing}")
        mid = f"{row['project']}__{row['ifc_model']}"
        if mid not in model_ids:
            raise ValueError(f"row {n} (id {row['id']}): {mid} is not in the manifest")
        if any(r["id"] == row["id"] for r in grouped[mid]):
            raise ValueError(f"row {n}: id {row['id']} repeats on {mid}")
        grouped[mid].append(row)
    return dict(grouped)


def latest_records(path: Path) -> Dict[Key, Dict[str, Any]]:
    """The latest record per cell: a re-run or a later judgement appends a new one."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["model_id"], r["question_id"], r["paradigm"]): r for r in map(json.loads, f)}


def judge_record(judge: ChatClient, rec: Dict[str, Any]) -> Dict[str, Any]:
    """``rec`` with its judgement, or with ``judge_error`` if the judge failed.

    The episode is paid for by then: a judge failure must not lose it, and a
    resume judges the stored record without running it again.
    """
    rec = {k: v for k, v in rec.items() if k != "judge_error"}
    try:
        rec["judgement"] = judge_answer(judge, question=rec["question"], category=rec["category"],
                                        ground_truth=rec["ground_truth"],
                                        answer=rec["episode"]["answer"]).to_dict()
    except Exception as exc:  # noqa: BLE001 - the judge must not take the episode with it
        rec["judgement"], rec["judge_error"] = None, f"{type(exc).__name__}: {exc}"
    return rec


def resume_settings(stored: Dict[str, Any], now: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The settings a resumed run goes on under, or None if they conflict.

    Everything but the judge must match. The judge does not shape the
    episodes, so a ``--no-judge`` run may be judged later (the stamp takes that
    judge on) and ``--no-judge`` on a judged run only leaves new cells
    unjudged; swapping one judge for another would mix verdicts and is refused.
    """
    def episode_settings(s: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in s.items() if k not in JUDGE_SETTINGS}

    if episode_settings(stored) != episode_settings(now):
        return None
    if now["judge"] is None:
        return stored
    if stored.get("judge") is None:
        return now
    return now if all(stored.get(k) == now[k] for k in JUDGE_SETTINGS) else None


def build_snapshot(driver, manifest, entry, snapshot: Path, etl_report_dir: Path,
                   state: str = "base") -> str:
    """Build a missing snapshot and export it: a fresh machine needs no hand-made files.

    ``base``: ETL the model's IFC file. ``topology``: make the base graph the
    loaded one (restoring or building it), apply the topology enrichment
    computed from the same IFC, and export that. ``generic``: wipe and load the
    IFC with the generic loader.
    """
    from src.etl.graph_cache import export_with_meta

    ifc = resolve(manifest, entry["paths"]["ifc"])
    if not ifc.is_file():
        raise RuntimeError(f"{entry['model_id']}: {ifc} missing; run scripts/ifcbench/fetch.py")
    etl_report_dir.mkdir(parents=True, exist_ok=True)
    if state == GENERIC:
        logger.info("No %s snapshot for %s: loading %s as the instance graph",
                    state, entry["model_id"], ifc.name)
        stats = generic_loader.load(driver, ifc)
        (etl_report_dir / f"{entry['model_id']}.{state}.json").write_text(json.dumps(stats, indent=2))
        return export_with_meta(driver, snapshot, entry["model_id"], state, ifc,
                                loader_sources=generic_loader.SOURCES)["fingerprint"]
    if state == ENRICHMENT:
        ensure_graph(driver, manifest, entry, "base", "auto", etl_report_dir)
        logger.info("No %s snapshot for %s: enriching the base graph", state, entry["model_id"])
        topo = compute_topology(ifc)
        written = apply_topology(driver, topo)
        (etl_report_dir / f"{entry['model_id']}.{state}.json").write_text(
            json.dumps({"stats": topo.stats, "written": written}, indent=2))
        return export_with_meta(driver, snapshot, entry["model_id"], state, ifc)["fingerprint"]
    from src.etl.loader import IFCToNeo4jLoader

    logger.info("No snapshot for %s: running the ETL on %s", entry["model_id"], ifc.name)
    stats = IFCToNeo4jLoader().run(ifc_path=ifc, report_path=etl_report_dir / f"{entry['model_id']}.json")
    if stats["errors"]:
        logger.warning("%s: ETL reported %d element errors", entry["model_id"], stats["errors"])
    header = export_with_meta(driver, snapshot, entry["model_id"], "base", ifc)
    return header["fingerprint"]


def stale_topology_rev(snapshot: Path) -> Optional[int]:
    """The topology revision of a snapshot built by older rules than ``TOPOLOGY_REV``, else None.

    The header's ``meta.topology_rev``; snapshots from before it was stamped
    carry none and are revision 1.
    """
    rev = int(read_header(snapshot).get("meta", {}).get("topology_rev") or 1)
    return rev if rev < TOPOLOGY_REV else None


def retire_stale_snapshot(manifest, entry, state: str, policy: str = "auto") -> Optional[Path]:
    """Move a stale topology snapshot aside, so that it counts as missing and is rebuilt.

    It is kept as ``<model_id>.topology-v<rev>.jsonl.gz``. ``never`` refuses
    a stale snapshot instead (nothing is moved): it may not be rebuilt.
    """
    snapshot = cache_path(manifest, entry, state)
    if state != ENRICHMENT or not snapshot.is_file():
        return None
    rev = stale_topology_rev(snapshot)
    if rev is None:
        return None
    if policy != "auto":
        raise RuntimeError(f"{entry['model_id']}: {snapshot.name} is topology rev {rev}, the rules are "
                           f"rev {TOPOLOGY_REV}; rebuild it (--restore auto)")
    kept = snapshot.with_name(f"{entry['model_id']}.{state}-v{rev}.jsonl.gz")
    snapshot.replace(kept)
    logger.info("%s: %s is topology rev %d (rules: rev %d), kept as %s; rebuilding",
                entry["model_id"], snapshot.name, rev, TOPOLOGY_REV, kept.name)
    return kept


def ensure_graph(driver, manifest, entry, state: str, policy: str,
                 etl_report_dir: Path) -> Dict[str, Any]:
    """Make the loaded graph the manifest's snapshot for this model, or refuse.

    ``auto``: restore the snapshot if the live graph differs, and build the
    snapshot first (ETL + export) if there is none or it is a topology
    snapshot of an older revision. ``never``: only accept a graph that already
    matches.
    """
    snapshot = cache_path(manifest, entry, state)
    retire_stale_snapshot(manifest, entry, state, policy)
    if not snapshot.is_file():
        if policy != "auto" or state not in BUILDABLE_STATES:
            raise RuntimeError(f"{entry['model_id']}: no {state} snapshot at {snapshot}")
        return {"fingerprint": build_snapshot(driver, manifest, entry, snapshot, etl_report_dir, state),
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


def prepare_snapshots(driver, manifest, entries, state: str, etl_report_dir: Path) -> int:
    """``--snapshots-only``: build every selected model's missing snapshot.

    A snapshot already on disk is left alone, not restored: restoring only
    re-checks it, and Tier-B prep would wipe and reload 31 graphs for that.
    A topology snapshot of an older revision counts as missing (it is kept as
    ``<model_id>.topology-v<rev>.jsonl.gz``). One model failing does not stop
    the others; the exit code reports it.
    """
    failed = []
    for entry in entries:
        mid = entry["model_id"]
        snapshot = cache_path(manifest, entry, state)
        try:
            if snapshot.is_file() and not (state == ENRICHMENT and stale_topology_rev(snapshot)):
                logger.info("%s: %s snapshot present (%s)", mid, state, snapshot.name)
                continue
            identity = ensure_graph(driver, manifest, entry, state, "auto", etl_report_dir)
        except Exception as exc:  # noqa: BLE001 - one model must not stop the batch
            logger.error("%s: no %s snapshot built: %s", mid, state, exc)
            failed.append(mid)
            continue
        logger.info("%s: built %s (%s)", mid, snapshot.name, identity["fingerprint"][:12])
    if failed:
        logger.error("%d of %d snapshots failed: %s", len(failed), len(entries), ", ".join(failed))
    return 1 if failed else 0


def _query_error_rate(records: List[Dict[str, Any]]) -> Optional[float]:
    """Share of executed queries Neo4j refused (syntax, type or timeout errors)."""
    results = [s.get("result") or {} for r in records for s in r["episode"]["steps"] if s.get("cypher")]
    return round(sum(bool(x.get("error")) for x in results) / len(results), 4) if results else None


def _breakdown(counts: Dict[str, Counter]) -> Dict[str, Dict[str, Any]]:
    return {c: {"n": sum(v.values()), "accuracy": round(v["correct"] / sum(v.values()), 4)}
            for c, v in sorted(counts.items())}


def summarise(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    cells: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("judgement") is not None:
            groups[r["paradigm"]].append(r)
    for paradigm, rs in sorted(groups.items()):
        verdicts = Counter(r["judgement"]["verdict"] for r in rs)
        by_cat: Dict[str, Counter] = defaultdict(Counter)
        by_kind: Dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            by_cat[str(r["category"])][r["judgement"]["verdict"]] += 1
            if r.get("kind") is not None:  # authored questions (--questions)
                by_kind[str(r["kind"])][r["judgement"]["verdict"]] += 1
        n = len(rs)
        attempted = verdicts["correct"] + verdicts["wrong"]
        # Hellin et al.'s metrics: primary accuracy = correct / n (abstentions count
        # as failures), secondary "accuracy among attempted answers".
        cells[paradigm] = {
            "n": n,
            "correct": verdicts["correct"], "wrong": verdicts["wrong"],
            "abstained": verdicts["abstained"],
            "accuracy": round(verdicts["correct"] / n, 4) if n else None,
            "accuracy_attempted": round(verdicts["correct"] / attempted, 4) if attempted else None,
            "abstention_rate": round(verdicts["abstained"] / n, 4) if n else None,
            "by_category": _breakdown(by_cat),
            "mean_queries": round(sum(r["episode"]["n_queries"] for r in rs) / n, 2) if n else None,
            "query_error_rate": _query_error_rate(rs),
            "cost_usd": round(sum(r["episode"]["cost_usd"] for r in rs), 4),
            "judge_cost_usd": round(sum(r["judgement"].get("cost_usd") or 0 for r in rs), 4),
            "mean_wall_s": round(sum(r["episode"]["wall_s"] for r in rs) / n, 1) if n else None,
            "step_limit": sum(r["episode"]["status"] == "step_limit" for r in rs),
            "token_limit": sum(r["episode"]["status"] == "token_limit" for r in rs),
            "parse_error": sum(r["episode"]["status"] == "parse_error" for r in rs),
            "truncated_calls": sum(c.get("finish") == "length"
                                   for r in rs for c in r["episode"].get("calls", [])),
            "mean_reasoning_tokens": round(
                sum(r["episode"].get("reasoning_tokens", 0) for r in rs) / n) if n else None,
        }
        if by_kind:
            cells[paradigm]["by_kind"] = _breakdown(by_kind)
    return cells


def main(argv: Optional[List[str]] = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--model-id", action="append", help="Manifest model_id (repeatable)")
    sel.add_argument("--manifest-split", choices=["test", "train"],
                     help="Every manifest model with questions in this split")
    p.add_argument("--questions", type=Path,
                   help="Ask the questions of this JSONL file (testset row schema) instead of each "
                        "model's IFC-Bench testset; without a model selection, every model it names")
    p.add_argument("--split", choices=["test", "train", "all"], default="test",
                   help="Which questions to ask (default: test)")
    p.add_argument("--ids", type=int, nargs="*", help="Only these question ids")
    p.add_argument("--paradigm", choices=sorted(PARADIGMS), action="append",
                   help="Default: both")
    p.add_argument("--llm", help="e.g. openrouter:z-ai/glm-4.5-air (not with --snapshots-only)")
    p.add_argument("--reasoning", choices=REASONING_MODES, default="default",
                   help="Backbone thinking: model default, or forced on/off (a run factor)")
    p.add_argument("--provider", help="Pin the backbone's OpenRouter provider(s), comma-separated, e.g. z-ai")
    p.add_argument("--grammar", choices=GRAMMARS, default="none",
                   help="cfg: constrain query-writing replies to the graph's grammar (vLLM only)")
    p.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="hellin",
                   help="hellin: Hellin's prompts as published; ours: plus our answer-writing "
                        "addendum on the answer-writing calls (prompts.ANSWER_ADDENDUM)")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help=f"Default: {DEFAULT_JUDGE}")
    p.add_argument("--judge-provider", help="Pin the judge's OpenRouter provider(s), e.g. z-ai")
    p.add_argument("--no-judge", action="store_true",
                   help="Store episodes unjudged; a later resume with the judge on judges them")
    p.add_argument("--run", required=True, help="Run name (results/ifcbench/runs/<run>)")
    p.add_argument("--state", default="base",
                   help=f"Graph state snapshot to use: base, {ENRICHMENT} (built from base if "
                        f"missing) or {GENERIC} (the ECPPM instance graph, built from the IFC)")
    p.add_argument("--restore", choices=["auto", "never"], default="auto",
                   help="auto: restore the snapshot if needed, building it (ETL) if missing")
    p.add_argument("--snapshots-only", action="store_true",
                   help="Build the selected models' missing --state snapshots and exit (no LLM)")
    p.add_argument("--max-steps", type=int, default=loop.MAX_STEPS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="At most N questions per model")
    p.add_argument("--manifest", type=Path, default=settings.ifcbench_manifest_path)
    p.add_argument("--out-root", type=Path, default=Path("results/ifcbench/runs"))
    args = p.parse_args(argv)
    if not (args.model_id or args.manifest_split or args.questions):
        p.error("one of --model-id, --manifest-split or --questions is required")
    if args.snapshots_only and args.restore == "never":
        p.error("--snapshots-only builds missing snapshots; it cannot go with --restore never")
    if not args.snapshots_only and not args.llm:
        p.error("--llm is required (except with --snapshots-only)")
    if args.grammar != "none" and args.state == GENERIC:
        p.error(f"--grammar {args.grammar} cannot go with --state {GENERIC}: a grammar over the "
                f"generic graph's labels ({', '.join(generic_loader.LABELS)}) and its one edge "
                f"type :{generic_loader.REL_TYPE} constrains nothing; use --grammar none")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    manifest = load_manifest(args.manifest)
    question_rows: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if args.questions:
        try:
            question_rows = questions_by_model(read_questions(args.questions),
                                               {m["model_id"] for m in manifest["models"]})
        except (OSError, ValueError) as exc:
            p.error(f"--questions {args.questions}: {exc}")
    if args.model_id:
        wanted = set(args.model_id)
        entries = [m for m in manifest["models"] if m["model_id"] in wanted]
        missing = wanted - {m["model_id"] for m in entries}
        if missing:
            p.error(f"not in manifest: {sorted(missing)}")
    elif args.manifest_split:
        entries = [m for m in manifest["models"] if m["questions"][args.manifest_split]]
    else:
        entries = [m for m in manifest["models"] if m["model_id"] in (question_rows or {})]
    paradigms = args.paradigm or ["single_shot", "agentic"]
    frame = "generic" if args.state == GENERIC else "typed"   # prompts.FRAMES
    out = args.out_root / args.run

    def connect():
        return GraphDatabase.driver(settings.neo4j_uri,
                                    auth=(settings.neo4j_user, settings.neo4j_password_value))

    if args.snapshots_only:
        driver = connect()
        try:
            return prepare_snapshots(driver, manifest, entries, args.state, out / "etl")
        finally:
            driver.close()

    def providers(value: Optional[str]) -> Optional[List[str]]:
        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    if args.grammar != "none" and not args.llm.startswith("vllm:"):
        p.error("--grammar needs a vllm: backbone (grammars go through vLLM's structured outputs)")
    if not args.provider and args.llm.startswith("openrouter:"):
        logger.warning("No --provider: OpenRouter may serve calls from different providers")
    llm = client_from_spec(args.llm, provider_order=providers(args.provider),
                           reasoning=args.reasoning)
    judge = None if args.no_judge else make_judge(args.judge, providers(args.judge_provider))

    (out / "context").mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.jsonl"
    stored = latest_records(episodes_path)
    config_path = out / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": manifest["dataset"], "models": {},
    }
    settings_stamp = {
        "llm": args.llm, "reasoning": args.reasoning, "provider": providers(args.provider),
        "temperature": llm.temperature, "max_tokens": llm.max_tokens,
        "judge": None if args.no_judge else args.judge,
        "judge_provider": providers(args.judge_provider),
        "state": args.state, "max_steps": args.max_steps,
    }
    if args.grammar != "none":  # absent when off, so older runs still resume
        settings_stamp["grammar"] = args.grammar
    if args.prompt_variant != "hellin":  # likewise
        settings_stamp["prompt_variant"] = args.prompt_variant
    if args.questions:  # likewise; question ids of two sets may collide, so a set is a cell
        questions_sha256 = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        settings_stamp["questions"] = questions_sha256
        config["questions"] = {"path": str(args.questions), "sha256": questions_sha256}
    if config_path.exists():
        resumed = resume_settings(config.get("settings") or {}, settings_stamp)
        if resumed is None:
            p.error(f"{out} was run with other settings; use a new --run.\n"
                    f"  stored: {config.get('settings')}\n  now:    {settings_stamp}")
        settings_stamp = resumed
    config.update({"settings": settings_stamp, "llm": args.llm,
                   "judge": settings_stamp["judge"], "paradigms": paradigms,
                   "split": args.split, "state": args.state, "max_steps": args.max_steps,
                   "prompt_variant": args.prompt_variant})

    driver = connect()
    tools = GraphTools(driver)
    write_lock = threading.Lock()
    try:
        for entry in entries:
            mid = entry["model_id"]
            rows = question_rows.get(mid, []) if question_rows is not None else \
                read_questions(resolve(manifest, entry["paths"]["testset"]))
            questions = select_questions(rows, args.split,
                                         set(args.ids) if args.ids else None)[: args.limit]
            # An episode needs running if it has no record or an infrastructure
            # error; a stored episode only needs the judge if it went unjudged.
            todo, unjudged = [], []
            for q in questions:
                for par in paradigms:
                    rec = stored.get((mid, q["id"], par))
                    if rec is None or rec["episode"]["status"] == "error":
                        todo.append((q, par))
                    elif judge is not None and rec.get("judgement") is None:
                        unjudged.append(rec)
            if not todo and not unjudged:
                logger.info("%s: nothing to do (%d questions)", mid, len(questions))
                continue
            if unjudged:
                logger.info("%s: judging %d stored episodes", mid, len(unjudged))
            if todo:
                identity = ensure_graph(driver, manifest, entry, args.state, args.restore,
                                        etl_report_dir=out / "etl")
                ctx = (build_generic_context(driver) if args.state == GENERIC else
                       build_graph_context(driver, resolve(manifest, entry["paths"]["ifc"])))
                schema = ctx.schema_prompt()
                (out / "context" / f"{mid}.txt").write_text(schema, encoding="utf-8")
                config["models"][mid] = {**ctx.graph, **identity}
                grammars: Dict[str, Optional[str]] = {par: None for par in paradigms}
                if args.grammar == "cfg":
                    for par in paradigms:
                        reply = GRAMMAR_REPLY[par]
                        grammars[par] = grammar_from_summary(ctx.graph_stats, reply=reply)
                        (out / "context" / f"{mid}.{reply}.lark").write_text(grammars[par], encoding="utf-8")
                    config["models"][mid]["grammar_sha256"] = {
                        GRAMMAR_REPLY[par]: hashlib.sha256(g.encode()).hexdigest()
                        for par, g in grammars.items() if g}
                logger.info("%s: %d cells (%d questions x %s), schema %d chars",
                            mid, len(todo), len(questions), paradigms, len(schema))
            config_path.write_text(json.dumps(config, indent=2))

            def cell(q: Dict[str, Any], paradigm: str) -> Dict[str, Any]:
                kwargs = {"max_steps": args.max_steps} if paradigm == "agentic" else {}
                kwargs["grammar"] = grammars[paradigm]
                kwargs["prompt_variant"] = args.prompt_variant
                kwargs["frame"] = frame
                episode = PARADIGMS[paradigm](q["id"], q["question"], schema, llm, tools, **kwargs)
                rec = {"model_id": mid, "question_id": q["id"], "paradigm": paradigm,
                       "category": q["category"], "split": q["split"],
                       "question": q["question"], "ground_truth": q["ground_truth"],
                       **{k: q[k] for k in EXTRA_KEYS if k in q},
                       "episode": episode.to_dict(), "judgement": None}
                if judge is not None and episode.status != "error":
                    rec = judge_record(judge, rec)
                return rec

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(cell, q, par): (q["id"], par) for q, par in todo}
                if judge is not None:
                    futures.update({pool.submit(judge_record, judge, rec): (rec["question_id"], rec["paradigm"])
                                    for rec in unjudged})
                for fut in as_completed(futures):
                    qid, par = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001 - one cell must not kill the run
                        logger.error("%s q%s %s failed: %s", mid, qid, par, exc)
                        continue
                    with write_lock, open(episodes_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    if rec.get("judge_error"):
                        logger.warning("%s q%s %s: episode kept unjudged, a resume judges it (%s)",
                                       mid, qid, par, rec["judge_error"][:300])
                    verdict = (rec["judgement"] or {}).get("verdict", rec["episode"]["status"])
                    logger.info("%s q%-4s %-11s %-9s queries=%d cost=$%.4f",
                                mid, qid, par, verdict, rec["episode"]["n_queries"],
                                rec["episode"]["cost_usd"])
    finally:
        driver.close()

    latest = latest_records(episodes_path)
    unjudged_total = sum(r["judgement"] is None and r["episode"]["status"] != "error"
                         for r in latest.values())
    if judge is not None and unjudged_total:
        logger.warning("%d stored episodes are unjudged; re-run the same command to judge them",
                       unjudged_total)
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
