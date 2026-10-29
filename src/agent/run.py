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

Re-running the same command resumes: cells already in ``episodes.jsonl`` with a
non-error status are skipped, so a dropped Colab session loses nothing. The
LLM settings (backbone, reasoning, providers, sampling) are stamped into
``config.json`` and a resume with different ones is refused: one run directory
is one experimental cell.

Usage:
    python -m src.agent.run --model-id dental_clinic__arc --split train \\
        --llm openrouter:z-ai/glm-4.5-air --provider z-ai --reasoning off --run clinic_smoke
    python -m src.agent.run --model-id dental_clinic__arc --split train \\
        --llm vllm:Qwen/Qwen3-8B-AWQ --reasoning off --grammar cfg --run clinic_qwen3_8b_cfg
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
from src.agent.context import build_graph_context
from src.agent.llm import REASONING_MODES, client_from_spec
from src.agent.tools import GraphTools
from src.config import get_settings
from src.constraints.cfg_grammar import grammar_from_summary
from src.etl.graph_cache import fingerprint, graph_summary, read_header, restore_graph
from src.etl.manifest import cache_path, load_manifest, resolve
from src.etl.topology import ENRICHMENT, apply_topology, compute_topology
from src.eval.judge import DEFAULT_JUDGE, judge_answer, make_judge

logger = logging.getLogger("agent.run")

PARADIGMS = {"single_shot": loop.single_shot, "agentic": loop.agentic}
# Which reply grammar constrains each paradigm's query-writing calls.
GRAMMAR_REPLY = {"single_shot": "generator", "agentic": "agent"}
GRAMMARS = ("none", "cfg")
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


def build_snapshot(driver, manifest, entry, snapshot: Path, etl_report_dir: Path,
                   state: str = "base") -> str:
    """Build a missing snapshot and export it: a fresh machine needs no hand-made files.

    ``base``: ETL the model's IFC file. ``topology``: make the base graph the
    loaded one (restoring or building it), apply the topology enrichment
    computed from the same IFC, and export that.
    """
    from src.etl.graph_cache import export_with_meta

    ifc = resolve(manifest, entry["paths"]["ifc"])
    if not ifc.is_file():
        raise RuntimeError(f"{entry['model_id']}: {ifc} missing; run scripts/ifcbench/fetch.py")
    etl_report_dir.mkdir(parents=True, exist_ok=True)
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


def ensure_graph(driver, manifest, entry, state: str, policy: str,
                 etl_report_dir: Path) -> Dict[str, Any]:
    """Make the loaded graph the manifest's snapshot for this model, or refuse.

    ``auto``: restore the snapshot if the live graph differs, and build the
    snapshot first (ETL + export) if there is none. ``never``: only accept a
    graph that already matches.
    """
    snapshot = cache_path(manifest, entry, state)
    if not snapshot.is_file():
        if policy != "auto" or state not in ("base", ENRICHMENT):
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


def _query_error_rate(records: List[Dict[str, Any]]) -> Optional[float]:
    """Share of executed queries Neo4j refused (syntax, type or timeout errors)."""
    results = [s.get("result") or {} for r in records for s in r["episode"]["steps"] if s.get("cypher")]
    return round(sum(bool(x.get("error")) for x in results) / len(results), 4) if results else None


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
            "by_category": {c: {"n": sum(v.values()),
                                "accuracy": round(v["correct"] / sum(v.values()), 4)}
                            for c, v in sorted(by_cat.items())},
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
    p.add_argument("--reasoning", choices=REASONING_MODES, default="default",
                   help="Backbone thinking: model default, or forced on/off (a run factor)")
    p.add_argument("--provider", help="Pin the backbone's OpenRouter provider(s), comma-separated, e.g. z-ai")
    p.add_argument("--grammar", choices=GRAMMARS, default="none",
                   help="cfg: constrain query-writing replies to the graph's grammar (vLLM only)")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help=f"Default: {DEFAULT_JUDGE}")
    p.add_argument("--judge-provider", help="Pin the judge's OpenRouter provider(s), e.g. z-ai")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--run", required=True, help="Run name (results/ifcbench/runs/<run>)")
    p.add_argument("--state", default="base",
                   help=f"Graph state snapshot to use: base, or {ENRICHMENT} (built from base if missing)")
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

    def providers(value: Optional[str]) -> Optional[List[str]]:
        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    if args.grammar != "none" and not args.llm.startswith("vllm:"):
        p.error("--grammar needs a vllm: backbone (grammars go through vLLM's structured outputs)")
    if not args.provider and args.llm.startswith("openrouter:"):
        logger.warning("No --provider: OpenRouter may serve calls from different providers")
    llm = client_from_spec(args.llm, provider_order=providers(args.provider),
                           reasoning=args.reasoning)
    judge = None if args.no_judge else make_judge(args.judge, providers(args.judge_provider))

    out = args.out_root / args.run
    (out / "context").mkdir(parents=True, exist_ok=True)
    episodes_path = out / "episodes.jsonl"
    done = done_keys(episodes_path, need_judgement=judge is not None)
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
    if config_path.exists() and config.get("settings") != settings_stamp:
        p.error(f"{out} was run with other settings; use a new --run.\n"
                f"  stored: {config.get('settings')}\n  now:    {settings_stamp}")
    config.update({"settings": settings_stamp, "llm": args.llm,
                   "judge": settings_stamp["judge"], "paradigms": paradigms,
                   "split": args.split, "state": args.state, "max_steps": args.max_steps})

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
            grammars: Dict[str, Optional[str]] = {par: None for par in paradigms}
            if args.grammar == "cfg":
                for par in paradigms:
                    reply = GRAMMAR_REPLY[par]
                    grammars[par] = grammar_from_summary(ctx.graph_stats, reply=reply)
                    (out / "context" / f"{mid}.{reply}.lark").write_text(grammars[par], encoding="utf-8")
                config["models"][mid]["grammar_sha256"] = {
                    GRAMMAR_REPLY[par]: hashlib.sha256(g.encode()).hexdigest()
                    for par, g in grammars.items() if g}
            config_path.write_text(json.dumps(config, indent=2))
            logger.info("%s: %d cells (%d questions x %s), schema %d chars",
                        mid, len(todo), len(questions), paradigms, len(schema))

            def cell(q: Dict[str, Any], paradigm: str) -> Dict[str, Any]:
                kwargs = {"max_steps": args.max_steps} if paradigm == "agentic" else {}
                kwargs["grammar"] = grammars[paradigm]
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
