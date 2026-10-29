#!/usr/bin/env python3
"""How much of what real models query does the reply grammar still allow?

Why this exists
---------------
A grammar that rejects queries a strong model needs would make the constrained
arm lose for the wrong reason. So before any constrained run, every query the
API backbones executed without error is replayed through the grammar of the
graph it ran on: wrapped in the agent envelope exactly as a model would emit it,
and checked byte by byte (``cfg_grammar.check_reply``). The acceptance rate is
the grammar's coverage; every rejection is listed so it can be judged as either
a grammar gap or a query the grammar is right to forbid.

The vocabulary comes from each model's snapshot header, so no Neo4j is needed.
Queries are compared with their newlines and tabs turned into spaces: the
grammar is single-line by design (the query sits in a JSON string), and Cypher
reads both the same.

Usage
-----
    python scripts/grammar_coverage.py results/ifcbench/runs/*/episodes.jsonl
    python scripts/grammar_coverage.py RUN/episodes.jsonl --state base --out coverage.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.constraints.cfg_grammar import check_reply, grammar_from_summary
from src.etl.manifest import cache_path, load_manifest


def executed_queries(paths):
    """Distinct (model_id, cypher) pairs that ran without a Neo4j error."""
    seen = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        for rec in records:
            for step in rec["episode"]["steps"]:
                result = step.get("result") or {}
                if step.get("cypher") and not result.get("error"):
                    seen.setdefault((rec["model_id"], step["cypher"]), Path(path).parent.name)
    return seen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("episodes", nargs="+", type=Path)
    p.add_argument("--state", default="base")
    p.add_argument("--manifest", type=Path, default=get_settings().ifcbench_manifest_path)
    p.add_argument("--out", type=Path, help="Write every verdict as JSON")
    args = p.parse_args(argv)

    manifest = load_manifest(args.manifest)
    entries = {m["model_id"]: m for m in manifest["models"]}
    grammars = {}
    verdicts = []
    for (model_id, cypher), run in sorted(executed_queries(args.episodes).items()):
        if model_id not in grammars:
            with gzip.open(cache_path(manifest, entries[model_id], args.state), "rt") as f:
                grammars[model_id] = grammar_from_summary(json.loads(f.readline())["summary"])
        one_line = re.sub(r"\s+", " ", cypher).strip()
        reply = json.dumps({"action": "cypher", "thoughts": "", "cypher": one_line})
        verdicts.append({"model_id": model_id, "run": run, "cypher": one_line,
                         "rejection": check_reply(grammars[model_id], reply)})

    rejected = [v for v in verdicts if v["rejection"]]
    n = len(verdicts)
    print(f"accepted {n - len(rejected)}/{n} = {(n - len(rejected)) / n:.1%}" if n else "no queries")
    for model_id, count in sorted(Counter(v["model_id"] for v in rejected).items()):
        print(f"  rejected on {model_id}: {count}")
    for v in rejected:
        print(f"\n[{v['model_id']}] {v['rejection']}\n  {v['cypher'][:300]}")
    if args.out:
        args.out.write_text(json.dumps(verdicts, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
