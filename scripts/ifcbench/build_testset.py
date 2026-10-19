#!/usr/bin/env python3
"""Build the IFC-Bench run manifest and one question slice per IFC file.

Why this exists
---------------
The graph method answers questions against *one* loaded graph, and IFC-Bench
questions each target one file (``project`` + ``ifc_model``). So a run is a
loop over files -- restore that file's graph, run its questions, move on -- and
something has to say, per file, where its bytes, graph cache, bundle and
questions live. That is the manifest. It is generated, not hand-edited, so it
cannot drift from the question CSVs it was built from.

Each file gets a ``model_id`` of ``<project>__<ifc_model>`` (``dental_clinic__arc``).
Discipline files of one project are separate models, exactly as Hellin's agent
sees them: a question about ``str.ifc`` is asked of the structural file alone.

Inputs (from ``fetch.py``): ``data/ifcbench/raw/questions/ifc-bench-v2.csv``,
``.../eval-split-hellin2026.csv`` and ``.../_fetch_record.json``.

Outputs:
  * ``data/ifcbench/manifest.yaml``
  * ``data/ifcbench/testsets/<model_id>.jsonl`` -- one record per question:
    ``id, question, ground_truth, category, split, project, ifc_model``.

Usage
-----
    python scripts/ifcbench/build_testset.py              # every file with questions
    python scripts/ifcbench/build_testset.py --split test # only files the test split targets
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path("data/ifcbench")
RAW = ROOT / "raw"
QUESTIONS = RAW / "questions/ifc-bench-v2.csv"
EVAL_SPLIT = RAW / "questions/eval-split-hellin2026.csv"
FETCH_RECORD = RAW / "_fetch_record.json"

CATEGORY_NAMES = {
    "1": "direct_retrieval",
    "2": "aggregation",
    "3": "geometric_spatial",
    "4": "incomplete_information",
}

_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", re.IGNORECASE)
_FILE_NAME_RE = re.compile(r"FILE_NAME\s*\(", re.IGNORECASE)


def model_id(project: str, ifc_model: str) -> str:
    return f"{project}__{ifc_model}"


def split_step_args(text: str) -> List[str]:
    """Split a STEP argument list at top-level commas, up to its closing paren.

    Commas inside quoted strings (``''`` is an escaped quote) and inside nested
    lists do not split, so ``(''),('A','B')`` stays two arguments.
    """
    args: List[str] = []
    depth, start, i, in_str = 0, 0, 0, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "'":
                if text[i + 1:i + 2] == "'":
                    i += 1
                else:
                    in_str = False
        elif c == "'":
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                args.append(text[start:i].strip())
                return args
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    return args


def _unquote(arg: str) -> Optional[str]:
    if len(arg) >= 2 and arg[0] == arg[-1] == "'":
        return arg[1:-1].replace("''", "'") or None
    return None


def read_header(path: Path, limit: int = 16384) -> Dict[str, Optional[str]]:
    """Schema and authoring fields from the STEP header, without parsing the file.

    ``FILE_NAME`` is ``(name, time_stamp, (author), (organization),
    preprocessor_version, originating_system, authorization)``. Both tool fields
    are recorded verbatim because exporters do not agree on them: the dental
    clinic carries ``Autodesk Revit Architecture 2011`` as the *preprocessor*
    and Solibri's optimizer as the *originating system*. A header that does not
    parse is reported as ``None`` rather than guessed.
    """
    head = path.read_bytes()[:limit].decode("latin-1", errors="replace")
    schema = _SCHEMA_RE.search(head)
    preprocessor = originating = None
    match = _FILE_NAME_RE.search(head)
    if match:
        args = split_step_args(head[match.end():])
        if len(args) >= 6:
            preprocessor, originating = _unquote(args[4]), _unquote(args[5])
    return {
        "schema": schema.group(1) if schema else None,
        "preprocessor_version": preprocessor,
        "originating_system": originating,
    }


def load_questions() -> List[Dict[str, str]]:
    with open(EVAL_SPLIT, newline="", encoding="utf-8") as f:
        split_of = {row["id"]: row["split"] for row in csv.DictReader(f)}
    with open(QUESTIONS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        # Every question should have a split row; one without is kept out of
        # both splits rather than guessed into either.
        row["split"] = split_of.get(row["id"], "excluded")
    return rows


def build(split: Optional[str]) -> Dict[str, Any]:
    record = json.loads(FETCH_RECORD.read_text())
    index: Dict[str, Dict[str, Any]] = record.get("ifc_index", {})
    questions = load_questions()

    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for q in questions:
        by_model[model_id(q["project"], q["ifc_model"])].append(q)

    testsets = ROOT / "testsets"
    testsets.mkdir(parents=True, exist_ok=True)
    models = []
    for mid in sorted(by_model):
        rows = by_model[mid]
        if split and not any(r["split"] == split for r in rows):
            continue
        project, ifc_model = rows[0]["project"], rows[0]["ifc_model"]
        hf_path = f"projects/{project}/{ifc_model}.ifc"
        raw_path = RAW / hf_path
        header = read_header(raw_path) if raw_path.exists() else {
            "schema": None, "preprocessor_version": None, "originating_system": None,
        }
        entry = index.get(hf_path, {})

        slice_path = testsets / f"{mid}.jsonl"
        with open(slice_path, "w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: int(r["id"])):
                f.write(json.dumps({
                    "id": int(r["id"]),
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "category": int(r["category"]),
                    "split": r["split"],
                    "project": project,
                    "ifc_model": ifc_model,
                }, ensure_ascii=False) + "\n")

        splits = Counter(r["split"] for r in rows)
        categories = Counter(CATEGORY_NAMES.get(r["category"], r["category"]) for r in rows)
        models.append({
            "model_id": mid,
            "project": project,
            "ifc_model": ifc_model,
            "hf_path": hf_path,
            "sha256": entry.get("sha256"),
            "size_bytes": entry.get("size"),
            "schema": header["schema"],
            "preprocessor_version": header["preprocessor_version"],
            "originating_system": header["originating_system"],
            "questions": {"test": splits.get("test", 0), "train": splits.get("train", 0),
                          "excluded": splits.get("excluded", 0)},
            "categories": dict(sorted(categories.items())),
            "paths": {
                "ifc": str(raw_path),
                "testset": str(slice_path),
                "graph_cache": str(ROOT / "cache" / f"{mid}.base.jsonl.gz"),
                "bundle": str(ROOT / "bundles" / f"{mid}.base.json"),
            },
        })

    if split:
        missing = [m["hf_path"] for m in models if m["sha256"] is None]
        if missing:
            raise RuntimeError(f"Not in the dataset index (re-run fetch.py): {missing}")

    return {
        "dataset": {"repo": record["repo"], "commit": record["commit"]},
        "selection": split or "all",
        "totals": {
            "models": len(models),
            "questions_test": sum(m["questions"]["test"] for m in models),
            "questions_train": sum(m["questions"]["train"] for m in models),
            "bytes": sum(m["size_bytes"] or 0 for m in models),
        },
        "models": models,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["test", "train"], default=None,
                        help="Keep only files with at least one question in this split")
    parser.add_argument("--out", type=Path, default=ROOT / "manifest.yaml")
    args = parser.parse_args(argv)

    manifest = build(args.split)
    args.out.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    t = manifest["totals"]
    print(f"{t['models']} models, {t['questions_test']} test / {t['questions_train']} train "
          f"questions, {t['bytes'] / 1e9:.2f} GB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
