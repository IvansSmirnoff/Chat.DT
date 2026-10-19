#!/usr/bin/env python3
"""Download IFC-Bench question sets and IFC models from the HuggingFace Hub.

Why this exists
---------------
IFC-Bench (Hellin et al., Automation in Construction 192, 2026) is the one
public BIM-QA benchmark with a published agentic baseline: 1,026 questions over
31 IFC files, a 514-question test split, and free-text ground truth. Running
the graph method on it means loading every file it targets, so the download is
a reproducibility step, not a convenience: the dataset is still being revised
(v2.0.3 at the time of writing), and a result is only comparable to Hellin's
if it was produced against the same bytes.

So this script
  * pins the dataset **commit sha**, not ``main``, and records it;
  * verifies every LFS file against the sha256 the Hub publishes for it;
  * writes atomically (``.part`` then rename), so an interrupted run never
    leaves a truncated IFC that ifcopenshell would half-parse.

No ``huggingface_hub`` dependency: two public HTTPS endpoints are enough.

Usage
-----
    # question CSVs only (small; always fetched)
    python scripts/ifcbench/fetch.py

    # plus every IFC file the published test split asks questions about
    python scripts/ifcbench/fetch.py --split test

    # plus specific projects (all of their IFC files)
    python scripts/ifcbench/fetch.py --project dental_clinic

Files land in ``data/ifcbench/raw/`` mirroring the dataset layout, with
``_fetch_record.json`` holding the commit sha and per-file sha256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("ifcbench.fetch")

REPO = "sylvainHellin/ifc-bench"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve"
QUESTION_DIR = "questions/"
QUESTIONS = "questions/ifc-bench-v2.csv"
EVAL_SPLIT = "questions/eval-split-hellin2026.csv"
DEFAULT_OUT = Path("data/ifcbench/raw")
CHUNK = 1 << 20


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def resolve_revision(revision: str) -> str:
    """Turn a branch name into the commit sha it currently points at."""
    info = _get_json(f"{API}/revision/{revision}")
    return info["sha"]


def list_files(sha: str) -> List[Dict[str, Any]]:
    """Every file in the dataset at ``sha``, with size and LFS sha256 if any."""
    entries = _get_json(f"{API}/tree/{sha}?recursive=true")
    return [e for e in entries if e.get("type") == "file"]


def ifc_path(project: str, ifc_model: str) -> str:
    """Dataset path of the IFC file a question targets.

    The CSV's ``ifc_model`` column is a discipline name (``arc``, ``str``,
    ``heating``...), not a file name; the file is ``projects/<p>/<model>.ifc``.
    """
    return f"projects/{project}/{ifc_model}.ifc"


def ifc_paths_for_split(questions_csv: Path, split_csv: Path, split: str) -> Set[str]:
    """IFC files that at least one question of ``split`` targets.

    The split CSV carries only ``id, split``; the target file comes from the
    question CSV, joined on ``id``.
    """
    with open(split_csv, newline="", encoding="utf-8") as f:
        ids = {row["id"] for row in csv.DictReader(f) if split == "all" or row["split"] == split}
    with open(questions_csv, newline="", encoding="utf-8") as f:
        return {
            ifc_path(row["project"], row["ifc_model"])
            for row in csv.DictReader(f)
            if row["id"] in ids
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(sha: str, entry: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Fetch one file unless an identical copy is already on disk.

    Identity is the LFS sha256 where the Hub publishes one, and the byte size
    otherwise (small text files are stored in git, not LFS).
    """
    rel = entry["path"]
    target = out_dir / rel
    expected = (entry.get("lfs") or {}).get("oid")
    size = entry.get("size")

    if target.exists():
        if expected and sha256_of(target) == expected:
            return {"path": rel, "sha256": expected, "size": size, "status": "cached"}
        if not expected and target.stat().st_size == size:
            return {"path": rel, "sha256": sha256_of(target), "size": size, "status": "cached"}

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    url = f"{RESOLVE}/{sha}/{urllib.request.quote(rel)}"
    logger.info("GET %s (%.1f MB)", rel, (size or 0) / 1e6)
    with urllib.request.urlopen(url, timeout=300) as resp, open(part, "wb") as f:
        for block in iter(lambda: resp.read(CHUNK), b""):
            digest.update(block)
            f.write(block)

    actual = digest.hexdigest()
    if expected and actual != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {rel}: expected {expected}, got {actual}")
    if size is not None and part.stat().st_size != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {rel}: expected {size}, got {part.stat().st_size}")
    part.replace(target)
    return {"path": rel, "sha256": actual, "size": size, "status": "downloaded"}


def select(
    entries: Iterable[Dict[str, Any]],
    wanted_ifc: Set[str],
    projects: Set[str],
) -> List[Dict[str, Any]]:
    """Question CSVs always; IFC files if named by split or by project."""
    chosen = []
    for e in entries:
        path = e["path"]
        parts = path.split("/")
        project = parts[1] if len(parts) == 3 and parts[0] == "projects" else None
        if path.startswith(QUESTION_DIR) and path.endswith(".csv"):
            chosen.append(e)
        elif path.endswith(".ifc") and (path in wanted_ifc or project in projects):
            chosen.append(e)
    return chosen


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--revision", default="main",
                        help="Branch or commit to pin (resolved to a sha). Default: main")
    parser.add_argument("--split", choices=["test", "train", "all"], default=None,
                        help="Also fetch every IFC file this split of the eval CSV targets")
    parser.add_argument("--project", action="append", default=[],
                        help="Also fetch all IFC files of this project (repeatable)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sha = resolve_revision(args.revision)
    logger.info("%s @ %s (%s)", REPO, sha, args.revision)
    entries = list_files(sha)
    by_path = {e["path"]: e for e in entries}

    # The two question CSVs decide which IFC files are needed, so they go first.
    records = [download(sha, by_path[p], args.out) for p in (QUESTIONS, EVAL_SPLIT)]
    wanted: Set[str] = set()
    if args.split:
        wanted = ifc_paths_for_split(args.out / QUESTIONS, args.out / EVAL_SPLIT, args.split)
        missing = sorted(p for p in wanted if p not in by_path)
        if missing:
            raise RuntimeError(f"Questions target files absent from the dataset: {missing}")

    for entry in select(entries, wanted, set(args.project)):
        if entry["path"] not in (QUESTIONS, EVAL_SPLIT):
            records.append(download(sha, entry, args.out))

    record_path = args.out / "_fetch_record.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    if previous.get("commit") not in (None, sha):
        logger.warning("Dataset commit changed: %s -> %s", previous.get("commit"), sha)
    files = {r["path"]: r for r in previous.get("files", [])} if previous.get("commit") == sha else {}
    files.update({r["path"]: {k: r[k] for k in ("path", "sha256", "size")} for r in records})
    # Every IFC file in the dataset, downloaded or not, so the manifest can be
    # built offline and say what a missing file *should* hash to.
    ifc_index = {
        e["path"]: {"sha256": (e.get("lfs") or {}).get("oid"), "size": e.get("size")}
        for e in entries if e["path"].endswith(".ifc")
    }
    record_path.write_text(json.dumps(
        {"repo": REPO, "revision": args.revision, "commit": sha,
         "files": sorted(files.values(), key=lambda r: r["path"]),
         "ifc_index": dict(sorted(ifc_index.items()))},
        indent=2,
    ))
    fetched = sum(r["status"] == "downloaded" for r in records)
    logger.info("%d files (%d downloaded, %d cached) -> %s",
                len(records), fetched, len(records) - fetched, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
