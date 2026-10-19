"""Read the IFC-Bench run manifest written by ``scripts/ifcbench/build_testset.py``.

Paths inside the manifest are relative to the repository root (``data/...``),
which is two directories above the manifest file itself, so they resolve the
same way on the host, in the ``app`` container and in the ``api`` container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = yaml.safe_load(Path(path).read_text())
    manifest["_root"] = Path(path).resolve().parent.parent.parent
    return manifest


def model_entry(manifest: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for entry in manifest["models"]:
        if entry["model_id"] == model_id:
            return entry
    raise KeyError(f"{model_id!r} is not in the manifest")


def resolve(manifest: Dict[str, Any], rel: str) -> Path:
    return manifest["_root"] / rel


def cache_path(manifest: Dict[str, Any], entry: Dict[str, Any], state: str = "base") -> Path:
    """Graph snapshot for one graph state (``base``, ``topology``, ...).

    The manifest records the ``base`` path; other states sit beside it as
    ``<model_id>.<state>.jsonl.gz``.
    """
    base = resolve(manifest, entry["paths"]["graph_cache"])
    return base.with_name(f"{entry['model_id']}.{state}.jsonl.gz")
