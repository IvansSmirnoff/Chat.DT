#!/usr/bin/env python3
"""
Build a server→Colab bundle artifact.

Produces a single ``bundle_<model>.json`` with everything the Colab-side LLM
runner needs to operate without mounting the server's filesystem:

- ``combined_vocabulary``: Super-schema from IDS + IFC merge
- ``ids_schema``: Parsed IDS constraints
- ``model_dump``: Flat element dump for Direct QA settings
- ``graph_stats``: Label/relationship counts from Neo4j (for prompt context)

No HTTP API — the Colab notebook loads this file plus a Bolt URL and is
self-sufficient from there.

Usage:
    python scripts/build_bundle.py \\
        --ifc /app/data/Barcelona.ifc \\
        --ids /app/data/requirements.ids \\
        --out /app/data/bundle_barcelona.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.constraints.ids_parser import parse_ids_file
from src.constraints.vocabulary_merger import build_combined_vocabulary
from scripts.create_model_dump import export_from_ifc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _BundleEncoder(json.JSONEncoder):
    """JSON encoder that handles sets and dataclasses used in the bundle."""

    def default(self, o: Any):
        if isinstance(o, set):
            return sorted(o)
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def collect_graph_stats(uri: str, user: str, password: str) -> Dict[str, Any]:
    """Query Neo4j for node/relationship counts per label and type.

    Returns empty dict on connection failure so bundle build still succeeds
    when Neo4j isn't running (useful for dev iteration).
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j driver not installed, skipping graph_stats")
        return {}

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as exc:
        logger.warning(f"Neo4j unreachable ({exc}), skipping graph_stats")
        return {}

    stats: Dict[str, Any] = {"labels": {}, "relationships": {}, "totals": {}}
    with driver.session() as session:
        label_result = session.run("CALL db.labels() YIELD label RETURN label")
        for record in label_result:
            label = record["label"]
            count = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]
            stats["labels"][label] = count

        rel_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        for record in rel_result:
            rtype = record["relationshipType"]
            count = session.run(f"MATCH ()-[r:`{rtype}`]->() RETURN count(r) AS c").single()["c"]
            stats["relationships"][rtype] = count

        stats["totals"]["nodes"] = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        stats["totals"]["relationships"] = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]

    driver.close()
    return stats


def build_bundle(
    ifc_path: Path,
    ids_path: Optional[Path],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    include_model_dump: bool = True,
) -> Dict[str, Any]:
    """Assemble the bundle dict. IDS is optional — when omitted, the vocabulary
    is built from IFC alone and ``ids_schema`` is ``None`` (Setting 2 / Setting 4
    fall back to soft constraints because there is no enforced vocabulary)."""
    if ids_path is not None:
        logger.info(f"Building vocabulary from {ids_path.name} + {ifc_path.name}")
    else:
        logger.info(f"Building vocabulary from {ifc_path.name} (no IDS)")
    vocab = build_combined_vocabulary(ids_path=ids_path, ifc_path=ifc_path)

    ids_schema = None
    if ids_path is not None:
        logger.info(f"Parsing IDS: {ids_path}")
        ids_schema = parse_ids_file(ids_path)

    model_dump = []
    if include_model_dump:
        logger.info(f"Exporting model dump from {ifc_path}")
        model_dump = export_from_ifc(ifc_path=ifc_path)

    logger.info(f"Collecting graph stats from {neo4j_uri}")
    graph_stats = collect_graph_stats(neo4j_uri, neo4j_user, neo4j_password)

    return {
        "schema_version": 1,
        "source": {
            "ifc_file": ifc_path.name,
            "ids_file": ids_path.name if ids_path is not None else None,
        },
        "combined_vocabulary": vocab,
        "ids_schema": ids_schema,
        "model_dump": model_dump,
        "graph_stats": graph_stats,
        "neo4j": {
            "uri_hint": neo4j_uri,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a server→Colab bundle (vocab + IDS + model dump + graph stats)",
    )
    parser.add_argument("--ifc", type=Path, help="Path to IFC file (default: from settings)")
    parser.add_argument("--ids", type=Path, help="Path to IDS file (default: from settings)")
    parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Force-skip IDS even if the settings default points at an existing file.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output bundle JSON path")
    parser.add_argument("--no-model-dump", action="store_true", help="Skip model dump (smaller bundle)")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON (larger file)")
    args = parser.parse_args()

    settings = get_settings()
    ifc_path = args.ifc or Path(settings.ifc_file_path)

    if not ifc_path.exists():
        logger.error(f"IFC file not found: {ifc_path}")
        sys.exit(1)

    # IDS is optional. Resolve from CLI > settings; treat empty/missing/--no-ids as "no IDS".
    ids_path: Optional[Path] = None
    if args.no_ids:
        if args.ids is not None:
            logger.error("--no-ids and --ids cannot be combined.")
            sys.exit(2)
        logger.info("--no-ids set, building bundle without IDS.")
    else:
        ids_candidate: Optional[Path]
        if args.ids is not None:
            ids_candidate = args.ids
        elif settings.ids_file_path and str(settings.ids_file_path).strip():
            ids_candidate = Path(settings.ids_file_path)
        else:
            ids_candidate = None

        if ids_candidate is not None:
            if ids_candidate.exists():
                ids_path = ids_candidate
            elif args.ids is not None:
                logger.error(f"IDS file not found: {ids_candidate}")
                sys.exit(1)
            else:
                logger.warning(
                    f"IDS path from settings does not exist ({ids_candidate}); "
                    "building bundle without IDS."
                )

    bundle = build_bundle(
        ifc_path=ifc_path,
        ids_path=ids_path,
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password_value,
        include_model_dump=not args.no_model_dump,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            bundle,
            f,
            cls=_BundleEncoder,
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )

    size_kb = args.out.stat().st_size / 1024
    logger.info(f"Bundle written: {args.out} ({size_kb:.1f} KB)")
    ids_count = (
        len(bundle["ids_schema"].entities) if bundle["ids_schema"] is not None else 0
    )
    logger.info(
        f"Contents: vocabulary ({len(bundle['combined_vocabulary'].entities)} entities), "
        f"IDS ({ids_count} entities), "
        f"model_dump ({len(bundle['model_dump'])} elements), "
        f"graph_stats ({len(bundle['graph_stats'].get('labels', {}))} labels)"
    )


if __name__ == "__main__":
    main()
