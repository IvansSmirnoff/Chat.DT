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
from src.constraints.value_sampler import (
    generate_few_shot_examples,
    sample_value_enumerations,
)
from src.constraints.vocabulary_merger import (
    build_combined_vocabulary,
    merge_graph_stats_into_vocabulary,
)
from src.eval.neo4j_exec import collect_graph_stats
from scripts.create_model_dump import export_from_ifc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Neo4j driver emits notification messages at WARNING (deprecated procedure
# output, dead property keys we already filter, etc.). They are not errors
# and clutter the bundle-build log; downgrade to ERROR so only real driver
# failures surface.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


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


def _open_driver(uri: str, user: str, password: str):
    """Open and verify a Neo4j driver, raising on failure.

    The bundle now depends on live graph data (value enumerations +
    few-shot examples) for prompt context, so an empty fallback would
    silently weaken the model. We hard-fail instead.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver is required to build a bundle: pip install neo4j"
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


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

    logger.info(f"Connecting to Neo4j at {neo4j_uri}")
    driver = _open_driver(neo4j_uri, neo4j_user, neo4j_password)
    try:
        logger.info("Collecting graph stats")
        graph_stats = collect_graph_stats(driver)

        # Must run before value sampling / few-shot generation so both see the
        # full label set.
        merge_graph_stats_into_vocabulary(vocab, graph_stats)

        logger.info("Sampling value enumerations from graph")
        value_enumerations = sample_value_enumerations(driver, vocab)

        logger.info("Generating graph-sourced few-shot examples")
        few_shot_examples = generate_few_shot_examples(
            driver, vocab, enumerations=value_enumerations,
        )
    finally:
        driver.close()

    return {
        "schema_version": 2,
        "source": {
            "ifc_file": ifc_path.name,
            "ids_file": ids_path.name if ids_path is not None else None,
        },
        "combined_vocabulary": vocab,
        "ids_schema": ids_schema,
        "model_dump": model_dump,
        "graph_stats": graph_stats,
        "value_enumerations": value_enumerations,
        "few_shot_examples": few_shot_examples,
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
        f"graph_stats ({len(bundle['graph_stats'].get('labels', {}))} labels), "
        f"value_enumerations ({len(bundle['value_enumerations'])} pairs), "
        f"few_shot_examples ({len(bundle['few_shot_examples'])})"
    )


if __name__ == "__main__":
    main()
