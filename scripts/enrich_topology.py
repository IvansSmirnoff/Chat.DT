#!/usr/bin/env python3
"""Add or remove the topology enrichment on the loaded graph.

Why this exists
---------------
The enrichment (``src/etl/topology.py``) is an ablation step: the same model
is asked the same questions with and without it. So it has to be applied to a
loaded base graph, exported as its own snapshot (``<model_id>.topology.jsonl.gz``)
and removed again without a trace -- ``--remove`` returns the graph to its base
fingerprint. ``python -m src.agent.run --state topology`` builds that snapshot
itself when it is missing; this script is the manual path and the dry run.

Usage
-----
    # what would be written, from the IFC alone (no Neo4j)
    python scripts/enrich_topology.py --dry-run --model-id dental_clinic__arc
    # the same from geometry alone, as if the IFC had no space boundaries
    # (how the geometry fallback is validated on the models that have them)
    python scripts/enrich_topology.py --dry-run --ignore-boundaries --model-id dental_clinic__arc
    # enrich the loaded graph (it must be that model's base graph) and export the snapshot
    python scripts/enrich_topology.py --apply --model-id dental_clinic__arc --export
    # back to base
    python scripts/enrich_topology.py --remove
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.etl.manifest import cache_path, load_manifest, resolve
from src.etl.topology import (
    ENRICHMENT,
    apply_topology,
    compute_topology,
    remove_topology,
)


def main(argv=None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Compute and report only")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--remove", action="store_true")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model-id", help="Manifest model_id (IFC and snapshot paths from the manifest)")
    src.add_argument("--ifc", type=Path)
    p.add_argument("--manifest", type=Path, default=settings.ifcbench_manifest_path)
    p.add_argument("--export", action="store_true",
                   help="After --apply, export the topology snapshot (needs --model-id)")
    p.add_argument("--report", type=Path, help="Write the computation stats as JSON")
    p.add_argument("--ignore-boundaries", action="store_true",
                   help="With --dry-run: compute from geometry alone, as if the IFC had no space boundaries")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    entry = manifest = None
    ifc = args.ifc
    if args.model_id:
        manifest = load_manifest(args.manifest)
        entry = next((m for m in manifest["models"] if m["model_id"] == args.model_id), None)
        if entry is None:
            p.error(f"not in manifest: {args.model_id}")
        ifc = resolve(manifest, entry["paths"]["ifc"])
    if (args.dry_run or args.apply) and ifc is None:
        p.error("--dry-run/--apply need --model-id or --ifc")
    if args.export and entry is None:
        p.error("--export needs --model-id")
    if args.ignore_boundaries and not args.dry_run:
        p.error("--ignore-boundaries is a validation mode: --dry-run only")

    if args.dry_run:
        topo = compute_topology(ifc, use_boundaries=not args.ignore_boundaries)
        print(json.dumps(topo.stats, indent=2))
        if args.report:
            args.report.write_text(json.dumps(topo.stats, indent=2))
        return 0

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password_value))
    try:
        if args.remove:
            print(json.dumps(remove_topology(driver), indent=2))
            return 0
        topo = compute_topology(ifc)
        written = apply_topology(driver, topo)
        print(json.dumps({"stats": topo.stats, "written": written}, indent=2))
        if args.report:
            args.report.write_text(json.dumps({"stats": topo.stats, "written": written}, indent=2))
        if args.export:
            from src.etl.graph_cache import export_with_meta

            out = cache_path(manifest, entry, ENRICHMENT)
            header = export_with_meta(driver, out, entry["model_id"], ENRICHMENT, ifc)
            print(f"exported {out} ({header['fingerprint'][:12]})")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
