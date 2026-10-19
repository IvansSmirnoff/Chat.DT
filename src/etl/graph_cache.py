"""Export a loaded graph to a portable file, and restore it.

Why this exists
---------------
IFC-Bench asks questions of 31 IFC files, and Neo4j Community holds one user
database. Keeping every model in that database behind a ``model_id`` property
does not work: the ``IfcRoot.GlobalId`` uniqueness constraint breaks on GUIDs
that federated discipline exports reuse, material nodes are keyed by name and
would merge across models, and every generated query would need a model filter
-- which the grammar, the prompt and SCR would all have to know about.

So a run holds **one model at a time**: load it once with the ETL, export the
graph here, and from then on *restore* it in seconds instead of re-parsing the
IFC (the ETL is the slow part; ~7 min for the 157 MB merged clinic).

The export is taken from the live graph, not from the loader, so it captures
whatever state the graph is in -- base, enriched, deduplicated -- and nothing
here has to change when the loader learns a new edge type.

Format (``chatdt-graph/1``, gzip JSON lines):

    {"format": ..., "meta": {...}, "summary": {...}, "fingerprint": "..."}
    {"n": 0, "l": ["IfcWall", ..., "IfcRoot"], "p": {...}}        # every node
    {"r": [0, 7], "t": "CONTAINS", "p": {...}}                     # every edge

Node ids are export-local integers. A ``(:_GraphMeta)`` node records which
model is loaded; it is excluded from the export, the summary and the
vocabulary (any label starting with ``_`` is internal).

The **fingerprint** is a sha256 over label counts, relationship-type counts,
totals and the property keys in use. A bundle built against one graph state
carries it, so a client can refuse to run against a different one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.eval.neo4j_exec import collect_label_and_type_counts, collect_property_keys_in_use

logger = logging.getLogger(__name__)

FORMAT = "chatdt-graph/1"
META_LABEL = "_GraphMeta"
RESTORE_LABEL = "_Restore"
RESTORE_KEY = "_cid"
RESTORE_INDEX = "chatdt_restore_cid"
BATCH_SIZE = 5000

# Files whose content defines how an IFC becomes a graph. Their hash is the
# loader revision recorded in the metadata, so a cache built by an older loader
# is identifiable without git (the api container has no .git).
LOADER_SOURCES = (
    "src/etl/loader.py",
    "src/utils/ifc_edges.py",
    "src/utils/ifc_attributes.py",
    "src/utils/property_names.py",
)


# -----------------------------------------------------------------------------
# Summary and fingerprint
# -----------------------------------------------------------------------------

def graph_summary(driver) -> Dict[str, Any]:
    """Counts and keys that identify a graph state, excluding internal nodes."""
    labels, rel_types = collect_label_and_type_counts(driver)
    with driver.session() as session:
        totals = session.run(
            f"MATCH (n) WHERE NOT n:`{META_LABEL}` "
            "RETURN count(n) AS nodes, sum(size(keys(n))) AS node_properties"
        ).single()
        rels = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS rels, sum(size(keys(r))) AS rel_properties"
        ).single()
    return {
        "labels": dict(sorted(labels.items())),
        "relationships": dict(sorted(rel_types.items())),
        "totals": {
            "nodes": totals["nodes"],
            "relationships": rels["rels"],
            "node_properties": totals["node_properties"] or 0,
            "relationship_properties": rels["rel_properties"] or 0,
        },
        "property_keys": collect_property_keys_in_use(driver),
    }


def fingerprint(summary: Dict[str, Any]) -> str:
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def loader_revision(root: Path = Path(".")) -> Optional[str]:
    digest = hashlib.sha256()
    found = False
    for rel in LOADER_SOURCES:
        path = root / rel
        if path.exists():
            digest.update(rel.encode())
            digest.update(path.read_bytes())
            found = True
    return digest.hexdigest()[:12] if found else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Metadata node
# -----------------------------------------------------------------------------

def read_meta(driver) -> Optional[Dict[str, Any]]:
    with driver.session() as session:
        record = session.run(f"MATCH (m:`{META_LABEL}`) RETURN properties(m) AS p").single()
    return dict(record["p"]) if record else None


def write_meta(driver, meta: Dict[str, Any]) -> None:
    """Replace the metadata node. Values must be Neo4j primitives."""
    with driver.session() as session:
        session.run(f"MATCH (m:`{META_LABEL}`) DETACH DELETE m").consume()
        session.run(f"CREATE (m:`{META_LABEL}`) SET m = $meta", meta=meta).consume()


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

def _dumps(obj: Any) -> str:
    # No default=: a value that is not a JSON primitive (a temporal or spatial
    # type the loader does not write today) must fail the export, not be
    # silently stringified and restored as a different type.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def export_graph(driver, out_path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Write the live graph to ``out_path``; return the header."""
    summary = graph_summary(driver)
    header = {
        "format": FORMAT,
        "meta": meta,
        "summary": summary,
        "fingerprint": fingerprint(summary),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")

    ids: Dict[str, int] = {}
    n_rels = 0
    with gzip.open(part, "wt", encoding="utf-8") as f:
        f.write(_dumps(header) + "\n")
        with driver.session() as session:
            for record in session.run(
                f"MATCH (n) WHERE NOT n:`{META_LABEL}` "
                "RETURN elementId(n) AS id, labels(n) AS l, properties(n) AS p"
            ):
                ids[record["id"]] = len(ids)
                f.write(_dumps({"n": ids[record["id"]], "l": record["l"], "p": record["p"]}) + "\n")
            for record in session.run(
                f"MATCH (a)-[r]->(b) WHERE NOT a:`{META_LABEL}` AND NOT b:`{META_LABEL}` "
                "RETURN elementId(a) AS s, elementId(b) AS t, type(r) AS type, properties(r) AS p"
            ):
                f.write(_dumps({"r": [ids[record["s"]], ids[record["t"]]],
                                "t": record["type"], "p": record["p"]}) + "\n")
                n_rels += 1

    if len(ids) != summary["totals"]["nodes"] or n_rels != summary["totals"]["relationships"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Graph changed during export: summary {summary['totals']}, "
            f"wrote {len(ids)} nodes / {n_rels} relationships"
        )
    part.replace(out_path)
    logger.info("Exported %d nodes, %d relationships -> %s (fingerprint %s)",
                len(ids), n_rels, out_path, header["fingerprint"][:12])
    return header


# -----------------------------------------------------------------------------
# Restore
# -----------------------------------------------------------------------------

def read_header(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported format {header.get('format')!r}")
    return header


def _records(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            yield json.loads(line)


def _flush_nodes(session, groups: Dict[Tuple[str, ...], List[Dict[str, Any]]]) -> None:
    for labels, rows in groups.items():
        label_clause = ":".join(f"`{l}`" for l in (*labels, RESTORE_LABEL))
        session.run(
            f"UNWIND $rows AS r CREATE (n:{label_clause}) SET n = r.p, n.`{RESTORE_KEY}` = r.n",
            rows=rows,
        ).consume()
    groups.clear()


def _flush_rels(session, groups: Dict[str, List[Dict[str, Any]]]) -> int:
    created = 0
    for rtype, rows in groups.items():
        created += session.run(
            f"UNWIND $rows AS r "
            f"MATCH (a:`{RESTORE_LABEL}` {{`{RESTORE_KEY}`: r.s}}) "
            f"MATCH (b:`{RESTORE_LABEL}` {{`{RESTORE_KEY}`: r.t}}) "
            f"CREATE (a)-[x:`{rtype}`]->(b) SET x = r.p",
            rows=rows,
        ).consume().counters.relationships_created
    groups.clear()
    return created


def restore_graph(driver, path: Path, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    """Wipe the database and rebuild it from an export; verify the fingerprint.

    Raises if the restored graph does not reproduce the exported fingerprint,
    leaving the (wrong) graph in place and no metadata node, so nothing can
    mistake it for the model it was meant to be.
    """
    from src.etl.loader import IFCToNeo4jLoader

    path = Path(path)
    header = read_header(path)

    # Reuse the loader's wipe and constraints so a restored graph is indexed
    # exactly like a freshly loaded one.
    loader = IFCToNeo4jLoader()
    loader.driver = driver
    loader.wipe_database()
    loader.create_constraints()

    with driver.session() as session:
        session.run(
            f"CREATE INDEX {RESTORE_INDEX} IF NOT EXISTS "
            f"FOR (n:`{RESTORE_LABEL}`) ON (n.`{RESTORE_KEY}`)"
        ).consume()
        session.run("CALL db.awaitIndexes(300)").consume()

        node_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        rel_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        pending = nodes = rels = 0
        for rec in _records(path):
            if "n" in rec:
                node_groups[tuple(rec["l"])].append({"n": rec["n"], "p": rec["p"]})
                nodes += 1
            else:
                if node_groups:
                    _flush_nodes(session, node_groups)
                    pending = 0
                rel_groups[rec["t"]].append({"s": rec["r"][0], "t": rec["r"][1], "p": rec["p"]})
            pending += 1
            if pending >= batch_size:
                if node_groups:
                    _flush_nodes(session, node_groups)
                if rel_groups:
                    rels += _flush_rels(session, rel_groups)
                pending = 0
        if node_groups:
            _flush_nodes(session, node_groups)
        if rel_groups:
            rels += _flush_rels(session, rel_groups)

        session.run(
            f"MATCH (n:`{RESTORE_LABEL}`) "
            f"CALL {{ WITH n REMOVE n:`{RESTORE_LABEL}`, n.`{RESTORE_KEY}` }} "
            f"IN TRANSACTIONS OF 10000 ROWS"
        ).consume()
        session.run(f"DROP INDEX {RESTORE_INDEX} IF EXISTS").consume()

    actual = fingerprint(graph_summary(driver))
    if actual != header["fingerprint"]:
        raise RuntimeError(
            f"Restore of {path} did not reproduce the graph: fingerprint "
            f"{actual[:12]} != exported {header['fingerprint'][:12]}"
        )
    write_meta(driver, {**header["meta"], "fingerprint": actual, "cache_path": str(path)})
    logger.info("Restored %d nodes, %d relationships from %s (fingerprint %s)",
                nodes, rels, path, actual[:12])
    return header


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from neo4j import GraphDatabase

    from src.config import get_settings

    parser = argparse.ArgumentParser(description="Export / restore / inspect the loaded graph.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="Write the live graph to a .jsonl.gz cache")
    ex.add_argument("out", type=Path)
    ex.add_argument("--model-id", required=True)
    ex.add_argument("--enrichment", default="base", help="Graph state label (base, topology, ...)")
    ex.add_argument("--ifc", type=Path, default=None, help="Source IFC, hashed into the metadata")
    rs = sub.add_parser("restore", help="Wipe the database and restore a cache")
    rs.add_argument("path", type=Path)
    sub.add_parser("info", help="Print the metadata node and the live fingerprint")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password_value)
    )
    try:
        if args.cmd == "export":
            meta = {"model_id": args.model_id, "enrichment": args.enrichment,
                    "loader_rev": loader_revision()}
            if args.ifc:
                meta["source_ifc"] = str(args.ifc)
                meta["source_sha256"] = sha256_file(args.ifc)
            meta = {k: v for k, v in meta.items() if v is not None}
            header = export_graph(driver, args.out, meta)
            # The graph that was exported is now a known state; say so in it.
            write_meta(driver, {**meta, "fingerprint": header["fingerprint"],
                                "cache_path": str(args.out)})
            print(json.dumps(header["summary"]["totals"]), header["fingerprint"])
        elif args.cmd == "restore":
            header = restore_graph(driver, args.path)
            print(json.dumps(header["summary"]["totals"]), header["fingerprint"])
        else:
            print(json.dumps({"meta": read_meta(driver),
                              "live_fingerprint": fingerprint(graph_summary(driver))}, indent=2))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
