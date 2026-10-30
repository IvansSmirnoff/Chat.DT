"""Rung 0: the generic IFC instance graph of the ECPPM 2026 Cypher pipeline.

Every STEP instance of the IFC file becomes one node and every entity
reference one edge; nothing is interpreted. It is the bottom rung of the
representation ladder (graph state ``generic``), set against our typed graph
(``base``) and its enrichment (``topology``).

Provenance
----------
A port of ``pipelines/cypher/ifc2neo4j/ifc_to_graph.py`` from the companion
code of Du, Hellin, Fuchs & Borrmann, "Querying BIM Models with LLM Agents: A
Comparison of Data Representations" (ECPPM 2026; commit 85eaf24), MIT,
Copyright (c) 2026 Changyu Du, Sylvain Hellin, Stefan Fuchs, André Borrmann.
Their importer is adapted from ConMan2 (https://github.com/seb-esser/ConMan2,
Sebastian Esser et al., TUM; MIT). The full copyright and permission notice is
in ``third_party/LICENSE-ecppm``.

Kept exactly (the point of the rung, so none of it is "improved"):
  * node kinds: ``PrimaryNode`` = IfcObjectDefinition + IfcPropertyDefinition
    (so property sets are nodes), ``ConnectionNode`` = IfcRelationship
    (objectified relationships stay nodes), ``SecondaryNode`` = every other
    instance (geometry, placements, properties, quantities, units, ...); all
    three also ``:GenericNode:Node``. ``InlineNode:Node`` = an id-0 instance (a
    typed value such as ``IfcLabel('x')``), one node per occurrence.
  * node properties: ``EntityType`` (the IFC class; never a label),
    ``p21_id`` (``"#<id>"``), ``timestamp``, ``GlobalId`` on primary and
    connection nodes, and every primitive direct attribute under its IFC name:
    ``None`` -> ``"$"``, a list or tuple without entities -> ``str()`` of it.
    Inline nodes carry ``EntityType``, ``timestamp`` and ``wrappedValue``.
  * edges: one type, ``:rel {rel_type: <attribute name>, list_index}``, from the
    referencing to the referenced instance; a list element carries its
    position. Direct attributes only (``get_info()``): no inverses, nothing
    derived, no geometry interpretation.
  * ``entity_attributes`` is their ``_process_ifc_attributes`` with the
    per-model accumulators made per-entity; the Cypher statements are theirs;
    so are the two indexes and ``timestamp = "current"`` (their ``run.py``).

Re-implemented (the graph that comes out is the same):
  * neomodel -> the neo4j 5 driver (``neo4j<6`` here; no neomodel, no APOC).
    Their neomodel classes are not ported: the importer never installs their
    constraints, so they shape nothing in the database.
  * Their skeleton ``CREATE`` and the later ``SET n += attributes`` are one
    ``CREATE``: the attribute keys exclude every skeleton key, so the
    properties are the same.
  * Streaming instead of whole-model lists: one pass over the model writes the
    nodes, a second writes the edges and inline nodes (edges need both ends).
    Beyond the opened model, Python holds one batch of rows at a time.
  * 5,000 rows per ``UNWIND`` transaction instead of 20,000, sized for our
    default 512m heap (see Size below).
  * Wiping uses our loader's ``CALL {...} IN TRANSACTIONS`` (theirs needs APOC).
  * Edge and inline writes are counted and a shortfall raises: an edge whose
    ``MATCH`` misses is silently dropped by the Cypher, and a short graph must
    not be exported as a snapshot. Their unused ``related_nodes`` set is gone.
  * Their two indexes are import-time lookup aids. A snapshot restore
    (``src.etl.graph_cache``) does not recreate them; query results do not
    depend on them.

Size (dental_clinic arc.ifc, 12.6 MB; converted offline, ifcopenshell 0.9):
209,148 STEP instances -> 18,501 primary + 22,567 connection + 168,080
secondary nodes, plus 7,485 inline nodes (216,633 nodes, ~0.93M node
properties) and 446,605 ``:rel`` edges (7,485 of them to inline nodes); 107
EntityTypes, 92 rel_types. Both conversion passes take ~5 s of Python. The
largest 5,000-row batch is ~1.1 MB of parameters, so a transaction's state
stays in the low MB: far inside the default 512m heap (their 20,000 would be
~4 MB, also fine, but leaves less room on the bigger IFC-Bench models).

Usage::

    python -m src.etl.generic_loader data/ifcbench/raw/projects/dental_clinic/arc.ifc --dry-run
    python -m src.agent.run --model-id dental_clinic__arc --state generic --snapshots-only --run prep
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import ifcopenshell

logger = logging.getLogger(__name__)

STATE = "generic"
TIMESTAMP = "current"          # what their run.py passes for every import
BATCH_SIZE = 5000
# Files whose content defines this graph: their hash is the snapshot's loader_rev.
SOURCES = ("src/etl/generic_loader.py",)

PRIMARY, CONNECTION, SECONDARY, INLINE = "PrimaryNode", "ConnectionNode", "SecondaryNode", "InlineNode"
NODE_KINDS = (PRIMARY, CONNECTION, SECONDARY)
LABELS = (*NODE_KINDS, INLINE, "GenericNode", "Node")
REL_TYPE = "rel"

# get_info() keys never copied as attributes (theirs, verbatim).
SKIPPED_KEYS = ("GlobalId", "EntityType", "type", "p21_id", "id", "inline_id", "timestamp")

# Their statements, verbatim.
CREATE_NODES = {kind: f"""
            UNWIND $batch AS props
            CREATE (n:{kind}:GenericNode:Node)
            SET n = props
            """ for kind in NODE_KINDS}
INDEXES = (
    "CREATE INDEX generic_p21_ts IF NOT EXISTS FOR (n:GenericNode) ON (n.p21_id, n.timestamp)",
    "CREATE INDEX generic_ts IF NOT EXISTS FOR (n:GenericNode) ON (n.timestamp)",
)
CREATE_RELS = """
            UNWIND $batch AS r
            MATCH (a:GenericNode {p21_id: r.source_p21_id, timestamp: r.timestamp})
            MATCH (b:GenericNode {p21_id: r.target_p21_id, timestamp: r.timestamp})
            CREATE (a)-[:rel {rel_type: r.rel_type, list_index: r.list_index}]->(b)
            """
CREATE_INLINE = """
            UNWIND $batch AS r
            MATCH (a:GenericNode {p21_id: r.relation.source_p21_id, timestamp: r.props.timestamp})
            CREATE (b:InlineNode:Node)
            SET b = r.props
            CREATE (a)-[:rel {rel_type: r.relation.rel_type, list_index: r.relation.list_index}]->(b)
            """

Row = Dict[str, Any]


# -----------------------------------------------------------------------------
# IFC -> rows (pure; no database)
# -----------------------------------------------------------------------------

def node_kind(entity: ifcopenshell.entity_instance) -> str:
    """Their partition: by_type(IfcObjectDefinition) + by_type(IfcPropertyDefinition),
    by_type(IfcRelationship), and everything else with an id."""
    if entity.is_a("IfcObjectDefinition") or entity.is_a("IfcPropertyDefinition"):
        return PRIMARY
    if entity.is_a("IfcRelationship"):
        return CONNECTION
    return SECONDARY


def entity_attributes(entity: ifcopenshell.entity_instance,
                      timestamp: str = TIMESTAMP) -> Tuple[Row, List[Row], List[Row]]:
    """One entity's (primitive properties, :rel rows, inline-node rows).

    Their ``_process_ifc_attributes`` with ``props_map[p21_id]`` as a local dict;
    the classification of every attribute value is unchanged.
    """
    p21_id = f"#{entity.id()}"
    props: Row = {}
    relationships: List[Row] = []
    inline_patterns: List[Row] = []

    def traverse(key, val, list_index=0):
        if isinstance(val, ifcopenshell.entity_instance):
            if val.id() == 0:
                node_props: dict = {
                    "EntityType": val.is_a(),
                    "timestamp": timestamp,
                }
                if hasattr(val, "wrappedValue"):
                    node_props["wrappedValue"] = val.wrappedValue
                else:
                    for k, v in val.get_info().items():
                        if k in ("id", "type"):
                            continue
                        if isinstance(v, ifcopenshell.entity_instance):
                            if v.id() == 0 and hasattr(v, "wrappedValue"):
                                node_props[k] = v.wrappedValue
                            else:
                                node_props[k] = str(v)
                        elif v is not None:
                            node_props[k] = str(v) if isinstance(v, (tuple, list)) else v
                inline_patterns.append(
                    {
                        "props": node_props,
                        "relation": {
                            "rel_type": key,
                            "list_index": list_index,
                            "source_p21_id": p21_id,
                        },
                    }
                )
            else:
                related_p21_id = f"#{val.id()}"
                relationships.append(
                    {
                        "source_p21_id": p21_id,
                        "target_p21_id": related_p21_id,
                        "timestamp": timestamp,
                        "rel_type": key,
                        "list_index": list_index,
                    }
                )
        elif isinstance(val, (tuple, list)):
            if any(isinstance(x, ifcopenshell.entity_instance) for x in val):
                for i, x in enumerate(val):
                    traverse(key, x, list_index=i)
            else:
                props[key] = str(val)
        elif val is None:
            props[key] = "$"
        else:
            props[key] = val

    for key, val in entity.get_info().items():
        if key in SKIPPED_KEYS:
            continue
        traverse(key, val)
    return props, relationships, inline_patterns


def node_row(entity: ifcopenshell.entity_instance, kind: str, timestamp: str = TIMESTAMP) -> Row:
    """Their skeleton dict plus the attributes their second pass SETs."""
    skeleton: Row = {"EntityType": entity.is_a(), "p21_id": f"#{entity.id()}", "timestamp": timestamp}
    if kind != SECONDARY:
        skeleton = {"GlobalId": entity.GlobalId, **skeleton}
    props, _, _ = entity_attributes(entity, timestamp)
    return {**skeleton, **props}


def iter_nodes(model: ifcopenshell.file, timestamp: str = TIMESTAMP) -> Iterator[Tuple[str, Row]]:
    """(kind, row) for every instance with an id: the first write pass."""
    for entity in model:
        if entity.id() == 0:
            continue
        kind = node_kind(entity)
        yield kind, node_row(entity, kind, timestamp)


def iter_links(model: ifcopenshell.file, timestamp: str = TIMESTAMP) -> Iterator[Tuple[str, Row]]:
    """("rel" | "inline", row) for every reference: the second write pass."""
    for entity in model:
        _, relationships, inline_patterns = entity_attributes(entity, timestamp)
        for row in relationships:
            yield "rel", row
        for row in inline_patterns:
            yield "inline", row


@dataclass
class GenericGraph:
    """The rows a model becomes, all in memory: for tests and ``--dry-run`` on small files."""
    nodes: Dict[str, List[Row]] = field(default_factory=lambda: {k: [] for k in NODE_KINDS})
    relationships: List[Row] = field(default_factory=list)
    inline: List[Row] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {**{k: len(v) for k, v in self.nodes.items()}, INLINE: len(self.inline),
                "rel": len(self.relationships) + len(self.inline)}


def convert(model: ifcopenshell.file, timestamp: str = TIMESTAMP) -> GenericGraph:
    graph = GenericGraph()
    for kind, row in iter_nodes(model, timestamp):
        graph.nodes[kind].append(row)
    for what, row in iter_links(model, timestamp):
        (graph.relationships if what == "rel" else graph.inline).append(row)
    return graph


# -----------------------------------------------------------------------------
# Rows -> Neo4j
# -----------------------------------------------------------------------------

def wipe_database(driver) -> None:
    """Our loader's batched wipe (theirs is APOC's periodic commit)."""
    from src.etl.loader import IFCToNeo4jLoader

    loader = IFCToNeo4jLoader()
    loader.driver = driver
    loader.wipe_database()


def _write(session, query: str, batch: List[Row]):
    return session.run(query, batch=batch).consume().counters


def load(driver, ifc: Union[str, Path, ifcopenshell.file], timestamp: str = TIMESTAMP,
         batch_size: int = BATCH_SIZE, wipe: bool = True) -> Dict[str, Any]:
    """Write the generic graph of ``ifc`` into an (by default wiped) database.

    Raises if fewer edges or inline nodes were created than rows sent, which
    means an endpoint ``MATCH`` missed.
    """
    start = time.monotonic()
    model = ifc if isinstance(ifc, ifcopenshell.file) else ifcopenshell.open(str(ifc))
    if wipe:
        wipe_database(driver)
    created: Counter = Counter()
    sent: Counter = Counter()
    with driver.session() as session:
        buffers: Dict[str, List[Row]] = {k: [] for k in NODE_KINDS}
        for kind, row in iter_nodes(model, timestamp):
            buffers[kind].append(row)
            if len(buffers[kind]) >= batch_size:
                created[kind] += _write(session, CREATE_NODES[kind], buffers[kind]).nodes_created
                buffers[kind] = []
        for kind, rows in buffers.items():
            if rows:
                created[kind] += _write(session, CREATE_NODES[kind], rows).nodes_created
        logger.info("Nodes: %s", {k: created[k] for k in NODE_KINDS})

        for statement in INDEXES:
            session.run(statement).consume()
        session.run("CALL db.awaitIndexes(300)").consume()

        queries = {"rel": CREATE_RELS, "inline": CREATE_INLINE}
        links: Dict[str, List[Row]] = {"rel": [], "inline": []}

        def flush(what: str) -> None:
            counters = _write(session, queries[what], links[what])
            sent[what] += len(links[what])
            created[f"{what}_edges"] += counters.relationships_created
            if what == "inline":
                created[INLINE] += counters.nodes_created
            links[what] = []

        for what, row in iter_links(model, timestamp):
            links[what].append(row)
            if len(links[what]) >= batch_size:
                flush(what)
        for what in links:
            if links[what]:
                flush(what)

    short = {what: (sent[what], created[f"{what}_edges"]) for what in sent
             if created[f"{what}_edges"] != sent[what]}
    if short or created[INLINE] != sent["inline"]:
        raise RuntimeError(f"Generic load is short: sent vs created {short}, inline nodes "
                           f"{created[INLINE]}/{sent['inline']} -- an endpoint MATCH missed")
    stats = {"nodes": {k: created[k] for k in (*NODE_KINDS, INLINE)},
             "relationships": created["rel_edges"] + created["inline_edges"],
             "seconds": round(time.monotonic() - start, 1)}
    logger.info("Generic graph loaded: %s", stats)
    return stats


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    import json

    p = argparse.ArgumentParser(description="Load an IFC file as the generic (ECPPM) instance graph.")
    p.add_argument("ifc", type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="Convert only and print the counts; no database")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-wipe", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.dry_run:
        start = time.monotonic()
        model = ifcopenshell.open(str(args.ifc))
        counts: Counter = Counter()
        properties = 0
        for kind, row in iter_nodes(model):
            counts[kind] += 1
            properties += len(row)
        for what, row in iter_links(model):
            counts[what] += 1
            properties += len(row["props"]) if what == "inline" else 0
        counts["node_properties"] = properties
        counts["seconds"] = round(time.monotonic() - start, 1)
        print(json.dumps(counts, indent=2))
        return 0

    from neo4j import GraphDatabase

    from src.config import get_settings

    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri,
                                  auth=(settings.neo4j_user, settings.neo4j_password_value))
    try:
        print(json.dumps(load(driver, args.ifc, batch_size=args.batch_size, wipe=not args.no_wipe)))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
