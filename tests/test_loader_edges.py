"""Regression tests on the current Barcelona load in Neo4j.

Read-only sanity checks. Skipped automatically when Neo4j is unreachable
so the suite still runs on Colab / dev machines without the DB.
"""

from __future__ import annotations

import pytest


def _count(driver, cypher: str) -> int:
    with driver.session() as s:
        return s.run(cypher).single()["c"]


def test_total_nodes_above_minimum(neo4j_driver):
    total = _count(neo4j_driver, "MATCH (n) RETURN count(n) AS c")
    # Plan expects ~11,786 for Barcelona. Allow drift but flag if it halves.
    assert total > 5_000, f"node count suspiciously low: {total}"


def test_contains_edges_present(neo4j_driver):
    literal = _count(
        neo4j_driver,
        "MATCH ()-[r:CONTAINS]->() WHERE r.derived IS NULL RETURN count(r) AS c",
    )
    derived = _count(
        neo4j_driver,
        "MATCH ()-[r:CONTAINS]->() WHERE r.derived = true RETURN count(r) AS c",
    )
    # Both literal IFC containment and transitive derived edges should exist.
    assert literal > 0, "no literal CONTAINS edges — loader regression?"
    assert derived > 0, "no derived CONTAINS edges — transitive walk broken?"


def test_has_opening_and_fills_present(neo4j_driver):
    openings = _count(neo4j_driver, "MATCH ()-[r:HAS_OPENING]->() RETURN count(r) AS c")
    fills = _count(neo4j_driver, "MATCH ()-[r:FILLS]->() RETURN count(r) AS c")
    assert openings > 0 and fills > 0


def test_is_of_type_present(neo4j_driver):
    type_edges = _count(
        neo4j_driver, "MATCH ()-[r:IS_OF_TYPE]->() RETURN count(r) AS c"
    )
    assert type_edges > 0


def test_relationship_entities_not_materialised_as_nodes(neo4j_driver):
    # IfcRelationship reifications must only exist as edges, not nodes.
    with neo4j_driver.session() as s:
        labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
    forbidden = [
        "IfcRelContainedInSpatialStructure",
        "IfcRelAggregates",
        "IfcRelVoidsElement",
        "IfcRelFillsElement",
        "IfcRelDefinesByType",
    ]
    leaked = [l for l in forbidden if l in labels]
    assert not leaked, f"IfcRelationship leaked as nodes: {leaked}"
