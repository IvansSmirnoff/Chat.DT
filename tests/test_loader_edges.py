"""Regression tests on whatever model is currently loaded in Neo4j.

Read-only sanity checks. Skipped automatically when Neo4j is unreachable
so the suite still runs on Colab / dev machines without the DB.
"""

from __future__ import annotations

import pytest


def _count(driver, cypher: str) -> int:
    with driver.session() as s:
        return s.run(cypher).single()["c"]


def test_graph_is_loaded_without_property_definition_nodes(neo4j_driver):
    # Model-independent: the database holds whichever IFC-Bench model was
    # restored last, so an absolute node count (this used to assert > 5,000,
    # sized for Barcelona) fails on small models. The defect it guarded is
    # property sets materialised as orphan nodes -- 82.5% of the clinic graph
    # before the loader fix -- and that is checkable on any model.
    total = _count(neo4j_driver, "MATCH (n) RETURN count(n) AS c")
    assert total > 0, "graph is empty"
    psets = _count(neo4j_driver, "MATCH (n:IfcPropertyDefinition) RETURN count(n) AS c")
    assert psets == 0, f"{psets} IfcPropertyDefinition nodes leaked into the graph"


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
    # Precondition-gated like the material-layer tests: a structural model
    # (dental_clinic__str) has openings in slabs but no doors or windows, so it
    # legitimately carries HAS_OPENING and no FILLS.
    opening_nodes = _count(neo4j_driver, "MATCH (o:IfcOpeningElement) RETURN count(o) AS c")
    if not opening_nodes:
        pytest.skip("loaded model has no opening elements")
    openings = _count(neo4j_driver, "MATCH ()-[r:HAS_OPENING]->() RETURN count(r) AS c")
    assert openings > 0, "opening elements exist but no HAS_OPENING edge was written"

    fillers = _count(neo4j_driver, "MATCH (n) WHERE n:IfcDoor OR n:IfcWindow RETURN count(n) AS c")
    if fillers:
        fills = _count(neo4j_driver, "MATCH ()-[r:FILLS]->() RETURN count(r) AS c")
        assert fills > 0, "doors/windows and openings exist but no FILLS edge was written"


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


def _live_property_names(driver) -> set[str]:
    """Property names actually present on nodes.

    ``db.propertyKeys()`` is the wrong source here: the token store never
    shrinks, so it still lists keys from every model ever loaded into this
    database. ``db.schema.nodeTypeProperties()`` reflects the current graph.
    """
    with driver.session() as s:
        return {
            record["propertyName"]
            for record in s.run(
                "CALL db.schema.nodeTypeProperties() YIELD propertyName "
                "RETURN DISTINCT propertyName"
            )
        }


def test_direct_ifc_attributes_are_written(neo4j_driver):
    """The D1 regression: the loader once read four attributes and no more.

    Which of these a given model carries depends on which classes it contains,
    so this asserts the *class* of failure -- no attribute beyond the original
    four reaching the graph at all -- rather than a specific model's counts.
    """
    live = _live_property_names(neo4j_driver)
    schema_attributes = {
        "LongName", "OverallWidth", "OverallHeight", "Elevation",
        "Tag", "PredefinedType", "CompositionType", "RefElevation",
    }
    found = schema_attributes & live
    assert found, (
        "no IFC direct attribute beyond GlobalId/Name/Description/ObjectType "
        "is in the graph — the schema-derived attribute walk is not running"
    )


def test_material_layers_carry_their_thickness(neo4j_driver):
    """Layer sets used to collapse to the first layer's name."""
    layers = _count(neo4j_driver, "MATCH (l:IfcMaterialLayer) RETURN count(l) AS c")
    if layers == 0:
        pytest.skip("loaded model has no material layer sets")
    with_thickness = _count(
        neo4j_driver,
        "MATCH (l:IfcMaterialLayer) WHERE l.LayerThickness IS NOT NULL "
        "RETURN count(l) AS c",
    )
    assert with_thickness > 0, "layer nodes exist but none carries a thickness"

    named = _count(
        neo4j_driver,
        "MATCH (l:IfcMaterialLayer) WHERE l.MaterialName IS NOT NULL "
        "RETURN count(l) AS c",
    )
    # The grammar admits one hop, so the material name has to be reachable
    # without a second traversal.
    assert named > 0, "layer nodes carry no MaterialName — single-hop queries break"


def test_material_layers_are_ordered(neo4j_driver):
    layers = _count(neo4j_driver, "MATCH (l:IfcMaterialLayer) RETURN count(l) AS c")
    if layers == 0:
        pytest.skip("loaded model has no material layer sets")
    indexed = _count(
        neo4j_driver,
        "MATCH (l:IfcMaterialLayer) WHERE l.LayerIndex IS NOT NULL RETURN count(l) AS c",
    )
    assert indexed == layers, "some layer nodes have no LayerIndex"


def test_materials_are_named(neo4j_driver):
    unnamed = _count(
        neo4j_driver,
        "MATCH (m:IfcMaterial) WHERE m.Name IS NULL RETURN count(m) AS c",
    )
    assert unnamed == 0, f"{unnamed} material nodes have no Name"


def test_graph_relationship_types_are_all_declared(neo4j_driver):
    """Nothing writes an edge type that is not in the shared spec table."""
    from src.utils.ifc_edges import DERIVED_EDGE_NOTES, EDGE_SPECS, SYNTHETIC_EDGE_NOTES

    declared = {spec.rel_type for spec in EDGE_SPECS}
    declared |= {rel_type for rel_type, _ in SYNTHETIC_EDGE_NOTES.values()}
    declared |= {"HAS_MATERIAL_LAYER"}
    # The topology enrichment's types (and db.relationshipTypes() is the token
    # store: it keeps them after a remove or a restore of the base graph).
    declared |= set(DERIVED_EDGE_NOTES)

    with neo4j_driver.session() as s:
        live = {
            r["relationshipType"]
            for r in s.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType"
            )
        }
    undeclared = live - declared
    assert not undeclared, f"graph carries undeclared edge types: {undeclared}"
