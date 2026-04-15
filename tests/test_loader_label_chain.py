"""Tests for IFCToNeo4jLoader._label_chain.

Walks the ifcopenshell schema to produce the multi-label supertype tuple.
Depends on ifcopenshell but NOT on Neo4j — safe to run without the DB.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from src.etl.loader import IFCToNeo4jLoader  # noqa: E402


@pytest.fixture(scope="module")
def loader_with_ifc4_schema():
    """A loader with its schema primed to IFC4 for pure-schema label walks."""
    loader = IFCToNeo4jLoader.__new__(IFCToNeo4jLoader)
    loader._label_chain_cache = {}
    loader._schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name("IFC4")
    return loader


def test_wall_standard_case_has_full_supertype_chain(loader_with_ifc4_schema):
    chain = loader_with_ifc4_schema._label_chain("IfcWallStandardCase")
    # Must include leaf + every canonical supertype up to IfcRoot.
    expected_ancestors = [
        "IfcWallStandardCase",
        "IfcWall",
        "IfcBuildingElement",
        "IfcElement",
        "IfcProduct",
        "IfcObject",
        "IfcObjectDefinition",
        "IfcRoot",
    ]
    for name in expected_ancestors:
        assert name in chain, f"{name} missing from label chain: {chain}"
    # Leaf must be first, IfcRoot last.
    assert chain[0] == "IfcWallStandardCase"
    assert chain[-1] == "IfcRoot"


def test_label_chain_is_cached(loader_with_ifc4_schema):
    first = loader_with_ifc4_schema._label_chain("IfcWall")
    second = loader_with_ifc4_schema._label_chain("IfcWall")
    assert first is second, "expected cached tuple reuse"


def test_unknown_class_falls_back_to_ifc_root(loader_with_ifc4_schema):
    chain = loader_with_ifc4_schema._label_chain("IfcNotAClass")
    assert chain[-1] == "IfcRoot"
