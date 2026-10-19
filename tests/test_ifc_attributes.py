"""Tests for the schema-derived IFC attribute classifier.

These pin the rule that replaced the loader's hand-written four-attribute list
(``GlobalId``/``Name``/``Description``/``ObjectType``). No Neo4j and no IFC file
-- everything comes from the ifcopenshell schema declarations.
"""

import pytest

pytest.importorskip("ifcopenshell")

from ifcopenshell import ifcopenshell_wrapper  # noqa: E402

from src.utils.ifc_attributes import (  # noqa: E402
    AGGREGATE_ENTITY,
    AGGREGATE_SIMPLE,
    ENTITY,
    SIMPLE,
    attribute_kinds,
    classify_attribute,
    storable_attributes,
)


def _attr_type(schema, ifc_class, attribute_name):
    declaration = ifcopenshell_wrapper.schema_by_name(schema).declaration_by_name(ifc_class)
    for attribute in declaration.all_attributes():
        if attribute.name() == attribute_name:
            return attribute.type_of_attribute()
    raise AssertionError(f"{ifc_class} has no attribute {attribute_name}")


@pytest.mark.parametrize(
    "schema,ifc_class,attribute,expected",
    [
        # A plain string type.
        ("IFC2X3", "IfcDoor", "Name", SIMPLE),
        # A measure, which unwraps through several named_type layers to 'real'.
        ("IFC2X3", "IfcDoor", "OverallWidth", SIMPLE),
        ("IFC2X3", "IfcBuildingStorey", "Elevation", SIMPLE),
        # An enumeration is a string once ifcopenshell is done with it.
        ("IFC2X3", "IfcSpace", "CompositionType", SIMPLE),
        ("IFC4", "IfcWall", "PredefinedType", SIMPLE),
        # Entity-valued attributes are relationship territory, never properties.
        ("IFC2X3", "IfcDoor", "OwnerHistory", ENTITY),
        ("IFC2X3", "IfcDoor", "ObjectPlacement", ENTITY),
        ("IFC2X3", "IfcDoor", "Representation", ENTITY),
        # A LIST of INTEGER is storable; a SET of entities is not.
        ("IFC2X3", "IfcSite", "RefLatitude", AGGREGATE_SIMPLE),
        ("IFC2X3", "IfcElementQuantity", "Quantities", AGGREGATE_ENTITY),
    ],
)
def test_classify_attribute(schema, ifc_class, attribute, expected):
    assert classify_attribute(_attr_type(schema, ifc_class, attribute)) == expected


def test_classify_attribute_handles_none():
    assert classify_attribute(None) == "unknown"


def test_door_gains_its_overall_dimensions():
    """The concrete D1 regression: these were readable and never read."""
    names = storable_attributes("IFC2X3", "IfcDoor")
    assert "OverallWidth" in names
    assert "OverallHeight" in names
    assert "Tag" in names


def test_space_gains_long_name():
    assert "LongName" in storable_attributes("IFC2X3", "IfcSpace")


def test_storey_gains_elevation():
    """Previously present only because a Revit pset happened to duplicate it."""
    assert "Elevation" in storable_attributes("IFC2X3", "IfcBuildingStorey")


@pytest.mark.parametrize("schema", ["IFC2X3", "IFC4"])
@pytest.mark.parametrize("excluded", ["OwnerHistory", "ObjectPlacement", "Representation"])
def test_entity_valued_attributes_are_never_storable(schema, excluded):
    assert excluded not in storable_attributes(schema, "IfcDoor")


@pytest.mark.parametrize("schema", ["IFC2X3", "IFC4"])
def test_the_original_four_are_still_covered(schema):
    """The hand-written list must be a subset of what the schema derives."""
    names = set(storable_attributes(schema, "IfcWall"))
    assert {"GlobalId", "Name", "Description", "ObjectType"} <= names


def test_schema_versions_differ_where_the_schema_differs():
    """IFC4 adds PredefinedType to IfcWall; IFC2X3 does not have it."""
    assert "PredefinedType" not in storable_attributes("IFC2X3", "IfcWall")
    assert "PredefinedType" in storable_attributes("IFC4", "IfcWall")


def test_results_are_cached():
    first = storable_attributes("IFC4", "IfcSpace")
    second = storable_attributes("IFC4", "IfcSpace")
    assert first is second


def test_unknown_class_degrades_to_empty():
    assert storable_attributes("IFC4", "IfcDefinitelyNotAnEntity") == ()


def test_attribute_kinds_explains_every_attribute():
    kinds = attribute_kinds("IFC2X3", "IfcDoor")
    assert kinds["OverallWidth"] == SIMPLE
    assert kinds["OwnerHistory"] == ENTITY
    # Every declared attribute is accounted for, so a gap is visible.
    declaration = ifcopenshell_wrapper.schema_by_name("IFC2X3").declaration_by_name("IfcDoor")
    assert set(kinds) == {a.name() for a in declaration.all_attributes()}
