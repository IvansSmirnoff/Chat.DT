"""Loader property extraction: attributes, collisions, provenance, layers.

Everything here runs against small in-memory ifcopenshell files, so no Neo4j
and no 157 MB model is needed. These pin the three defects the hardening fixed:

* only four IFC direct attributes were ever read;
* property sets silently overwrote those attributes;
* pset/pset collisions were resolved with a ``_1``/``_2`` suffix that recorded
  nothing about where either value came from.
"""

import pytest

pytest.importorskip("ifcopenshell")

import ifcopenshell  # noqa: E402

from src.etl.loader import (  # noqa: E402
    IFCToNeo4jLoader,
    flatten_psets,
    resolve_material,
    scan_property_name_ambiguity,
)
from src.utils.ifc_attributes import storable_attributes  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _single_value(ifc_file, name, value):
    return ifc_file.create_entity(
        "IfcPropertySingleValue",
        Name=name,
        NominalValue=ifc_file.create_entity("IfcLabel", value),
    )


def _attach_pset(ifc_file, element, pset_name, properties):
    pset = ifc_file.create_entity(
        "IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(),
        Name=pset_name,
        HasProperties=[_single_value(ifc_file, k, v) for k, v in properties.items()],
    )
    ifc_file.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[element],
        RelatingPropertyDefinition=pset,
    )
    return pset


@pytest.fixture
def space_model():
    """An IfcSpace whose IFC Name and Revit pset Name disagree.

    This is the dental-clinic shape: the IFC attribute holds the room number
    and the exporter's property set holds the room name, under the same key.
    """
    ifc_file = ifcopenshell.file(schema="IFC4")
    space = ifc_file.create_entity(
        "IfcSpace",
        GlobalId=ifcopenshell.guid.new(),
        Name="1AC1",
        LongName="CENTRAL WAITING",
    )
    _attach_pset(
        ifc_file, space, "PSet_Revit_Identity Data",
        {"Name": "CENTRAL WAITING", "Reference": "R-A"},
    )
    _attach_pset(ifc_file, space, "Pset_SpaceCommon", {"Reference": "R-B"})
    return ifc_file, space


def _flatten(ifc_file, element, reserved):
    owners = scan_property_name_ambiguity(ifc_file)
    ambiguous = {name for name, sets in owners.items() if len(sets) > 1}
    return flatten_psets(element, ambiguous=ambiguous, reserved=reserved)


# ---------------------------------------------------------------------------
# Ambiguity scan
# ---------------------------------------------------------------------------

def test_ambiguity_scan_records_every_owning_pset(space_model):
    ifc_file, _ = space_model
    owners = scan_property_name_ambiguity(ifc_file)
    assert owners["Reference"] == {"PSet_Revit_Identity Data", "Pset_SpaceCommon"}
    assert owners["Name"] == {"PSet_Revit_Identity Data"}


def test_ambiguity_scan_covers_quantity_sets():
    ifc_file = ifcopenshell.file(schema="IFC4")
    wall = ifc_file.create_entity("IfcWall", GlobalId=ifcopenshell.guid.new())
    _attach_pset(ifc_file, wall, "PSet_Revit_Dimensions", {"Area": "12.0"})
    quantity_set = ifc_file.create_entity(
        "IfcElementQuantity",
        GlobalId=ifcopenshell.guid.new(),
        Name="Qto_WallBaseQuantities",
        Quantities=[ifc_file.create_entity("IfcQuantityArea", Name="Area", AreaValue=12.0)],
    )
    ifc_file.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[wall],
        RelatingPropertyDefinition=quantity_set,
    )

    owners = scan_property_name_ambiguity(ifc_file)
    assert owners["Area"] == {"PSet_Revit_Dimensions", "Qto_WallBaseQuantities"}


# ---------------------------------------------------------------------------
# D1 -- IFC direct attributes are read
# ---------------------------------------------------------------------------

def test_space_attributes_include_long_name(space_model):
    ifc_file, space = space_model
    names = storable_attributes(ifc_file.schema, space.is_a())
    assert "LongName" in names
    assert space.LongName == "CENTRAL WAITING"


def test_loader_writes_every_storable_attribute(space_model):
    ifc_file, space = space_model
    loader = IFCToNeo4jLoader.__new__(IFCToNeo4jLoader)
    loader._schema_name = ifc_file.schema
    loader.stats = {"attributes_written": 0}

    attributes = loader._extract_attributes(space)

    assert attributes["Name"] == "1AC1"
    assert attributes["LongName"] == "CENTRAL WAITING"
    assert attributes["GlobalId"] == space.GlobalId
    # Entity-valued attributes never become properties.
    assert "OwnerHistory" not in attributes
    assert "ObjectPlacement" not in attributes


# ---------------------------------------------------------------------------
# D2 -- psets no longer overwrite attributes
# ---------------------------------------------------------------------------

def test_pset_does_not_overwrite_an_ifc_attribute(space_model):
    ifc_file, space = space_model
    properties, _ = _flatten(ifc_file, space, reserved={"GlobalId", "Name", "LongName"})
    assert "Name" not in properties


def test_displaced_pset_value_survives_under_its_qualified_key(space_model):
    ifc_file, space = space_model
    properties, counters = _flatten(
        ifc_file, space, reserved={"GlobalId", "Name", "LongName"}
    )
    assert properties["PSet_Revit_Identity_Data_Name"] == "CENTRAL WAITING"
    assert counters["displaced"] >= 1


# ---------------------------------------------------------------------------
# D3 -- collisions carry provenance instead of a numeric suffix
# ---------------------------------------------------------------------------

def test_both_sides_of_a_collision_survive(space_model):
    ifc_file, space = space_model
    properties, _ = _flatten(ifc_file, space, reserved={"GlobalId", "Name", "LongName"})
    assert properties["PSet_Revit_Identity_Data_Reference"] == "R-A"
    assert properties["Pset_SpaceCommon_Reference"] == "R-B"


def test_bare_alias_is_kept_for_the_first_writer(space_model):
    """Nothing that already queries the bare name moves."""
    ifc_file, space = space_model
    properties, _ = _flatten(ifc_file, space, reserved={"GlobalId", "Name", "LongName"})
    assert properties["Reference"] in {"R-A", "R-B"}


def test_no_numeric_suffix_fallback_is_used(space_model):
    ifc_file, space = space_model
    _, counters = _flatten(ifc_file, space, reserved={"GlobalId", "Name", "LongName"})
    assert counters["fallback"] == 0


def test_unambiguous_properties_stay_bare():
    """Qualification is scoped to ambiguity; it does not rename the whole model."""
    ifc_file = ifcopenshell.file(schema="IFC4")
    wall = ifc_file.create_entity("IfcWall", GlobalId=ifcopenshell.guid.new())
    _attach_pset(ifc_file, wall, "Pset_WallCommon", {"FireRating": "EI60"})

    properties, counters = _flatten(ifc_file, wall, reserved={"GlobalId"})

    assert properties == {"FireRating": "EI60"}
    assert counters["qualified"] == 0


def test_ambiguous_name_is_qualified_on_every_node_that_has_it():
    """Key sets stay consistent across nodes of a label.

    If qualification were decided per element, a node carrying only one of the
    two psets would get the bare key and its neighbour a qualified one, and a
    query written against the label would silently miss half the rows.
    """
    ifc_file = ifcopenshell.file(schema="IFC4")
    both = ifc_file.create_entity("IfcWall", GlobalId=ifcopenshell.guid.new())
    one = ifc_file.create_entity("IfcWall", GlobalId=ifcopenshell.guid.new())
    _attach_pset(ifc_file, both, "Pset_WallCommon", {"Reference": "A"})
    _attach_pset(ifc_file, both, "PSet_Revit_Other", {"Reference": "B"})
    _attach_pset(ifc_file, one, "Pset_WallCommon", {"Reference": "C"})

    owners = scan_property_name_ambiguity(ifc_file)
    ambiguous = {name for name, sets in owners.items() if len(sets) > 1}

    properties, _ = flatten_psets(one, ambiguous=ambiguous, reserved={"GlobalId"})
    assert properties["Pset_WallCommon_Reference"] == "C"
    assert properties["Reference"] == "C"


# ---------------------------------------------------------------------------
# D4 -- material layers
# ---------------------------------------------------------------------------

def _layered_wall(schema="IFC4"):
    ifc_file = ifcopenshell.file(schema=schema)
    wall = ifc_file.create_entity("IfcWall", GlobalId=ifcopenshell.guid.new())
    layers = [
        ifc_file.create_entity(
            "IfcMaterialLayer",
            Material=ifc_file.create_entity("IfcMaterial", Name=name),
            LayerThickness=thickness,
        )
        for name, thickness in (
            ("Insulation - Insulated Panel", 0.042),
            ("Metal - Stud Layer", 0.152),
            ("Plasterboard", 0.016),
        )
    ]
    layer_set = ifc_file.create_entity(
        "IfcMaterialLayerSet", MaterialLayers=layers, LayerSetName="Exterior"
    )
    usage = ifc_file.create_entity(
        "IfcMaterialLayerSetUsage",
        ForLayerSet=layer_set,
        LayerSetDirection="AXIS2",
        DirectionSense="POSITIVE",
        OffsetFromReferenceLine=0.0,
    )
    ifc_file.create_entity(
        "IfcRelAssociatesMaterial",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[wall],
        RelatingMaterial=usage,
    )
    return ifc_file, wall, usage


def test_every_layer_survives_with_its_thickness():
    """The whole set, not just the first layer's name."""
    _, _, usage = _layered_wall()
    names, layers = resolve_material(usage)

    assert names == [
        "Insulation - Insulated Panel",
        "Metal - Stud Layer",
        "Plasterboard",
    ]
    assert [layer["LayerThickness"] for layer in layers] == [0.042, 0.152, 0.016]


def test_layers_keep_their_order():
    _, _, usage = _layered_wall()
    _, layers = resolve_material(usage)
    assert [layer["LayerIndex"] for layer in layers] == [0, 1, 2]


def test_layer_carries_its_material_name_for_single_hop_queries():
    """The grammar admits one hop, so the material name is on the layer node."""
    _, _, usage = _layered_wall()
    _, layers = resolve_material(usage)
    insulation = next(l for l in layers if "Insulation" in (l["MaterialName"] or ""))
    assert insulation["LayerThickness"] == pytest.approx(0.042)
    assert insulation["LayerSetName"] == "Exterior"


def test_bare_material_resolves_without_layers():
    ifc_file = ifcopenshell.file(schema="IFC4")
    material = ifc_file.create_entity("IfcMaterial", Name="Concrete")
    names, layers = resolve_material(material)
    assert names == ["Concrete"]
    assert layers == []


def test_material_list_resolves_every_member():
    ifc_file = ifcopenshell.file(schema="IFC4")
    material_list = ifc_file.create_entity(
        "IfcMaterialList",
        Materials=[
            ifc_file.create_entity("IfcMaterial", Name="Steel"),
            ifc_file.create_entity("IfcMaterial", Name="Concrete"),
        ],
    )
    names, layers = resolve_material(material_list)
    assert names == ["Steel", "Concrete"]
    assert layers == []


def test_constituent_set_resolves_every_member():
    ifc_file = ifcopenshell.file(schema="IFC4")
    constituent_set = ifc_file.create_entity(
        "IfcMaterialConstituentSet",
        MaterialConstituents=[
            ifc_file.create_entity(
                "IfcMaterialConstituent",
                Name="Frame",
                Material=ifc_file.create_entity("IfcMaterial", Name="Aluminium"),
            ),
            ifc_file.create_entity(
                "IfcMaterialConstituent",
                Name="Glazing",
                Material=ifc_file.create_entity("IfcMaterial", Name="Glass"),
            ),
        ],
    )
    names, _ = resolve_material(constituent_set)
    assert names == ["Aluminium", "Glass"]


def test_ifc2x3_layer_sets_resolve_through_the_same_code():
    _, _, usage = _layered_wall(schema="IFC2X3")
    names, layers = resolve_material(usage)
    assert len(layers) == 3
    assert names[0] == "Insulation - Insulated Panel"


def test_resolve_material_tolerates_none():
    assert resolve_material(None) == ([], [])
