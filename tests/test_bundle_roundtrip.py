"""Round-trip tests for bundle dataclass serialization.

The Colab side rebuilds ``CombinedVocabulary`` and ``IDSSchema`` from the JSON
produced by ``scripts/build_bundle.py``. ``build_bundle.py`` dumps via
``dataclasses.asdict`` (sets get sorted into lists by ``_BundleEncoder``), so
``from_dict(asdict(x))`` must reconstruct semantically equivalent objects.
"""

from dataclasses import asdict

from src.constraints.ids_parser import (
    EntityConstraint,
    IDSSchema,
    PropertyConstraint,
)
from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    EntityVocabulary,
    PropertyType,
    PropertyVocabulary,
)


def test_property_constraint_roundtrip():
    pc = PropertyConstraint(
        name="FireRating",
        property_set="Pset_WallCommon",
        data_type="IFCLABEL",
        allowed_values={"EI30", "EI60"},
        cardinality="required",
    )
    rebuilt = PropertyConstraint.from_dict(asdict(pc))
    assert rebuilt == pc


def test_entity_constraint_roundtrip():
    ec = EntityConstraint(
        name="IFCWALL",
        predefined_type="STANDARD",
        properties=[
            PropertyConstraint(name="FireRating", allowed_values={"EI30"}),
            PropertyConstraint(name="LoadBearing", data_type="IFCBOOLEAN"),
        ],
    )
    rebuilt = EntityConstraint.from_dict(asdict(ec))
    assert rebuilt == ec


def test_ids_schema_roundtrip():
    schema = IDSSchema(
        title="Sample",
        version="1.0",
        entities={"IFCWALL", "IFCDOOR"},
        properties={"FireRating", "Name"},
        property_sets={"Pset_WallCommon"},
        entity_constraints={
            "IFCWALL": EntityConstraint(
                name="IFCWALL",
                properties=[
                    PropertyConstraint(name="FireRating", allowed_values={"EI30"})
                ],
            )
        },
        property_values={"FireRating": {"EI30", "EI60"}},
    )
    rebuilt = IDSSchema.from_dict(asdict(schema))
    assert rebuilt == schema


def test_property_vocabulary_roundtrip():
    pv = PropertyVocabulary(
        name="FireRating",
        property_type=PropertyType.STRICT,
        allowed_values={"EI30", "EI60"},
        data_type="IFCLABEL",
        source="both",
        description="d",
    )
    rebuilt = PropertyVocabulary.from_dict(asdict(pv))
    assert rebuilt == pv


def test_entity_vocabulary_roundtrip():
    ev = EntityVocabulary(
        name="IfcWall",
        properties={
            "FireRating": PropertyVocabulary(
                name="FireRating",
                property_type=PropertyType.STRICT,
                allowed_values={"EI30"},
            )
        },
        relations={"CONTAINS", "DECOMPOSES"},
        count=42,
        from_ids=True,
        from_ifc=True,
    )
    rebuilt = EntityVocabulary.from_dict(asdict(ev))
    assert rebuilt == ev


def test_combined_vocabulary_roundtrip_drops_ifc_schema():
    vocab = CombinedVocabulary(
        entities={
            "IfcWall": EntityVocabulary(
                name="IfcWall",
                properties={
                    "Name": PropertyVocabulary(name="Name", property_type=PropertyType.STRING)
                },
                relations={"CONTAINS"},
                count=10,
                from_ids=True,
                from_ifc=True,
            )
        },
        all_properties={
            "Name": PropertyVocabulary(name="Name", property_type=PropertyType.STRING),
            "FireRating": PropertyVocabulary(
                name="FireRating",
                property_type=PropertyType.STRICT,
                allowed_values={"EI30"},
            ),
        },
        strict_properties={"FireRating"},
        open_properties={"Name"},
        all_relations={"CONTAINS"},
        ids_schema=IDSSchema(entities={"IFCWALL"}, properties={"Name"}),
        ifc_schema=None,
    )
    rebuilt = CombinedVocabulary.from_dict(asdict(vocab))

    assert rebuilt.entities == vocab.entities
    assert rebuilt.all_properties == vocab.all_properties
    assert rebuilt.strict_properties == vocab.strict_properties
    assert rebuilt.open_properties == vocab.open_properties
    assert rebuilt.all_relations == vocab.all_relations
    assert rebuilt.ids_schema == vocab.ids_schema
    # ifc_schema is intentionally not rehydrated client-side.
    assert rebuilt.ifc_schema is None


def test_get_entity_names_and_property_names_after_roundtrip():
    """Sanity check: the methods used by the LLM prompt builders still work."""
    vocab = CombinedVocabulary(
        entities={
            "IfcWall": EntityVocabulary(name="IfcWall", from_ifc=True),
            "IfcDoor": EntityVocabulary(name="IfcDoor", from_ifc=True),
        },
        all_properties={
            "Name": PropertyVocabulary(name="Name"),
            "FireRating": PropertyVocabulary(name="FireRating"),
        },
    )
    rebuilt = CombinedVocabulary.from_dict(asdict(vocab))
    assert rebuilt.get_entity_names() == {"IfcWall", "IfcDoor"}
    assert rebuilt.get_all_property_names() == {"Name", "FireRating"}
