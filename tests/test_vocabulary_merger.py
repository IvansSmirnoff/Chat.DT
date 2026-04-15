"""Smoke tests for the IFC+IDS vocabulary merger.

Uses the real project data fixtures. The merger is non-trivial and has no
unit tests today — this locks in the invariants the experiment depends on.
"""

from __future__ import annotations

from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    PropertyType,
    build_combined_vocabulary,
)


def test_build_combined_vocabulary_returns_populated_object(
    barcelona_ifc, requirements_ids
):
    vocab = build_combined_vocabulary(
        ids_path=requirements_ids, ifc_path=barcelona_ifc
    )
    assert isinstance(vocab, CombinedVocabulary)
    assert len(vocab.get_entity_names()) > 0
    assert len(vocab.get_all_property_names()) > 0


def test_ids_entities_appear_in_merged_vocabulary(barcelona_ifc, requirements_ids):
    vocab = build_combined_vocabulary(
        ids_path=requirements_ids, ifc_path=barcelona_ifc
    )
    # IDS-sourced entities should round-trip into the combined vocabulary as
    # Neo4j-style labels (PascalCase) once merged with the IFC scan.
    labels = vocab.get_entity_names()
    # At least one Ifc* label should be present (common IDS entities).
    assert any(name.startswith("Ifc") for name in labels)


def test_strict_properties_flagged(barcelona_ifc, requirements_ids):
    vocab = build_combined_vocabulary(
        ids_path=requirements_ids, ifc_path=barcelona_ifc
    )
    # Every strict property in the aggregate set must be of type STRICT in the
    # detailed property map.
    for prop_name in vocab.strict_properties:
        prop = vocab.get_property(prop_name)
        assert prop is not None
        assert prop.property_type == PropertyType.STRICT
        assert prop.allowed_values, f"{prop_name} marked strict but has no values"


def test_all_properties_union_matches_aggregate(barcelona_ifc, requirements_ids):
    vocab = build_combined_vocabulary(
        ids_path=requirements_ids, ifc_path=barcelona_ifc
    )
    union = vocab.strict_properties | vocab.open_properties
    # Some merged properties might not appear in either bucket (e.g. numeric /
    # boolean types are neither strict nor open). So assert the buckets are a
    # subset of the master set, not equal.
    assert union.issubset(vocab.get_all_property_names())
