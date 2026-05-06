"""Tests for the prompt builder's graph-sourced sections."""

from __future__ import annotations

from src.constraints.context_builder import ContextBuilder
from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    EntityVocabulary,
    PropertyType,
    PropertyVocabulary,
)


def _vocab() -> CombinedVocabulary:
    vocab = CombinedVocabulary()
    vocab.entities["IfcBuildingStorey"] = EntityVocabulary(
        name="IfcBuildingStorey", from_ifc=True, count=3,
        properties={"Name": PropertyVocabulary(name="Name", property_type=PropertyType.STRING)},
    )
    vocab.all_properties["Name"] = PropertyVocabulary(
        name="Name", property_type=PropertyType.STRING
    )
    return vocab


def test_known_values_block_renders_when_enumerations_present():
    builder = ContextBuilder(
        _vocab(),
        value_enumerations={"IfcBuildingStorey.Name": ["Level 0", "Level 1"]},
    )
    ctx = builder.build_context()

    assert "## Known Values" in ctx.system_prompt
    assert "IfcBuildingStorey.Name" in ctx.system_prompt
    assert "'Level 0'" in ctx.system_prompt
    assert "'Level 1'" in ctx.system_prompt


def test_known_values_block_omitted_when_no_enumerations():
    builder = ContextBuilder(_vocab())
    ctx = builder.build_context()

    assert "## Known Values" not in ctx.system_prompt


def test_examples_block_omitted_when_no_few_shots():
    """Default fallback must be no-examples, not the legacy hardcoded list."""
    builder = ContextBuilder(_vocab())
    ctx = builder.build_context()

    assert "## Examples" not in ctx.system_prompt
    # Old hardcoded few-shots used W-001 / IsExternal=true / window_count —
    # none of those should leak in once we drop DEFAULT_FEW_SHOT_EXAMPLES.
    assert "W-001" not in ctx.system_prompt
    assert "window_count" not in ctx.system_prompt
    assert "IfcWall.FireRating = 'EI60'" not in ctx.system_prompt


def test_few_shot_examples_render_when_provided():
    examples = [
        {
            "question": "How many doors are on 'Level 2'?",
            "cypher": (
                "MATCH (s:IfcBuildingStorey)-[:CONTAINS]->(d:IfcDoor) "
                "WHERE s.Name = 'Level 2' RETURN count(d)"
            ),
        }
    ]
    builder = ContextBuilder(_vocab(), few_shot_examples=examples)
    ctx = builder.build_context()

    assert "## Examples" in ctx.system_prompt
    assert "Level 2" in ctx.system_prompt
    assert "RETURN count(d)" in ctx.system_prompt
