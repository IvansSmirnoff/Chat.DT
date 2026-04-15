"""Round-trip tests for the Cypher grammar regex builder.

Gold Cypher queries from ``data/test_set.csv`` must match the regex we
generate from the combined vocabulary. If ``build_cypher_regex`` drifts,
``CYPHER_STRICT`` (Outlines) stops producing executable queries.
"""

from __future__ import annotations

import re

import pytest

from src.constraints.grammar import build_cypher_regex


ENTITIES = {
    "IfcWall",
    "IfcDoor",
    "IfcWindow",
    "IfcSpace",
    "IfcBuildingStorey",
    "IfcColumn",
}
PROPERTIES = {"GlobalId", "Name", "FireRating", "IsExternal"}


@pytest.fixture(scope="module")
def cypher_regex() -> str:
    return build_cypher_regex(ENTITIES, PROPERTIES)


# The default grammar has force_return_node=True: RETURN is constrained to a
# single-letter variable (e.g. "RETURN n"). GlobalId extraction happens in
# execution, not in the regex. Gold queries in data/test_set.csv follow the
# same convention.
@pytest.mark.parametrize(
    "gold_query",
    [
        "MATCH (n:IfcDoor) RETURN n",
        "MATCH (n:IfcWall) WHERE n.FireRating = 'F60' RETURN n",
        "MATCH (n:IfcDoor) WHERE n.IsExternal = true RETURN n",
    ],
)
def test_gold_queries_match_regex(cypher_regex, gold_query):
    assert re.fullmatch(cypher_regex, gold_query) is not None, (
        f"gold query not matched by grammar regex: {gold_query!r}"
    )


def test_regex_rejects_unknown_label(cypher_regex):
    assert re.fullmatch(cypher_regex, "MATCH (n:IfcUnicorn) RETURN n") is None


def test_regex_rejects_unknown_property(cypher_regex):
    bad = "MATCH (n:IfcWall) WHERE n.NotAProp = 'x' RETURN n"
    assert re.fullmatch(cypher_regex, bad) is None
