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


# Default grammar has force_return_node=False, allow_aggregations=True,
# allow_order_by=True, allow_limit=True. Variable references (in WHERE,
# RETURN, ORDER BY) are bound to a single lowercase letter (REF_VARIABLE)
# to prevent the FSM self-loop trap that small models fall into on the
# open-ended VARIABLE pattern. AS-alias names are capped at 20 chars; the
# RETURN list is capped at 8 items.
@pytest.mark.parametrize(
    "gold_query",
    [
        "MATCH (n:IfcDoor) RETURN n",
        "MATCH (n:IfcWall) WHERE n.FireRating = 'F60' RETURN n",
        "MATCH (n:IfcDoor) WHERE n.IsExternal = true RETURN n",
        "MATCH (n:IfcWall) RETURN count(n)",
        "MATCH (d:IfcDoor) RETURN d.FireRating",
        "MATCH (s:IfcBuildingStorey) RETURN s.Name, count(s)",
        "MATCH (c:IfcColumn) RETURN c ORDER BY c.GlobalId DESC LIMIT 1",
        "MATCH (n:IfcWall) RETURN count(n) AS total",
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


def test_regex_rejects_alias_self_loop(cypher_regex):
    # 21-char alias exceeds the 20-char cap (q0 self-loop trap).
    bad = "MATCH (n:IfcWall) RETURN count(n) AS abcdefghijabcdefghijK"
    assert re.fullmatch(cypher_regex, bad) is None


def test_regex_rejects_oversized_return_list(cypher_regex):
    # 9-item RETURN exceeds the 8-item cap (q10 / q22 dup-column trap).
    bad = (
        "MATCH (n:IfcWall) RETURN n.Name, n.FireRating, n.IsExternal, "
        "n.GlobalId, n.Name, n.FireRating, n.IsExternal, n.GlobalId, n.Name"
    )
    assert re.fullmatch(cypher_regex, bad) is None
