"""Reply grammar: the envelope, the Cypher subset, and the graph vocabulary.

Acceptance is checked byte by byte with each vLLM backend that reads Lark:
xgrammar (the one the spike uses) and llguidance, so both read the grammar the
same way. A backend that is not installed is skipped (llguidance is in
requirements-dev.txt; xgrammar pulls torch, so install it by hand:
``pip install xgrammar==0.2.8``, the version vLLM 0.32 pins).
"""

import json

import importlib.util

import pytest

from src.constraints.cfg_grammar import build_reply_grammar, check_reply

BACKENDS = [pytest.param(b, marks=pytest.mark.skipif(importlib.util.find_spec(b) is None,
                                                     reason=f"{b} not installed"))
            for b in ("xgrammar", "llguidance")]

LABELS = ["IfcSpace", "IfcDoor", "IfcWall", "IfcWallStandardCase", "IfcBuildingStorey"]
RELS = ["BOUNDED_BY", "CONTAINS", "HAS_MATERIAL_LAYER"]
PROPS = ["Name", "LongName", "Width", "GlobalId", "LayerThickness", "derived"]


@pytest.fixture(scope="module")
def agent():
    return build_reply_grammar(LABELS, RELS, PROPS, reply="agent")


@pytest.fixture(scope="module")
def generator():
    return build_reply_grammar(LABELS, RELS, PROPS, reply="generator")


def cypher(q, thoughts="t"):
    return json.dumps({"action": "cypher", "thoughts": thoughts, "cypher": q})


@pytest.mark.parametrize("query", [
    "MATCH (s:IfcSpace) WHERE s.Name = '1B07' RETURN s.LongName",
    ("MATCH (s1:IfcSpace)-[:BOUNDED_BY]->(e)<-[:BOUNDED_BY]-(s2:IfcSpace) "
     "WHERE s1.Name = '1B07' AND s2.Name <> '1B07' RETURN DISTINCT s2.Name ORDER BY s2.Name"),
    "MATCH (n:IfcDoor) RETURN count(n) as c",                      # lowercase keyword; count is also a VAR
    "MATCH (n) WHERE ANY(k IN keys(n) WHERE toLower(k) CONTAINS 'width') RETURN labels(n)[0], count(*) LIMIT 5",
    ("MATCH (st:IfcBuildingStorey)-[c:CONTAINS]->(w:IfcWall) WHERE c.derived IS NULL "
     "WITH st, count(w) AS n RETURN st.Name, n ORDER BY n DESC"),
    "MATCH (w:IfcWall)-[:HAS_MATERIAL_LAYER]->(m) RETURN sum(m.LayerThickness) / 1000.0 AS t",
    "MATCH (d:IfcDoor) RETURN d.Name UNION ALL MATCH (w:IfcWall) RETURN w.Name",
    "MATCH (n) WHERE n:IfcDoor OR n:IfcWall RETURN CASE WHEN n.Width > 1.5 THEN 'wide' ELSE 'narrow' END",
    "MATCH (s:IfcSpace)-[:BOUNDED_BY*1..2]-(x) RETURN [p IN collect(x.Name) WHERE p STARTS WITH 'D' | toUpper(p)]",
])
@pytest.mark.parametrize("backend", BACKENDS)
def test_accepts_read_queries_over_the_vocabulary(agent, query, backend):
    assert check_reply(agent, cypher(query), backend) is None


@pytest.mark.parametrize("query, why", [
    ("MATCH (s:IfcRoom) RETURN s", "label not in the graph"),
    ("MATCH (s:IfcSpace) RETURN s.RoomNumber", "property not in the graph"),
    ("MATCH (s)-[:ADJACENT_TO]->(t) RETURN t", "relationship type not in the graph"),
    ("MATCH (s:IfcSpace) DETACH DELETE s", "write clause"),
    ("MATCH (s:IfcSpace) SET s.Name = 'x' RETURN s", "write clause"),
    ("MATCH (s:IfcSpace)-[:BOUNDED_BY|:CONTAINS]->(t) RETURN t", "deprecated |: separator"),
    ("MATCH (n) WHERE EXISTS(n.Name) RETURN n", "removed exists(property) syntax"),
    ("MATCH (s:IfcSpace) RETURN s.Name AS " + "a" * 40, "alias over the length cap"),
])
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_what_the_graph_or_neo4j_would_not_take(agent, query, why, backend):
    assert check_reply(agent, cypher(query), backend) is not None, why


@pytest.mark.parametrize("backend", BACKENDS)
def test_envelope_is_one_line_and_in_prompt_order(agent, backend):
    assert check_reply(agent, '{"action": "cypher", "thoughts": "t", "cypher": "MATCH (n:IfcDoor)\\nRETURN n"}', backend)
    assert check_reply(agent, '{"thoughts": "t", "action": "cypher", "cypher": "MATCH (n:IfcDoor) RETURN n"}', backend)
    answer = json.dumps({"action": "final_answer", "thoughts": "found \"it\"", "answer": "1.8 m\nper IfcDoor.Width"})
    assert check_reply(agent, answer, backend) is None  # JSON strings keep their escapes


@pytest.mark.parametrize("backend", BACKENDS)
def test_generator_reply_may_query_decline_or_answer(generator, backend):
    def ok(reply):
        return check_reply(generator, json.dumps(reply), backend) is None
    assert ok({"thoughts": "t", "cypher": "MATCH (d:IfcDoor) RETURN count(d)"})
    assert ok({"thoughts": "not in the model", "cypher": ""})
    assert ok({"thoughts": "t", "answer": "unknown"})
    assert not ok({"action": "cypher", "thoughts": "t", "cypher": "MATCH (d:IfcDoor) RETURN d"})  # agent envelope


def test_grammar_text_is_deterministic():
    a = build_reply_grammar(LABELS, RELS, PROPS)
    b = build_reply_grammar(reversed(LABELS), reversed(RELS), reversed(PROPS))
    assert a == b


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_without_edges_still_compiles(backend):
    g = build_reply_grammar(LABELS, [], PROPS)
    assert check_reply(g, cypher("MATCH (s:IfcSpace) RETURN s.Name"), backend) is None
