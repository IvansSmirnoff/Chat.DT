"""Agent slice: action parsing, both paradigms, LLM client retries.

No Neo4j and no network: the LLM is scripted and the graph tools are faked.
"""

import json

import httpx
import pytest

from src.agent import loop
from src.agent.actions import ActionParseError, first_json_object, parse_action
from src.agent.llm import ChatClient, Completion, LLMError
from src.agent.tools import QueryResult


# -----------------------------------------------------------------------------
# Action parsing
# -----------------------------------------------------------------------------

def test_parse_fenced_cypher_action():
    text = 'Sure.\n```json\n{"action": "cypher", "thoughts": "t", "cypher": "MATCH (d:IfcDoor) RETURN count(d)"}\n```'
    a = parse_action(text)
    assert (a.kind, a.cypher) == ("cypher", "MATCH (d:IfcDoor) RETURN count(d)")


def test_parse_infers_kind_and_accepts_class_names():
    assert parse_action('{"answer": "254 doors (counted from IfcDoor)"}').kind == "final_answer"
    assert parse_action('{"action": "FinalAnswer", "answer": "x"}').kind == "final_answer"
    assert parse_action('{"action": "CypherAction", "cypher": "RETURN 1"}').kind == "cypher"


def test_parse_tolerates_trailing_commas_but_not_missing_fields():
    assert first_json_object('{"a": 1,}') == {"a": 1}
    with pytest.raises(ActionParseError):
        parse_action('{"action": "cypher", "cypher": ""}')
    with pytest.raises(ActionParseError):
        parse_action("MATCH (n) RETURN n")
    with pytest.raises(ActionParseError):
        parse_action('{"action": "final_answer", "answer": "x"}', allowed=("cypher",))


# -----------------------------------------------------------------------------
# Paradigms
# -----------------------------------------------------------------------------

class ScriptedLLM:
    spec = "fake:scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, **_):
        self.calls.append(messages)
        return Completion(text=self.replies.pop(0), prompt_tokens=100,
                          completion_tokens=10, cost_usd=0.001, provider="P")


class FakeTools:
    max_rows = 50
    timeout_s = 10.0

    def __init__(self, results=None):
        self.results = dict(results or {})
        self.queries = []

    def run_cypher(self, query):
        self.queries.append(query)
        r = self.results.get(query, {"columns": ["c"], "rows": [[254]], "total_rows": 1})
        return QueryResult(query=query, **r)


def _act(**kw):
    return json.dumps(kw)


def test_single_shot_generates_executes_once_and_synthesises():
    llm = ScriptedLLM([
        _act(thoughts="count doors", cypher="MATCH (d:IfcDoor) RETURN count(d) AS c"),
        _act(thoughts="done", answer="There are 254 doors (counted from IfcDoor nodes)."),
    ])
    tools = FakeTools()
    ep = loop.single_shot(124, "How many doors?", "SCHEMA", llm, tools)
    assert ep.status == "ok"
    assert ep.answer.startswith("There are 254 doors")
    assert tools.queries == ["MATCH (d:IfcDoor) RETURN count(d) AS c"]
    assert ep.llm_calls == 2 and ep.cost_usd == pytest.approx(0.002)
    # The synthesiser saw the executed query and its result, not the schema.
    synth_user = llm.calls[1][1]["content"]
    assert "MATCH (d:IfcDoor)" in synth_user and "254" in synth_user
    assert "SCHEMA" not in llm.calls[1][0]["content"]


def test_single_shot_reasks_once_on_unparseable_json():
    llm = ScriptedLLM([
        "MATCH (d:IfcDoor) RETURN count(d)",
        _act(cypher="MATCH (d:IfcDoor) RETURN count(d)"),
        _act(answer="254 doors (counted from IfcDoor nodes)."),
    ])
    ep = loop.single_shot(124, "How many doors?", "SCHEMA", llm, FakeTools())
    assert ep.status == "ok" and ep.llm_calls == 3


def test_agentic_feeds_back_results_and_stops_on_final_answer():
    llm = ScriptedLLM([
        _act(action="cypher", thoughts="try", cypher="MATCH (s:IfcSpace {Name:'1B07'}) RETURN s"),
        _act(action="cypher", thoughts="neighbours", cypher="Q2"),
        _act(action="final_answer", thoughts="ok", answer="1B07 connects to 1BC2 (shared IfcDoor boundary)."),
    ])
    tools = FakeTools({"Q2": {"columns": ["n"], "rows": [["1BC2"]], "total_rows": 1}})
    ep = loop.agentic(120, "Which rooms connect to 1B07?", "SCHEMA", llm, tools)
    assert ep.status == "ok" and "1BC2" in ep.answer
    assert len(ep.steps) == 2 and ep.final_cypher == "Q2"
    third_prompt = llm.calls[2][1]["content"]
    assert "## PREVIOUS ATTEMPTS" in third_prompt and "1BC2" in third_prompt


def test_agentic_parse_error_costs_a_step_and_step_limit_abstains():
    llm = ScriptedLLM(["not json"] + [_act(action="cypher", cypher="RETURN 1")] * 2)
    ep = loop.agentic(1, "q", "SCHEMA", llm, FakeTools(), max_steps=3)
    assert ep.status == "step_limit"
    assert ep.answer.startswith("I cannot determine")
    assert ep.steps[0].parse_error and len(ep.steps) == 3
    assert "could not be parsed" in llm.calls[1][1]["content"]


def test_query_result_render_reports_truncation_and_errors():
    r = QueryResult(query="q", columns=["n"], rows=[[1], [2]], total_rows=5)
    text = r.render()
    assert text.startswith("5 row(s)") and "3 more row(s) not shown" in text
    assert QueryResult(query="q", error="Neo.ClientError: boom").render() == "ERROR: Neo.ClientError: boom"
    assert QueryResult(query="q", columns=["n"]).render() == "0 rows (columns: n)"


# -----------------------------------------------------------------------------
# LLM client
# -----------------------------------------------------------------------------

def _client():
    return ChatClient(model="m", base_url="http://llm.invalid/v1", api_key="k",
                      max_retries=3, spec="fake:m")


def test_client_retries_rate_limits_then_parses_usage(monkeypatch):
    replies = iter([
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.0001},
            "provider": "Z.AI",
        }),
    ])
    monkeypatch.setattr("src.agent.llm.httpx.post", lambda *a, **k: next(replies))
    monkeypatch.setattr("src.agent.llm.time.sleep", lambda s: None)
    c = _client().complete([{"role": "user", "content": "x"}])
    assert (c.text, c.prompt_tokens, c.cost_usd, c.provider) == ("hi", 7, 0.0001, "Z.AI")


def test_client_does_not_retry_client_errors(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return httpx.Response(400, text="bad model")

    monkeypatch.setattr("src.agent.llm.httpx.post", post)
    with pytest.raises(LLMError, match="HTTP 400"):
        _client().complete([{"role": "user", "content": "x"}])
    assert len(calls) == 1
