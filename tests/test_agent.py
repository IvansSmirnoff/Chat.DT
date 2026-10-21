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
        reply = self.replies.pop(0)
        if isinstance(reply, Completion):
            return reply
        return Completion(text=reply, prompt_tokens=100,
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


def _cut_off():
    """What GLM-4.5-air returns when its thinking eats the whole budget."""
    return Completion(text="", prompt_tokens=100, completion_tokens=8192,
                      reasoning_tokens=8192, cost_usd=0.007, provider="P", finish_reason="length")


def test_single_shot_records_every_call():
    llm = ScriptedLLM([_act(cypher="MATCH (d:IfcDoor) RETURN count(d)"), _act(answer="254 doors.")])
    ep = loop.single_shot(124, "How many doors?", "SCHEMA", llm, FakeTools())
    assert [c["role"] for c in ep.calls] == ["generator", "synthesizer"]
    assert ep.calls[0]["completion_tokens"] == 10 and ep.calls[0]["provider"] == "P"


def test_single_shot_truncated_generator_is_a_token_limit_abstention_not_reasked():
    llm = ScriptedLLM([_cut_off()])
    tools = FakeTools()
    ep = loop.single_shot(93, "Insulation thickness?", "SCHEMA", llm, tools)
    assert ep.status == "token_limit" and ep.answer == loop.TOKEN_LIMIT_ANSWER
    assert ep.llm_calls == 1 and tools.queries == []
    assert ep.reasoning_tokens == 8192 and ep.calls[0]["finish"] == "length"
    assert "generator" in ep.error


def test_single_shot_truncated_but_complete_reply_is_used():
    # A reply can hit the limit after its JSON closed; it parses, so it counts.
    llm = ScriptedLLM([
        Completion(text=_act(cypher="RETURN 1"), finish_reason="length"),
        _act(answer="1 (from RETURN 1)."),
    ])
    ep = loop.single_shot(1, "q", "SCHEMA", llm, FakeTools())
    assert ep.status == "ok" and ep.llm_calls == 2


def test_single_shot_generator_may_answer_directly():
    llm = ScriptedLLM([_act(action="final_answer", answer="I cannot determine X: not in the model.")])
    tools = FakeTools()
    ep = loop.single_shot(130, "q", "SCHEMA", llm, tools)
    assert ep.status == "ok" and ep.answer.startswith("I cannot determine")
    assert ep.llm_calls == 1 and tools.queries == [] and ep.steps == []


def test_single_shot_decline_is_synthesised_from_no_query():
    llm = ScriptedLLM([
        _act(thoughts="no property describes the structural concept", cypher=""),
        _act(answer="I cannot determine the structural concept: it is not in the model."),
    ])
    tools = FakeTools()
    ep = loop.single_shot(133, "Structural concept?", "SCHEMA", llm, tools)
    assert ep.status == "ok" and ep.answer.startswith("I cannot determine")
    assert tools.queries == [] and ep.steps == []
    synth_user = llm.calls[1][1]["content"]
    assert "(none)" in synth_user and loop.NO_QUERY_RESULT in synth_user


def test_single_shot_persistent_garbage_is_a_scored_parse_error():
    llm = ScriptedLLM(["no json here"] * (loop.PARSE_RETRIES + 1))
    ep = loop.single_shot(1, "q", "SCHEMA", llm, FakeTools())
    assert ep.status == "parse_error" and ep.answer == loop.PARSE_ERROR_ANSWER
    assert ep.llm_calls == loop.PARSE_RETRIES + 1


def test_decline_is_only_parsed_when_allowed():
    reply = '{"thoughts": "nothing to query", "cypher": ""}'
    assert parse_action(reply, ("cypher", "final_answer", "decline")).kind == "decline"
    with pytest.raises(ActionParseError):
        parse_action(reply)


def test_agentic_truncated_step_costs_a_step_and_says_why():
    llm = ScriptedLLM([_cut_off(), _act(action="final_answer", answer="42 mm (from LayerThickness).")])
    ep = loop.agentic(93, "q", "SCHEMA", llm, FakeTools())
    assert ep.status == "ok" and ep.answer.startswith("42 mm")
    assert ep.steps[0].parse_error == loop.TRUNCATED_STEP
    assert "token limit" in llm.calls[1][1]["content"]


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


def test_client_parses_reasoning_tokens_and_truncation(monkeypatch):
    monkeypatch.setattr("src.agent.llm.httpx.post", lambda *a, **k: httpx.Response(200, json={
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 8192,
                  "completion_tokens_details": {"reasoning_tokens": 8192}},
    }))
    c = _client().complete([{"role": "user", "content": "x"}])
    assert c.truncated and c.reasoning_tokens == 8192 and c.text == ""


def test_client_from_spec_pins_provider_and_forces_reasoning(monkeypatch):
    from src.agent.llm import client_from_spec

    bodies = []

    def post(*a, json=None, **k):
        bodies.append(json)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("src.agent.llm.httpx.post", post)
    client_from_spec("openrouter:z-ai/glm-4.5-air", provider_order=["z-ai"],
                     reasoning="off").complete([{"role": "user", "content": "x"}])
    client_from_spec("openrouter:z-ai/glm-4.5-air").complete([{"role": "user", "content": "x"}])
    forced, default = bodies
    assert forced["reasoning"] == {"enabled": False}
    assert forced["provider"] == {"allow_fallbacks": False, "order": ["z-ai"]}
    assert forced["temperature"] == 0.1
    assert "reasoning" not in default and "order" not in default["provider"]


def test_reasoning_and_pinning_are_refused_off_openrouter(monkeypatch):
    from src.agent.llm import client_from_spec

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.invalid/v1")
    with pytest.raises(ValueError, match="OpenRouter"):
        client_from_spec("vllm:Qwen/Qwen2.5-7B-Instruct", reasoning="off")
    with pytest.raises(ValueError, match="OpenRouter"):
        client_from_spec("vllm:Qwen/Qwen2.5-7B-Instruct", provider_order=["x"])
    with pytest.raises(ValueError, match="reasoning"):
        ChatClient(model="m", base_url="http://x", reasoning="low")


def test_run_refuses_to_resume_with_other_settings(monkeypatch, tmp_path):
    from src.agent import run

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "config.json").write_text(json.dumps({"settings": {"reasoning": "on"}}))
    with pytest.raises(SystemExit):
        run.main(["--model-id", "dental_clinic__arc", "--split", "train", "--no-judge",
                  "--llm", "openrouter:z-ai/glm-4.5-air", "--reasoning", "off",
                  "--run", "r", "--out-root", str(tmp_path),
                  "--manifest", "data/ifcbench/manifest.yaml"])
