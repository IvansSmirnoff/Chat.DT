"""Agent slice: action parsing, both paradigms, LLM client retries.

No Neo4j and no network: the LLM is scripted and the graph tools are faked.
"""

import hashlib
import json
import re
from pathlib import Path

import httpx
import pytest

from src.agent import loop, prompts
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
    assert ep.steps[0].reply == "not json"  # kept for diagnosis, not shown to the model
    assert "could not be parsed" in llm.calls[1][1]["content"]
    assert "not json" not in llm.calls[1][1]["content"]


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
    assert "last reply: 'no json here'" in ep.error


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


def test_stored_steps_keep_only_what_the_model_was_shown():
    big = {"columns": ["n"], "rows": [[{"labels": ["IfcSpace"], "Name": "x" * 200}]] * 50,
           "total_rows": 5000}
    llm = ScriptedLLM([_act(action="cypher", cypher="BIG"),
                       _act(action="cypher", cypher="SMALL"),
                       _act(action="final_answer", answer="5000 spaces (count of IfcSpace).")])
    tools = FakeTools({"BIG": big, "SMALL": {"columns": ["c"], "rows": [[3]], "total_rows": 1}})
    ep = loop.agentic(1, "q", "SCHEMA", llm, tools)
    full = ep.steps[0].observation
    assert len(full) > loop.RESULT_CHARS
    # The model's view is unchanged: the next prompt holds the RESULT_CHARS clip.
    assert loop._clip(full, loop.RESULT_CHARS) in llm.calls[1][1]["content"]

    big_step, small_step = ep.to_dict()["steps"]
    assert big_step["observation"] == loop._clip(full, loop.RESULT_CHARS)
    assert big_step["observation_chars"] == len(full) and big_step["observation_clipped"]
    assert "rows" not in big_step["result"]
    assert big_step["result"]["total_rows"] == 5000 and big_step["result"]["error"] is None
    assert small_step["observation"] == "1 row(s); columns: c\n[3]"
    assert not small_step["observation_clipped"]
    assert len(json.dumps(ep.to_dict())) < 3 * loop.RESULT_CHARS
    assert ep.steps[0].result["rows"]  # in memory nothing is dropped


def test_stored_episodes_still_feed_the_readers(tmp_path):
    import importlib.util

    from src.agent.run import summarise

    llm = ScriptedLLM([_act(action="cypher", cypher="BAD"), _act(action="cypher", cypher="GOOD"),
                       _act(action="final_answer", answer="3 (from GOOD).")])
    tools = FakeTools({"BAD": {"error": "Neo.ClientError.Statement.SyntaxError: x"}})
    rec = {"model_id": "m", "question_id": 1, "paradigm": "agentic", "category": 1,
           "episode": loop.agentic(1, "q", "SCHEMA", llm, tools).to_dict(),
           "judgement": {"verdict": "correct"}}
    rec = json.loads(json.dumps(rec))
    assert rec["episode"]["final_cypher"] == "GOOD"
    assert summarise([rec])["agentic"]["query_error_rate"] == 0.5

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps(rec) + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "grammar_coverage.py"
    spec = importlib.util.spec_from_file_location("grammar_coverage", script)
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    assert list(coverage.executed_queries([episodes])) == [("m", "GOOD")]


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
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ValueError, match="OpenRouter"):
        client_from_spec("gemini:gemini-2.5-flash", reasoning="off")
    with pytest.raises(ValueError, match="OpenRouter"):
        client_from_spec("vllm:Qwen/Qwen2.5-7B-Instruct", provider_order=["x"])
    with pytest.raises(ValueError, match="reasoning"):
        ChatClient(model="m", base_url="http://x", reasoning="low")


def test_vllm_reasoning_switch_goes_through_the_chat_template(monkeypatch):
    from src.agent.llm import client_from_spec

    bodies = []
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.invalid/v1")
    monkeypatch.setattr("src.agent.llm.httpx.post", lambda *a, json=None, **k: (
        bodies.append(json), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))[1])
    client = client_from_spec("vllm:Qwen/Qwen3-8B-AWQ", reasoning="off")
    client.complete([{"role": "user", "content": "x"}], structured_outputs={"grammar": "start: \"x\""})
    assert client.timeout_s == 600.0
    body = bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["structured_outputs"] == {"grammar": "start: \"x\""}
    assert "reasoning" not in body  # OpenRouter's field means nothing to vLLM


def test_grammar_constrains_query_writing_calls_only():
    class Recording(ScriptedLLM):
        def complete(self, messages, **kw):
            self.kwargs = getattr(self, "kwargs", []) + [kw]
            return super().complete(messages)

    g = {"structured_outputs": {"grammar": "G"}}
    llm = Recording([_act(thoughts="q", cypher="MATCH (d:IfcDoor) RETURN count(d)"),
                     _act(thoughts="s", answer="254 doors")])
    loop.single_shot(1, "How many doors?", "SCHEMA", llm, FakeTools(), grammar="G")
    assert llm.kwargs == [g, {}]  # generator constrained, synthesiser free

    llm = Recording([_act(action="cypher", thoughts="q", cypher="MATCH (d:IfcDoor) RETURN count(d)"),
                     _act(action="final_answer", thoughts="s", answer="254")])
    loop.agentic(1, "How many doors?", "SCHEMA", llm, FakeTools(), grammar="G")
    assert llm.kwargs == [g, g]

    llm = Recording([_act(action="final_answer", thoughts="s", answer="254")])
    loop.agentic(1, "How many doors?", "SCHEMA", llm, FakeTools())
    assert llm.kwargs == [{}]


def test_run_refuses_a_grammar_off_vllm(monkeypatch, tmp_path):
    from src.agent import run

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(SystemExit):
        run.main(["--model-id", "dental_clinic__arc", "--split", "train", "--no-judge",
                  "--llm", "openrouter:z-ai/glm-4.5-air", "--grammar", "cfg",
                  "--run", "r", "--out-root", str(tmp_path),
                  "--manifest", "data/ifcbench/manifest.yaml"])


def test_summary_reports_both_hellin_accuracies():
    from src.agent.run import summarise

    def rec(verdict, cat=1):
        return {"paradigm": "agentic", "category": cat, "judgement": {"verdict": verdict},
                "episode": {"n_queries": 1, "cost_usd": 0.0, "wall_s": 1.0, "status": "ok",
                            "steps": [{"cypher": "q", "result": {"error": err}}
                                      for err in ([None, "SyntaxError"] if verdict == "wrong" else [None])]}}

    cells = summarise([rec("correct"), rec("wrong"), rec("abstained"), rec("abstained", 4)])
    a = cells["agentic"]
    assert (a["accuracy"], a["accuracy_attempted"], a["abstention_rate"]) == (0.25, 0.5, 0.5)
    assert a["query_error_rate"] == 0.2  # 1 refused of 5 executed
    # Nothing attempted: the secondary metric is undefined, not zero.
    assert summarise([rec("abstained")])["agentic"]["accuracy_attempted"] is None


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


# -----------------------------------------------------------------------------
# Prompt variants
# -----------------------------------------------------------------------------

# sha256 of the prompts as ported from Hellin, rendered with the inputs below
# (taken before --prompt-variant existed). Update only on purpose: a change
# here changes what the hellin arm is compared against.
HELLIN_SHA256 = {
    "generator": "c96f204a7695809b574b1791f5b4255237acc4206eb530da4aebc530a10260f8",
    "synthesizer": "950e4b0eeaa0b0ae60aface2ad28538003223bab00af1c2e158a9a7a5870ae0f",
    "agent": "7651257f4303a80561afa1bd8030d2fbb95c8fafeeeb73c33abafa633ec31870",
}
AGENT_KW = {"max_steps": 25, "max_rows": 50, "timeout": 30.0}


def _sha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


def _rendered(**variant):
    return {
        "generator": prompts.static_generator_messages("Q?", "SCHEMA"),
        "synthesizer": prompts.synthesizer_messages("Q?", "MATCH (n) RETURN n", "RESULT", **variant),
        "agent": prompts.agent_messages("Q?", "SCHEMA", "PREV", **AGENT_KW, **variant),
    }


def test_default_prompts_are_hellins_byte_for_byte():
    default = _rendered()
    assert {k: _sha(m) for k, m in default.items()} == HELLIN_SHA256
    assert _rendered(variant="hellin") == default


def test_ours_adds_the_addendum_to_the_answer_writing_prompts_only():
    hellin, ours = _rendered(), _rendered(variant="ours")
    block = "\n" + prompts.ANSWER_ADDENDUM
    for call in ("synthesizer", "agent"):
        system = ours[call][0]["content"]
        assert system.count(prompts.ANSWER_ADDENDUM) == 1
        assert system.replace(block, "", 1) == hellin[call][0]["content"]  # nothing else moved
        assert ours[call][1] == hellin[call][1]                             # user turn untouched
    # After Hellin's criteria (and the synthesiser's abstain line), before the reply format.
    synth = ours["synthesizer"][0]["content"]
    assert synth.index("(see ABSTENTION).") < synth.index("## ANSWER WRITING") < synth.index("## OUTPUT FORMAT")
    agent = ours["agent"][0]["content"]
    assert agent.index("### 5. ABSTENTION") < agent.index("## ANSWER WRITING") < agent.index("## ACTION SELECTION")
    # The generator only writes a query: it has no variant.
    assert "ANSWER WRITING" not in json.dumps(ours["generator"])
    with pytest.raises(ValueError, match="prompt variant"):
        prompts.synthesizer_messages("Q?", "", "R", variant="mine")


def test_addendum_is_generic_and_short():
    assert not re.search(r"categor|Ifc[A-Z]|\bQ\d", prompts.ANSWER_ADDENDUM, re.IGNORECASE)
    assert len(prompts.ANSWER_ADDENDUM.split()) < 160


def test_ours_reaches_the_answer_calls_and_leaves_the_grammar_alone():
    class Recording(ScriptedLLM):
        def complete(self, messages, **kw):
            self.kwargs = getattr(self, "kwargs", []) + [kw]
            return super().complete(messages)

    g = {"structured_outputs": {"grammar": "G"}}
    query = "MATCH (d:IfcDoor) RETURN count(d)"
    llm = Recording([_act(thoughts="q", cypher=query), _act(thoughts="s", answer="254 doors")])
    ep = loop.single_shot(1, "How many doors?", "SCHEMA", llm, FakeTools(), grammar="G",
                          prompt_variant="ours")
    assert llm.kwargs == [g, {}]
    assert llm.calls[0] == prompts.static_generator_messages("How many doors?", "SCHEMA")
    assert prompts.ANSWER_ADDENDUM in llm.calls[1][0]["content"]
    assert ep.to_dict()["prompt_variant"] == "ours"

    llm = Recording([_act(action="cypher", thoughts="q", cypher=query),
                     _act(action="final_answer", thoughts="s", answer="254")])
    ep = loop.agentic(1, "How many doors?", "SCHEMA", llm, FakeTools(), grammar="G",
                      prompt_variant="ours")
    assert llm.kwargs == [g, g]
    assert all(prompts.ANSWER_ADDENDUM in call[0]["content"] for call in llm.calls)
    assert ep.to_dict()["prompt_variant"] == "ours"

    llm = ScriptedLLM([_act(action="final_answer", thoughts="s", answer="254")])
    ep = loop.agentic(1, "How many doors?", "SCHEMA", llm, FakeTools())
    assert prompts.ANSWER_ADDENDUM not in llm.calls[0][0]["content"]
    assert ep.to_dict()["prompt_variant"] == "hellin"


def test_an_answer_in_the_addendums_style_fits_the_agent_grammar():
    pytest.importorskip("xgrammar")
    from src.constraints.cfg_grammar import build_reply_grammar, check_reply

    grammar = build_reply_grammar(["IfcSpace"], ["CONTAINS"], ["Name"], reply="agent")
    answer = ('Found 12 of the 14 rooms (IfcSpace nodes on the storey, 2 without a Name excluded). '
              'Assuming "office" means a room whose name contains "Office", the answer is 12.\n'
              'Assuming a 2.5 m ceiling height (general knowledge, not from the BIM model), ...')
    reply = json.dumps({"action": "final_answer", "thoughts": "partial", "answer": answer})
    assert check_reply(grammar, reply) is None
