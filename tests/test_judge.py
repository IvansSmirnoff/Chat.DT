"""Judge port: verdict rule and reply parsing (no network)."""

import pytest

from src.agent.llm import Completion
from src.eval.judge import JudgeParseError, derive_verdict, judge_answer, parse_judgement

ALL_YES = {"faithfulness": "Yes", "completeness": "Yes", "transparency": "Yes", "relevance": "Yes"}


def test_verdict_rule_matches_derive_binary_classification():
    assert derive_verdict(False, ALL_YES) == "correct"
    assert derive_verdict(False, {**ALL_YES, "transparency": "No"}) == "wrong"
    # 'Na' on an answered question is a failure, as in Hellin's rule.
    assert derive_verdict(False, {**ALL_YES, "completeness": "Na"}) == "wrong"
    assert derive_verdict(True, {c: "Na" for c in ALL_YES}) == "abstained"


def test_parse_normalises_case_and_string_booleans():
    parsed = parse_judgement('```json\n{"abstention": "false", "faithfulness": "yes", '
                             '"completeness": "YES", "transparency": "No", "relevance": "Yes", '
                             '"justification": "no source cited",}\n```')
    assert parsed["abstention"] is False
    assert parsed["faithfulness"] == "Yes" and parsed["transparency"] == "No"


def test_parse_rejects_unknown_values():
    with pytest.raises(JudgeParseError):
        parse_judgement('{"abstention": false, "faithfulness": "maybe", "completeness": "Yes", '
                        '"transparency": "Yes", "relevance": "Yes"}')


class OneReplyJudge:
    spec = "fake:judge"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.messages = None

    def complete(self, messages, **_):
        self.messages = messages
        return Completion(text=self.texts.pop(0), cost_usd=0.002)


def test_judge_sees_question_reference_and_answer_but_no_trace():
    client = OneReplyJudge('{"abstention": false, "faithfulness": "Yes", "completeness": "Yes", '
                           '"transparency": "Yes", "relevance": "Yes", "justification": "ok"}')
    r = judge_answer(client, question="How many doors?", category=2,
                     ground_truth="There are 254 doors", answer="254 doors (counted from IfcDoor)")
    assert r.verdict == "correct" and r.cost_usd == pytest.approx(0.002)
    user = client.messages[1]["content"]
    assert "QUESTION CATEGORY: 2" in user and "REFERENCE ANSWER: There are 254 doors" in user
    assert "SYSTEM RESPONSE: 254 doors (counted from IfcDoor)" in user


def test_judge_reasks_after_an_unparseable_reply():
    client = OneReplyJudge("I think it is correct.",
                           '{"abstention": true, "faithfulness": "Na", "completeness": "Na", '
                           '"transparency": "Na", "relevance": "Na", "justification": "refused"}')
    r = judge_answer(client, question="q", category=4, ground_truth="g", answer="I cannot determine X")
    assert r.verdict == "abstained" and r.cost_usd == pytest.approx(0.004)
