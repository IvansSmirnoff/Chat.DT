"""Answer-key scoring of the authored questions (fake extractor, no network, no Neo4j)."""

import json
import sys
from pathlib import Path

import pytest

from src.agent.llm import Completion
from src.eval.answer_key import (
    EXTRACTOR_SYSTEM,
    check_claim,
    compare_claim,
    derive_schema,
    extract_claim,
    norm_text,
    render_template,
    same,
    score_run,
)

ROUTE_Q = {"doors": 6, "routes": [["2B3", "2BC5", "2BC4", "2BC3", "2D11"]]}
PAIRS_Q = {"toilets": [["2C07", "2CC3"], ["2C24", "2AC2"]]}
WALL_Q = {"wall_types": ["Interior - Rated 1-HR (92mm Stud)"], "segments": 3,
          "layer_sets": [[["Type-X Plasterboard", 0.016], ["Metal - Stud Layer", 0.092],
                          ["Type-X Plasterboard", 0.016]]], "thickness_m": 0.124}


class FakeExtractor:
    spec = "fake:extractor"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return Completion(text=self.texts.pop(0), cost_usd=0.001, provider="Fake")


def matches(claim, answer):
    return compare_claim(claim, answer, derive_schema(answer))


# -- schema ------------------------------------------------------------------

def test_schema_hides_values_and_route_lengths():
    schema = derive_schema(ROUTE_Q)
    template = render_template(schema)
    assert template == ('{\n  "doors": integer,\n'
                        '  "routes": list of (ordered list of string), any order, '
                        'written [["<string>", ...], ...]\n}')
    for value in ("2B3", "2BC5", "2D11", "6"):
        assert value not in template and value not in json.dumps(schema)
    # A longer gold route gives the same schema: the route's length is not shown.
    longer = {"doors": 9, "routes": [["A", "B", "C", "D", "E", "F", "G", "H"]]}
    assert derive_schema(longer) == schema


def test_schema_shows_tuple_arity_and_scalar_types():
    assert render_template(derive_schema(PAIRS_Q)) == (
        '{\n  "toilets": list of [string, string], any order, written [["<string>", "<string>"], ...]\n}')
    template = render_template(derive_schema(WALL_Q))
    assert ('"layer_sets": list of (ordered list of [string, number]), any order, '
            'written [[["<string>", <number>], ...], ...]') in template
    assert '"segments": integer' in template and '"thickness_m": number' in template
    assert render_template(derive_schema({"reachable": False})) == '{\n  "reachable": true or false\n}'
    assert "Plasterboard" not in template and "0.016" not in template


# -- comparison --------------------------------------------------------------

def test_sets_ignore_order_case_and_duplicates():
    answer = {"rooms": ["1C14", "1C15", "1CC3"]}
    assert matches({"rooms": ["1cc3", "1C15", "1C14", "1C15"]}, answer) == {"rooms": True}
    assert matches({"rooms": ["1C14", "1C15"]}, answer) == {"rooms": False}
    assert matches({"rooms": ["1C14", "1C15", "1CC3", "1D16"]}, answer) == {"rooms": False}
    assert matches({"rooms": None}, answer) == {"rooms": False}


def test_numbers_within_one_percent_and_integers_exact():
    answer = {"rooms": ["1D28"], "total_area_m2": 110.817}
    assert matches({"rooms": ["1D28"], "total_area_m2": 110.82}, answer)["total_area_m2"]
    assert matches({"rooms": ["1D28"], "total_area_m2": 111.9}, answer)["total_area_m2"]       # 0.98%
    assert not matches({"rooms": ["1D28"], "total_area_m2": 103.73}, answer)["total_area_m2"]
    assert not matches({"doors": 8, "routes": None}, ROUTE_Q)["doors"]
    assert matches({"doors": 6, "routes": None}, ROUTE_Q)["doors"]


def test_booleans_are_compared_by_identity():
    answer = {"reachable": False, "reachable_spaces": ["1E22"]}
    assert matches({"reachable": False, "reachable_spaces": ["1E22"]}, answer)["reachable"]
    assert not matches({"reachable": True, "reachable_spaces": ["1E22"]}, answer)["reachable"]
    assert not matches({"reachable": None, "reachable_spaces": ["1E22"]}, answer)["reachable"]


def test_routes_match_any_gold_route_and_complete_missing_endpoints():
    answer = {"doors": 2, "routes": [["1D14", "1D12", "1DC3"], ["1D14", "1D13", "1DC3"]]}
    assert matches({"doors": 2, "routes": [["1D14", "1D12", "1DC3"]]}, answer)["routes"]
    assert matches({"doors": 2, "routes": [["1D14", "1D13", "1DC3"]]}, answer)["routes"]
    # "you pass through 1D12": start and destination come from the question
    assert matches({"doors": 2, "routes": [["1D12"]]}, answer)["routes"]
    assert matches({"doors": 2, "routes": [["1D14", "1D12"]]}, answer)["routes"]
    # order matters inside a route; every claimed route must be a gold one
    assert not matches({"doors": 2, "routes": [["1DC3", "1D12", "1D14"]]}, answer)["routes"]
    assert not matches({"doors": 2, "routes": [["1D14", "1D12", "1DC3"], ["1D14", "1D11", "1DC3"]]},
                       answer)["routes"]
    assert not matches({"doors": 2, "routes": []}, answer)["routes"]
    assert not matches({"doors": 2, "routes": [[]]}, answer)["routes"]


def test_pairs_are_multisets_inside_a_set():
    assert matches({"toilets": [["2CC3", "2C07"], ["2C24", "2AC2"]]}, PAIRS_Q) == {"toilets": True}
    assert not matches({"toilets": [["2C07", "2CC3"], ["2C24", "2CC1"]]}, PAIRS_Q)["toilets"]
    assert not matches({"toilets": [["2C07", "2CC3"], ["2C24", None]]}, PAIRS_Q)["toilets"]
    assert not matches({"toilets": [["2C07", "2CC3"], ["2C08", "2CC1"], ["2C24", "2AC2"]]},
                       PAIRS_Q)["toilets"]


def test_layer_build_up_keeps_order_and_revit_names_reduce_to_the_type():
    claim = {"wall_types": ["Basic Wall: Interior - Rated 1-HR (92mm Stud)"], "segments": 3,
             "layer_sets": [[["Type-X Plasterboard", 0.016], ["Metal – Stud Layer", 0.092],
                             ["type-x plasterboard", 0.016]]], "thickness_m": 0.124}
    assert all(matches(claim, WALL_Q).values())
    swapped = {**claim, "layer_sets": [[["Metal - Stud Layer", 0.092], ["Type-X Plasterboard", 0.016],
                                        ["Type-X Plasterboard", 0.016]]]}
    assert not matches(swapped, WALL_Q)["layer_sets"]
    assert norm_text("Basic Wall:Interior - Rated 1-HR (92mm Stud):128426") == "interior - rated 1-hr (92mm stud)"
    assert norm_text("  First   Floor ") == "first floor"


def test_space_numbers_given_with_their_long_names():
    route = {"doors": 3, "routes": [["2C16", "2C17", "2CC5", "2C15"]]}
    claim = {"doors": 3, "routes": [["CERAMIC LAB 2C16", "2C17 \u2013 PROSTH. LAB", "CORRIDOR (2CC5)", "2C15"]]}
    assert matches(claim, route)["routes"]
    assert matches({"rooms": ["1E15-A GAS STORAGE"]}, {"rooms": ["1E15-A"]})["rooms"]
    # a name without the number, or with two numbers, is not that space
    assert not matches({"doors": 3, "routes": [["2C16", "CORRIDOR", "2CC5", "2C15"]]}, route)["routes"]
    assert not matches({"rooms": ["2C16 or 2C17"]}, {"rooms": ["2C16"]})["rooms"]
    # and the rule only applies to space numbers, not to other names
    assert not matches({"floors": ["Floor First"]}, {"floors": ["First Floor"]})["floors"]


def test_same_keeps_the_verify_script_semantics():
    assert same([3, [1, 2]], [[1, 2], 3], 1e-3, ordered=False)
    assert not same([[2, 1]], [[1, 2]], 1e-3, ordered=False)        # nested lists keep their order
    assert same(110.8177, 110.817, 1e-3) and not same(111.0, 110.817, 1e-3)
    assert not same(1, True, 1e-3)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ifcbench"))
    import verify_authored
    assert verify_authored.same is same
    assert verify_authored.compare([{"rooms": ["B", "A"]}], {"rooms": ["A", "B"]}, 1e-3) == (True, "")


# -- extraction ----------------------------------------------------------------

def test_extractor_sees_schema_not_values_and_is_reasked_once():
    client = FakeExtractor('{"doors": "eight", "routes": null}',
                           '```json\n{"doors": 8.0, "routes": [["2B3", "2D11"]]}\n```')
    ex = extract_claim(client, "How many doors from 2B3 to 2D11?", derive_schema(ROUTE_Q), "Eight doors.")
    assert ex.claim == {"doors": 8, "routes": [["2B3", "2D11"]]} and ex.error is None
    assert ex.attempts == 2 and ex.cost_usd == pytest.approx(0.002)
    first_messages, kwargs = client.calls[0]
    assert first_messages[0]["content"] == EXTRACTOR_SYSTEM
    user = first_messages[1]["content"]
    assert "Eight doors." in user and "2BC5" not in user and '"doors": integer' in user
    assert kwargs["response_format"] == {"type": "json_object"}
    retry = client.calls[1][0][-1]["content"]
    assert "doors: expected an integer" in retry


def test_malformed_twice_is_recorded_as_a_failure():
    client = FakeExtractor("no json here", '{"toilets": [["2C07"]]}')
    ex = extract_claim(client, "q", derive_schema(PAIRS_Q), "a")
    assert ex.claim is None and ex.attempts == 2
    assert "expected 2 parts" in ex.error
    assert check_claim({"toilets": [], "extra": 1}, derive_schema(PAIRS_Q))[1] == ["keys not in the template ['extra']"]


# -- runs: abstentions, cache, summary ------------------------------------------

def _question(qid, answer, needs="base", kind="path_route"):
    return {"id": qid, "question": f"question {qid}", "ground_truth": "gt", "category": 3,
            "split": "train", "project": "dental_clinic", "ifc_model": "arc", "source": "authored",
            "kind": kind, "hops": 2, "needs": needs, "answer": answer}


def _record(qid, paradigm, answer, verdict, status="ok"):
    return {"model_id": "dental_clinic__arc", "question_id": qid, "paradigm": paradigm,
            "category": 3, "question": f"question {qid}",
            "episode": {"status": status, "answer": answer},
            "judgement": None if verdict is None else {"verdict": verdict}}


def _write_run(tmp_path, records):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    with open(run / "episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return run


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "authored.jsonl"
    rows = [_question(1, {"doors": 6, "routes": [["A", "B", "C"]]}),
            _question(2, {"rooms": ["1C14", "1C15"]}, needs="geometry", kind="adjacency")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from src.eval.answer_key import load_questions
    return path, load_questions(path)


EXTRACTOR = {"spec": "fake:extractor", "provider": None, "reasoning": "default"}


def test_score_run_abstentions_confusion_and_cache(tmp_path, questions):
    qpath, qs = questions
    run = _write_run(tmp_path, [
        _record(1, "single_shot", "An earlier record of this cell", "wrong"),  # superseded: latest wins
        _record(1, "agentic", "Eight doors: A, B, C.", "correct"),           # judge passes, key fails
        _record(2, "agentic", "1C14 and 1C15.", "wrong"),                     # judge fails, key passes
        _record(1, "single_shot", "I cannot determine the answer: the step limit was reached.",
                "abstained", status="step_limit"),                            # no extraction
        _record(2, "single_shot", "I cannot tell which rooms.", "abstained"),  # extracted, all null
    ])
    client = FakeExtractor('{"doors": 8, "routes": [["A", "B", "C"]]}',
                           '{"rooms": ["1C15", "1C14"]}',
                           '{"rooms": null}')
    summary = score_run(run, qs, qpath, client, EXTRACTOR, workers=1)
    assert len(client.calls) == 3

    rows = {(r["question_id"], r["paradigm"]): r
            for r in map(json.loads, (run / "answer_key.jsonl").read_text().splitlines())}
    assert rows[(1, "agentic")]["match"] == {"doors": False, "routes": True}
    assert rows[(1, "agentic")]["stated_match"] is False and rows[(1, "agentic")]["correct"] is False
    assert rows[(2, "agentic")]["correct"] is True
    assert rows[(1, "single_shot")]["extraction_status"] == "not_extracted"
    assert rows[(1, "single_shot")]["abstained_by"] == ["judge", "episode_step_limit"]
    assert rows[(2, "single_shot")]["abstained_by"] == ["judge", "no_claim"]
    assert rows[(2, "single_shot")]["correct"] is False

    agentic = summary["paradigms"]["agentic"]
    assert agentic["key_correct"] == 1 and agentic["judge_correct"] == 1
    assert agentic["disagreements"]["judge_correct_key_mismatch"] == [1]
    assert agentic["disagreements"]["judge_wrong_key_match"] == [2]
    assert agentic["confusion"] == {"correct": {"match": 0, "mismatch": 1},
                                    "wrong": {"match": 1, "mismatch": 0}}
    assert agentic["by_needs"]["geometry"]["key_accuracy"] == 1.0
    assert summary["paradigms"]["single_shot"]["key_abstained"] == 2
    assert summary["extraction_cost_usd"]["this_invocation"] == pytest.approx(0.003)

    # Re-running costs nothing: every extraction comes from the cache.
    again = FakeExtractor()
    summary2 = score_run(run, qs, qpath, again, EXTRACTOR, workers=1)
    assert again.calls == [] and summary2["extraction_cost_usd"]["this_invocation"] == 0
    assert summary2["paradigms"] == summary["paradigms"]

    # A changed answer (a resumed run appended a new record) is extracted again; others are not.
    with open(run / "episodes.jsonl", "a") as f:
        f.write(json.dumps(_record(1, "agentic", "Six doors: A, B, C.", "correct")) + "\n")
    third = FakeExtractor('{"doors": 6, "routes": [["A", "B", "C"]]}')
    summary3 = score_run(run, qs, qpath, third, EXTRACTOR, workers=1)
    assert len(third.calls) == 1
    assert summary3["paradigms"]["agentic"]["key_correct"] == 2

    # Another extractor spec is another cache key; --cache-only (client None) calls nothing.
    other = {**EXTRACTOR, "spec": "fake:other"}
    summary4 = score_run(run, qs, qpath, None, other, workers=1)
    assert summary4["paradigms"]["agentic"]["extraction_failed"] == [1, 2]


def test_failed_extraction_is_cached_and_retried_only_on_request(tmp_path, questions):
    qpath, qs = questions
    run = _write_run(tmp_path, [_record(2, "agentic", "1C14 and 1C15.", "correct")])
    bad = FakeExtractor("nope", "still nope")
    summary = score_run(run, qs, qpath, bad, EXTRACTOR, workers=1)
    assert summary["paradigms"]["agentic"]["extraction_failed"] == [2]
    assert summary["paradigms"]["agentic"]["key_correct"] == 0

    idle = FakeExtractor()
    score_run(run, qs, qpath, idle, EXTRACTOR, workers=1)
    assert idle.calls == []

    good = FakeExtractor('{"rooms": ["1C14", "1C15"]}')
    summary = score_run(run, qs, qpath, good, EXTRACTOR, workers=1, retry_failed=True)
    assert len(good.calls) == 1 and summary["paradigms"]["agentic"]["key_correct"] == 1
