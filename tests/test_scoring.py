"""Unit tests for the pure SVR/SCR/EA scoring primitives.

No Neo4j required. Locks in the contracts that ``src.eval.aggregate``
and the Colab re-scoring flow rely on.
"""

from __future__ import annotations

from src.eval.scoring import (
    calculate_ea_direct,
    calculate_ea_from_ids,
    calculate_scr_cypher,
    calculate_svr_json,
)


class TestSVRJson:
    def test_valid_list_of_strings(self):
        svr, ok, err, parsed = calculate_svr_json('["a", "b", "c"]')
        assert (svr, ok, err) == (1.0, True, None)
        assert parsed == ["a", "b", "c"]

    def test_valid_list_wrapped_in_code_fence(self):
        svr, ok, _, parsed = calculate_svr_json('```json\n["x"]\n```')
        assert ok is True and svr == 1.0
        assert parsed == ["x"]

    def test_invalid_json_scores_zero(self):
        svr, ok, err, parsed = calculate_svr_json("not json at all")
        assert svr == 0.0 and ok is False
        assert err is not None
        assert parsed is None

    def test_non_list_scores_zero(self):
        svr, ok, _, _ = calculate_svr_json('{"a": 1}')
        assert ok is False and svr == 0.0


class TestSCRCypher:
    def test_all_labels_and_props_valid(self):
        query = "MATCH (w:IfcWall) WHERE w.FireRating = 'F60' RETURN w.GlobalId"
        scr, invalid_labels, invalid_props = calculate_scr_cypher(
            query,
            valid_labels={"IfcWall"},
            valid_properties={"FireRating", "GlobalId"},
        )
        assert scr == 1.0
        assert invalid_labels == set()
        assert invalid_props == set()

    def test_unknown_label_penalises(self):
        query = "MATCH (w:IfcUnicorn) RETURN w.GlobalId"
        scr, invalid_labels, _ = calculate_scr_cypher(
            query,
            valid_labels={"IfcWall"},
            valid_properties={"GlobalId"},
        )
        assert scr < 1.0
        assert "IfcUnicorn" in invalid_labels


class TestEAFromIDs:
    def test_identical_sets_score_one(self):
        assert calculate_ea_from_ids({"A", "B"}, {"A", "B"}) == 1.0

    def test_disjoint_sets_score_zero(self):
        assert calculate_ea_from_ids({"A"}, {"B"}) == 0.0

    def test_partial_overlap_is_jaccard(self):
        # |∩|=1, |∪|=3 → 1/3
        assert abs(calculate_ea_from_ids({"A", "B"}, {"A", "C"}) - 1 / 3) < 1e-9

    def test_both_empty_scores_one(self):
        assert calculate_ea_from_ids(set(), set()) == 1.0


class TestEADirect:
    def test_parses_json_and_scores_jaccard(self):
        ea, parsed_ids, err = calculate_ea_direct('["A", "B"]', {"A", "C"})
        assert err is None
        assert parsed_ids == {"A", "B"}
        assert abs(ea - 1 / 3) < 1e-9

    def test_invalid_json_returns_zero_with_error(self):
        ea, parsed_ids, err = calculate_ea_direct("oops", {"A"})
        assert ea == 0.0
        assert err is not None
