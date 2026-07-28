"""Unit tests for the pure SVR/SCR/EA scoring primitives.

No Neo4j required. Locks in the contracts that ``src.eval.aggregate``
and the Colab re-scoring flow rely on.
"""

from __future__ import annotations

from src.config import ExperimentSetting
from src.eval.aggregate import _is_scalar_gold, aggregate_results
from src.eval.scoring import (
    LIST_SEP,
    RECORD_SEP,
    EvaluationResult,
    OutputType,
    _coerce_direct_qa_token,
    _extract_global_id,
    calculate_ea_direct,
    calculate_ea_from_ids,
    calculate_scr_cypher,
    calculate_svr_json,
    tokenize_record,
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

    def test_scalar_int_matches_num_gold(self):
        """Gold ``RETURN count(n) = 5`` tokenises as ``{"num:5"}``. A Direct QA
        answer of ``[5]`` (raw int) must coerce to the same token, not stay as
        a string ``"5"`` that misses the gold set."""
        ea, parsed_ids, err = calculate_ea_direct("[5]", {"num:5"})
        assert err is None
        assert parsed_ids == {"num:5"}
        assert ea == 1.0

    def test_scalar_string_matches_num_gold(self):
        ea, parsed_ids, err = calculate_ea_direct('["5"]', {"num:5"})
        assert err is None
        assert parsed_ids == {"num:5"}
        assert ea == 1.0

    def test_scalar_float_matches_num_gold(self):
        ea, _, _ = calculate_ea_direct("[3.14]", {"num:3.14"})
        assert ea == 1.0

    def test_globalid_strings_untouched(self):
        gid = "1xS3BCk291UvhgP2dvNMKI"
        ea, parsed_ids, err = calculate_ea_direct(f'["{gid}"]', {gid})
        assert err is None
        assert parsed_ids == {gid}
        assert ea == 1.0


class TestScalarSCRGuards:
    def test_empty_query_does_not_inflate_scr(self):
        """Empty/whitespace output extracts zero labels + zero properties; the
        legacy code returned SCR=1.0 (vacuous compliance) and inflated batch
        summaries on runs where the model emitted nothing."""
        scr, invalid_labels, invalid_props = calculate_scr_cypher(
            "",
            valid_labels={"IfcWall"},
            valid_properties={"GlobalId"},
        )
        assert scr == 0.0
        assert invalid_labels == set() and invalid_props == set()

    def test_whitespace_only_query_does_not_inflate_scr(self):
        scr, _, _ = calculate_scr_cypher(
            "   \n\t  ",
            valid_labels={"IfcWall"},
            valid_properties={"GlobalId"},
        )
        assert scr == 0.0


class TestTokenizeRecord:
    """Locks the contract that group-by gold (``RETURN s.Name, count(d)``) does
    not collapse to a flat-cell set where mis-paired predictions score EA=1.0.
    """

    def test_single_cell_record_is_just_the_token(self):
        # Backwards compat: ``RETURN n.GlobalId`` returning one cell per row
        # should still yield the bare GlobalId in the set.
        assert tokenize_record(["gid-abc"]) == "gid-abc"
        assert tokenize_record([5]) == "num:5"
        assert tokenize_record([True]) == "bool:1"

    def test_multi_cell_record_joins_with_separator(self):
        token = tokenize_record(["Level 1", 5])
        assert token == f"Level 1{RECORD_SEP}num:5"

    def test_group_by_pairs_do_not_collide(self):
        # The classic group-by bug: same names, same counts, mis-paired.
        # Pre-fix the flat-cell set was {Level 1, Level 2, num:3, num:5} on
        # both sides → EA=1.0. Post-fix the tuples differ.
        gold = {tokenize_record(["Level 1", 5]), tokenize_record(["Level 2", 3])}
        pred = {tokenize_record(["Level 1", 3]), tokenize_record(["Level 2", 5])}
        assert gold != pred
        ea = calculate_ea_from_ids(pred, gold)
        # |∩|=0, |∪|=4 → EA must be 0 not 1.
        assert ea == 0.0

    def test_group_by_correct_pairing_scores_one(self):
        gold = {tokenize_record(["Level 1", 5]), tokenize_record(["Level 2", 3])}
        pred = {tokenize_record(["Level 1", 5]), tokenize_record(["Level 2", 3])}
        assert calculate_ea_from_ids(pred, gold) == 1.0

    def test_all_none_record_dropped(self):
        assert tokenize_record([None, None]) is None

    def test_record_with_values_method(self):
        class FakeRecord:
            def __init__(self, vals):
                self._vals = vals

            def values(self):
                return self._vals

        token = tokenize_record(FakeRecord(["Level 1", 5]))
        assert token == f"Level 1{RECORD_SEP}num:5"


class TestExtractGlobalIdList:
    """``RETURN collect(n.GlobalId)`` returns a list inside one cell. Pre-fix
    the fall-through returned None and the gold set ended up empty, sending
    the case into the trivial bucket and silently excluding it from EA_nt."""

    def test_collect_list_tokenises_to_sorted_bag(self):
        token = _extract_global_id(["g-b", "g-a", "g-c"])
        # Sorted to make the bag order-insensitive.
        assert token == f"g-a{LIST_SEP}g-b{LIST_SEP}g-c"

    def test_empty_list_returns_none(self):
        assert _extract_global_id([]) is None

    def test_mixed_list_with_nones_skips_them(self):
        token = _extract_global_id(["g-a", None, "g-b"])
        assert token == f"g-a{LIST_SEP}g-b"

    def test_record_with_storey_and_collect(self):
        # ``RETURN s.Name, collect(d.GlobalId)`` shape: per-storey bag of doors.
        token = tokenize_record(["Level 1", ["d-2", "d-1"]])
        assert token == f"Level 1{RECORD_SEP}d-1{LIST_SEP}d-2"


class TestCoerceDirectQAToken:
    """Defends against silent regressions in the Direct-QA → gold-token mapping
    that decides whether scalar gold (``num:5``) can ever match a JSON answer."""

    def test_int_becomes_num_prefixed(self):
        assert _coerce_direct_qa_token(5) == "num:5"

    def test_bool_becomes_bool_prefixed_not_num(self):
        # bool is a subclass of int — must be caught first.
        assert _coerce_direct_qa_token(True) == "bool:1"
        assert _coerce_direct_qa_token(False) == "bool:0"

    def test_numeric_string_coerces(self):
        assert _coerce_direct_qa_token("5") == "num:5"
        assert _coerce_direct_qa_token("3.14") == "num:3.14"

    def test_non_numeric_string_passes_through(self):
        gid = "1xS3BCk291UvhgP2dvNMKI"
        assert _coerce_direct_qa_token(gid) == gid
        assert _coerce_direct_qa_token("Level 1") == "Level 1"


class TestAggregateScalarSetSplit:
    @staticmethod
    def _result(gold_ids, ea=0.0, gen_ids=None):
        return EvaluationResult(
            question="q",
            experiment_setting=ExperimentSetting.CYPHER_SOFT,
            output_type=OutputType.CYPHER,
            generated_cypher="MATCH (n) RETURN n",
            gold_ids=set(gold_ids),
            generated_ids=set(gen_ids) if gen_ids is not None else set(),
            svr=1.0,
            scr=1.0,
            ea=ea,
        )

    def test_is_scalar_gold(self):
        assert _is_scalar_gold({"num:5"}) is True
        assert _is_scalar_gold({"num:5", "num:3"}) is True
        assert _is_scalar_gold({"bool:1"}) is True
        # Mixed: a count + a GlobalId → not pure scalar.
        assert _is_scalar_gold({"num:5", "gid-abc"}) is False
        assert _is_scalar_gold({"gid-abc"}) is False
        # Empty gold is trivial, not scalar.
        assert _is_scalar_gold(set()) is False

    def test_aggregate_splits_means(self):
        results = [
            self._result({"num:5"}, ea=1.0, gen_ids={"num:5"}),
            self._result({"num:3"}, ea=0.0, gen_ids={"num:7"}),
            self._result({"gid-a", "gid-b"}, ea=1.0, gen_ids={"gid-a", "gid-b"}),
            self._result({"gid-c"}, ea=0.5, gen_ids={"gid-c", "gid-extra"}),
        ]
        summary = aggregate_results(results)
        assert summary["scalar_count"] == 2
        assert summary["set_count"] == 2
        # Scalars: mean(1.0, 0.0) = 0.5
        assert summary["ea_mean_scalar"] == 0.5
        # Sets: mean(1.0, 0.5) = 0.75
        assert summary["ea_mean_set"] == 0.75
