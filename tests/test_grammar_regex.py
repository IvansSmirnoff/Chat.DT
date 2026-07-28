"""Round-trip tests for the Cypher grammar regex builder.

Every gold Cypher query we ship must full-match the regex we generate from the
combined vocabulary. If ``build_cypher_regex`` drifts, ``CYPHER_STRICT``
(Outlines) stops being able to produce the right answer at all — the FSM
simply cannot emit a query the grammar does not describe.

Two fixtures, deliberately:

* ``synthetic_regex`` — a tiny hand-made vocabulary, for the negative tests
  (what the grammar must REJECT). Cheap and independent of any data file.
* ``bundle_regex`` — the real Building 15 vocabulary. The positive coverage
  tests run against this. An earlier version of this module tested gold
  coverage against the synthetic vocabulary only, which is why a grammar that
  could not express 13 of the 33 shipped gold queries passed CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.constraints.grammar import build_cypher_regex


DATA_DIR = Path(__file__).parent.parent / "data"
BUNDLE_PATH = DATA_DIR / "bundle_b15_ids.json"
GOLD_FILES = ("ch9_demo_b15.json", "test_set_b15.json")


# ---------------------------------------------------------------------------
# Synthetic fixture — negative tests only
# ---------------------------------------------------------------------------

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
def synthetic_regex() -> str:
    return build_cypher_regex(ENTITIES, PROPERTIES)


# ---------------------------------------------------------------------------
# Real-vocabulary fixture — gold coverage
# ---------------------------------------------------------------------------


def _load_bundle_vocabulary():
    from src.constraints.vocabulary_merger import CombinedVocabulary

    with open(BUNDLE_PATH, encoding="utf-8") as f:
        bundle = json.load(f)
    return CombinedVocabulary.from_dict(bundle["combined_vocabulary"])


def _load_gold_queries():
    """(file, index, query) for every gold query shipped in data/."""
    cases = []
    for name in GOLD_FILES:
        path = DATA_DIR / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for i, case in enumerate(json.load(f)):
                cases.append(pytest.param(case["gold_cypher"], id=f"{name}[{i}]"))
    return cases


requires_bundle = pytest.mark.skipif(
    not BUNDLE_PATH.exists(),
    reason=f"{BUNDLE_PATH.name} not built; run scripts/build_bundle.py",
)


@pytest.fixture(scope="module")
def bundle_regex() -> str:
    from src.constraints.grammar import build_cypher_regex_from_vocabulary

    return build_cypher_regex_from_vocabulary(_load_bundle_vocabulary())


@requires_bundle
@pytest.mark.parametrize("gold_query", _load_gold_queries())
def test_gold_queries_match_regex(bundle_regex, gold_query):
    assert re.fullmatch(bundle_regex, gold_query) is not None, (
        f"gold query not matched by grammar regex: {gold_query!r}"
    )


@requires_bundle
def test_vocabulary_carries_graph_relationships():
    """The grammar's relationship set must come from the graph, not a default.

    ``BOUNDED_BY`` is the canary: it is how IfcRelSpaceBoundary is loaded, it
    is what the space/window and space/door queries traverse, and it is absent
    from both the IFC product scan and the hardcoded fallback.
    """
    vocab = _load_bundle_vocabulary()
    assert "BOUNDED_BY" in vocab.all_relations, (
        "bundle predates the graph-stats merge; rebuild with scripts/build_bundle.py"
    )


@requires_bundle
def test_grammar_and_vocabulary_agree_on_relationships(bundle_regex):
    """Every relationship the vocabulary knows must be decodable."""
    vocab = _load_bundle_vocabulary()
    for rel in vocab.all_relations:
        assert f"\\[:{rel}" in bundle_regex or f"|{rel}" in bundle_regex or f"({rel}" in bundle_regex, (
            f"relationship {rel} is in the vocabulary but not in the grammar"
        )


# ---------------------------------------------------------------------------
# Negative tests — what the grammar must reject
# ---------------------------------------------------------------------------


def test_regex_rejects_unknown_label(synthetic_regex):
    assert re.fullmatch(synthetic_regex, "MATCH (n:IfcUnicorn) RETURN n") is None


def test_regex_rejects_unknown_property(synthetic_regex):
    bad = "MATCH (n:IfcWall) WHERE n.NotAProp = 'x' RETURN n"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_regex_rejects_alias_self_loop(synthetic_regex):
    # 21-char alias exceeds the 20-char cap (q0 self-loop trap).
    bad = "MATCH (n:IfcWall) RETURN count(n) AS abcdefghijabcdefghijK"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_regex_rejects_oversized_return_list(synthetic_regex):
    # 9-item RETURN exceeds the 8-item cap (q10 / q22 dup-column trap).
    bad = (
        "MATCH (n:IfcWall) RETURN n.Name, n.FireRating, n.IsExternal, "
        "n.GlobalId, n.Name, n.FireRating, n.IsExternal, n.GlobalId, n.Name"
    )
    assert re.fullmatch(synthetic_regex, bad) is None


# --- FSM self-loop traps ---------------------------------------------------
#
# Each of these is a token class the decoder could previously stay inside
# indefinitely, burning the whole max_new_tokens budget and returning a
# truncated query that neither full-matches nor executes.


def test_regex_rejects_runaway_string_literal(synthetic_regex):
    bad = "MATCH (n:IfcWall) WHERE n.Name = '" + "a" * 200 + "' RETURN n"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_regex_rejects_runaway_identifier(synthetic_regex):
    bad = f"MATCH ({'n' * 40}:IfcWall) RETURN n"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_regex_rejects_runaway_whitespace(synthetic_regex):
    bad = "MATCH" + " " * 30 + "(n:IfcWall) RETURN n"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_regex_rejects_runaway_number(synthetic_regex):
    bad = "MATCH (n:IfcWall) WHERE n.Name > 123456789012345 RETURN n"
    assert re.fullmatch(synthetic_regex, bad) is None


def test_binding_and_reference_share_one_language(synthetic_regex):
    """A name bound in MATCH must be expressible in WHERE and vice versa.

    The old grammar bound with an unbounded identifier but referenced with a
    single letter, so this query full-matched the regex and then failed in
    Neo4j with "variable s not defined".
    """
    bad = "MATCH (space:IfcWall) WHERE s.Name = 'x' RETURN s"
    assert re.fullmatch(synthetic_regex, bad) is None

    good = "MATCH (st:IfcBuildingStorey) WHERE st.Name = 'x' RETURN st"
    assert re.fullmatch(synthetic_regex, good) is not None


# --- newly admitted constructs ---------------------------------------------


def test_regex_allows_distinct(synthetic_regex):
    for good in (
        "MATCH (n:IfcWall) RETURN DISTINCT n.Name",
        "MATCH (n:IfcWall) RETURN count(DISTINCT n)",
    ):
        assert re.fullmatch(synthetic_regex, good) is not None, good


def test_regex_allows_negated_pattern_predicate():
    """The construct a property dump cannot answer: absence of a relationship."""
    regex = build_cypher_regex(
        ENTITIES, PROPERTIES, config=None
    )
    # Default relationships only; use one that is in the default set.
    good = "MATCH (s:IfcSpace) WHERE NOT (s)-[:CONTAINS]->(:IfcWindow) RETURN s.Name"
    assert re.fullmatch(regex, good) is not None


def test_regex_allows_anonymous_and_untyped_nodes(synthetic_regex):
    for good in (
        "MATCH (s:IfcSpace)-[:CONTAINS]->(e) RETURN count(e)",
        "MATCH (s:IfcSpace)-[:CONTAINS]->(:IfcWindow) RETURN count(s)",
    ):
        assert re.fullmatch(synthetic_regex, good) is not None, good
