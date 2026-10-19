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

from src.constraints.grammar import CypherGrammar, build_cypher_regex


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


@pytest.fixture(scope="module")
def bundle_vocabulary():
    return _load_bundle_vocabulary()


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


# --- IDS value binding --------------------------------------------------------


@pytest.fixture(scope="module")
def bound_regex() -> str:
    """Grammar with an IDS-style enumeration on FireRating."""
    return build_cypher_regex(
        ENTITIES,
        PROPERTIES,
        config=CypherGrammar(
            entities=ENTITIES,
            properties=PROPERTIES,
            value_enumerations={"FireRating": {"EI30", "EI60", "EI90"}},
        ),
    )


@pytest.mark.parametrize(
    "query,allowed",
    [
        ("MATCH (n:IfcWall) WHERE n.FireRating = 'EI60' RETURN n", True),
        ("MATCH (n:IfcWall) WHERE n.FireRating <> 'EI30' RETURN n", True),
        # Invented value — the whole point of binding.
        ("MATCH (n:IfcWall) WHERE n.FireRating = 'EI-60' RETURN n", False),
        ("MATCH (n:IfcWall) WHERE n.FireRating = 'ei60' RETURN n", False),
        # CONTAINS would smuggle a free literal back in against a closed set.
        ("MATCH (n:IfcWall) WHERE n.FireRating CONTAINS 'EI' RETURN n", False),
        # Unenumerated properties keep free values.
        ("MATCH (n:IfcWall) WHERE n.Name = 'whatever' RETURN n", True),
        # Positions that carry no literal stay open to enumerated properties.
        ("MATCH (n:IfcWall) WHERE n.FireRating IS NOT NULL RETURN n.FireRating", True),
        ("MATCH (n:IfcWall) RETURN n.FireRating ORDER BY n.FireRating DESC", True),
    ],
)
def test_ids_value_binding(bound_regex, query, allowed):
    assert (re.fullmatch(bound_regex, query) is not None) is allowed, query


@requires_bundle
def test_bundle_binds_the_ids_enumerated_properties(bundle_regex):
    """The four IDS-enumerated Building 15 properties must be value-bound."""
    good = "MATCH (d:IfcDoor) WHERE d.PBSa_DESCRIZIONE = 'Porta, Portafinestra REI/EI' RETURN count(d)"
    # The value the model invented on the first Building 15 run.
    bad = "MATCH (d:IfcDoor) WHERE d.PBSa_DESCRIZIONE = 'MAT_TEM_Fiamme' RETURN count(d)"
    assert re.fullmatch(bundle_regex, good) is not None
    assert re.fullmatch(bundle_regex, bad) is None


def test_regex_allows_anonymous_and_untyped_nodes(synthetic_regex):
    for good in (
        "MATCH (s:IfcSpace)-[:CONTAINS]->(e) RETURN count(e)",
        "MATCH (s:IfcSpace)-[:CONTAINS]->(:IfcWindow) RETURN count(s)",
    ):
        assert re.fullmatch(synthetic_regex, good) is not None, good


# ---------------------------------------------------------------------------
# FSM budget guard
# ---------------------------------------------------------------------------
#
# The DFA the regex compiles to is the scarce resource, not the regex itself:
# outlines-core fails outright once the state count exceeds its limit, and the
# measured table at grammar.py:519-538 shows the shipped caps
# (max_where_clauses=1, max_return_items=3) sitting at 31,892 states / 868 MB
# for a 230-property vocabulary. The property alternation is duplicated about
# eight times across the pattern, so vocabulary growth is the thing that pushes
# it over.
#
# outlines is not installed in the workbench container (it lives in the Colab
# GPU runtime), so the real compile cannot run here. These guard the inputs
# instead: if a change inflates the vocabulary, this fails in seconds rather
# than after an eleven-minute DFA build on a GPU.

MAX_VOCABULARY_PROPERTIES = 320
MAX_VOCABULARY_ENTITIES = 110
MAX_REGEX_CHARS = 75_000


@requires_bundle
def test_vocabulary_stays_within_the_fsm_budget(bundle_vocabulary):
    properties = len(bundle_vocabulary.all_properties)
    entities = len(bundle_vocabulary.entities)
    assert properties <= MAX_VOCABULARY_PROPERTIES, (
        f"{properties} properties in the vocabulary; the measured DFA budget was "
        f"established at 230. Do not raise this cap without re-running the "
        f"outlines compile on a GPU runtime and updating grammar.py:519-538."
    )
    assert entities <= MAX_VOCABULARY_ENTITIES, f"{entities} entities in the vocabulary"


@requires_bundle
def test_regex_size_stays_within_the_fsm_budget(bundle_regex):
    assert len(bundle_regex) <= MAX_REGEX_CHARS, (
        f"regex is {len(bundle_regex)} chars; a DFA compile at this size has not "
        f"been measured"
    )


@requires_bundle
def test_graph_sourced_relationships_reach_the_grammar(bundle_regex, bundle_vocabulary):
    """The loader's newer edge types must be decodable, not just present.

    HAS_MATERIAL_LAYER is the one that matters for material-layer questions:
    without it in the grammar, layer thickness is undecodable under STRICT even
    though the graph carries it.
    """
    assert "HAS_MATERIAL_LAYER" in bundle_vocabulary.all_relations
    assert "HAS_MATERIAL_LAYER" in bundle_regex
