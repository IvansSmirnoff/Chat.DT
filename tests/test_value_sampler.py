"""Tests for graph-sourced value enumeration + few-shot generation.

Uses a small in-process fake Neo4j driver so the suite runs without a
live database. The fake replays scripted result rows for each Cypher
string the sampler can issue.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pytest

from src.constraints.value_sampler import (
    EXCLUDE_PROPS,
    generate_few_shot_examples,
    sample_value_enumerations,
)
from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    EntityVocabulary,
    PropertyType,
    PropertyVocabulary,
)


# =============================================================================
# Fake Neo4j driver
# =============================================================================


class _FakeRecord(dict):
    pass


class _FakeResult:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = [_FakeRecord(r) for r in rows]

    def data(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._rows]

    def single(self) -> Optional[_FakeRecord]:
        return self._rows[0] if self._rows else None

    def consume(self) -> None:
        return None

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, handler: Callable[[str, Dict[str, Any]], _FakeResult]):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params) -> _FakeResult:
        return self._handler(cypher, params)


class _FakeDriver:
    def __init__(
        self,
        responses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        regex_responses: Optional[List[tuple]] = None,
    ):
        self.responses = responses or {}
        self.regex_responses = regex_responses or []
        self.executed: List[str] = []

    def session(self):
        return _FakeSession(self._handle)

    def _handle(self, cypher: str, params: Dict[str, Any]) -> _FakeResult:
        self.executed.append(cypher)
        if cypher in self.responses:
            return _FakeResult(self.responses[cypher])
        for matcher, rows in self.regex_responses:
            if matcher(cypher):
                return _FakeResult(rows)
        # Default: empty result for unknown queries (e.g. count() probes).
        return _FakeResult([])


def _build_vocab(properties: Dict[str, Dict[str, PropertyType]]) -> CombinedVocabulary:
    """Build a minimal CombinedVocabulary for sampler tests.

    ``properties`` maps entity name -> {property_name: type}. Only structure
    needed by the sampler is populated.
    """
    vocab = CombinedVocabulary()
    for entity_name, props in properties.items():
        entity = EntityVocabulary(name=entity_name, from_ifc=True, count=1)
        for prop_name, ptype in props.items():
            entity.properties[prop_name] = PropertyVocabulary(
                name=prop_name, property_type=ptype
            )
            vocab.all_properties[prop_name] = entity.properties[prop_name]
        vocab.entities[entity_name] = entity
    return vocab


# =============================================================================
# sample_value_enumerations
# =============================================================================


def test_sample_keeps_low_cardinality_string_property():
    vocab = _build_vocab({"IfcBuildingStorey": {"Name": PropertyType.STRING}})
    cypher = (
        "MATCH (n:`IfcBuildingStorey`) "
        "WHERE n.`Name` IS NOT NULL "
        "RETURN DISTINCT n.`Name` AS v "
        "LIMIT $cap"
    )
    driver = _FakeDriver({cypher: [{"v": "Level 0"}, {"v": "Level 1"}, {"v": "Level 2"}]})

    enums = sample_value_enumerations(driver, vocab, max_cardinality=50)

    assert enums == {"IfcBuildingStorey.Name": ["Level 0", "Level 1", "Level 2"]}


def test_sample_drops_high_cardinality_property():
    vocab = _build_vocab({"IfcWall": {"Name": PropertyType.STRING}})
    # max_cardinality=2 ⇒ pulling 3 distinct values trips the threshold.
    rows = [{"v": f"W-{i:03}"} for i in range(3)]
    driver = _FakeDriver(
        regex_responses=[(lambda q: "IfcWall" in q, rows)],
    )

    enums = sample_value_enumerations(driver, vocab, max_cardinality=2)

    assert enums == {}


def test_sample_excludes_globalid_and_other_identity_props():
    assert "GlobalId" in EXCLUDE_PROPS
    vocab = _build_vocab({"IfcWall": {"GlobalId": PropertyType.STRING}})
    driver = _FakeDriver({})  # no responses needed; GlobalId must be skipped

    enums = sample_value_enumerations(driver, vocab)

    assert enums == {}
    # Sampler must not probe the excluded property — only the schema
    # introspection probe is allowed (and tolerated when it returns nothing).
    assert all("WHERE n.`GlobalId`" not in q for q in driver.executed)


def test_sample_drops_numeric_only_values():
    vocab = _build_vocab({"IfcSpace": {"GrossFloorArea": PropertyType.NUMERIC}})
    driver = _FakeDriver(
        regex_responses=[
            (lambda q: "GrossFloorArea" in q, [{"v": 12.5}, {"v": 30.1}]),
        ],
    )

    enums = sample_value_enumerations(driver, vocab)

    assert enums == {}


def test_sample_drops_long_free_text():
    vocab = _build_vocab({"IfcSpace": {"Comments": PropertyType.STRING}})
    long_text = "x" * 200
    driver = _FakeDriver(
        regex_responses=[
            (lambda q: "Comments" in q, [{"v": long_text}, {"v": long_text}]),
        ],
    )

    enums = sample_value_enumerations(driver, vocab)

    assert enums == {}


def test_sample_skips_strict_properties():
    """STRICT (IDS-enumerated) properties already carry their values; we
    should not duplicate them via graph sampling."""
    vocab = CombinedVocabulary()
    entity = EntityVocabulary(name="IfcWall", from_ifc=True, count=1)
    entity.properties["FireRating"] = PropertyVocabulary(
        name="FireRating",
        property_type=PropertyType.STRICT,
        allowed_values={"EI30", "EI60"},
    )
    vocab.all_properties["FireRating"] = entity.properties["FireRating"]
    vocab.entities["IfcWall"] = entity

    driver = _FakeDriver({})

    enums = sample_value_enumerations(driver, vocab)

    assert enums == {}
    # STRICT path short-circuits before any DISTINCT probe — the only
    # query allowed is the schema-introspection probe.
    assert all("RETURN DISTINCT" not in q for q in driver.executed)


def test_sample_handles_empty_graph_gracefully():
    vocab = _build_vocab({"IfcSpace": {"LongName": PropertyType.STRING}})
    driver = _FakeDriver(regex_responses=[(lambda q: True, [])])

    enums = sample_value_enumerations(driver, vocab)

    assert enums == {}


# =============================================================================
# generate_few_shot_examples
# =============================================================================


def _label_responses(label_counts: Dict[str, int]) -> List[tuple]:
    """Build regex_responses that reproduce graph_stats-style probing."""
    matchers: List[tuple] = []

    def labels_query(cypher: str) -> bool:
        return "db.labels()" in cypher

    matchers.append(
        (
            labels_query,
            [{"label": lbl} for lbl in label_counts],
        )
    )

    def make_count_matcher(target_label: str, cnt: int):
        def _m(cypher: str) -> bool:
            return f"MATCH (n:`{target_label}`) RETURN count(n) AS c" == cypher
        return _m, [{"c": cnt}]

    for lbl, cnt in label_counts.items():
        m, rows = make_count_matcher(lbl, cnt)
        matchers.append((m, rows))

    return matchers


def _rel_responses(rel_types: List[str]) -> List[tuple]:
    def rel_query(cypher: str) -> bool:
        return "db.relationshipTypes()" in cypher

    return [(rel_query, [{"relationshipType": r} for r in rel_types])]


def _smoke_ok(cypher: str) -> bool:
    """Default: every smoke-test query passes."""
    return cypher.upper().startswith("MATCH") or cypher.upper().startswith("CALL")


def test_few_shots_use_storey_value_when_available():
    vocab = _build_vocab({
        "IfcBuildingStorey": {"Name": PropertyType.STRING},
        "IfcDoor": {"Name": PropertyType.STRING},
    })
    enumerations = {"IfcBuildingStorey.Name": ["Level 2"]}

    label_counts = {"IfcBuildingStorey": 3, "IfcDoor": 50}
    regex_responses: List[tuple] = []
    regex_responses += _label_responses(label_counts)
    regex_responses += _rel_responses(["CONTAINS"])
    # All smoke executions succeed.
    regex_responses.append((lambda q: True, []))

    driver = _FakeDriver(regex_responses=regex_responses)

    examples = generate_few_shot_examples(driver, vocab, enumerations, n=5)

    storey_examples = [
        ex for ex in examples if "IfcBuildingStorey" in ex["cypher"] and "Level 2" in ex["cypher"]
    ]
    assert storey_examples, f"Expected a storey-relationship example, got {examples}"
    # Single-letter variable bindings (REF_VARIABLE FSM trap).
    assert "(s:IfcBuildingStorey)" in storey_examples[0]["cypher"]
    assert "(d:IfcDoor)" in storey_examples[0]["cypher"]


def test_few_shots_skip_entities_with_zero_count():
    vocab = _build_vocab({
        "IfcWall": {"Name": PropertyType.STRING},
        "IfcDoor": {"Name": PropertyType.STRING},
    })

    # Wall present, Door at zero count => no Door examples.
    label_counts = {"IfcWall": 10, "IfcDoor": 0}
    regex_responses: List[tuple] = []
    regex_responses += _label_responses(label_counts)
    regex_responses += _rel_responses([])
    regex_responses.append((lambda q: True, []))

    driver = _FakeDriver(regex_responses=regex_responses)

    examples = generate_few_shot_examples(driver, vocab, {}, n=5)

    for ex in examples:
        assert "IfcDoor" not in ex["cypher"], f"Door at zero count leaked into {ex}"


def test_few_shots_drop_examples_that_fail_to_execute():
    vocab = _build_vocab({"IfcWall": {"Name": PropertyType.STRING}})

    label_counts = {"IfcWall": 5}

    # Smoke-test handler raises for anything except the metadata + count
    # probes, so every generated example is dropped.
    def handler(cypher: str, params: Dict[str, Any]) -> _FakeResult:
        if "db.labels()" in cypher:
            return _FakeResult([{"label": "IfcWall"}])
        if "db.relationshipTypes()" in cypher:
            return _FakeResult([])
        if "RETURN count(n) AS c" in cypher:
            return _FakeResult([{"c": label_counts["IfcWall"]}])
        # Smoke test: pretend Cypher is invalid.
        raise RuntimeError("simulated cypher failure")

    driver = _FakeDriver({})
    driver._handle = handler  # type: ignore[assignment]

    examples = generate_few_shot_examples(driver, vocab, {}, n=5)

    assert examples == []


def test_few_shots_respect_n_cap():
    vocab = _build_vocab({
        "IfcWall": {"Name": PropertyType.STRING},
        "IfcWindow": {"Name": PropertyType.STRING},
        "IfcDoor": {"Name": PropertyType.STRING},
    })
    enumerations = {
        "IfcWall.Name": ["WallA"],
        "IfcWindow.Name": ["WindowA"],
        "IfcDoor.Name": ["DoorA"],
    }
    label_counts = {"IfcWall": 1, "IfcWindow": 1, "IfcDoor": 1}

    regex_responses: List[tuple] = []
    regex_responses += _label_responses(label_counts)
    regex_responses += _rel_responses([])
    regex_responses.append((lambda q: True, []))
    driver = _FakeDriver(regex_responses=regex_responses)

    examples = generate_few_shot_examples(driver, vocab, enumerations, n=2)

    assert len(examples) <= 2
