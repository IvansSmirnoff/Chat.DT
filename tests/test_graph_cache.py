"""Graph cache: fingerprint, file format, restore batching, live round-trip.

The live round-trip wipes and rebuilds the database, so it runs only with
``NL2C_ALLOW_DB_WIPE=1``. It restores the graph it found, so a passing run
leaves the database as it was.
"""

import gzip
import json
import os

import pytest

from src.etl import graph_cache as gc
from src.eval.neo4j_exec import is_internal_label


SUMMARY = {
    "labels": {"IfcDoor": 254, "IfcSpace": 269},
    "relationships": {"BOUNDED_BY": 2915},
    "totals": {"nodes": 3815, "relationships": 17553,
               "node_properties": 200025, "relationship_properties": 726},
    "property_keys": ["GlobalId", "Name"],
}


def test_fingerprint_is_deterministic_and_order_independent():
    reordered = {
        "property_keys": ["GlobalId", "Name"],
        "totals": dict(reversed(list(SUMMARY["totals"].items()))),
        "relationships": {"BOUNDED_BY": 2915},
        "labels": {"IfcSpace": 269, "IfcDoor": 254},
    }
    assert gc.fingerprint(SUMMARY) == gc.fingerprint(reordered)


def test_fingerprint_sees_a_single_count_change():
    changed = json.loads(json.dumps(SUMMARY))
    changed["labels"]["IfcSpace"] = 270
    assert gc.fingerprint(SUMMARY) != gc.fingerprint(changed)


def test_internal_labels_are_underscore_prefixed():
    assert is_internal_label(gc.META_LABEL)
    assert is_internal_label(gc.RESTORE_LABEL)
    assert not is_internal_label("IfcSpace")


def test_dumps_refuses_values_it_cannot_round_trip():
    # A temporal or spatial Neo4j value must fail the export, not be
    # stringified and restored as a different type.
    with pytest.raises(TypeError):
        gc._dumps({"p": {"when": object()}})


def test_read_header_rejects_an_unknown_format(tmp_path):
    path = tmp_path / "other.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"format": "something-else/9"}) + "\n")
    with pytest.raises(ValueError, match="unsupported format"):
        gc.read_header(path)


class _Counters:
    relationships_created = 0


class _Result:
    counters = _Counters()

    def consume(self):
        return self


class _RecordingSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return _Result()


def test_nodes_are_created_once_per_label_set_with_the_restore_label():
    session = _RecordingSession()
    groups = {
        ("IfcDoor", "IfcRoot"): [{"n": 0, "p": {"GlobalId": "a"}}, {"n": 1, "p": {"GlobalId": "b"}}],
        ("IfcMaterial",): [{"n": 2, "p": {"material_id": "m"}}],
    }
    gc._flush_nodes(session, groups)
    assert groups == {}
    assert len(session.calls) == 2
    door_query, door_params = session.calls[0]
    assert "CREATE (n:`IfcDoor`:`IfcRoot`:`_Restore`)" in door_query
    assert len(door_params["rows"]) == 2


@pytest.mark.skipif(os.environ.get("NL2C_ALLOW_DB_WIPE") != "1",
                    reason="wipes the live database; set NL2C_ALLOW_DB_WIPE=1")
def test_live_round_trip_reproduces_the_graph(neo4j_driver, tmp_path):
    before = gc.fingerprint(gc.graph_summary(neo4j_driver))
    path = tmp_path / "roundtrip.jsonl.gz"
    header = gc.export_graph(neo4j_driver, path, {"model_id": "roundtrip_test"})
    assert header["fingerprint"] == before

    gc.restore_graph(neo4j_driver, path, batch_size=97)  # odd size: exercises partial flushes
    assert gc.fingerprint(gc.graph_summary(neo4j_driver)) == before
    assert gc.read_meta(neo4j_driver)["model_id"] == "roundtrip_test"
