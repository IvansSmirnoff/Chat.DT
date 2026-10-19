"""Tests for scripts/bind_sensors.py (offline; no Neo4j needed)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bind_sensors as bs  # noqa: E402

CONFIG = [{"building": 15, "rooms": [
    {"airthings_room_id": "000_001", "ifc_room_id": "MIA0206000001", "sensor_id": "1000000001"},
    {"airthings_room_id": "000_017b", "ifc_room_id": "MIA0206000017b", "sensor_id": "1000000002"},
    {"airthings_room_id": "00S_008", "ifc_room_id": "MIA020600S008", "sensor_id": "1000000003"},
]}]
SPACES = [
    {"GlobalId": "g1", "Numero": "001", "Livello": "PT", "PBSs_III_PIANO": "000", "PBSs_IV_VANO": "001"},
    {"GlobalId": "g2", "Numero": "017b", "Livello": "PT", "PBSs_III_PIANO": "000", "PBSs_IV_VANO": "017b"},
    # the same Numero on two floors: the join must use floor + room, not Numero
    {"GlobalId": "g3", "Numero": "008", "Livello": "PS1", "PBSs_III_PIANO": "00S", "PBSs_IV_VANO": "008"},
    {"GlobalId": "g4", "Numero": "008", "Livello": "PT", "PBSs_III_PIANO": "000", "PBSs_IV_VANO": "008"},
    {"GlobalId": "g5", "Numero": "009", "Livello": "PT"},  # a space without the institutional coding
]


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "building_config.json"
    p.write_text(json.dumps(CONFIG))
    return p


def test_parse_room_code():
    assert bs.parse_room_code("MIA0206000017b") == ("000", "017b")
    assert bs.parse_room_code("MIA020600S008") == ("00S", "008")


@pytest.mark.parametrize("bad", ["", "XYZ0206000001", "MIA0206", "MIA0206000"])
def test_parse_room_code_rejects(bad):
    with pytest.raises(ValueError):
        bs.parse_room_code(bad)


def test_load_bindings(config):
    rows = bs.load_bindings(config)
    assert [r["room_code"] for r in rows] == ["MIA0206000001", "MIA0206000017b", "MIA020600S008"]
    assert rows[2]["piano"] == "00S" and rows[2]["serial"] == "1000000003"


def test_load_bindings_rejects_duplicate_serial(tmp_path):
    cfg = [{"building": 15, "rooms": [CONFIG[0]["rooms"][0], {**CONFIG[0]["rooms"][1], "sensor_id": "1000000001"}]}]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        bs.load_bindings(p)


def test_match_uses_floor_and_room_not_numero(config):
    matched, unmatched, ambiguous = bs.match_spaces(bs.load_bindings(config), SPACES)
    assert not unmatched and not ambiguous
    by_code = {m["room_code"]: m["global_id"] for m in matched}
    assert by_code["MIA020600S008"] == "g3"  # the basement 008, not the ground-floor 008


def test_match_reports_unmatched_and_ambiguous(config):
    rows = bs.load_bindings(config)
    spaces = [s for s in SPACES if s["GlobalId"] != "g2"] + [dict(SPACES[0], GlobalId="g1bis")]
    matched, unmatched, ambiguous = bs.match_spaces(rows, spaces)
    assert [r["room_code"] for r in unmatched] == ["MIA0206000017b"]
    assert [r["room_code"] for r in ambiguous] == ["MIA0206000001"]


def test_demo_query_stays_inside_the_compiled_profile():
    # one property condition, one pattern predicate, at most three projected columns
    q = bs.DEMO_QUERY
    where = q.split("WHERE", 1)[1].split("RETURN", 1)[0]
    assert where.count(" AND ") == 1 and "NOT (s)-[:BOUNDED_BY]->(:IfcWindow)" in where
    assert len(q.split("RETURN", 1)[1].split(",")) == 3


def test_check_against_committed_bundle_if_present(config, tmp_path):
    bundle = Path(__file__).resolve().parents[1] / "data" / "bundle_b15_ids.json"
    if not bundle.exists():
        pytest.skip("bundle not available")
    spaces = bs.spaces_from_bundle(bundle)
    assert sum(1 for s in spaces if s.get("PBSs_III_PIANO")) >= 3
