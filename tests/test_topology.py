"""Topology enrichment: geometry rules, the clinic checks, the prompt notes, the live round trip.

Three layers:
* pure rules on synthetic rooms (shapely only);
* the plan's train-data checks on the dental clinic IFC files (skipped where the
  files or ifcopenshell are missing): Q28 connected = {1BC2}, Q29 adjacent =
  {1B04, 1B05, 1B09, 1BC2};
* the live apply/remove round trip, which wipes the database and so runs only
  with ``NL2C_ALLOW_DB_WIPE=1``; it restores the graph it found.
"""

import os

import pytest
from shapely.geometry import box

from src.constraints.vocabulary_merger import (
    CombinedVocabulary,
    get_system_schema_string,
)
from src.etl import topology as tp
from src.etl.geometry import ElementGeometry, merge, mesh_geometry, triangles_footprint


def unit_box(x0=0.0, y0=0.0, z0=0.0, dx=1.0, dy=1.0, dz=1.0):
    """A closed box mesh: 8 points, 12 triangles (top, bottom and four vertical sides)."""
    pts = [(x0 + i * dx, y0 + j * dy, z0 + k * dz) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    faces = [0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6,              # bottom, top
             0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,              # y = y0, y = y0 + dy
             0, 2, 6, 0, 6, 4, 1, 3, 7, 1, 7, 5]              # x = x0, x = x0 + dx
    return pts, faces


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def test_footprint_is_the_plan_outline_and_vertical_faces_are_ignored():
    pts, faces = unit_box(2, 3, 0, dx=4, dy=2, dz=3)
    g = mesh_geometry(pts, faces)
    assert g.footprint.area == pytest.approx(8.0)
    assert g.centroid == pytest.approx((4.0, 4.0))
    assert (g.z_min, g.z_max) == (0, 3)
    vertical_only = faces[12:]
    assert triangles_footprint(pts, vertical_only).is_empty


def test_empty_mesh_has_no_geometry_and_parts_merge():
    assert mesh_geometry([], []) is None
    a = mesh_geometry(*unit_box(0, 0, 0))
    b = mesh_geometry(*unit_box(1, 0, 3, dz=2))
    m = merge([None, a, b])
    assert m.footprint.area == pytest.approx(2.0)
    assert (m.z_min, m.z_max) == (0, 5)
    assert merge([None, None]) is None


# -----------------------------------------------------------------------------
# Doors
# -----------------------------------------------------------------------------

ROOMS = {"A": box(0, 0, 4, 3), "B": box(4.12, 0, 8, 3), "C": box(0, 3.12, 4, 6)}


def test_a_door_on_two_rooms_connects_them():
    pairs, stats = tp.door_space_pairs({"d1": {"B", "A"}}, {}, ROOMS)
    assert pairs == [("A", "B", "d1")]
    assert stats == {"doors_bounding_2": 1}


def test_a_door_on_three_rooms_connects_the_two_nearest():
    # The door sits in the A|B wall; C is recorded on it too (a boundary artefact).
    pairs, _ = tp.door_space_pairs({"d1": {"A", "B", "C"}}, {"d1": (4.06, 1.0)}, ROOMS)
    assert pairs == [("A", "B", "d1")]


def test_a_door_on_three_rooms_without_a_position_is_skipped_not_guessed():
    pairs, stats = tp.door_space_pairs({"d1": {"A", "B", "C"}}, {}, ROOMS)
    assert pairs == [] and stats["multi_room_doors_skipped"] == 1


def test_exterior_doors_from_external_boundaries_or_a_single_room():
    flags = tp.exterior_doors({"front": {"A"}, "inner": {"A", "B"}, "glass": {"A", "B"}, "orphan": set()},
                              {"glass": True})
    assert flags == {"front": True, "inner": False, "glass": True}   # no boundary record: no flag


# -----------------------------------------------------------------------------
# Adjacency
# -----------------------------------------------------------------------------

def test_rooms_across_a_wall_are_adjacent_and_far_rooms_sharing_it_are_not():
    rooms = {"A": box(0, 0, 4, 3), "B": box(4.12, 0, 8, 3), "F": box(10, 0, 12, 3)}
    storey = {k: "s1" for k in rooms}
    # One long wall bounds all three; only A and B face each other across it.
    wall_pairs = {("A", "B"): ["w1"], ("A", "F"): ["w1"], ("B", "F"): ["w1"]}
    out, stats = tp.adjacent_pairs(wall_pairs, rooms, storey)
    assert [(p["a"], p["b"], p["method"], p["via_gid"]) for p in out] == [("A", "B", "wall", "w1")]
    assert out[0]["gap_m"] == pytest.approx(0.12)
    assert out[0]["shared_len_m"] == pytest.approx(3.0, abs=2 * tp.EDGE_TOL_M + 1e-6)  # documented end bias
    assert stats["wall_pairs_gap_above_max"] == 2 and stats["wall_pairs_over_return"] == pytest.approx(2 / 3, abs=1e-3)


def test_rooms_meeting_at_a_corner_are_not_adjacent():
    rooms = {"A": box(0, 0, 4, 3), "D": box(4.12, 3.12, 8, 6)}
    out, stats = tp.adjacent_pairs({("A", "D"): ["w1"]}, rooms, {"A": "s1", "D": "s1"})
    assert out == [] and stats["dropped"] == {"shared edge below min": 1}


def test_rooms_on_different_or_unknown_storeys_are_never_adjacent():
    rooms = {"A": box(0, 0, 4, 3), "B": box(4.12, 0, 8, 3)}
    out, _ = tp.adjacent_pairs({("A", "B"): ["w1"]}, rooms, {"A": "s1", "B": "s2"})
    assert out == []
    out, _ = tp.adjacent_pairs({("A", "B"): ["w1"]}, rooms, {"A": "s1", "B": None})
    assert out == []


def test_open_plan_rooms_touching_without_a_wall_are_adjacent_once():
    rooms = {"A": box(0, 0, 4, 3), "B": box(4, 0, 8, 3), "C": box(4.12, 3.12, 8, 6)}
    storey = {k: "s1" for k in rooms}
    out, _ = tp.adjacent_pairs({}, rooms, storey)
    assert [(p["a"], p["b"], p["method"], p["via_gid"]) for p in out] == [("A", "B", "open", None)]
    # A pair already found across a wall is not repeated as open plan.
    out, _ = tp.adjacent_pairs({("A", "B"): ["w1"]}, rooms, storey)
    assert [(p["a"], p["b"], p["method"]) for p in out] == [("A", "B", "wall")]


def test_vertical_access_serves_storeys_within_the_height_range():
    pairs = tp.vertical_access({"stair": (-0.01, 5.67), "ramp": (-1.0, 0.0)},
                               {"footing": -1.0, "first": 0.0, "second": 4.57, "roof": 9.25})
    assert sorted(pairs) == [("ramp", "first"), ("ramp", "footing"), ("stair", "first"), ("stair", "second")]


# -----------------------------------------------------------------------------
# Assembly: what gets written is exactly what remove_topology removes
# -----------------------------------------------------------------------------

def _small_topology():
    geom = {g: ElementGeometry(f, 0.0, 3.0) for g, f in ROOMS.items()}
    adjacency, _ = tp.adjacent_pairs({("A", "B"): ["w1"]}, ROOMS, {k: "s1" for k in ROOMS})
    return tp.assemble([("A", "B", "d1")], {"d1": "Door 1"}, adjacency, [("stair", "s2")],
                       geom, {"d1": False}, {("A", "w1"): True, ("B", "w1"): False})


def test_symmetric_edges_are_stored_both_ways_and_all_enriched():
    topo = _small_topology()
    assert topo.neighbours("CONNECTED_VIA", "A") == {"B"} and topo.neighbours("CONNECTED_VIA", "B") == {"A"}
    assert topo.neighbours("ADJACENT_TO", "B") == {"A"}
    assert topo.neighbours("VERTICAL_ACCESS", "stair") == {"s2"} and not topo.neighbours("VERTICAL_ACCESS", "s2")
    assert all(e.props["enriched"] is True for e in topo.edges)
    assert {e.rel_type for e in topo.edges} == set(tp.DERIVED_REL_TYPES)
    door_edge = next(e for e in topo.edges if e.rel_type == "CONNECTED_VIA")
    assert door_edge.props == {"door_gid": "d1", "door_name": "Door 1", "enriched": True}


def test_every_written_property_is_one_remove_knows_about():
    topo = _small_topology()
    assert {k for props in topo.node_props.values() for k in props} <= set(tp.NODE_PROPS)
    assert {k for _, _, props in topo.bounded_by for k in props} <= set(tp.BOUNDED_BY_PROPS)
    assert topo.node_props["A"]["geo_footprint_m2"] == pytest.approx(12.0)
    assert topo.node_props["d1"] == {"is_exterior_door": False}


# -----------------------------------------------------------------------------
# Prompt notes
# -----------------------------------------------------------------------------

def test_base_graph_prompt_has_no_derived_block():
    vocab = CombinedVocabulary(relationship_signatures=[("IfcSpace", "BOUNDED_BY", "IfcWall")])
    assert "DERIVED FROM GEOMETRY" not in get_system_schema_string(vocab)


def test_enriched_graph_prompt_explains_only_the_types_it_has():
    vocab = CombinedVocabulary(relationship_signatures=[
        ("IfcSpace", "BOUNDED_BY", "IfcWall"), ("IfcSpace", "ADJACENT_TO", "IfcSpace")])
    text = get_system_schema_string(vocab)
    assert "DERIVED FROM GEOMETRY" in text
    assert "ADJACENT_TO: rooms that physically touch" in text
    assert "CONNECTED_VIA:" not in text
    assert "geo_footprint_m2" in text


# -----------------------------------------------------------------------------
# The clinic (train-data checks from the plan)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clinic(data_dir):
    pytest.importorskip("ifcopenshell")
    import ifcopenshell

    path = data_dir / "ifcbench/raw/projects/dental_clinic/arc.ifc"
    if not path.is_file():
        pytest.skip(f"clinic IFC not at {path}")
    model = ifcopenshell.open(str(path))
    gid = {s.Name: s.GlobalId for s in model.by_type("IfcSpace")}
    name = {v: k for k, v in gid.items()}
    storeys = {s.GlobalId: s.Name for s in model.by_type("IfcBuildingStorey")}
    return tp.compute_topology(path), gid, name, storeys


def test_clinic_q28_rooms_connected_to_1b07(clinic):
    topo, gid, name, _ = clinic
    assert {name[g] for g in topo.neighbours("CONNECTED_VIA", gid["1B07"])} == {"1BC2"}


def test_clinic_q29_rooms_adjacent_to_1b07(clinic):
    topo, gid, name, _ = clinic
    assert {name[g] for g in topo.neighbours("ADJACENT_TO", gid["1B07"])} == {"1B04", "1B05", "1B09", "1BC2"}


def test_clinic_doors_walls_and_stairs(clinic):
    topo, _, _, storeys = clinic
    s = topo.stats
    assert s["doors"] == {"doors_bounding_0": 18, "doors_bounding_1": 10,
                          "doors_bounding_2": 222, "doors_bounding_3+": 4}
    # The reportable over-return of "shares a wall" as adjacency.
    assert s["adjacency"]["wall_pairs_same_storey"] == 1037
    assert s["adjacency"]["wall_pairs_over_return"] == pytest.approx(0.375, abs=0.01)
    assert s["edges"]["CONNECTED_VIA"] == 2 * 226
    served = {storeys[e.target] for e in topo.edges if e.rel_type == "VERTICAL_ACCESS"}
    assert served == {"First Floor", "Second Floor"}


def test_structural_model_has_stairs_but_no_rooms(data_dir):
    pytest.importorskip("ifcopenshell")
    path = data_dir / "ifcbench/raw/projects/dental_clinic/str.ifc"
    if not path.is_file():
        pytest.skip(f"clinic IFC not at {path}")
    topo = tp.compute_topology(path)
    assert topo.stats["spaces"] == 0
    assert set(topo.stats["edges"]) == {"VERTICAL_ACCESS"}


# -----------------------------------------------------------------------------
# Runner wiring
# -----------------------------------------------------------------------------

def test_runner_refuses_unknown_graph_states_it_cannot_build(tmp_path):
    from src.agent import run

    entry = {"model_id": "m", "paths": {"graph_cache": "data/ifcbench/cache/m.base.jsonl.gz",
                                        "ifc": "data/m.ifc"}}
    manifest = {"_root": tmp_path, "models": [entry]}
    with pytest.raises(RuntimeError, match="no dedup snapshot"):
        run.ensure_graph(None, manifest, entry, "dedup", "auto", tmp_path)
    with pytest.raises(RuntimeError, match="no topology snapshot"):
        run.ensure_graph(None, manifest, entry, "topology", "never", tmp_path)


# -----------------------------------------------------------------------------
# Live round trip
# -----------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("NL2C_ALLOW_DB_WIPE") != "1",
                    reason="wipes the live database; set NL2C_ALLOW_DB_WIPE=1")
def test_live_apply_then_remove_restores_the_base_graph(neo4j_driver, data_dir, tmp_path, clinic):
    from src.etl import graph_cache as gc

    snapshot = data_dir / "ifcbench/cache/dental_clinic__arc.base.jsonl.gz"
    if not snapshot.is_file():
        pytest.skip(f"no base snapshot at {snapshot}")
    topo, _, _, _ = clinic
    found = tmp_path / "found.jsonl.gz"
    gc.export_graph(neo4j_driver, found, gc.read_meta(neo4j_driver) or {"model_id": "found"})
    try:
        gc.restore_graph(neo4j_driver, snapshot)
        base = gc.fingerprint(gc.graph_summary(neo4j_driver))

        written = tp.apply_topology(neo4j_driver, topo)
        assert written["unmatched_edges"] == {}
        assert written["CONNECTED_VIA"] == topo.stats["edges"]["CONNECTED_VIA"]
        with neo4j_driver.session() as s:
            adj = s.run("MATCH (:IfcSpace {Name: '1B07'})-[:ADJACENT_TO]->(b:IfcSpace) "
                        "RETURN collect(DISTINCT b.Name) AS n").single()["n"]
            con = s.run("MATCH (:IfcSpace {Name: '1B07'})-[:CONNECTED_VIA]-(b:IfcSpace) "
                        "RETURN collect(DISTINCT b.Name) AS n").single()["n"]
        assert set(adj) == {"1B04", "1B05", "1B09", "1BC2"} and set(con) == {"1BC2"}
        meta = gc.read_meta(neo4j_driver)
        assert meta["enrichment"] == "topology"
        assert meta["fingerprint"] == gc.fingerprint(gc.graph_summary(neo4j_driver)) != base

        with pytest.raises(RuntimeError, match="already carries"):
            tp.apply_topology(neo4j_driver, topo)

        tp.remove_topology(neo4j_driver)
        assert gc.fingerprint(gc.graph_summary(neo4j_driver)) == base
        assert gc.read_meta(neo4j_driver)["enrichment"] == "base"
    finally:
        gc.restore_graph(neo4j_driver, found)
