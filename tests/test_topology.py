"""Topology enrichment: geometry rules, the clinic checks, the prompt notes, the live round trip.

Three layers:
* pure rules on synthetic rooms (shapely only);
* the plan's train-data checks on the dental clinic IFC files (skipped where the
  files or ifcopenshell are missing): Q28 connected = {1BC2}, Q29 adjacent =
  {1B04, 1B05, 1B09, 1BC2}; open passages, door sides and doors without
  boundaries;
* the live apply/remove round trip, which wipes the database and so runs only
  with ``NL2C_ALLOW_DB_WIPE=1``; it restores the graph it found.
"""

import os

import pytest
from shapely.geometry import LineString, box

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
# A door in the wall x = 4..4.12 whose far side is split between two rooms
# joined by a separation line at y = 2 (B below, C above): a three-room door.
SPLIT = {"A": box(0, 0, 4, 4), "B": box(4.12, 0, 8, 2), "C": box(4.12, 2, 8, 4)}
SPLIT_DOOR = box(4.0, 1.5, 4.12, 2.5)


def test_a_door_on_two_rooms_connects_them():
    pairs, exterior, stats = tp.door_space_pairs({"d1": {"B", "A"}}, {}, ROOMS)
    assert pairs == [("A", "B", "d1", "boundary")] and exterior == {}
    assert stats == {"doors_bounding_2": 1}


def test_door_sides_follow_the_leaf_and_swap_axes_for_a_deep_footprint():
    plus, minus = tp.door_side_boxes(SPLIT_DOOR, (0.0, 1.0))      # x-axis along the wall (y)
    sides = tp.room_sides(SPLIT, (plus, minus))
    assert sides == {"A": {1}, "B": {-1}, "C": {-1}}
    # The same door given an x-axis across the wall: the footprint says it is wrong.
    swapped = tp.room_sides(SPLIT, tp.door_side_boxes(SPLIT_DOOR, (1.0, 0.0)))
    assert {r: len(s) for r, s in swapped.items()} == {"A": 1, "B": 1, "C": 1}
    assert swapped["B"] == swapped["C"] != swapped["A"]
    assert tp.split_sides(sides) == ({"A"}, {"B", "C"})


def test_a_door_on_three_rooms_connects_across_its_leaf_only():
    sides = tp.room_sides(SPLIT, tp.door_side_boxes(SPLIT_DOOR, (0.0, 1.0)))
    pairs, _, stats = tp.door_space_pairs({"d1": {"A", "B", "C"}}, {"d1": (4.06, 2.0)}, SPLIT, {"d1": sides})
    assert pairs == [("A", "B", "d1", "boundary"), ("A", "C", "d1", "boundary")]   # never B-C
    assert stats["multi_room_doors_side_test"] == 1


def test_a_door_on_three_rooms_with_unclear_sides_falls_back_to_the_two_nearest():
    # The door sits in the A|B wall; C is recorded on it too (a boundary artefact).
    pairs, _, stats = tp.door_space_pairs({"d1": {"A", "B", "C"}}, {"d1": (4.06, 1.0)}, ROOMS)
    assert pairs == [("A", "B", "d1", "boundary")] and stats["multi_room_doors_nearest_two"] == 1
    pairs, _, stats = tp.door_space_pairs({"d1": {"A", "B", "C"}}, {}, ROOMS)
    assert pairs == [] and stats["multi_room_doors_skipped"] == 1


def test_doors_without_boundaries_use_their_sides_only_when_asked():
    door_sides = {
        "cubicle": {"A": {1, -1}},                    # inside one room: joins nothing
        "front": {"A": {1}},                          # one side faces nothing: leads outside
        "glass": {"A": {1}, "B": {-1}},               # joins two rooms
        "lost": {},                                   # touches no room
    }
    spaces = {d: set() for d in door_sides}
    pairs, exterior, stats = tp.door_space_pairs(spaces, {}, ROOMS, door_sides, use_geometry=True)
    assert pairs == [("A", "B", "glass", "geometry")]
    assert exterior == {"cubicle": False, "front": True, "glass": False}
    assert stats == {"doors_bounding_0": 4, "unbounded_inside_one_room": 1, "unbounded_exterior": 1,
                     "unbounded_joining_rooms": 1, "unbounded_no_room": 1}
    pairs, exterior, _ = tp.door_space_pairs(spaces, {}, ROOMS, door_sides)
    assert pairs == [] and exterior == {}


def test_the_side_test_is_checked_against_doors_with_boundaries():
    spaces = {"two": {"A", "B"}, "one": {"A"}, "three": {"A", "B", "C"}, "none": set()}
    good = {"two": {"A": {1}, "B": {-1}}, "one": {"A": {-1}}, "three": {"A": {1}, "B": {-1}, "C": {-1}},
            "none": {"A": {1}}}
    assert tp.validate_door_sides(spaces, good) == {"checked": 3, "agree": 3, "disagree": [], "trusted": True}
    bad = dict(good, two={"A": {1}, "B": {1}}, one={"A": {1}, "B": {-1}})
    check = tp.validate_door_sides(spaces, bad)
    assert check["agree"] == 1 and sorted(check["disagree"]) == ["one", "two"] and not check["trusted"]


def test_exterior_doors_from_external_boundaries_or_a_single_room():
    flags = tp.exterior_doors({"front": {"A"}, "inner": {"A", "B"}, "glass": {"A", "B"}, "orphan": set()},
                              {"glass": True})
    assert flags == {"front": True, "inner": False, "glass": True}   # no boundary record: no flag
    flags = tp.exterior_doors({"front": {"A"}, "orphan": set()}, {}, {"orphan": True, "front": False})
    assert flags == {"front": True, "orphan": True}                  # boundaries win over geometry


# -----------------------------------------------------------------------------
# Open passages (virtual boundaries)
# -----------------------------------------------------------------------------

def test_virtual_boundaries_recorded_on_both_spaces_pair_them_once():
    storey = {"A": "s1", "B": "s1", "C": "s1", "U": "s2"}
    lines = [
        ("A", LineString([(4, 0), (4, 3)]), False),            # A|B separation line, from A
        ("B", LineString([(4, 3.0), (4, 0.0)]), False),        # ...and from B, drawn the other way
        ("B", LineString([(4, 3), (4, 3.05)]), False),         # a sliver at the A|B|C junction
        ("C", LineString([(4, 3), (6, 3)]), False),            # touches A|B only at its end
        ("C", LineString([(0, 6), (8, 6)]), True),             # an edge onto the outside
        ("U", LineString([(4, 0), (4, 3)]), False),            # same plan line, another storey
    ]
    pairs, stats = tp.open_passages(lines, storey)
    assert pairs == [("A", "B")]
    assert stats["boundaries_matching_one_space"] == 2              # A's and B's long lines
    assert stats["boundaries_matching_two_or_more_spaces"] == 1     # the sliver touches A's and C's
    assert stats["boundaries_unmatched_external"] == 1
    assert stats["boundaries_unmatched_internal"] == 2              # C's end-on line, U's other storey
    assert stats["pairs_one_sided"] == 0 and stats["pairs_shorter_than_min"] == 2   # A-C, B-C at a point


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


def test_an_open_passage_is_adjacent_as_open_even_across_a_shared_wall():
    rooms = {"A": box(0, 0, 4, 3), "B": box(4.12, 0, 8, 3), "F": box(10, 0, 12, 3)}
    storey = {k: "s1" for k in rooms}
    wall_pairs = {("A", "B"): ["w1"]}
    out, stats = tp.adjacent_pairs(wall_pairs, rooms, storey, open_pairs=[("A", "B"), ("B", "F")])
    got = {(p["a"], p["b"]): (p["method"], p["via_gid"]) for p in out}
    # B|F are 2 m apart, but a passage joins them: still adjacent, as open.
    assert got == {("A", "B"): ("open", None), ("B", "F"): ("open", None)}
    assert stats["open_passages_sharing_a_wall"] == 1 and stats["open_passages_below_thresholds"] == 1
    assert stats["adjacent_open"] == 2 and stats["adjacent_wall"] == 0


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
    return tp.assemble([("A", "B", "d1", "boundary")], {"d1": "Door 1"}, adjacency, [("stair", "s2")],
                       geom, {"d1": False}, {("A", "w1"): True, ("B", "w1"): False}, [("A", "C")])


def test_symmetric_edges_are_stored_both_ways_and_all_enriched():
    topo = _small_topology()
    assert topo.neighbours("CONNECTED_VIA", "A") == {"B"} and topo.neighbours("CONNECTED_VIA", "B") == {"A"}
    assert topo.neighbours("ADJACENT_TO", "B") == {"A"}
    assert topo.neighbours("OPEN_TO", "A") == {"C"} and topo.neighbours("OPEN_TO", "C") == {"A"}
    assert topo.neighbours("VERTICAL_ACCESS", "stair") == {"s2"} and not topo.neighbours("VERTICAL_ACCESS", "s2")
    assert all(e.props["enriched"] is True for e in topo.edges)
    assert {e.rel_type for e in topo.edges} == set(tp.DERIVED_REL_TYPES)
    door_edge = next(e for e in topo.edges if e.rel_type == "CONNECTED_VIA")
    assert door_edge.props == {"door_gid": "d1", "door_name": "Door 1", "door_source": "boundary", "enriched": True}
    open_edge = next(e for e in topo.edges if e.rel_type == "OPEN_TO")
    assert open_edge.props == {"source": "virtual_boundary", "enriched": True}


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
    assert "CONNECTED_VIA:" not in text and "OPEN_TO:" not in text
    assert "geo_footprint_m2" in text
    # The flag means "leads outside", not "is the entrance": a model filtered on
    # it for the main entrance and picked a service door.
    assert "does not say which one is the main entrance" in text


def test_open_to_note_says_what_is_walkable():
    vocab = CombinedVocabulary(relationship_signatures=[
        ("IfcSpace", "CONNECTED_VIA", "IfcSpace"), ("IfcSpace", "OPEN_TO", "IfcSpace")])
    text = get_system_schema_string(vocab)
    assert "OPEN_TO: spaces joined by a doorless opening" in text
    assert "Walkable connections between spaces are CONNECTED_VIA or OPEN_TO" in text


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
                          "doors_bounding_2": 222, "doors_bounding_3+": 4,
                          "multi_room_doors_side_test": 4,
                          "unbounded_exterior": 6, "unbounded_inside_one_room": 12}
    # The side test reproduces every door that has boundaries before it is used.
    assert s["door_side_check"] == {"checked": 236, "agree": 236, "disagree": [], "trusted": True}
    # The reportable over-return of "shares a wall" as adjacency.
    assert s["adjacency"]["wall_pairs_same_storey"] == 1037
    assert s["adjacency"]["wall_pairs_over_return"] == pytest.approx(0.375, abs=0.01)
    assert s["edges"]["CONNECTED_VIA"] == 2 * (222 + 4 * 2)
    served = {storeys[e.target] for e in topo.edges if e.rel_type == "VERTICAL_ACCESS"}
    assert served == {"First Floor", "Second Floor"}


def test_clinic_open_passages(clinic):
    topo, gid, name, _ = clinic
    s = topo.stats["open_passages"]
    # 54 straight passages + corridor 2AC1 onto its two curved "open to below" voids.
    assert s["pairs"] == 56 and s["pairs_one_sided"] == 0 and s["virtual_boundaries"] == 209
    assert topo.stats["edges"]["OPEN_TO"] == 2 * 56
    assert {name[g] for g in topo.neighbours("OPEN_TO", gid["1C01"])} == {"1C02", "1CC1"}
    assert {name[g] for g in topo.neighbours("OPEN_TO", gid["2AC1"])} == {"2AC1-A", "2AC1-B", "2AS1", "2BC1"}
    # Every passage is adjacency with nothing between, and nothing else is.
    open_adj = {(e.source, e.target) for e in topo.edges if e.rel_type == "ADJACENT_TO" and e.props["method"] == "open"}
    assert open_adj == {(e.source, e.target) for e in topo.edges if e.rel_type == "OPEN_TO"}
    assert topo.stats["adjacency"]["open_passages_sharing_a_wall"] == 33


def test_clinic_three_room_doors_connect_across_the_leaf(clinic):
    topo, _, name, _ = clinic
    joined = {}
    for e in topo.edges:
        if e.rel_type == "CONNECTED_VIA" and e.source < e.target:
            joined.setdefault(e.props["door_gid"], set()).add(frozenset((name[e.source], name[e.target])))
    expected = {("1A10", "1A16", "1AC4"), ("1A15", "1AC2", "1AS1"), ("1B14", "1BC2", "1BC3"),
                ("2C12", "2CC3", "2CC4")}
    multi = {d: pairs for d, pairs in joined.items() if len(pairs) > 1}
    assert len(multi) == 4
    for room, *far in expected:
        pairs = next(p for p in multi.values() if frozenset((room, far[0])) in p)
        assert pairs == {frozenset((room, far[0])), frozenset((room, far[1]))}   # never far[0]-far[1]


def test_clinic_doors_without_boundaries(clinic):
    topo, _, _, _ = clinic
    exterior = {g for g, p in topo.node_props.items() if p.get("is_exterior_door")}
    assert len(exterior) == 18
    geometry_edges = [e for e in topo.edges if e.rel_type == "CONNECTED_VIA" and e.props["door_source"] == "geometry"]
    assert geometry_edges == []        # cubicles and the storefront in 1DC1 sit inside one room
    main_entrance = {"3o5ntlA212uQCF0q6eQsZI", "0otfaO0qPDAhynjJ6DnLXx"}   # 1AC1, exported twice
    assert main_entrance <= exterior


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
        assert written["OPEN_TO"] == topo.stats["edges"]["OPEN_TO"]
        with neo4j_driver.session() as s:
            adj = s.run("MATCH (:IfcSpace {Name: '1B07'})-[:ADJACENT_TO]->(b:IfcSpace) "
                        "RETURN collect(DISTINCT b.Name) AS n").single()["n"]
            con = s.run("MATCH (:IfcSpace {Name: '1B07'})-[:CONNECTED_VIA]-(b:IfcSpace) "
                        "RETURN collect(DISTINCT b.Name) AS n").single()["n"]
        assert set(adj) == {"1B04", "1B05", "1B09", "1BC2"} and set(con) == {"1BC2"}
        meta = gc.read_meta(neo4j_driver)
        assert meta["enrichment"] == "topology" and meta["topology_rev"] == tp.TOPOLOGY_REV
        assert meta["fingerprint"] == gc.fingerprint(gc.graph_summary(neo4j_driver)) != base

        with pytest.raises(RuntimeError, match="already carries"):
            tp.apply_topology(neo4j_driver, topo)

        tp.remove_topology(neo4j_driver)
        assert gc.fingerprint(gc.graph_summary(neo4j_driver)) == base
        assert gc.read_meta(neo4j_driver)["enrichment"] == "base"
        assert "topology_rev" not in gc.read_meta(neo4j_driver)
    finally:
        gc.restore_graph(neo4j_driver, found)
