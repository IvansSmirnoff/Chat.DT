"""Topology enrichment: which rooms open onto each other, which touch, which stairs serve which floors.

Why this exists
---------------
The base graph has ``(:IfcSpace)-[:BOUNDED_BY]->(:IfcElement)`` but no
room-to-room relation, so "which rooms are connected to 1B07?" has to be
answered by joining through shared boundary elements -- and that join is wrong
in three different ways:

* **Connected** (through a door): two rooms bounded by the same door. Right for
  a door between two rooms; a door recorded on three spaces needs to know which
  side of the leaf each space is on.
* **Open** (no door): the loader drops virtual space boundaries (they have no
  element), and they are the model's room-separation lines. On the dental
  clinic (arc) they join 56 pairs of spaces -- corridor segments, alcoves,
  open waiting areas -- so door-only connectivity splits the plan into islands.
* **Adjacent** (sharing a wall): two rooms bounded by the same wall is *not*
  adjacency. A long corridor wall bounds rooms metres apart. On the clinic,
  38% of the 1,037 same-storey room pairs that share a physical wall are 0.5 m
  or more apart; 1B07 shares walls with 8 rooms and touches 4.

So the enrichment computes the relations once, from the IFC (boundaries) and a
little geometry (room footprints, door placements, ``src/etl/geometry.py``),
and writes them as edges a query can follow in one hop:

* ``(:IfcSpace)-[:CONNECTED_VIA {door_gid, door_name, door_source}]->(:IfcSpace)``
  -- rooms on opposite sides of the same door. ``door_source = "boundary"``: the
  rooms come from the door's space boundaries; for a door recorded on three or
  more rooms, only rooms on opposite sides of the leaf are joined (the side
  test, below). ``door_source = "geometry"``: the door has no space boundary
  and the rooms on either side were found from the door's position.
* ``(:IfcSpace)-[:OPEN_TO {source: "virtual_boundary"}]->(:IfcSpace)`` -- spaces
  joined by a virtual space boundary: no wall, no door. Paired by geometry:
  the separation line recorded on one space coincides with one recorded on the
  other. Walkable connections are ``CONNECTED_VIA`` or ``OPEN_TO``.
* ``(:IfcSpace)-[:ADJACENT_TO {method, via_gid, gap_m, shared_len_m}]->(:IfcSpace)``
  -- same storey, footprints at most ``MAX_GAP_M`` apart along at least
  ``MIN_SHARED_M`` of edge. ``method = "open"``: nothing between them -- an
  ``OPEN_TO`` pair (always adjacent, whatever else they share), or footprints
  touching with no element recorded between them. ``method = "wall"``: they
  share a physical boundary element (``via_gid`` is the first such wall).
* ``(:IfcStair|IfcRamp|IfcTransportElement)-[:VERTICAL_ACCESS]->(:IfcBuildingStorey)``
  -- every storey whose elevation lies within the element's height range.
* node properties: ``geo_cx/geo_cy`` (footprint centroid), ``geo_z_min/geo_z_max``,
  ``geo_footprint_m2`` on rooms, doors, windows and vertical-access elements;
  ``is_exterior_door`` on doors: an EXTERNAL boundary or exactly one bounded
  room, or, for a door without boundaries, a room on one side only.
* ``BOUNDED_BY.is_external`` from the boundaries' InternalOrExternalBoundary.

The three symmetric relations are stored in **both directions**, so a directed
pattern written either way finds them; count them with an undirected pattern
or ``DISTINCT``.

Door sides. A door's placement x-axis runs along the wall, so the leaf
separates two half-planes; a room is on a side if a fair share of its
footprint near the door lies in that half (``door_side_boxes``,
``room_sides``). Before any door without boundaries is trusted to geometry,
the same rule is checked against every door that has them: it must find the
same rooms, with two-room doors split across the leaf and one-room doors facing
nothing (on the clinic: all 236). Doors inside one room (toilet-cubicle
partitions, an interior storefront) have that room on both sides and join
nothing.

Thresholds: room gaps on the clinic are bimodal (561 pairs at 0.12 m, a tail
to 0.5 m, then nothing until ~2 m), so ``MAX_GAP_M = 0.5`` sits in the valley.
Per-wall thickness was tried and rejected: the gap between neighbours is the
whole wall stack (core plus furring), while only the core bounds both rooms.

Reversible: every edge carries ``enriched = true`` and every property name is
listed below, so ``remove_topology`` returns the graph to its base fingerprint
(checked in tests). ``apply_topology`` refuses a graph that already carries
any of them, so removal can never delete base data. ``TOPOLOGY_REV`` names the
rules; it is stamped on the graph's metadata node.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from src.etl.geometry import ElementGeometry, element_geometry, shape_settings

logger = logging.getLogger(__name__)

ENRICHMENT = "topology"
#: Revision of the rules below. 1: boundary doors, nearest-two tie-break, open
#: plan from touching footprints. 2: OPEN_TO from virtual boundaries (open
#: takes precedence in ADJACENT_TO), door side test, doors without boundaries.
TOPOLOGY_REV = 2
SYMMETRIC_TYPES = ("CONNECTED_VIA", "ADJACENT_TO", "OPEN_TO")
DERIVED_REL_TYPES = SYMMETRIC_TYPES + ("VERTICAL_ACCESS",)
NODE_PROPS = ("geo_cx", "geo_cy", "geo_z_min", "geo_z_max", "geo_footprint_m2", "is_exterior_door")
BOUNDED_BY_PROPS = ("is_external",)

MAX_GAP_M = 0.5          # footprints farther apart than this are not adjacent
MIN_SHARED_M = 0.3       # ...nor if they face each other along less edge than this
OPEN_GAP_M = 0.05        # "touching" for open-plan rooms with no element between
EDGE_TOL_M = 0.05        # slack when measuring the shared edge
STOREY_TOL_M = 0.1       # a storey counts as served if within this of the height range

DOOR_TOUCH_M = 0.05      # a room touches a door if its footprint is this close to the door's
DOOR_Z_OVERLAP = 0.5     # ...and their height ranges overlap by this share of the door's height
SIDE_DEPTH_M = 0.6       # how far beyond the door's faces a side reaches
SIDE_MARGIN_M = 0.1      # ...and beyond its ends, along the wall
SIDE_SHARE = 0.2         # a room is on a side holding at least this share of its area near the door
SIDE_MIN_AREA_M2 = 0.01
GEOMETRY_TRUST = 0.95    # doors with boundaries the side test must reproduce before it is used

VERTICAL_CLASSES = ("IfcStair", "IfcRamp", "IfcTransportElement")
GEOMETRY_CLASSES = ("IfcSpace", "IfcDoor", "IfcWindow") + VERTICAL_CLASSES

Pair = Tuple[str, str]


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


Sides = Dict[str, Set[int]]          # room -> the sides of a door it is on (+1, -1)


@dataclass
class Edge:
    rel_type: str
    source: str            # GlobalId
    target: str            # GlobalId
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Topology:
    edges: List[Edge] = field(default_factory=list)
    node_props: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bounded_by: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def neighbours(self, rel_type: str, gid: str) -> Set[str]:
        return {e.target for e in self.edges if e.rel_type == rel_type and e.source == gid}


# -----------------------------------------------------------------------------
# Pure rules (no IFC, no Neo4j): unit-tested on synthetic shapes
# -----------------------------------------------------------------------------

def door_side_boxes(footprint: BaseGeometry, x_axis: Tuple[float, float],
                    depth: float = SIDE_DEPTH_M, margin: float = SIDE_MARGIN_M) -> Tuple[Polygon, Polygon]:
    """The two half-boxes either side of a door leaf: (+y side, -y side).

    ``x_axis`` is the door's placement x-axis, which runs along the wall. If the
    footprint is deeper than it is wide along it, the axes are swapped.
    """
    ux, uy = x_axis
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    pts = list(footprint.convex_hull.exterior.coords) if footprint.convex_hull.geom_type == "Polygon" \
        else list(footprint.convex_hull.coords)
    ox, oy = footprint.centroid.x, footprint.centroid.y

    def extents(ax, ay):
        along = [(x - ox) * ax + (y - oy) * ay for x, y in pts]
        across = [-(x - ox) * ay + (y - oy) * ax for x, y in pts]
        return along, across

    along, across = extents(ux, uy)
    if max(along) - min(along) < max(across) - min(across):
        ux, uy = -uy, ux
        along, across = extents(ux, uy)
    x0, x1 = min(along) - margin, max(along) + margin
    y0, y1 = min(across), max(across)
    mid = (y0 + y1) / 2

    def box(a0, a1, b0, b1):
        return Polygon([(ox + a * ux - b * uy, oy + a * uy + b * ux)
                        for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))])

    return box(x0, x1, mid, y1 + depth), box(x0, x1, y0 - depth, mid)


def room_sides(rooms: Mapping[str, BaseGeometry], boxes: Tuple[Polygon, Polygon],
               share: float = SIDE_SHARE, min_area: float = SIDE_MIN_AREA_M2) -> Sides:
    """Which side(s) of a door each room is on; a room inside both halves is on both."""
    out: Sides = {}
    for room, fp in rooms.items():
        areas = {side: fp.intersection(b).area for side, b in zip((1, -1), boxes)}
        total = sum(areas.values())
        out[room] = {side for side, a in areas.items() if a >= min_area and a >= share * total}
    return out


def split_sides(sides: Sides) -> Optional[Tuple[Set[str], Set[str]]]:
    """(rooms on +, rooms on -) when every room is on exactly one side and both sides have one."""
    if any(len(s) != 1 for s in sides.values()):
        return None
    plus = {r for r, s in sides.items() if s == {1}}
    minus = set(sides) - plus
    return (plus, minus) if plus and minus else None


def validate_door_sides(door_spaces: Mapping[str, Set[str]],
                        door_sides: Mapping[str, Sides]) -> Dict[str, Any]:
    """Does the geometric rule reproduce the doors that have space boundaries?

    Agreement: the same rooms touch the door; two or more rooms split across
    the leaf; a single room on one side, facing nothing.
    """
    checked = agree = 0
    disagree: List[str] = []
    for door, rooms in sorted(door_spaces.items()):
        if not rooms or door not in door_sides:
            continue
        checked += 1
        sides = door_sides[door]
        ok = set(sides) == set(rooms)
        if ok and len(rooms) == 1:
            ok = len(next(iter(sides.values()))) == 1
        elif ok:
            ok = split_sides(sides) is not None
        agree += ok
        if not ok:
            disagree.append(door)
    return {"checked": checked, "agree": agree, "disagree": disagree[:20],
            "trusted": bool(checked) and agree / checked >= GEOMETRY_TRUST}


def door_space_pairs(
    door_spaces: Mapping[str, Set[str]],
    door_points: Mapping[str, Tuple[float, float]],
    footprints: Mapping[str, BaseGeometry],
    door_sides: Optional[Mapping[str, Sides]] = None,
    use_geometry: bool = False,
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, bool], Dict[str, int]]:
    """(room, room, door, door_source) for every pair of rooms a door joins.

    ``door_spaces`` (from the space boundaries) decides which rooms a door
    bounds. A door on three or more rooms joins the rooms on opposite sides of
    its leaf (``door_sides``); if the sides are not clear, the two rooms nearest
    its centre, and without a position or footprints it is skipped.
    ``use_geometry``: a door with no boundary joins the rooms on its two sides
    (``door_source = "geometry"``). Returns the pairs, ``is_exterior_door`` for
    those doors (a room on one side only), and counts.
    """
    from shapely.geometry import Point

    door_sides = door_sides or {}
    pairs: List[Tuple[str, str, str, str]] = []
    exterior: Dict[str, bool] = {}
    stats: Counter = Counter()
    for door in sorted(set(door_spaces) | (set(door_sides) if use_geometry else set())):
        rooms = sorted(door_spaces.get(door, ()))
        if rooms:
            stats[f"doors_bounding_{min(len(rooms), 3)}{'+' if len(rooms) >= 3 else ''}"] += 1
        if len(rooms) == 2:
            pairs.append((rooms[0], rooms[1], door, "boundary"))
        elif len(rooms) > 2:
            split = split_sides({r: door_sides.get(door, {}).get(r, set()) for r in rooms})
            if split is not None:
                stats["multi_room_doors_side_test"] += 1
                pairs += [(*_pair(a, b), door, "boundary") for a in sorted(split[0]) for b in sorted(split[1])]
                continue
            point = door_points.get(door)
            if point is None or any(r not in footprints for r in rooms):
                stats["multi_room_doors_skipped"] += 1
                continue
            stats["multi_room_doors_nearest_two"] += 1
            near = sorted(sorted(rooms, key=lambda r: (footprints[r].distance(Point(point)), r))[:2])
            pairs.append((near[0], near[1], door, "boundary"))
        elif not rooms:
            stats["doors_bounding_0"] += 1
            if not use_geometry or door not in door_sides:
                continue
            sides = door_sides[door]
            plus = {r for r, s in sides.items() if 1 in s}
            minus = {r for r, s in sides.items() if -1 in s}
            if not plus and not minus:
                stats["unbounded_no_room"] += 1
            elif not plus or not minus:
                stats["unbounded_exterior"] += 1
                exterior[door] = True
            else:
                joined = sorted({_pair(a, b) for a in plus for b in minus if a != b})
                stats["unbounded_joining_rooms" if joined else "unbounded_inside_one_room"] += 1
                exterior[door] = False
                pairs += [(a, b, door, "geometry") for a, b in joined]
    return pairs, exterior, dict(stats)


def exterior_doors(door_spaces: Mapping[str, Set[str]], external: Mapping[str, bool],
                   geometric: Optional[Mapping[str, bool]] = None) -> Dict[str, bool]:
    """``is_exterior_door`` for every door with boundary records, then the geometric ones."""
    flags = {door: bool(external.get(door)) or len(spaces) == 1
             for door, spaces in door_spaces.items() if spaces}
    for door, flag in (geometric or {}).items():
        flags.setdefault(door, flag)
    return flags


def open_passages(
    lines: Sequence[Tuple[str, BaseGeometry, bool]],
    storey_of: Mapping[str, Optional[str]],
    tol: float = EDGE_TOL_M,
    min_len: float = MIN_SHARED_M,
) -> Tuple[List[Pair], Dict[str, Any]]:
    """Space pairs joined by a virtual boundary recorded on both of them.

    ``lines``: (space, plan line of the boundary, is EXTERNAL) for every virtual
    boundary. A boundary *matches* another space on its storey when at least
    half of it coincides (within ``tol``) with that space's virtual boundaries.
    Two spaces are paired when each one's boundaries coincide with the other's
    along at least ``min_len`` in total. A boundary matching nothing faces the
    outside or a space the model does not have on that storey.
    """
    tree = STRtree([line for _, line, _ in lines])
    coincide: Dict[Pair, float] = defaultdict(float)      # (space, other) -> length of space's lines on other's
    stats: Counter = Counter()
    unmatched_internal: List[float] = []
    for space, line, external in lines:
        on: Dict[str, float] = defaultdict(float)
        for j in tree.query(line.buffer(tol)):
            other, other_line, _ = lines[int(j)]
            if other == space or storey_of.get(space) is None or storey_of.get(space) != storey_of.get(other):
                continue
            on[other] += line.intersection(other_line.buffer(tol)).length
        for other, length in on.items():
            coincide[(space, other)] += length
        matched = [o for o, length in on.items() if length >= 0.5 * line.length]
        if len(matched) > 1:
            stats["boundaries_matching_two_or_more_spaces"] += 1
        elif matched:
            stats["boundaries_matching_one_space"] += 1
        elif external:
            stats["boundaries_unmatched_external"] += 1
        else:
            stats["boundaries_unmatched_internal"] += 1
            unmatched_internal.append(round(line.length, 3))
    candidates = {_pair(*k) for k in coincide}
    pairs = sorted(p for p in candidates
                   if min(coincide.get(p, 0.0), coincide.get((p[1], p[0]), 0.0)) >= min_len)
    one_sided = sum(1 for a, b in candidates
                    if (coincide.get((a, b), 0.0) >= min_len) != (coincide.get((b, a), 0.0) >= min_len))
    out = {"virtual_boundaries": len(lines), **dict(sorted(stats.items())), "pairs": len(pairs),
           "pairs_one_sided": one_sided, "pairs_shorter_than_min": len(candidates) - len(pairs) - one_sided}
    if unmatched_internal:
        out["unmatched_internal_lengths_m"] = sorted(unmatched_internal)
    return pairs, out


def shared_length(a: BaseGeometry, b: BaseGeometry, gap: float, tol: float = EDGE_TOL_M) -> float:
    """How much of each footprint's outline lies within ``gap + tol`` of the other (the smaller).

    Overestimates a straight shared wall by up to ``2 * tol``: the reach also
    takes in that much of the perpendicular edges at its two ends.
    """
    reach = gap + tol
    return min(a.boundary.intersection(b.buffer(reach)).length,
               b.boundary.intersection(a.buffer(reach)).length)


def adjacent_pairs(
    wall_pairs: Mapping[Pair, Sequence[str]],
    footprints: Mapping[str, BaseGeometry],
    storey_of: Mapping[str, Optional[str]],
    max_gap: float = MAX_GAP_M,
    min_shared: float = MIN_SHARED_M,
    open_gap: float = OPEN_GAP_M,
    open_pairs: Iterable[Pair] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rooms that really touch: through a shared wall (``wall``) or with nothing between (``open``).

    ``wall_pairs`` maps a sorted room pair to the physical elements bounding
    both. ``open_pairs`` (sorted) are joined by a virtual boundary: they are
    adjacent with ``method = "open"`` even where they also share a wall
    elsewhere or their footprints miss the thresholds. Pairs on different (or
    unknown) storeys are never adjacent.
    """
    open_set = set(open_pairs)
    out: List[Dict[str, Any]] = []
    dropped: Counter = Counter()
    gaps: List[float] = []
    reclassified = 0
    for (a, b), walls in sorted(wall_pairs.items()):
        if a not in footprints or b not in footprints:
            dropped["no footprint"] += 1
            continue
        if storey_of.get(a) is None or storey_of.get(a) != storey_of.get(b):
            dropped["different or unknown storey"] += 1
            continue
        gap = footprints[a].distance(footprints[b])
        gaps.append(gap)
        if gap > max_gap:
            dropped["gap above max"] += 1
            continue
        shared = shared_length(footprints[a], footprints[b], gap)
        if shared < min_shared:
            dropped["shared edge below min"] += 1
            continue
        is_open = (a, b) in open_set
        reclassified += is_open
        out.append({"a": a, "b": b, "method": "open" if is_open else "wall",
                    "via_gid": None if is_open else min(walls),
                    "gap_m": round(gap, 3), "shared_len_m": round(shared, 3)})

    # Open plan: rooms whose footprints touch with no element recorded between them.
    keys = sorted(k for k in footprints if storey_of.get(k) is not None)
    tree = STRtree([footprints[k] for k in keys])
    seen = {(p["a"], p["b"]) for p in out} | set(wall_pairs)
    open_plan = 0
    for i, a in enumerate(keys):
        for j in tree.query(footprints[a].buffer(open_gap)):
            b = keys[int(j)]
            if a >= b or (a, b) in seen or storey_of[a] != storey_of[b]:
                continue
            gap = footprints[a].distance(footprints[b])
            if gap > open_gap:
                continue
            shared = shared_length(footprints[a], footprints[b], gap)
            if shared >= min_shared:
                out.append({"a": a, "b": b, "method": "open", "via_gid": None,
                            "gap_m": round(gap, 3), "shared_len_m": round(shared, 3)})
                open_plan += (a, b) not in open_set

    # Passages the footprint rules missed: still adjacent, they share an opening.
    found = {(p["a"], p["b"]) for p in out}
    forced = 0
    for a, b in sorted(open_set - found):
        if a not in footprints or b not in footprints:
            continue
        gap = footprints[a].distance(footprints[b])
        out.append({"a": a, "b": b, "method": "open", "via_gid": None,
                    "gap_m": round(gap, 3), "shared_len_m": round(shared_length(footprints[a], footprints[b], gap), 3)})
        forced += 1
    n_open = sum(p["method"] == "open" for p in out)
    stats = {
        "wall_pairs_same_storey": len(gaps),
        "wall_pairs_gap_above_max": sum(g > max_gap for g in gaps),
        "wall_pairs_over_return": round(sum(g > max_gap for g in gaps) / len(gaps), 4) if gaps else None,
        "adjacent_wall": len(out) - n_open,
        "adjacent_open": n_open,
        "open_passages_sharing_a_wall": reclassified,
        "open_passages_below_thresholds": forced,
        "open_plan_without_passage": open_plan,
        "dropped": dict(dropped),
    }
    return out, stats


def vertical_access(
    z_ranges: Mapping[str, Tuple[float, float]],
    storey_z: Mapping[str, float],
    tol: float = STOREY_TOL_M,
) -> List[Tuple[str, str]]:
    """(element, storey) for each storey whose elevation lies in the element's height range."""
    return [(el, st) for el, (lo, hi) in sorted(z_ranges.items())
            for st, z in sorted(storey_z.items()) if lo - tol <= z <= hi + tol]


def assemble(
    connections: Iterable[Tuple[str, str, str, str]],
    door_names: Mapping[str, Optional[str]],
    adjacency: Iterable[Mapping[str, Any]],
    access: Iterable[Tuple[str, str]],
    geometry: Mapping[str, ElementGeometry],
    exterior: Mapping[str, bool],
    boundary_external: Mapping[Pair, bool],
    passages: Iterable[Pair] = (),
) -> Topology:
    """Edges (symmetric ones both ways) and properties, every name from the declared lists."""
    topo = Topology()

    def both_ways(rel: str, a: str, b: str, props: Dict[str, Any]) -> None:
        props = {k: v for k, v in props.items() if v is not None}
        props["enriched"] = True
        topo.edges.extend([Edge(rel, a, b, dict(props)), Edge(rel, b, a, dict(props))])

    for a, b, door, source in connections:
        both_ways("CONNECTED_VIA", a, b, {"door_gid": door, "door_name": door_names.get(door),
                                          "door_source": source})
    for a, b in passages:
        both_ways("OPEN_TO", a, b, {"source": "virtual_boundary"})
    for p in adjacency:
        both_ways("ADJACENT_TO", p["a"], p["b"], {k: v for k, v in p.items() if k not in ("a", "b")})
    for el, storey in access:
        topo.edges.append(Edge("VERTICAL_ACCESS", el, storey, {"enriched": True}))
    for gid, g in geometry.items():
        cx, cy = g.centroid
        topo.node_props[gid] = {"geo_cx": round(cx, 3), "geo_cy": round(cy, 3),
                                "geo_z_min": round(g.z_min, 3), "geo_z_max": round(g.z_max, 3),
                                "geo_footprint_m2": round(g.footprint.area, 3)}
    for door, flag in exterior.items():
        topo.node_props.setdefault(door, {})["is_exterior_door"] = flag
    topo.bounded_by = [(s, e, {"is_external": flag}) for (s, e), flag in sorted(boundary_external.items())]
    return topo


# -----------------------------------------------------------------------------
# From an IFC file
# -----------------------------------------------------------------------------

def _storey(element) -> Optional[Any]:
    import ifcopenshell.util.element as ue

    for parent in (ue.get_aggregate(element), ue.get_container(element)):
        if parent is not None and parent.is_a("IfcBuildingStorey"):
            return parent
    return None


def _curve_points(curve, angle_scale: float) -> Optional[List[Tuple[float, float]]]:
    """2D points along a boundary's swept curve: a polyline, or an arc of a circle (5° steps)."""
    import ifcopenshell.util.placement as up

    if curve is None:
        return None
    if curve.is_a("IfcPolyline"):
        return [tuple(p.Coordinates[:2]) for p in curve.Points]
    if curve.is_a("IfcTrimmedCurve") and curve.BasisCurve.is_a("IfcCircle"):
        circle = curve.BasisCurve
        place = up.get_axis2placement(circle.Position)

        def angle(trim) -> Optional[float]:
            for t in trim:
                if t.is_a("IfcParameterValue"):
                    return t.wrappedValue * angle_scale
            for t in trim:
                if t.is_a("IfcCartesianPoint"):
                    x, y = t.Coordinates[0] - place[0][3], t.Coordinates[1] - place[1][3]
                    return math.atan2(place[0][1] * x + place[1][1] * y, place[0][0] * x + place[1][0] * y)
            return None

        t1, t2 = angle(curve.Trim1), angle(curve.Trim2)
        if t1 is None or t2 is None:
            return None
        if not curve.SenseAgreement:
            t1, t2 = t2, t1
        if t2 < t1:
            t2 += 2 * math.pi
        n = max(2, math.ceil((t2 - t1) / math.radians(5)) + 1)
        out = []
        for i in range(n):
            t = t1 + (t2 - t1) * i / (n - 1)
            x, y = circle.Radius * math.cos(t), circle.Radius * math.sin(t)
            out.append((place[0][0] * x + place[0][1] * y + place[0][3],
                        place[1][0] * x + place[1][1] * y + place[1][3]))
        return out
    return None


def virtual_boundary_line(rb, scale: float, angle_scale: float) -> Optional[LineString]:
    """Plan line (metres, world) of a virtual boundary given as a vertical extruded curve."""
    import ifcopenshell.util.placement as up
    import numpy as np

    surface = getattr(rb.ConnectionGeometry, "SurfaceOnRelatingElement", None) \
        if rb.ConnectionGeometry is not None else None
    if surface is None or not surface.is_a("IfcSurfaceOfLinearExtrusion") \
            or rb.RelatingSpace.ObjectPlacement is None:
        return None
    points = _curve_points(getattr(surface.SweptCurve, "Curve", None), angle_scale)
    if not points or len(points) < 2:
        return None
    matrix = up.get_local_placement(rb.RelatingSpace.ObjectPlacement)
    if surface.Position is not None:
        matrix = matrix @ up.get_axis2placement(surface.Position)
    world = [matrix @ np.array([x, y, 0.0, 1.0]) for x, y in points]
    return LineString([(float(w[0]) * scale, float(w[1]) * scale) for w in world])


def compute_topology(ifc_path: Path, max_gap: float = MAX_GAP_M,
                     min_shared: float = MIN_SHARED_M) -> Topology:
    """Everything the enrichment writes, computed from the IFC alone (no Neo4j)."""
    import ifcopenshell
    import ifcopenshell.util.placement as up
    import ifcopenshell.util.unit as uu

    start = time.monotonic()
    model = ifcopenshell.open(str(ifc_path))
    scale = uu.calculate_unit_scale(model)
    angle_scale = uu.calculate_unit_scale(model, "PLANEANGLEUNIT")
    settings = shape_settings()

    geometry: Dict[str, ElementGeometry] = {}
    for cls in GEOMETRY_CLASSES:
        for el in model.by_type(cls):
            g = element_geometry(el, settings, include_parts=cls in VERTICAL_CLASSES)
            if g is not None:
                geometry[el.GlobalId] = g

    spaces = {s.GlobalId: s for s in model.by_type("IfcSpace")}
    doors = {d.GlobalId: d for d in model.by_type("IfcDoor")}
    storey_of = {gid: (st.GlobalId if (st := _storey(s)) is not None else None) for gid, s in spaces.items()}

    element_spaces: Dict[str, Set[str]] = defaultdict(set)      # physical boundaries only
    element_class: Dict[str, str] = {}
    door_external: Dict[str, bool] = {}
    boundary_external: Dict[Pair, bool] = {}
    virtual_lines: List[Tuple[str, BaseGeometry, bool]] = []
    virtual_skipped = 0
    for rb in model.by_type("IfcRelSpaceBoundary"):
        space, el = rb.RelatingSpace, rb.RelatedBuildingElement
        if space is None or not space.is_a("IfcSpace"):
            continue
        external = rb.InternalOrExternalBoundary == "EXTERNAL"
        if rb.PhysicalOrVirtualBoundary == "VIRTUAL":
            line = virtual_boundary_line(rb, scale, angle_scale)
            if line is None:
                virtual_skipped += 1
            else:
                virtual_lines.append((space.GlobalId, line, external))
        if el is None:
            continue
        key = (space.GlobalId, el.GlobalId)
        boundary_external[key] = boundary_external.get(key, False) or external
        if el.GlobalId in doors:
            door_external[el.GlobalId] = door_external.get(el.GlobalId, False) or external
        if rb.PhysicalOrVirtualBoundary == "PHYSICAL":
            element_spaces[el.GlobalId].add(space.GlobalId)
            element_class[el.GlobalId] = el.is_a()

    footprints = {gid: geometry[gid].footprint for gid in spaces if gid in geometry}
    passages, passage_stats = open_passages(virtual_lines, storey_of)
    if virtual_skipped:
        passage_stats["virtual_boundaries_unreadable"] = virtual_skipped

    # Doors: which rooms touch each one, and on which side of its leaf.
    room_keys = sorted(footprints)
    room_tree = STRtree([footprints[k] for k in room_keys])
    door_sides: Dict[str, Sides] = {}
    for gid, door in doors.items():
        g = geometry.get(gid)
        if g is None or door.ObjectPlacement is None:
            continue
        matrix = up.get_local_placement(door.ObjectPlacement)
        height = max(g.z_max - g.z_min, 1e-6)
        touching = {}
        for j in room_tree.query(g.footprint.buffer(DOOR_TOUCH_M)):
            room = room_keys[int(j)]
            rg = geometry[room]
            if min(g.z_max, rg.z_max) - max(g.z_min, rg.z_min) < DOOR_Z_OVERLAP * height:
                continue
            if footprints[room].distance(g.footprint) <= DOOR_TOUCH_M:
                touching[room] = footprints[room]
        boxes = door_side_boxes(g.footprint, (float(matrix[0][0]), float(matrix[1][0])))
        door_sides[gid] = room_sides(touching, boxes)

    door_spaces = {d: set(element_spaces.get(d, ())) for d in doors}
    door_check = validate_door_sides(door_spaces, door_sides)
    if not door_check["trusted"]:
        logger.warning("Door side test reproduces %d/%d doors with boundaries: not used for doors without",
                       door_check["agree"], door_check["checked"])
    door_points = {d: geometry[d].centroid for d in doors if d in geometry}
    connections, geometric_exterior, door_stats = door_space_pairs(
        door_spaces, door_points, footprints, door_sides, use_geometry=door_check["trusted"])

    wall_pairs: Dict[Pair, List[str]] = defaultdict(list)
    for el, rooms in element_spaces.items():
        if element_class[el] in ("IfcDoor", "IfcWindow"):
            continue
        for a, b in itertools.combinations(sorted(rooms), 2):
            wall_pairs[(a, b)].append(el)
    adjacency, adj_stats = adjacent_pairs(wall_pairs, footprints, storey_of, max_gap, min_shared,
                                          open_pairs=passages)

    storey_z = {st.GlobalId: float(up.get_local_placement(st.ObjectPlacement)[2][3]) * scale
                for st in model.by_type("IfcBuildingStorey") if st.ObjectPlacement is not None}
    z_ranges = {el.GlobalId: (geometry[el.GlobalId].z_min, geometry[el.GlobalId].z_max)
                for cls in VERTICAL_CLASSES for el in model.by_type(cls) if el.GlobalId in geometry}
    access = vertical_access(z_ranges, storey_z)

    exterior = exterior_doors(door_spaces, door_external, geometric_exterior)
    topo = assemble(connections, {d: doors[d].Name for d in doors}, adjacency, access, geometry,
                    exterior, boundary_external, passages)
    topo.stats = {
        "ifc": str(ifc_path),
        "rev": TOPOLOGY_REV,
        "params": {"max_gap_m": max_gap, "min_shared_m": min_shared, "open_gap_m": OPEN_GAP_M,
                   "storey_tol_m": STOREY_TOL_M, "door_touch_m": DOOR_TOUCH_M, "side_depth_m": SIDE_DEPTH_M,
                   "side_share": SIDE_SHARE},
        "spaces": len(spaces), "spaces_with_footprint": len(footprints),
        "spaces_without_storey": sum(v is None for v in storey_of.values()),
        "doors": dict(sorted(door_stats.items())),
        "door_side_check": door_check,
        "exterior_doors": sum(exterior.values()),
        "open_passages": passage_stats,
        "adjacency": adj_stats,
        "vertical_access": len(access),
        "edges": dict(Counter(e.rel_type for e in topo.edges)),
        "seconds": round(time.monotonic() - start, 1),
    }
    return topo


# -----------------------------------------------------------------------------
# Neo4j: apply and remove
# -----------------------------------------------------------------------------

def _existing_enrichment(session) -> List[str]:
    found = []
    for rel in DERIVED_REL_TYPES:
        if session.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c").single()["c"]:
            found.append(rel)
    for prop in NODE_PROPS:
        if session.run(f"MATCH (n) WHERE n.`{prop}` IS NOT NULL RETURN count(n) AS c").single()["c"]:
            found.append(f"node.{prop}")
    for prop in BOUNDED_BY_PROPS:
        if session.run(f"MATCH ()-[r:BOUNDED_BY]->() WHERE r.`{prop}` IS NOT NULL "
                       "RETURN count(r) AS c").single()["c"]:
            found.append(f"BOUNDED_BY.{prop}")
    return found


def _restamp_meta(driver, enrichment: str, extra: Optional[Dict[str, Any]] = None) -> None:
    from src.etl.graph_cache import fingerprint, graph_summary, read_meta, write_meta

    meta = read_meta(driver)
    if meta is None:
        return
    meta = {k: v for k, v in meta.items() if k not in ("cache_path", "topology_params", "topology_rev")}
    meta.update({"enrichment": enrichment, "fingerprint": fingerprint(graph_summary(driver)), **(extra or {})})
    write_meta(driver, meta)


def apply_topology(driver, topo: Topology) -> Dict[str, Any]:
    """Write the enrichment into the loaded graph; refuse a graph that already has any of it."""
    with driver.session() as session:
        found = _existing_enrichment(session)
        if found:
            raise RuntimeError(f"graph already carries enrichment names {found}; run remove_topology first")
        written: Dict[str, Any] = {}
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in topo.edges:
            by_type[e.rel_type].append({"a": e.source, "b": e.target, "p": e.props})
        for rel, rows in sorted(by_type.items()):
            written[rel] = session.run(
                f"UNWIND $rows AS r MATCH (a:IfcRoot {{GlobalId: r.a}}) MATCH (b:IfcRoot {{GlobalId: r.b}}) "
                f"CREATE (a)-[x:`{rel}`]->(b) SET x = r.p RETURN count(x) AS c", rows=rows).single()["c"]
        rows = [{"g": g, "p": p} for g, p in sorted(topo.node_props.items())]
        written["node_props"] = session.run(
            "UNWIND $rows AS r MATCH (n:IfcRoot {GlobalId: r.g}) SET n += r.p RETURN count(n) AS c",
            rows=rows).single()["c"]
        rows = [{"s": s, "e": e, "p": p} for s, e, p in topo.bounded_by]
        written["bounded_by_props"] = session.run(
            "UNWIND $rows AS r MATCH (:IfcRoot {GlobalId: r.s})-[b:BOUNDED_BY]->(:IfcRoot {GlobalId: r.e}) "
            "SET b += r.p RETURN count(b) AS c", rows=rows).single()["c"]
    expected = Counter(e.rel_type for e in topo.edges)
    written["unmatched_edges"] = {rel: expected[rel] - written.get(rel, 0)
                                  for rel in expected if expected[rel] != written.get(rel, 0)}
    if written["unmatched_edges"]:
        logger.warning("Enrichment edges whose endpoints are not in the graph: %s", written["unmatched_edges"])
    _restamp_meta(driver, ENRICHMENT, {"topology_params": json.dumps(topo.stats.get("params", {})),
                                       "topology_rev": TOPOLOGY_REV})
    return written


def remove_topology(driver) -> Dict[str, int]:
    """Delete every enrichment edge and property; the base graph is left as it was."""
    removed: Dict[str, int] = {}
    with driver.session() as session:
        for rel in DERIVED_REL_TYPES:
            removed[rel] = session.run(
                f"MATCH ()-[r:`{rel}`]->() WHERE r.enriched = true DELETE r RETURN count(r) AS c").single()["c"]
        for prop in NODE_PROPS:
            removed[prop] = session.run(
                f"MATCH (n) WHERE n.`{prop}` IS NOT NULL REMOVE n.`{prop}` RETURN count(n) AS c").single()["c"]
        for prop in BOUNDED_BY_PROPS:
            removed[f"BOUNDED_BY.{prop}"] = session.run(
                f"MATCH ()-[r:BOUNDED_BY]->() WHERE r.`{prop}` IS NOT NULL REMOVE r.`{prop}` "
                "RETURN count(r) AS c").single()["c"]
    _restamp_meta(driver, "base")
    return removed
