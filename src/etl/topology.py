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
little geometry (room and wall footprints, door placements,
``src/etl/geometry.py``), and writes them as edges a query can follow in one hop:

* ``(:IfcSpace)-[:CONNECTED_VIA {door_gid, door_name, door_source}]->(:IfcSpace)``
  -- rooms on opposite sides of the same door. ``door_source = "boundary"``: the
  rooms come from the door's space boundaries; for a door recorded on three or
  more rooms, only rooms on opposite sides of the leaf are joined (the side
  test, below). ``door_source = "geometry"``: the door has no space boundary
  and the rooms on either side were found from the door's position.
* ``(:IfcSpace)-[:OPEN_TO {source}]->(:IfcSpace)`` -- spaces joined with no
  wall and no door. ``source = "virtual_boundary"``: a virtual space boundary
  (the model's room-separation line) recorded on each coincides with one on
  the other. ``source = "geometry"``: footprints touching along an edge no
  wall crosses (below). Walkable connections are ``CONNECTED_VIA`` or ``OPEN_TO``.
* ``(:IfcSpace)-[:ADJACENT_TO {method, via_gid, gap_m, shared_len_m, source}]->(:IfcSpace)``
  -- same storey, footprints at most ``MAX_GAP_M`` apart along at least
  ``MIN_SHARED_M`` of edge. ``method = "open"``: nothing between them -- an
  ``OPEN_TO`` pair (always adjacent, whatever else they share), or footprints
  touching with no element recorded between them. ``method = "wall"``: a wall
  between them (``via_gid``): with ``source = "boundary"`` a physical boundary
  element of both, with ``source = "geometry"`` the wall filling the strip
  between their footprints.
* ``(:IfcStair|IfcRamp|IfcTransportElement)-[:VERTICAL_ACCESS]->(:IfcBuildingStorey)``
  -- every storey whose elevation lies within the element's height range.
* node properties: ``geo_cx/geo_cy`` (footprint centroid), ``geo_z_min/geo_z_max``,
  ``geo_footprint_m2`` on rooms, doors, windows and vertical-access elements;
  ``is_exterior_door`` on doors: an EXTERNAL boundary or exactly one bounded
  room; for a door without boundaries, its ``Pset_DoorCommon.IsExternal`` where
  the model gives it, else rooms on one side only.
* ``BOUNDED_BY.is_external`` from the boundaries' InternalOrExternalBoundary.

The three symmetric relations are stored in **both directions**, so a directed
pattern written either way finds them; count them with an undirected pattern
or ``DISTINCT``.

Boundaries first, geometry where they are missing. 8 of the 19 IFC-Bench
models with spaces carry no ``IfcRelSpaceBoundary`` (digital_hub, the five
fantasy_* models, the samuel_macalister house, wbdg_office's MEP file), so
every rule has a geometric form, used per door (a door without a physical
boundary) and per room pair (a pair with a room that has no boundary record);
a model whose boundaries are complete, like the clinic, never reaches it.
``compute_topology(..., use_boundaries=False)`` ignores the boundaries, which
is how the fallback was validated on the seven models that have them
(dental_clinic, ac20, smiley_west, wbdg_office, duplex, fzk_house, 4351):
geometry alone gives every boundary door link but one (a clinic corridor that
touches only a door's jamb) and nothing more, and every open passage.

Door sides. A door's placement x-axis runs along the wall, so the leaf
separates two half-planes; a room is on a side if a fair share of its
footprint near the door lies in that half and it fills a quarter of the
half-box (``door_side_boxes``, ``room_sides``). Doors are often thinner than
their walls (ac20: 0.10 m doors in 0.30 m walls) and rooms stop at the wall's
faces, so the door is widened across its host wall first (``door_passage``:
the host from IfcRelFillsElement/IfcRelVoidsElement, else the wall its
footprint overlaps; thickness measured on the wall's solid footprint). Before
any door without boundaries is trusted to geometry, the same rule is checked
against every door that has them (on the clinic: 235 of 236). A space on both
sides of a door surrounds it -- the room of a toilet cubicle, a gross-area
zone, two spaces exported with one footprint -- and is joined to nothing.

Geometry pairs (``geometric_pairs``): the boundary rule's thresholds without
a recorded wall; what lies between two rooms is read from the walls, curtain
walls and columns (with their parts) in the strip between their footprints.
Touching footprints with no wall across ``MIN_SHARED_M`` of their shared edge
are an open passage; an unfilled opening at floor level (a cased opening, a
lift entrance) cuts its wall. Rooms overlapping by more than half the smaller
are one room twice or a zone, not neighbours. A file with spaces but no walls
(an MEP model) gets no geometric pairs.

Storeys. A space aggregated under the site or the building (fantasy_office_building_2:
22 of 27) is placed by its floor height on the storey at that elevation, else on a
level of its own (``assign_levels``); only the same-storey tests use it.

Thresholds: room gaps on the clinic are bimodal (561 pairs at 0.12 m, a tail
to 0.5 m, then nothing until ~2 m), so ``MAX_GAP_M = 0.5`` sits in the valley.
Per-wall thickness was tried and rejected: the gap between neighbours is the
whole wall stack (core plus furring), while only the core bounds both rooms.
Rooms behind a door fill 70-95% of its side box, a corridor running past the
jamb under 11% (seven boundary models): ``SIDE_BOX_SHARE = 0.25``.

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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.etl.geometry import (
    ElementGeometry,
    element_geometry,
    iterate_geometry,
    merge,
    shape_settings,
)

logger = logging.getLogger(__name__)

ENRICHMENT = "topology"
#: Revision of the rules below. 1: boundary doors, nearest-two tie-break, open
#: plan from touching footprints. 2: OPEN_TO from virtual boundaries (open
#: takes precedence in ADJACENT_TO), door side test, doors without boundaries.
#: 3: geometry fallback for spaces and doors without space boundaries (door
#: passage through the host wall, ADJACENT_TO/OPEN_TO from footprints and wall
#: geometry, ``source`` on every edge), storeys from floor heights.
TOPOLOGY_REV = 3
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
SIDE_BOX_SHARE = 0.25    # ...and filling at least this share of that side's box
GEOMETRY_TRUST = 0.95    # doors with boundaries the side test must reproduce before it is used
MAX_WALL_M = 1.0         # a host wall measured thicker than this at a door is a measuring error
WALL_STRIP_M = 0.05      # half-width, along the wall, of the strip that measures it at a door

OVERLAP_SHARE = 0.5      # spaces overlapping by more than this share of the smaller one are not neighbours
BLOCK_Z_SHARE = 0.5      # an element separates two rooms if it spans this share of their common height
SILL_TOL_M = 0.3         # an unfilled opening is walkable if its sill is this close to the floor
HEADROOM_M = 1.8         # ...and its head at least this high above it (or the room's full height)
LEVEL_TOL_M = 0.3        # a space with no storey goes on the storey whose elevation is this close to its floor

VERTICAL_CLASSES = ("IfcStair", "IfcRamp", "IfcTransportElement")
GEOMETRY_CLASSES = ("IfcSpace", "IfcDoor", "IfcWindow") + VERTICAL_CLASSES
#: What separates rooms (and hosts doors) in the geometry fallback, with the
#: parts they aggregate.
WALL_CLASSES = ("IfcWall", "IfcCurtainWall", "IfcColumn")

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

Axis = Tuple[float, float]           # unit vector along the wall a door sits in
Origin = Tuple[float, float]


def _extents(shape: BaseGeometry, axis: Axis, origin: Origin) -> Tuple[float, float, float, float]:
    """(along min, along max, across min, across max) of a shape's hull in the frame (origin, axis)."""
    ux, uy = axis
    ox, oy = origin
    hull = shape.convex_hull
    pts = list(hull.exterior.coords) if hull.geom_type == "Polygon" else list(hull.coords)
    along = [(x - ox) * ux + (y - oy) * uy for x, y in pts]
    across = [-(x - ox) * uy + (y - oy) * ux for x, y in pts]
    return min(along), max(along), min(across), max(across)


def _box(origin: Origin, axis: Axis, a0: float, a1: float, b0: float, b1: float) -> Polygon:
    """The rectangle [a0, a1] along x [b0, b1] across, in the frame (origin, axis)."""
    (ox, oy), (ux, uy) = origin, axis
    return Polygon([(ox + a * ux - b * uy, oy + a * uy + b * ux)
                    for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))])


def door_axis(footprint: BaseGeometry, x_axis: Tuple[float, float],
              host: Optional[BaseGeometry] = None) -> Axis:
    """Unit vector along the wall: the door's placement x-axis, swapped if the footprint is deeper than wide.

    A footprint deeper than wide is usually a door exported with its leaves
    open, not a rotated axis; with the host wall known (``host``: its
    footprint), the swap follows the wall's direction at the door instead.
    """
    ux, uy = x_axis
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    origin = (footprint.centroid.x, footprint.centroid.y)
    a0, a1, b0, b1 = _extents(footprint, (ux, uy), origin)
    if a1 - a0 >= b1 - b0:
        return ux, uy
    along = _wall_direction(host, origin) if host is not None else None
    if along is not None and abs(along[0] * ux + along[1] * uy) >= abs(-along[0] * uy + along[1] * ux):
        return ux, uy
    return -uy, ux


def _wall_direction(host: BaseGeometry, origin: Origin, reach: float = MAX_WALL_M) -> Optional[Axis]:
    """Direction of a wall near a point: the long side of its footprint's bounding rectangle there."""
    from shapely.geometry import Point

    local = host.intersection(Point(origin).buffer(reach))
    if local.is_empty or local.area <= 0:
        return None
    corners = list(local.minimum_rotated_rectangle.exterior.coords)
    edges = [(corners[i + 1][0] - corners[i][0], corners[i + 1][1] - corners[i][1]) for i in (0, 1)]
    dx, dy = max(edges, key=lambda e: math.hypot(*e))
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 0 else None


def side_boxes(shape: BaseGeometry, axis: Axis, depth: float = SIDE_DEPTH_M,
               margin: float = SIDE_MARGIN_M) -> Tuple[Polygon, Polygon]:
    """The two half-boxes either side of the line through ``shape`` along ``axis``: (+ side, - side)."""
    origin = (shape.centroid.x, shape.centroid.y)
    a0, a1, b0, b1 = _extents(shape, axis, origin)
    mid = (b0 + b1) / 2
    return (_box(origin, axis, a0 - margin, a1 + margin, mid, b1 + depth),
            _box(origin, axis, a0 - margin, a1 + margin, b0 - depth, mid))


def door_side_boxes(footprint: BaseGeometry, x_axis: Tuple[float, float],
                    depth: float = SIDE_DEPTH_M, margin: float = SIDE_MARGIN_M) -> Tuple[Polygon, Polygon]:
    """The two half-boxes either side of a door leaf: (+y side, -y side).

    ``x_axis`` is the door's placement x-axis, which runs along the wall. If the
    footprint is deeper than it is wide along it, the axes are swapped.
    """
    return side_boxes(footprint, door_axis(footprint, x_axis), depth, margin)


def door_passage(footprint: BaseGeometry, axis: Axis, host: Optional[BaseGeometry] = None,
                 max_wall: float = MAX_WALL_M, strip: float = WALL_STRIP_M) -> Tuple[BaseGeometry, Optional[float]]:
    """The way through the wall at a door: its footprint, widened across the wall to the wall's faces.

    A door is often thinner than its host wall (ac20: 0.10 m in 0.30 m walls),
    and rooms stop at the wall's faces, so they lie up to half the difference
    away from the door itself. The host wall's thickness is measured on a strip
    across it through the door's centre (``host``: the wall's solid footprint,
    openings not cut). Returns the passage and that thickness; with no host,
    or a measure above ``max_wall`` (the strip caught another part of the
    element), the door's footprint alone and ``None``.
    """
    if host is None or host.is_empty:
        return footprint, None
    origin = (footprint.centroid.x, footprint.centroid.y)
    a0, a1, b0, b1 = _extents(footprint, axis, origin)
    local = host.intersection(_box(origin, axis, -strip, strip, -max_wall, max_wall))
    if local.is_empty or local.area <= 0:
        return footprint, None
    _, _, w0, w1 = _extents(local, axis, origin)
    if w1 - w0 >= max_wall:
        return footprint, None
    if w0 >= b0 and w1 <= b1:
        return footprint, w1 - w0
    return footprint.union(_box(origin, axis, a0, a1, min(b0, w0), max(b1, w1))), w1 - w0


def room_sides(rooms: Mapping[str, BaseGeometry], boxes: Tuple[Polygon, Polygon],
               share: float = SIDE_SHARE, min_area: float = SIDE_MIN_AREA_M2,
               box_share: float = SIDE_BOX_SHARE) -> Sides:
    """Which side(s) of a door each room is on; a room inside both halves is on both.

    A room is on a side holding at least ``share`` of its area near the door
    and filling at least ``box_share`` of that side's box: a room that only
    touches the door's jamb (a corridor running past its end) is on neither.
    """
    out: Sides = {}
    for room, fp in rooms.items():
        areas = {side: fp.intersection(b).area for side, b in zip((1, -1), boxes)}
        total = sum(areas.values())
        out[room] = {side for (side, a), b in zip(areas.items(), boxes)
                     if a >= min_area and a >= share * total and a >= box_share * b.area}
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
    (``door_source = "geometry"``), each on one side only. Returns the pairs,
    ``is_exterior_door`` for those doors (rooms on one side only and none
    around the door), and counts.
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
            # A space on both sides surrounds the door (a toilet cubicle's
            # room, a gross-area zone, two spaces exported with one footprint):
            # it is across the leaf from nothing, but the door is inside it.
            sides = door_sides[door]
            plus = {r for r, s in sides.items() if s == {1}}
            minus = {r for r, s in sides.items() if s == {-1}}
            around = {r for r, s in sides.items() if len(s) == 2}
            if plus and minus:
                stats["unbounded_joining_rooms"] += 1
                exterior[door] = False
                pairs += [(*_pair(a, b), door, "geometry") for a in sorted(plus) for b in sorted(minus)]
            elif (plus or minus) and not around:
                stats["unbounded_exterior"] += 1
                exterior[door] = True
            elif plus or minus:
                stats["unbounded_into_surrounding_space"] += 1
                exterior[door] = False
            elif around:
                stats["unbounded_inside_one_room"] += 1
                exterior[door] = False
            else:
                stats["unbounded_no_room"] += 1
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
    scope: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rooms that really touch: through a shared wall (``wall``) or with nothing between (``open``).

    ``wall_pairs`` maps a sorted room pair to the physical elements bounding
    both. ``open_pairs`` (sorted) are joined by a virtual boundary: they are
    adjacent with ``method = "open"`` even where they also share a wall
    elsewhere or their footprints miss the thresholds. Pairs on different (or
    unknown) storeys are never adjacent. ``scope``: the rooms whose boundaries
    are recorded (default: all); touching footprints with no element between
    are open plan only between two of them -- for the others, nothing recorded
    means nothing known, and the geometry rules (``geometric_pairs``) decide.
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
                    "gap_m": round(gap, 3), "shared_len_m": round(shared, 3), "source": "boundary"})

    # Open plan: rooms whose footprints touch with no element recorded between them.
    keys = sorted(k for k in footprints if storey_of.get(k) is not None and (scope is None or k in scope))
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
                            "gap_m": round(gap, 3), "shared_len_m": round(shared, 3), "source": "boundary"})
                open_plan += (a, b) not in open_set

    # Passages the footprint rules missed: still adjacent, they share an opening.
    found = {(p["a"], p["b"]) for p in out}
    forced = 0
    for a, b in sorted(open_set - found):
        if a not in footprints or b not in footprints:
            continue
        gap = footprints[a].distance(footprints[b])
        out.append({"a": a, "b": b, "method": "open", "via_gid": None,
                    "gap_m": round(gap, 3), "shared_len_m": round(shared_length(footprints[a], footprints[b], gap), 3),
                    "source": "boundary"})
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


Blocker = Tuple[str, BaseGeometry, float, float]     # (GlobalId, footprint, z_min, z_max) of a wall-like element


def overlapping(a: BaseGeometry, b: BaseGeometry, share: float = OVERLAP_SHARE) -> bool:
    """Footprints overlapping by more than ``share`` of the smaller: one room twice, or a zone and a room in it."""
    return a.intersection(b).area > share * min(a.area, b.area)


def geometric_pairs(
    footprints: Mapping[str, BaseGeometry],
    storey_of: Mapping[str, Optional[str]],
    z_ranges: Mapping[str, Tuple[float, float]],
    blockers: Sequence[Blocker],
    include: Optional[Callable[[str, str], bool]] = None,
    max_gap: float = MAX_GAP_M,
    min_shared: float = MIN_SHARED_M,
    open_gap: float = OPEN_GAP_M,
    tol: float = EDGE_TOL_M,
    openings: Sequence[Blocker] = (),
) -> Tuple[List[Dict[str, Any]], List[Pair], Dict[str, Any]]:
    """Adjacency and open passages from footprints and wall geometry alone, for rooms without boundaries.

    Same storey, footprints at most ``max_gap`` apart along at least
    ``min_shared`` of edge (the boundary rule's thresholds, without asking for
    a recorded wall). What lies between them is read from ``blockers`` (walls,
    curtain walls, columns spanning at least ``BLOCK_Z_SHARE`` of the two
    rooms' common height) in the strip between the footprints:

    * ``open``: the footprints touch (gap <= ``open_gap``) and at least
      ``min_shared`` of the shared edge is crossed by no blocker -- an open
      passage (returned as a pair, written as ``OPEN_TO``). ``openings``
      (unfilled openings in walls: a cased opening, a lift entrance) cut
      through their wall where they are walkable: sill within ``SILL_TOL_M``
      of the rooms' common floor, head at least ``HEADROOM_M`` above it;
    * ``wall``: a blocker lies between them (``via_gid``: the one filling most
      of the strip).

    Pairs further apart than ``open_gap`` with nothing between are dropped
    (``nothing_between``): no element says what separates them. Spaces
    overlapping by more than ``OVERLAP_SHARE`` of the smaller one are the same
    room recorded twice, or a zone and a room in it, not neighbours.
    ``include(a, b)`` limits the pairs (to those the boundaries cannot decide).
    A model with no wall-like element at all (an MEP or structural file with
    spaces) cannot tell a wall from an opening: nothing is returned.
    """
    if not blockers:
        return [], [], {"no_wall_geometry": True}
    keys = sorted(k for k in footprints if storey_of.get(k) is not None)
    tree = STRtree([footprints[k] for k in keys])
    btree = STRtree([b[1] for b in blockers]) if blockers else None
    otree = STRtree([o[1] for o in openings]) if openings else None
    out: List[Dict[str, Any]] = []
    passages: List[Pair] = []
    stats: Counter = Counter()
    for a in keys:
        fa = footprints[a]
        for j in tree.query(fa.buffer(max_gap)):
            b = keys[int(j)]
            if a >= b or storey_of[a] != storey_of[b] or (include is not None and not include(a, b)):
                continue
            fb = footprints[b]
            if overlapping(fa, fb):
                stats["overlapping"] += 1
                continue
            gap = fa.distance(fb)
            if gap > max_gap:
                continue
            shared = shared_length(fa, fb, gap, tol)
            if shared < min_shared:
                stats["shared_edge_below_min"] += 1
                continue
            reach = gap + tol
            strip = fa.buffer(reach).intersection(fb.buffer(reach))
            lo = max(z_ranges[a][0], z_ranges[b][0])
            hi = min(z_ranges[a][1], z_ranges[b][1])
            between: List[Tuple[float, str, BaseGeometry]] = []
            if btree is not None and not strip.is_empty:
                for k in btree.query(strip):
                    gid, fp, z0, z1 = blockers[int(k)]
                    if min(z1, hi) - max(z0, lo) < BLOCK_Z_SHARE * max(hi - lo, 1e-6):
                        continue
                    area = fp.intersection(strip).area
                    if area > 0:
                        between.append((area, gid, fp))
            entry = {"a": a, "b": b, "gap_m": round(gap, 3), "shared_len_m": round(shared, 3),
                     "source": "geometry"}
            if gap <= open_gap:
                walls = unary_union([fp for _, _, fp in between]).buffer(tol) if between else None
                if walls is not None and otree is not None:
                    walkable = [fp for _, fp, z0, z1 in (openings[int(k)] for k in otree.query(strip))
                                if z0 <= lo + SILL_TOL_M and z1 >= lo + min(HEADROOM_M, hi - lo)]
                    if walkable:
                        walls = walls.difference(unary_union(walkable).buffer(2 * tol))
                free = min(_free_length(fa.boundary.intersection(fb.buffer(reach)), walls),
                           _free_length(fb.boundary.intersection(fa.buffer(reach)), walls))
                if free >= min_shared:
                    out.append({**entry, "method": "open", "via_gid": None})
                    passages.append((a, b))
                    stats["open"] += 1
                    continue
            if not between:
                stats["nothing_between"] += 1
                continue
            out.append({**entry, "method": "wall", "via_gid": max(between, key=lambda t: (t[0], t[1]))[1]})
            stats["wall"] += 1
    return out, passages, dict(sorted(stats.items()))


def _free_length(segment: BaseGeometry, walls: Optional[BaseGeometry]) -> float:
    """Length of a shared edge no wall crosses."""
    return segment.length if walls is None else segment.difference(walls).length


def assign_levels(
    floor_z: Mapping[str, float],
    storey_of: Mapping[str, Optional[str]],
    storey_z: Mapping[str, float],
    tol: float = LEVEL_TOL_M,
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """A storey for every space the IFC leaves without one, from the height of its floor.

    The storey whose elevation is within ``tol`` of the space's floor; failing
    that a level of its own (``level@<z>``, never written to the graph) shared
    with the other loose spaces whose floors are within ``tol`` of it -- the
    adjacency rules only ask whether two rooms are on the same floor.
    (fantasy_office_building_2 aggregates 22 spaces under its site, on three
    floors that have no storey.)
    """
    out = dict(storey_of)
    stats: Counter = Counter()
    level: Optional[Tuple[float, str]] = None
    for z, space in sorted((z, s) for s, z in floor_z.items() if out.get(s) is None):
        near = min(storey_z, key=lambda st: (abs(storey_z[st] - z), st)) if storey_z else None
        if near is not None and abs(storey_z[near] - z) <= tol:
            out[space] = near
            stats["storey_from_elevation"] += 1
            continue
        if level is None or z - level[0] > tol:
            level = (z, f"level@{z:.2f}")
        out[space] = level[1]
        stats["level_from_floor_height"] += 1
    return out, dict(stats)


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
    geometric_passages: Iterable[Pair] = (),
) -> Topology:
    """Edges (symmetric ones both ways) and properties, every name from the declared lists.

    ``passages`` come from virtual boundaries, ``geometric_passages`` from
    footprints and wall geometry (``OPEN_TO.source`` says which).
    """
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
    for a, b in geometric_passages:
        both_ways("OPEN_TO", a, b, {"source": "geometry"})
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
    """The storey above an element in its aggregation/containment chain (a space in a space in a storey)."""
    import ifcopenshell.util.element as ue

    for _ in range(8):
        parents = [p for p in (ue.get_aggregate(element), ue.get_container(element)) if p is not None]
        storey = next((p for p in parents if p.is_a("IfcBuildingStorey")), None)
        if storey is not None:
            return storey
        element = next((p for p in parents if not p.is_a("IfcBuilding") and not p.is_a("IfcSite")
                        and not p.is_a("IfcProject")), None)
        if element is None:
            return None
    return None


def _is_external(door) -> Optional[bool]:
    """The door's ``Pset_DoorCommon.IsExternal`` (occurrence or type), or ``None`` if not given."""
    import ifcopenshell.util.element as ue

    value = ue.get_pset(door, "Pset_DoorCommon", "IsExternal")
    return value if isinstance(value, bool) else None


def _ifc_host(door) -> Optional[Any]:
    """The element whose opening the door fills (IfcRelFillsElement -> IfcRelVoidsElement)."""
    for fill in getattr(door, "FillsVoids", None) or ():
        for void in getattr(fill.RelatingOpeningElement, "VoidsElements", None) or ():
            return void.RelatingBuildingElement
    return None


def _geometric_host(door: ElementGeometry, walls: Mapping[str, ElementGeometry],
                    tree: Optional[STRtree], keys: Sequence[str]) -> Optional[str]:
    """The wall-like element the door's footprint overlaps most, at the door's height."""
    if tree is None:
        return None
    height = max(door.z_max - door.z_min, 1e-6)
    best: Optional[Tuple[float, str]] = None
    for j in tree.query(door.footprint):
        gid = keys[int(j)]
        w = walls[gid]
        if min(door.z_max, w.z_max) - max(door.z_min, w.z_min) < DOOR_Z_OVERLAP * height:
            continue
        area = w.footprint.intersection(door.footprint).area
        if area > 0 and (best is None or (area, gid) > best):
            best = (area, gid)
    return best[1] if best else None


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


def _bounded_plane_line(surface, matrix, scale: float, angle_scale: float) -> Optional[LineString]:
    """Plan line of a vertical bounded plane (2nd-level boundaries): the span of its outline in plan."""
    import ifcopenshell.util.placement as up
    import numpy as np

    plane = surface.BasisSurface
    if plane is None or not plane.is_a("IfcPlane"):
        return None
    matrix = matrix @ up.get_axis2placement(plane.Position)
    if abs(float(matrix[2][2])) > 0.5:            # a floor or ceiling opening, not a separation line
        return None
    boundary = surface.OuterBoundary
    curves = [s.ParentCurve for s in boundary.Segments] if boundary.is_a("IfcCompositeCurve") else [boundary]
    points = [p for c in curves for p in (_curve_points(c, angle_scale) or [])]
    plan = [(float(w[0]) * scale, float(w[1]) * scale)
            for w in (matrix @ np.array([x, y, 0.0, 1.0]) for x, y in points)]
    if len(plan) < 2:
        return None
    a, b = max(itertools.combinations(plan, 2), key=lambda ab: math.dist(*ab))
    return LineString([a, b]) if math.dist(a, b) > 0 else None


def virtual_boundary_line(rb, scale: float, angle_scale: float) -> Optional[LineString]:
    """Plan line (metres, world) of a virtual boundary: a vertical extruded curve or bounded plane."""
    import ifcopenshell.util.placement as up
    import numpy as np

    surface = getattr(rb.ConnectionGeometry, "SurfaceOnRelatingElement", None) \
        if rb.ConnectionGeometry is not None else None
    if surface is None or rb.RelatingSpace.ObjectPlacement is None:
        return None
    if surface.is_a("IfcCurveBoundedPlane"):
        return _bounded_plane_line(surface, up.get_local_placement(rb.RelatingSpace.ObjectPlacement),
                                   scale, angle_scale)
    if not surface.is_a("IfcSurfaceOfLinearExtrusion"):
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
                     min_shared: float = MIN_SHARED_M, use_boundaries: bool = True) -> Topology:
    """Everything the enrichment writes, computed from the IFC alone (no Neo4j).

    ``use_boundaries=False`` ignores the space boundaries, as if the file had
    none: the geometry fallback then decides everything (how it is validated
    against the boundaries of the models that have them).
    """
    import ifcopenshell
    import ifcopenshell.util.placement as up
    import ifcopenshell.util.unit as uu

    start = time.monotonic()
    model = ifcopenshell.open(str(ifc_path))
    scale = uu.calculate_unit_scale(model)
    angle_scale = uu.calculate_unit_scale(model, "PLANEANGLEUNIT")
    settings = shape_settings()

    flat = [el for cls in GEOMETRY_CLASSES if cls not in VERTICAL_CLASSES for el in model.by_type(cls)]
    geometry: Dict[str, ElementGeometry] = iterate_geometry(model, flat, settings)
    for cls in VERTICAL_CLASSES:
        for el in model.by_type(cls):
            g = element_geometry(el, settings, include_parts=True)
            if g is not None:
                geometry[el.GlobalId] = g

    spaces = {s.GlobalId: s for s in model.by_type("IfcSpace")}
    doors = {d.GlobalId: d for d in model.by_type("IfcDoor")}
    storey_z = {st.GlobalId: float(up.get_local_placement(st.ObjectPlacement)[2][3]) * scale
                for st in model.by_type("IfcBuildingStorey") if st.ObjectPlacement is not None}
    storey_of = {gid: (st.GlobalId if (st := _storey(s)) is not None else None) for gid, s in spaces.items()}
    unassigned = sum(v is None for v in storey_of.values())
    storey_of, level_stats = assign_levels({gid: geometry[gid].z_min for gid in spaces if gid in geometry},
                                           storey_of, storey_z)

    element_spaces: Dict[str, Set[str]] = defaultdict(set)      # physical boundaries only
    element_class: Dict[str, str] = {}
    door_external: Dict[str, bool] = {}
    boundary_external: Dict[Pair, bool] = {}
    virtual_lines: List[Tuple[str, BaseGeometry, bool]] = []
    virtual_skipped = 0
    bounded: Set[str] = set()                                     # spaces with any boundary record
    for rb in model.by_type("IfcRelSpaceBoundary") if use_boundaries else ():
        space, el = rb.RelatingSpace, rb.RelatedBuildingElement
        if space is None or not space.is_a("IfcSpace"):
            continue
        bounded.add(space.GlobalId)
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
    # Two spaces exported with one footprint carry the same separation lines (smiley_west).
    twins = [p for p in passages if p[0] in footprints and p[1] in footprints
             and overlapping(footprints[p[0]], footprints[p[1]])]
    if twins:
        passages = [p for p in passages if p not in twins]
        passage_stats["pairs_overlapping_dropped"] = len(twins)

    # Walls, solid (openings not cut): they host doors and separate rooms. A
    # wall's shape may sit in its parts (curtain-wall plates and mullions,
    # layered walls exported as IfcBuildingElementPart: molio).
    solid = shape_settings(solid=True)
    wall_elements = [w for cls in WALL_CLASSES for w in model.by_type(cls)] if footprints else []
    parts = {w.GlobalId: [p for rel in getattr(w, "IsDecomposedBy", None) or () for p in rel.RelatedObjects]
             for w in wall_elements}
    shapes = iterate_geometry(model, wall_elements + [p for ps in parts.values() for p in ps], solid)
    walls: Dict[str, ElementGeometry] = {}
    for w in wall_elements:
        g = merge([shapes.get(w.GlobalId)] + [shapes.get(p.GlobalId) for p in parts[w.GlobalId]])
        if g is not None:
            walls[w.GlobalId] = g
    wall_keys = sorted(walls)
    wall_tree = STRtree([walls[k].footprint for k in wall_keys]) if wall_keys else None
    unfilled = [rel.RelatedOpeningElement for w in wall_elements
                for rel in getattr(w, "HasOpenings", None) or ()
                if not getattr(rel.RelatedOpeningElement, "HasFillings", None)]
    openings = iterate_geometry(model, unfilled, settings)

    # Doors: which rooms touch the way through the wall, and on which side of the leaf.
    room_keys = sorted(footprints)
    room_tree = STRtree([footprints[k] for k in room_keys])
    door_spaces = {d: set(element_spaces.get(d, ())) for d in doors}
    door_sides: Dict[str, Sides] = {}        # the rule for doors without boundaries
    bounded_sides: Dict[str, Sides] = {}     # which side each of a door's recorded rooms is on
    hosts: Counter = Counter()
    thickness: List[float] = []
    for gid, door in doors.items():
        g = geometry.get(gid)
        if g is None or door.ObjectPlacement is None:
            continue
        host = _ifc_host(door)
        host_gid = host.GlobalId if host is not None and host.GlobalId in walls else None
        hosts["ifc" if host_gid else "geometry"] += 1
        if host_gid is None:
            host_gid = _geometric_host(g, walls, wall_tree, wall_keys)
            if host_gid is None:
                hosts["geometry"] -= 1
                hosts["none"] += 1
        host_fp = walls[host_gid].footprint if host_gid else None
        matrix = up.get_local_placement(door.ObjectPlacement)
        axis = door_axis(g.footprint, (float(matrix[0][0]), float(matrix[1][0])), host_fp)
        passage, t = door_passage(g.footprint, axis, host_fp)
        if t is not None:
            thickness.append(t)
        height = max(g.z_max - g.z_min, 1e-6)
        touching = {}
        for j in room_tree.query(passage.buffer(DOOR_TOUCH_M)):
            room = room_keys[int(j)]
            rg = geometry[room]
            if min(g.z_max, rg.z_max) - max(g.z_min, rg.z_min) < DOOR_Z_OVERLAP * height:
                continue
            if footprints[room].distance(passage) <= DOOR_TOUCH_M:
                touching[room] = footprints[room]
        boxes = side_boxes(passage, axis)
        door_sides[gid] = room_sides(touching, boxes)
        if door_spaces[gid]:
            # The boundaries already say the room is on the door; only its side is asked.
            bounded_sides[gid] = room_sides(touching, boxes, box_share=0.0)

    door_check = validate_door_sides(door_spaces, door_sides)
    # No door with boundaries to check the rule against: it stands on its
    # validation against the boundary models (results in the commit history).
    use_geometry = door_check["trusted"] or not door_check["checked"]
    if not use_geometry:
        logger.warning("Door side test reproduces %d/%d doors with boundaries: not used for doors without",
                       door_check["agree"], door_check["checked"])
    door_points = {d: geometry[d].centroid for d in doors if d in geometry}
    connections, geometric_exterior, door_stats = door_space_pairs(
        door_spaces, door_points, footprints, {**door_sides, **bounded_sides}, use_geometry=use_geometry)
    # Where the model says whether a door without boundaries is external, that
    # wins over its sides: a space on the far side can be a yard or a roof
    # terrace, and nothing on the far side can be a corridor nobody modelled.
    for d in sorted(doors) if use_geometry else ():
        flag = _is_external(doors[d]) if not door_spaces[d] else None
        if flag is not None and geometric_exterior.get(d) != flag:
            geometric_exterior[d] = flag
            key = f"unbounded_exterior_{'set' if flag else 'cleared'}_by_property"
            door_stats[key] = door_stats.get(key, 0) + 1

    wall_pairs: Dict[Pair, List[str]] = defaultdict(list)
    for el, rooms in element_spaces.items():
        if element_class[el] in ("IfcDoor", "IfcWindow"):
            continue
        for a, b in itertools.combinations(sorted(rooms), 2):
            wall_pairs[(a, b)].append(el)
    adjacency, adj_stats = adjacent_pairs(wall_pairs, footprints, storey_of, max_gap, min_shared,
                                          open_pairs=passages, scope=bounded)
    # Rooms without boundaries: from footprints and wall geometry.
    geo_adjacency, geo_passages, geo_stats = geometric_pairs(
        footprints, storey_of, {gid: (geometry[gid].z_min, geometry[gid].z_max) for gid in footprints},
        [(k, walls[k].footprint, walls[k].z_min, walls[k].z_max) for k in wall_keys],
        include=lambda a, b: a not in bounded or b not in bounded, max_gap=max_gap, min_shared=min_shared,
        openings=[(k, openings[k].footprint, openings[k].z_min, openings[k].z_max) for k in sorted(openings)])

    z_ranges = {el.GlobalId: (geometry[el.GlobalId].z_min, geometry[el.GlobalId].z_max)
                for cls in VERTICAL_CLASSES for el in model.by_type(cls) if el.GlobalId in geometry}
    access = vertical_access(z_ranges, storey_z)

    exterior = exterior_doors(door_spaces, door_external, geometric_exterior)
    topo = assemble(connections, {d: doors[d].Name for d in doors}, adjacency + geo_adjacency, access, geometry,
                    exterior, boundary_external, passages, geo_passages)
    topo.stats = {
        "ifc": str(ifc_path),
        "rev": TOPOLOGY_REV,
        "params": {"max_gap_m": max_gap, "min_shared_m": min_shared, "open_gap_m": OPEN_GAP_M,
                   "storey_tol_m": STOREY_TOL_M, "door_touch_m": DOOR_TOUCH_M, "side_depth_m": SIDE_DEPTH_M,
                   "side_share": SIDE_SHARE, "side_box_share": SIDE_BOX_SHARE, "max_wall_m": MAX_WALL_M,
                   "overlap_share": OVERLAP_SHARE, "block_z_share": BLOCK_Z_SHARE, "sill_tol_m": SILL_TOL_M,
                   "headroom_m": HEADROOM_M,
                   "level_tol_m": LEVEL_TOL_M},
        "boundaries_used": use_boundaries,
        "spaces": len(spaces), "spaces_with_footprint": len(footprints),
        "spaces_with_boundaries": len(bounded),
        "spaces_without_storey": sum(v is None for v in storey_of.values()),
        "spaces_without_storey_in_ifc": unassigned,
        "storey_levels": level_stats,
        "doors": dict(sorted(door_stats.items())),
        "door_hosts": dict(sorted(hosts.items())),
        "door_wall_thickness_m": _quantiles(thickness),
        "door_side_check": {**door_check, "geometry_used": use_geometry},
        "exterior_doors": sum(exterior.values()),
        "open_passages": passage_stats,
        "adjacency": adj_stats,
        "geometry_pairs": geo_stats,
        "vertical_access": len(access),
        "edges": dict(Counter(e.rel_type for e in topo.edges)),
        "edges_by_source": _edges_by_source(topo.edges),
        "seconds": round(time.monotonic() - start, 1),
    }
    return topo


def _quantiles(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    v = sorted(values)
    return {f"p{p}": round(v[min(len(v) - 1, len(v) * p // 100)], 3) for p in (10, 50, 90)}


def _edges_by_source(edges: Sequence[Edge]) -> Dict[str, int]:
    """Undirected pair counts per relation and provenance, e.g. ``ADJACENT_TO/geometry/wall``."""
    out: Counter = Counter()
    for e in edges:
        if e.rel_type in SYMMETRIC_TYPES and e.source < e.target:
            src = e.props.get("door_source") or e.props.get("source")
            out["/".join(str(x) for x in (e.rel_type, src, e.props.get("method")) if x is not None)] += 1
    return dict(sorted(out.items()))


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
