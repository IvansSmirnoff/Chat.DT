"""Topology enrichment: which rooms open onto each other, which touch, which stairs serve which floors.

Why this exists
---------------
The base graph has ``(:IfcSpace)-[:BOUNDED_BY]->(:IfcElement)`` but no
room-to-room relation, so "which rooms are connected to 1B07?" has to be
answered by joining through shared boundary elements -- and that join is wrong
in two different ways:

* **Connected** (through a door): two rooms bounded by the same door. Right for
  a door between two rooms; a door recorded on three spaces needs a tie-break.
* **Adjacent** (sharing a wall): two rooms bounded by the same wall is *not*
  adjacency. A long corridor wall bounds rooms metres apart. On the dental
  clinic (arc), 38% of the 1,037 same-storey room pairs that share a physical
  wall are 0.5 m or more apart; 1B07 shares walls with 8 rooms and touches 4.

So the enrichment computes the relations once, from the IFC (boundaries) and a
little geometry (room footprints, ``src/etl/geometry.py``), and writes them as
edges a query can follow in one hop:

* ``(:IfcSpace)-[:CONNECTED_VIA {door_gid, door_name}]->(:IfcSpace)`` -- both
  rooms bounded by the same door; for a door on three or more rooms, the two
  rooms nearest the door.
* ``(:IfcSpace)-[:ADJACENT_TO {method, via_gid, gap_m, shared_len_m}]->(:IfcSpace)``
  -- same storey, footprints at most ``MAX_GAP_M`` apart along at least
  ``MIN_SHARED_M`` of edge. ``method = "wall"``: they also share a physical
  boundary element (``via_gid`` is the first such wall). ``method = "open"``:
  footprints touch with no element between them (open plan).
* ``(:IfcStair|IfcRamp|IfcTransportElement)-[:VERTICAL_ACCESS]->(:IfcBuildingStorey)``
  -- every storey whose elevation lies within the element's height range.
* node properties: ``geo_cx/geo_cy`` (footprint centroid), ``geo_z_min/geo_z_max``,
  ``geo_footprint_m2`` on rooms, doors, windows and vertical-access elements;
  ``is_exterior_door`` on doors that have boundary records (an EXTERNAL
  boundary, or exactly one bounded room).
* ``BOUNDED_BY.is_external`` from the boundaries' InternalOrExternalBoundary.

The two symmetric relations are stored in **both directions**, so a directed
pattern written either way finds them; count them with an undirected pattern
or ``DISTINCT``.

Thresholds: room gaps on the clinic are bimodal (561 pairs at 0.12 m, a tail
to 0.5 m, then nothing until ~2 m), so ``MAX_GAP_M = 0.5`` sits in the valley.
Per-wall thickness was tried and rejected: the gap between neighbours is the
whole wall stack (core plus furring), while only the core bounds both rooms.

Reversible: every edge carries ``enriched = true`` and every property name is
listed below, so ``remove_topology`` returns the graph to its base fingerprint
(checked in tests). ``apply_topology`` refuses a graph that already carries
any of them, so removal can never delete base data.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from src.etl.geometry import ElementGeometry, element_geometry, shape_settings

logger = logging.getLogger(__name__)

ENRICHMENT = "topology"
SYMMETRIC_TYPES = ("CONNECTED_VIA", "ADJACENT_TO")
DERIVED_REL_TYPES = SYMMETRIC_TYPES + ("VERTICAL_ACCESS",)
NODE_PROPS = ("geo_cx", "geo_cy", "geo_z_min", "geo_z_max", "geo_footprint_m2", "is_exterior_door")
BOUNDED_BY_PROPS = ("is_external",)

MAX_GAP_M = 0.5          # footprints farther apart than this are not adjacent
MIN_SHARED_M = 0.3       # ...nor if they face each other along less edge than this
OPEN_GAP_M = 0.05        # "touching" for open-plan rooms with no element between
EDGE_TOL_M = 0.05        # slack when measuring the shared edge
STOREY_TOL_M = 0.1       # a storey counts as served if within this of the height range

VERTICAL_CLASSES = ("IfcStair", "IfcRamp", "IfcTransportElement")
GEOMETRY_CLASSES = ("IfcSpace", "IfcDoor", "IfcWindow") + VERTICAL_CLASSES

Pair = Tuple[str, str]


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

def door_space_pairs(
    door_spaces: Mapping[str, Set[str]],
    door_points: Mapping[str, Tuple[float, float]],
    footprints: Mapping[str, BaseGeometry],
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """(room, room, door) for every door bounding two rooms or more.

    A door recorded on three or more rooms connects the two nearest its centre;
    without a door position or room footprints that cannot be decided, and the
    door is skipped (counted in the returned stats).
    """
    from shapely.geometry import Point

    pairs: List[Tuple[str, str, str]] = []
    stats: Counter = Counter()
    for door, spaces in sorted(door_spaces.items()):
        rooms = sorted(spaces)
        stats[f"doors_bounding_{min(len(rooms), 3)}{'+' if len(rooms) >= 3 else ''}"] += 1
        if len(rooms) < 2:
            continue
        if len(rooms) > 2:
            point = door_points.get(door)
            if point is None or any(r not in footprints for r in rooms):
                stats["multi_room_doors_skipped"] += 1
                continue
            rooms = sorted(sorted(rooms, key=lambda r: (footprints[r].distance(Point(point)), r))[:2])
        pairs.append((rooms[0], rooms[1], door))
    return pairs, dict(stats)


def exterior_doors(door_spaces: Mapping[str, Set[str]], external: Mapping[str, bool]) -> Dict[str, bool]:
    """``is_exterior_door`` for every door with boundary records."""
    return {door: bool(external.get(door)) or len(spaces) == 1
            for door, spaces in door_spaces.items() if spaces}


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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rooms that really touch: through a shared wall (``wall``) or open plan (``open``).

    ``wall_pairs`` maps a sorted room pair to the physical elements bounding
    both. Pairs on different (or unknown) storeys are never adjacent.
    """
    out: List[Dict[str, Any]] = []
    dropped: Counter = Counter()
    gaps: List[float] = []
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
        out.append({"a": a, "b": b, "method": "wall", "via_gid": min(walls),
                    "gap_m": round(gap, 3), "shared_len_m": round(shared, 3)})

    # Open plan: rooms whose footprints touch with no element recorded between them.
    keys = sorted(k for k in footprints if storey_of.get(k) is not None)
    tree = STRtree([footprints[k] for k in keys])
    seen = {(p["a"], p["b"]) for p in out} | set(wall_pairs)
    open_count = 0
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
                open_count += 1
    stats = {
        "wall_pairs_same_storey": len(gaps),
        "wall_pairs_gap_above_max": sum(g > max_gap for g in gaps),
        "wall_pairs_over_return": round(sum(g > max_gap for g in gaps) / len(gaps), 4) if gaps else None,
        "adjacent_wall": len(out) - open_count,
        "adjacent_open": open_count,
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
    connections: Iterable[Tuple[str, str, str]],
    door_names: Mapping[str, Optional[str]],
    adjacency: Iterable[Mapping[str, Any]],
    access: Iterable[Tuple[str, str]],
    geometry: Mapping[str, ElementGeometry],
    exterior: Mapping[str, bool],
    boundary_external: Mapping[Pair, bool],
) -> Topology:
    """Edges (symmetric ones both ways) and properties, every name from the declared lists."""
    topo = Topology()
    for a, b, door in connections:
        props = {"door_gid": door, "door_name": door_names.get(door), "enriched": True}
        props = {k: v for k, v in props.items() if v is not None}
        topo.edges += [Edge("CONNECTED_VIA", a, b, dict(props)), Edge("CONNECTED_VIA", b, a, dict(props))]
    for p in adjacency:
        props = {k: v for k, v in p.items() if k not in ("a", "b") and v is not None}
        props["enriched"] = True
        topo.edges += [Edge("ADJACENT_TO", p["a"], p["b"], dict(props)),
                       Edge("ADJACENT_TO", p["b"], p["a"], dict(props))]
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


def compute_topology(ifc_path: Path, max_gap: float = MAX_GAP_M,
                     min_shared: float = MIN_SHARED_M) -> Topology:
    """Everything the enrichment writes, computed from the IFC alone (no Neo4j)."""
    import ifcopenshell
    import ifcopenshell.util.placement as up
    import ifcopenshell.util.unit as uu

    start = time.monotonic()
    model = ifcopenshell.open(str(ifc_path))
    scale = uu.calculate_unit_scale(model)
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
    for rb in model.by_type("IfcRelSpaceBoundary"):
        space, el = rb.RelatingSpace, rb.RelatedBuildingElement
        if space is None or el is None or not space.is_a("IfcSpace"):
            continue
        external = rb.InternalOrExternalBoundary == "EXTERNAL"
        key = (space.GlobalId, el.GlobalId)
        boundary_external[key] = boundary_external.get(key, False) or external
        if el.GlobalId in doors:
            door_external[el.GlobalId] = door_external.get(el.GlobalId, False) or external
        if rb.PhysicalOrVirtualBoundary == "PHYSICAL":
            element_spaces[el.GlobalId].add(space.GlobalId)
            element_class[el.GlobalId] = el.is_a()

    door_spaces = {d: set(element_spaces.get(d, ())) for d in doors}
    door_points = {d: geometry[d].centroid for d in doors if d in geometry}
    footprints = {gid: geometry[gid].footprint for gid in spaces if gid in geometry}
    connections, door_stats = door_space_pairs(door_spaces, door_points, footprints)

    wall_pairs: Dict[Pair, List[str]] = defaultdict(list)
    for el, rooms in element_spaces.items():
        if element_class[el] in ("IfcDoor", "IfcWindow"):
            continue
        for a, b in itertools.combinations(sorted(rooms), 2):
            wall_pairs[(a, b)].append(el)
    adjacency, adj_stats = adjacent_pairs(wall_pairs, footprints, storey_of, max_gap, min_shared)

    storey_z = {st.GlobalId: float(up.get_local_placement(st.ObjectPlacement)[2][3]) * scale
                for st in model.by_type("IfcBuildingStorey") if st.ObjectPlacement is not None}
    z_ranges = {el.GlobalId: (geometry[el.GlobalId].z_min, geometry[el.GlobalId].z_max)
                for cls in VERTICAL_CLASSES for el in model.by_type(cls) if el.GlobalId in geometry}
    access = vertical_access(z_ranges, storey_z)

    topo = assemble(connections, {d: doors[d].Name for d in doors}, adjacency, access, geometry,
                    exterior_doors(door_spaces, door_external), boundary_external)
    topo.stats = {
        "ifc": str(ifc_path),
        "params": {"max_gap_m": max_gap, "min_shared_m": min_shared, "open_gap_m": OPEN_GAP_M,
                   "storey_tol_m": STOREY_TOL_M},
        "spaces": len(spaces), "spaces_with_footprint": len(footprints),
        "spaces_without_storey": sum(v is None for v in storey_of.values()),
        "doors": dict(sorted(door_stats.items())),
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
    meta = {k: v for k, v in meta.items() if k not in ("cache_path", "topology_params")}
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
    _restamp_meta(driver, ENRICHMENT, {"topology_params": json.dumps(topo.stats.get("params", {}))})
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
