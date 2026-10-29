"""Plan footprints and height ranges of IFC elements, for the topology enrichment.

The loader is topology-only by design (no geometry in the graph). The
enrichment (``src/etl/topology.py``) needs a little geometry to decide which
rooms really touch: a room's **footprint** (its triangulated shape projected
onto the floor plan and merged into one polygon) and its **height range**.
Nothing else from the shape is kept.

Shapes come from ``ifcopenshell.geom`` in world coordinates, which it returns in
metres whatever the file's length unit. The ``create_shape`` result must stay
referenced while its buffers are read: ``create_shape(...).geometry`` on a
temporary dangles them (garbage face indices).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

Point3 = Tuple[float, float, float]

MIN_TRIANGLE_AREA = 1e-9  # m²; vertical faces project to slivers of zero area


@dataclass
class ElementGeometry:
    footprint: BaseGeometry          # plan polygon (may be a MultiPolygon)
    z_min: float
    z_max: float

    @property
    def centroid(self) -> Tuple[float, float]:
        c = self.footprint.centroid
        return c.x, c.y


def triangles_footprint(points: Sequence[Point3], faces: Sequence[int]) -> BaseGeometry:
    """Union of a triangle mesh projected onto the XY plane."""
    triangles = []
    for i in range(0, len(faces) - 2, 3):
        a, b, c = points[faces[i]], points[faces[i + 1]], points[faces[i + 2]]
        tri = Polygon([a[:2], b[:2], c[:2]])
        if tri.area > MIN_TRIANGLE_AREA:
            triangles.append(tri)
    return unary_union(triangles)


def mesh_geometry(points: Sequence[Point3], faces: Sequence[int]) -> Optional[ElementGeometry]:
    if not points or not faces:
        return None
    footprint = triangles_footprint(points, faces)
    if footprint.is_empty:
        return None
    zs = [p[2] for p in points]
    return ElementGeometry(footprint, min(zs), max(zs))


def merge(parts: Iterable[Optional[ElementGeometry]]) -> Optional[ElementGeometry]:
    """One geometry for an element whose shape lives in its parts (a stair's flights)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return ElementGeometry(unary_union([p.footprint for p in parts]),
                           min(p.z_min for p in parts), max(p.z_max for p in parts))


def shape_settings():
    import ifcopenshell.geom

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    return settings


def element_geometry(element, settings, include_parts: bool = False) -> Optional[ElementGeometry]:
    """Footprint and height range of one IFC element, or ``None`` if it has no shape.

    ``include_parts``: also merge the elements it aggregates (``IsDecomposedBy``),
    for containers such as ``IfcStair`` whose own representation is often empty.
    """
    import ifcopenshell.geom

    own: Optional[ElementGeometry] = None
    if getattr(element, "Representation", None) is not None:
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
        except RuntimeError:
            shape = None
        if shape is not None:
            geom = shape.geometry          # `shape` stays referenced while buffers are read
            v, f = list(geom.verts), list(geom.faces)
            points: List[Point3] = [(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v), 3)]
            own = mesh_geometry(points, f)
    if not include_parts:
        return own
    parts = [p for rel in (getattr(element, "IsDecomposedBy", None) or []) for p in rel.RelatedObjects]
    return merge([own] + [element_geometry(p, settings) for p in parts])
