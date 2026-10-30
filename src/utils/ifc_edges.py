"""Which ``IfcRelationship`` subtypes become graph edges, and how.

One declarative table, shared by the ETL loader (which materialises the edges)
and the schema scanner (which advertises them to the prompt and the grammar).
Those two used to carry separate hand-written lists -- the loader materialised
eight edge types while the scanner knew about four -- so the vocabulary
disagreed with the graph, and anything missing was undecodable under STRICT,
invisible under SOFT, and counted as a violation by SCR.

Deliberately free of any Neo4j import: ``src.constraints`` is reachable from the
Colab client, which must not drag the database driver into its process.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class EdgeSpec:
    """One materialised ``IfcRelationship`` → graph edge mapping.

    ``source_attr`` and ``target_attr`` are named in **graph direction**, which
    is not always the IFC one: ``FILLS`` is authored as
    ``IfcRelFillsElement(RelatingOpeningElement, RelatedBuildingElement)`` but is
    written ``(:IfcDoor)-[:FILLS]->(:IfcOpeningElement)``.

    Declaring these in one table rather than as eight near-identical extractor
    methods is what lets the coverage census (§ :meth:`IFCToNeo4jLoader.census_relationships`)
    be generated from the same source that drives extraction, instead of being
    hand-written and drifting.
    """

    rel_type: str
    ifc_class: str
    source_attr: str
    target_attr: str
    note: str = ""


#: Every ``IfcRoot`` → ``IfcRoot`` edge the loader materialises.
#:
#: ``IfcRelConnectsElements`` is subtype-inclusive, so ``IfcRelConnectsPathElements``
#: (wall-to-wall path connectivity) is picked up by the same spec.
EDGE_SPECS: Tuple[EdgeSpec, ...] = (
    EdgeSpec(
        "CONTAINS", "IfcRelContainedInSpatialStructure",
        "RelatingStructure", "RelatedElements",
        "spatial containment; propagated through aggregation as r.derived = true",
    ),
    EdgeSpec(
        "DECOMPOSES", "IfcRelAggregates",
        "RelatingObject", "RelatedObjects",
        "hierarchical decomposition (project/site/building/storey, assemblies)",
    ),
    EdgeSpec(
        "IS_OF_TYPE", "IfcRelDefinesByType",
        "RelatedObjects", "RelatingType",
        "instance -> its type definition, and thence to type-level psets",
    ),
    EdgeSpec(
        "HAS_OPENING", "IfcRelVoidsElement",
        "RelatingBuildingElement", "RelatedOpeningElement",
        "host element -> the opening it hosts",
    ),
    EdgeSpec(
        "FILLS", "IfcRelFillsElement",
        "RelatedBuildingElement", "RelatingOpeningElement",
        "door/window -> the opening it fills (inverted from the IFC direction)",
    ),
    EdgeSpec(
        "BOUNDED_BY", "IfcRelSpaceBoundary",
        "RelatingSpace", "RelatedBuildingElement",
        "space -> bounding element; virtual boundaries carry no element and are skipped",
    ),
    EdgeSpec(
        "CONNECTED_TO", "IfcRelConnectsElements",
        "RelatingElement", "RelatedElement",
        "physical connection, directed as authored but physically symmetric",
    ),
    EdgeSpec(
        "NESTS", "IfcRelNests",
        "RelatingObject", "RelatedObjects",
        "nesting (ports on a flow element, panels on a door type)",
    ),
    EdgeSpec(
        "ASSIGNED_TO_GROUP", "IfcRelAssignsToGroup",
        "RelatedObjects", "RelatingGroup",
        "element -> the group/system/zone it belongs to",
    ),
)

#: ``IfcRelationship`` subtypes handled outside :data:`EDGE_SPECS` because their
#: far side is not an ``IfcRoot`` and needs a synthetic node.
SYNTHETIC_EDGE_NOTES = {
    "IfcRelAssociatesMaterial": ("HAS_MATERIAL", "material association; layer sets also become :IfcMaterialLayer nodes"),
    "IfcRelAssociatesClassification": ("HAS_CLASSIFICATION", "classification assignment (Uniclass/OmniClass/…)"),
}

#: Reified as node properties rather than edges.
FLATTENED_RELATIONSHIPS = {
    "IfcRelDefinesByProperties": "flattened onto the owning product by flatten_psets",
}

#: Edges the topology enrichment derives (``src/etl/topology.py``); absent from a
#: base graph. Unlike the IFC edges above, their meaning cannot be read off the
#: pattern -- sharing a BOUNDED_BY wall looks like adjacency and is not -- so
#: the schema prompt explains whichever of them the loaded graph carries.
DERIVED_EDGE_NOTES = {
    "CONNECTED_VIA": (
        "rooms connected through a door (props door_gid, door_name). Stored in both "
        "directions: use DISTINCT or an undirected pattern when counting"
    ),
    "ADJACENT_TO": (
        "rooms that physically touch on the same storey: across a shared wall "
        "(method='wall', via_gid = that wall) or open plan (method='open'); props gap_m, "
        "shared_len_m. Stored in both directions. Two rooms bounded by the same "
        "element (BOUNDED_BY) are NOT necessarily adjacent: a corridor wall bounds rooms metres apart"
    ),
    "VERTICAL_ACCESS": "stair, ramp or lift -> each storey it serves",
}

#: Node properties the same enrichment writes.
DERIVED_PROPERTY_NOTES = {
    "geo_cx, geo_cy": "plan centroid in metres (rooms, doors, windows, stairs, ramps)",
    "geo_z_min, geo_z_max": "height range in metres",
    "geo_footprint_m2": "plan footprint area in m² (rooms: floor outline)",
    "is_exterior_door": (
        "on doors: true if the door leads outside (external boundary, or it bounds one room). "
        "True for every such door -- entrances, service doors, emergency exits alike; "
        "it does not say which one is the main entrance"
    ),
}
