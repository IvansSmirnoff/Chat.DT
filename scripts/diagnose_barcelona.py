"""Diagnostic pass comparing Barcelona.ifc vs. the graph we just loaded."""
from collections import Counter
import ifcopenshell
from neo4j import GraphDatabase

IFC_PATH = "/app/data/Barcelona.ifc"

f = ifcopenshell.open(IFC_PATH)
print(f"=== IFC level ({f.schema}) ===")

counts_of_interest = [
    "IfcSpace", "IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow",
    "IfcOpeningElement", "IfcRamp", "IfcSlab", "IfcStair", "IfcColumn",
    "IfcBeam", "IfcBuildingStorey", "IfcBuilding", "IfcSite",
    "IfcRelSpaceBoundary", "IfcRelVoidsElement", "IfcRelFillsElement",
    "IfcRelContainedInSpatialStructure", "IfcRelAggregates",
    "IfcRelConnectsPathElements", "IfcRelAssociatesMaterial",
]
for t in counts_of_interest:
    try:
        print(f"  {t:<42} {len(f.by_type(t))}")
    except Exception:
        print(f"  {t:<42} n/a")

# --- Anomaly checks directly on IFC ---
print("\n=== Anomalies (from IFC directly) ===")

# 1. Spaces that appear in NO IfcRelSpaceBoundary (i.e., not bounded by any element)
bounded_space_ids = set()
for rel in f.by_type("IfcRelSpaceBoundary"):
    sp = getattr(rel, "RelatingSpace", None)
    if sp is not None:
        bounded_space_ids.add(sp.id())
spaces = f.by_type("IfcSpace")
unbounded_spaces = [s for s in spaces if s.id() not in bounded_space_ids]
print(f"  Spaces with no IfcRelSpaceBoundary: {len(unbounded_spaces)} / {len(spaces)}")

# 2. Doors: for each door, is it the Filling of some OpeningElement (via IfcRelFillsElement)
door_ids = {d.id() for d in f.by_type("IfcDoor")}
filled_door_ids = set()
for rel in f.by_type("IfcRelFillsElement"):
    filler = getattr(rel, "RelatedBuildingElement", None)
    if filler is not None and filler.id() in door_ids:
        filled_door_ids.add(filler.id())
orphan_doors = door_ids - filled_door_ids
print(f"  Doors not filling any IfcOpeningElement: {len(orphan_doors)} / {len(door_ids)}")

# 3. Openings: for each IfcOpeningElement, is it voiding a wall/slab (IfcRelVoidsElement)?
opening_ids = {o.id() for o in f.by_type("IfcOpeningElement")}
voiding_openings = set()
opening_host_types = Counter()
for rel in f.by_type("IfcRelVoidsElement"):
    op = getattr(rel, "RelatedOpeningElement", None)
    host = getattr(rel, "RelatingBuildingElement", None)
    if op is not None:
        voiding_openings.add(op.id())
        if host is not None:
            opening_host_types[host.is_a()] += 1
orphan_openings = opening_ids - voiding_openings
print(f"  Openings not hosted by any element: {len(orphan_openings)} / {len(opening_ids)}")
print(f"  Opening host entity distribution:     {dict(opening_host_types)}")

# 4. Ramps: check spatial containment (some storey) AND relation to slab if any
ramps = f.by_type("IfcRamp")
contained_ramp_ids = set()
for rel in f.by_type("IfcRelContainedInSpatialStructure"):
    for e in rel.RelatedElements:
        if e.is_a("IfcRamp"):
            contained_ramp_ids.add(e.id())
loose_ramps = [r for r in ramps if r.id() not in contained_ramp_ids]
print(f"  Ramps not contained in any spatial structure: {len(loose_ramps)} / {len(ramps)}")

# 5. Walls with at least one door opening
walls = f.by_type("IfcWall") + f.by_type("IfcWallStandardCase")
wall_ids = {w.id() for w in walls}
walls_with_door = set()
walls_with_opening = set()
for rel in f.by_type("IfcRelVoidsElement"):
    host = getattr(rel, "RelatingBuildingElement", None)
    op = getattr(rel, "RelatedOpeningElement", None)
    if host is None or host.id() not in wall_ids or op is None:
        continue
    walls_with_opening.add(host.id())
    # Does this opening get filled by a door?
    for fills in f.by_type("IfcRelFillsElement"):
        if getattr(fills, "RelatingOpeningElement", None) is op:
            filler = getattr(fills, "RelatedBuildingElement", None)
            if filler is not None and filler.is_a("IfcDoor"):
                walls_with_door.add(host.id())
print(f"  Walls total:             {len(wall_ids)}")
print(f"  Walls with ≥1 opening:   {len(walls_with_opening)}")
print(f"  Walls with ≥1 door:      {len(walls_with_door)}")

# --- Graph level ---
print("\n=== Graph level (Neo4j) ===")
driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))
with driver.session() as s:
    r = s.run("CALL db.labels() YIELD label RETURN label ORDER BY label").values()
    print(f"  Labels: {[x[0] for x in r]}")
    r = s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType").values()
    print(f"  Rel types: {[x[0] for x in r]}")
    r = s.run("MATCH (n) RETURN count(n)").single()
    print(f"  Total nodes: {r[0]}")
    for lbl in ["IfcSpace", "IfcWall", "IfcWallStandardCase", "IfcDoor",
                "IfcOpeningElement", "IfcRamp", "IfcBuildingStorey"]:
        r = s.run(f"MATCH (n:`{lbl}`) RETURN count(n)").single()
        print(f"  {lbl:<24} {r[0]}")
    # Are spaces reached by any relationship in the graph at all?
    r = s.run("""
        MATCH (s:IfcSpace)
        OPTIONAL MATCH (s)-[r]-()
        WITH s, count(r) AS deg
        RETURN deg, count(*) AS spaces
        ORDER BY deg
    """).values()
    print(f"  Space degree distribution in graph: {r}")

driver.close()
