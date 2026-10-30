"""Rung 0: the generic (ECPPM) instance graph -- conversion and writes.

Conversion runs on a tiny in-memory IFC, writes against a recording fake
driver, so no Neo4j is needed. The live test at the bottom loads the tiny model
into the real database and round-trips it through a snapshot; it wipes the
database (restoring what it found afterwards), so it runs only with
``NL2C_ALLOW_DB_WIPE=1``.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("ifcopenshell")

import ifcopenshell  # noqa: E402
import ifcopenshell.guid  # noqa: E402

from src.etl import generic_loader as gl  # noqa: E402
from src.etl import graph_cache as gc  # noqa: E402


def _tiny_model():
    """Project > storey > space; a wall (with a pset and a placement) and a door on the storey."""
    f = ifcopenshell.file(schema="IFC4")
    root = lambda cls, **kw: f.create_entity(cls, GlobalId=ifcopenshell.guid.new(), **kw)  # noqa: E731
    project = root("IfcProject", Name="P")
    storey = root("IfcBuildingStorey", Name="Level 1", Elevation=0.0)
    space = root("IfcSpace", Name="101", LongName="Office")
    point = f.create_entity("IfcCartesianPoint", Coordinates=(1.0, 2.0, 0.0))
    placement = f.create_entity("IfcLocalPlacement", RelativePlacement=f.create_entity(
        "IfcAxis2Placement3D", Location=point))
    wall = root("IfcWall", Name="W1", ObjectPlacement=placement)
    door = root("IfcDoor", Name="D1", OverallWidth=0.9)
    root("IfcRelAggregates", RelatingObject=project, RelatedObjects=[storey])
    root("IfcRelAggregates", RelatingObject=storey, RelatedObjects=[space])
    contained = root("IfcRelContainedInSpatialStructure", RelatingStructure=storey,
                     RelatedElements=[wall, door])
    pset = root("IfcPropertySet", Name="Pset_WallCommon", HasProperties=[
        f.create_entity("IfcPropertySingleValue", Name="FireRating",
                        NominalValue=f.create_entity("IfcLabel", "EI60")),
        f.create_entity("IfcPropertySingleValue", Name="IsExternal",
                        NominalValue=f.create_entity("IfcBoolean", True)),
    ])
    root("IfcRelDefinesByProperties", RelatedObjects=[wall], RelatingPropertyDefinition=pset)
    return f, SimpleNamespace(project=project, storey=storey, space=space, wall=wall, door=door,
                              point=point, placement=placement, contained=contained, pset=pset)


def _p21(entity):
    return f"#{entity.id()}"


@pytest.fixture
def tiny():
    model, e = _tiny_model()
    return model, e, gl.convert(model)


def _row(graph, entity):
    return next(r for rows in graph.nodes.values() for r in rows if r["p21_id"] == _p21(entity))


# -----------------------------------------------------------------------------
# Conversion (pure)
# -----------------------------------------------------------------------------

def test_every_instance_is_one_node_of_its_kind(tiny):
    model, e, graph = tiny
    kinds = {kind: sorted(r["EntityType"] for r in rows) for kind, rows in graph.nodes.items()}
    assert kinds[gl.PRIMARY] == sorted(["IfcProject", "IfcBuildingStorey", "IfcSpace", "IfcWall",
                                        "IfcDoor", "IfcPropertySet"])
    assert kinds[gl.CONNECTION] == sorted(["IfcRelAggregates", "IfcRelAggregates",
                                           "IfcRelContainedInSpatialStructure",
                                           "IfcRelDefinesByProperties"])
    assert kinds[gl.SECONDARY] == sorted(["IfcCartesianPoint", "IfcAxis2Placement3D",
                                          "IfcLocalPlacement", "IfcPropertySingleValue",
                                          "IfcPropertySingleValue"])
    assert sum(len(rows) for rows in graph.nodes.values()) == len(list(model))
    assert sorted(r["props"]["EntityType"] for r in graph.inline) == ["IfcBoolean", "IfcLabel"]


def test_node_properties_follow_their_conventions(tiny):
    _, e, graph = tiny
    wall = _row(graph, e.wall)
    assert wall["EntityType"] == "IfcWall" and wall["p21_id"] == _p21(e.wall)
    assert wall["timestamp"] == gl.TIMESTAMP == "current"
    assert wall["GlobalId"] == e.wall.GlobalId and wall["Name"] == "W1"
    assert wall["Description"] == "$" and wall["OwnerHistory"] == "$"   # None, entity-typed or not
    assert "ObjectPlacement" not in wall                                 # an entity: an edge instead
    assert not {"id", "type"} & set(wall)
    assert _row(graph, e.door)["OverallWidth"] == 0.9
    point = _row(graph, e.point)
    assert point["Coordinates"] == "(1.0, 2.0, 0.0)" and "GlobalId" not in point
    assert _row(graph, e.contained)["GlobalId"] == e.contained.GlobalId


def test_references_are_rel_rows_with_attribute_name_and_list_index(tiny):
    _, e, graph = tiny
    out = {(r["rel_type"], r["target_p21_id"], r["list_index"])
           for r in graph.relationships if r["source_p21_id"] == _p21(e.contained)}
    assert out == {("RelatingStructure", _p21(e.storey), 0),
                   ("RelatedElements", _p21(e.wall), 0),
                   ("RelatedElements", _p21(e.door), 1)}
    assert all(r["timestamp"] == "current" for r in graph.relationships)
    # The objectified relationship is the only link: no wall->storey edge, no inverses.
    assert {r["rel_type"] for r in graph.relationships
            if r["source_p21_id"] == _p21(e.wall)} == {"ObjectPlacement"}
    assert not any(r["source_p21_id"] == _p21(e.storey) for r in graph.relationships)


def test_typed_values_become_inline_nodes_under_their_owner(tiny):
    _, e, graph = tiny
    props = {p.Name: p for p in e.pset.HasProperties}
    by_owner = {r["relation"]["source_p21_id"]: r for r in graph.inline}
    label = by_owner[_p21(props["FireRating"])]
    assert label["props"] == {"EntityType": "IfcLabel", "timestamp": "current", "wrappedValue": "EI60"}
    assert label["relation"] == {"rel_type": "NominalValue", "list_index": 0,
                                 "source_p21_id": _p21(props["FireRating"])}
    assert by_owner[_p21(props["IsExternal"])]["props"]["wrappedValue"] is True


def test_psets_are_nodes_reached_through_their_relationship(tiny):
    _, e, graph = tiny
    assert gl.node_kind(e.pset) == gl.PRIMARY
    has = sorted(r["list_index"] for r in graph.relationships
                 if r["source_p21_id"] == _p21(e.pset) and r["rel_type"] == "HasProperties")
    assert has == [0, 1]


# -----------------------------------------------------------------------------
# Writes (fake driver)
# -----------------------------------------------------------------------------

class _Session:
    """Records statements; reports every row as created, minus ``short`` edges."""

    def __init__(self, short=0):
        self.calls, self.short = [], short

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        n = len(params.get("batch", []))
        nodes = n if ("CREATE (n:" in query or "CREATE (b:InlineNode" in query) else 0
        rels = max(n - self.short, 0) if "-[:rel" in query else 0
        counters = SimpleNamespace(nodes_created=nodes, relationships_created=rels)
        return SimpleNamespace(consume=lambda: SimpleNamespace(counters=counters))


def _load(monkeypatch, short=0, batch_size=2):
    wiped = []
    monkeypatch.setattr(gl, "wipe_database", lambda driver: wiped.append(driver))
    session = _Session(short)
    driver = SimpleNamespace(session=lambda: session)
    model, _ = _tiny_model()
    stats = gl.load(driver, model, batch_size=batch_size)
    return stats, session, wiped, gl.convert(model)


def test_load_writes_their_statements_in_batches(monkeypatch):
    stats, session, wiped, graph = _load(monkeypatch)
    assert len(wiped) == 1
    queries = [q for q, _ in session.calls]
    assert all(len(p.get("batch", [])) <= 2 for _, p in session.calls)
    first_rel = queries.index(gl.CREATE_RELS)
    assert queries.index(gl.INDEXES[0]) < first_rel and queries.index(gl.INDEXES[1]) < first_rel
    assert max(i for i, q in enumerate(queries) if q in gl.CREATE_NODES.values()) < first_rel
    counts = graph.counts()
    assert stats["nodes"] == {k: counts[k] for k in (*gl.NODE_KINDS, gl.INLINE)}
    assert stats["relationships"] == counts["rel"]
    sent = [r for q, p in session.calls if q == gl.CREATE_NODES[gl.PRIMARY] for r in p["batch"]]
    assert sorted(r["p21_id"] for r in sent) == sorted(r["p21_id"] for r in graph.nodes[gl.PRIMARY])


def test_load_refuses_a_graph_with_missing_edges(monkeypatch):
    with pytest.raises(RuntimeError, match="short"):
        _load(monkeypatch, short=1)


# -----------------------------------------------------------------------------
# Live (wipes the database; restores what it found)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("NL2C_ALLOW_DB_WIPE") != "1",
                    reason="wipes the live database; set NL2C_ALLOW_DB_WIPE=1")
def test_live_generic_load_answers_their_recipes_and_round_trips(neo4j_driver, tmp_path):
    saved = tmp_path / "saved.jsonl.gz"
    found = gc.export_graph(neo4j_driver, saved, gc.read_meta(neo4j_driver) or {"model_id": "unknown"})
    try:
        model, e = _tiny_model()
        stats = gl.load(neo4j_driver, model, batch_size=3)
        assert stats["relationships"] == gl.convert(model).counts()["rel"]
        with neo4j_driver.session() as s:
            on_storey = s.run(
                "MATCH (r:ConnectionNode {EntityType:'IfcRelContainedInSpatialStructure'})"
                "-[:rel {rel_type:'RelatingStructure'}]->(s:PrimaryNode), "
                "(r)-[:rel {rel_type:'RelatedElements'}]->(x:PrimaryNode) "
                "RETURN s.Name AS s, x.EntityType AS t ORDER BY t").data()
            fire = s.run(
                "MATCH (w:PrimaryNode {EntityType:'IfcWall'})<-[:rel {rel_type:'RelatedObjects'}]-"
                "(:ConnectionNode {EntityType:'IfcRelDefinesByProperties'})"
                "-[:rel {rel_type:'RelatingPropertyDefinition'}]->"
                "(:PrimaryNode {EntityType:'IfcPropertySet'})"
                "-[:rel {rel_type:'HasProperties'}]->(p {Name:'FireRating'})"
                "-[:rel {rel_type:'NominalValue'}]->(v:InlineNode) RETURN v.wrappedValue AS v").single()
        assert on_storey == [{"s": "Level 1", "t": "IfcDoor"}, {"s": "Level 1", "t": "IfcWall"}]
        assert fire["v"] == "EI60"

        snap = tmp_path / "tiny.generic.jsonl.gz"
        header = gc.export_with_meta(neo4j_driver, snap, "tiny", gl.STATE, loader_sources=gl.SOURCES)
        gc.restore_graph(neo4j_driver, snap, batch_size=5)
        assert gc.fingerprint(gc.graph_summary(neo4j_driver)) == header["fingerprint"]
        assert gc.read_meta(neo4j_driver)["enrichment"] == gl.STATE
    finally:
        gc.restore_graph(neo4j_driver, saved)
    assert gc.fingerprint(gc.graph_summary(neo4j_driver)) == found["fingerprint"]
