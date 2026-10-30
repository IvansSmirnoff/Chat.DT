"""Which label names each side of a relationship signature in the schema prompt.

Runs without Neo4j: a fake driver replays ``labels(a), labels(b)`` rows the way
``collect_relationship_signatures`` asks for them. Label counts are the dental
clinic's (arc), where every opening is an ``IfcOpeningElement`` and its two
supertypes therefore count exactly as many nodes.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List

import pytest

from src.eval.neo4j_exec import collect_relationship_signatures, most_specific_label

pytest.importorskip("ifcopenshell")

ROOT = ["IfcRoot", "IfcObjectDefinition", "IfcObject", "IfcProduct", "IfcElement"]
OPENING = ROOT + ["IfcFeatureElement", "IfcFeatureElementSubtraction", "IfcOpeningElement"]
WALL = ROOT + ["IfcBuildingElement", "IfcWall"]
WALL_SC = WALL + ["IfcWallStandardCase"]
DOOR = ROOT + ["IfcBuildingElement", "IfcDoor"]
SPACE = ["IfcRoot", "IfcObjectDefinition", "IfcObject", "IfcProduct",
         "IfcSpatialStructureElement", "IfcSpace"]
DOOR_STYLE = ["IfcRoot", "IfcObjectDefinition", "IfcTypeObject", "IfcTypeProduct", "IfcDoorStyle"]
FURNITURE_TYPE = ["IfcRoot", "IfcObjectDefinition", "IfcTypeObject", "IfcTypeProduct",
                  "IfcElementType", "IfcFurnishingElementType", "IfcFurnitureType"]
FURNISHING = ROOT + ["IfcFurnishingElement"]

COUNTS = {
    "IfcRoot": 3780, "IfcObjectDefinition": 3780, "IfcObject": 3299, "IfcProduct": 3298,
    "IfcElement": 3023, "IfcBuildingElement": 2400,
    "IfcFeatureElement": 403, "IfcFeatureElementSubtraction": 403, "IfcOpeningElement": 403,
    "IfcWall": 1080, "IfcWallStandardCase": 1065, "IfcDoor": 254,
    "IfcSpatialStructureElement": 275, "IfcSpace": 269,
    "IfcTypeObject": 481, "IfcTypeProduct": 481, "IfcElementType": 461, "IfcDoorStyle": 19,
    "IfcFurnishingElementType": 71, "IfcFurnitureType": 71, "IfcFurnishingElement": 118,
}


@pytest.mark.parametrize("order", list(permutations(OPENING[-3:])))
def test_tied_supertypes_resolve_to_the_leaf_class(order):
    """All three count 403; whichever order labels() lists them in, the leaf wins."""
    assert most_specific_label(ROOT + list(order), COUNTS) == "IfcOpeningElement"


def test_rarest_label_still_wins_without_a_tie():
    assert most_specific_label(WALL_SC, COUNTS) == "IfcWallStandardCase"
    assert most_specific_label(WALL, COUNTS) == "IfcWall"
    assert most_specific_label(SPACE, COUNTS) == "IfcSpace"


def test_every_wall_a_standard_case_names_the_standard_case():
    counts = {**COUNTS, "IfcWall": 1065}
    assert most_specific_label(WALL_SC, counts) == "IfcWallStandardCase"


def test_tie_between_names_no_schema_declares_is_deterministic():
    assert most_specific_label(["Zeta", "Alpha"], {"Zeta": 2, "Alpha": 2}) == "Alpha"
    assert most_specific_label([], COUNTS) is None


class _Session:
    def __init__(self, rows: Dict[str, List[Dict[str, Any]]]):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher: str, **params):
        if "db.relationshipTypes()" in cypher:
            return [{"relationshipType": r} for r in self._rows]
        for rtype, rows in self._rows.items():
            if f"[r:`{rtype}`]" in cypher:
                return rows
        return []


class _Driver:
    def __init__(self, rows: Dict[str, List[Dict[str, Any]]]):
        self._rows = rows

    def session(self):
        return _Session(self._rows)


def test_signatures_name_the_leaf_class_on_both_sides():
    # Supertypes listed after the leaf, as labels() did on the clinic graph.
    opening = ROOT + ["IfcFeatureElement", "IfcOpeningElement", "IfcFeatureElementSubtraction"]
    driver = _Driver({
        "FILLS": [{"la": DOOR, "lb": opening}],
        "HAS_OPENING": [{"la": WALL_SC, "lb": opening}],
        "BOUNDED_BY": [{"la": SPACE, "lb": WALL}, {"la": SPACE, "lb": WALL_SC},
                       {"la": SPACE, "lb": DOOR}],
        "IS_OF_TYPE": [{"la": DOOR, "lb": DOOR_STYLE},
                       {"la": FURNISHING, "lb": FURNITURE_TYPE}],
    })

    signatures = collect_relationship_signatures(driver, COUNTS)

    assert signatures == [
        ["IfcDoor", "FILLS", "IfcOpeningElement"],
        ["IfcDoor", "IS_OF_TYPE", "IfcDoorStyle"],
        ["IfcFurnishingElement", "IS_OF_TYPE", "IfcFurnitureType"],
        ["IfcSpace", "BOUNDED_BY", "IfcDoor"],
        ["IfcSpace", "BOUNDED_BY", "IfcWall"],
        ["IfcSpace", "BOUNDED_BY", "IfcWallStandardCase"],
        ["IfcWallStandardCase", "HAS_OPENING", "IfcOpeningElement"],
    ]
