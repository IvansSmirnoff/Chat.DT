#!/usr/bin/env python3
"""Re-run the 43-question dental-clinic capability matrix against the live graph.

Why this exists
---------------
``data/question_groundtruth_match.csv`` records, for each of the 43 benchmark
questions, whether the graph method can answer it — YES / PARTIAL / NO — plus
the root cause when it cannot. Those verdicts were assessed on the *pre-dedup*
graph, where the three merged discipline trees triple-counted most rooms. Twelve
of the 43 reasons name the duplication explicitly, so those verdicts are stale
the moment ``scripts/dedup_graph.py`` runs.

This script re-probes every question against whatever is currently loaded and
prints the live answer beside the recorded ground truth, so the matrix can be
re-scored from evidence rather than from memory.

It is a *capability* probe, not a benchmark. The Cypher here is written by hand
to establish what the graph can express — it is not a gold standard, and it is
not what a model would be asked to produce. Authoring gold from these is a
separate, later step, and the probes deliberately return raw values (counts,
sums, names) rather than a fixed projection so that the answer is legible.

Usage:
    python scripts/probe_capability_matrix.py              # all questions
    python scripts/probe_capability_matrix.py 3 5 7 25     # only these
    python scripts/probe_capability_matrix.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase

from src.config import get_settings

CSV_PATH = Path(__file__).parent.parent / "data" / "question_groundtruth_match.csv"

# (question #, probe Cypher or None, short note on what the probe establishes)
# None means "no probe is possible" — the capability is absent by construction,
# not merely unmeasured. The note says which.
PROBES: Dict[int, Tuple[Optional[str], str]] = {
    1: (
        "MATCH (w:IfcWall)-[:HAS_MATERIAL_LAYER]->(l:IfcMaterialLayer) "
        "WHERE l.MaterialName CONTAINS 'Insulation' "
        "RETURN DISTINCT l.MaterialName AS material, l.LayerThickness AS thickness_m, "
        "count(w) AS walls",
        "layer thickness now materialised on :IfcMaterialLayer; 0.042 m = the GT's 42 mm",
    ),
    2: (
        "MATCH (w:IfcWall)-[:HAS_MATERIAL_LAYER]->(l:IfcMaterialLayer) "
        "WHERE l.MaterialName CONTAINS 'Insulat' "
        "RETURN DISTINCT l.MaterialName AS material, l.LayerThickness AS thickness_m, "
        "l.LayerSetName AS layer_set",
        "material name and layer thickness both present",
    ),
    3: (
        "MATCH (s:IfcSpace) WHERE toUpper(coalesce(s.LongName,'')) CONTAINS 'X-RAY' "
        "RETURN count(s) AS by_name, collect(s.Number) AS rooms",
        "can the '1 X-ray room' answer be reached from labels alone?",
    ),
    4: (
        "MATCH (s:IfcSpace {Number:'2A12'}) RETURN s.Unbounded_Height AS height, s.Number AS name",
        "ceiling height of a named room",
    ),
    5: (
        "MATCH (s:IfcSpace) RETURN count(DISTINCT s.OmniClass_Table_13_Category) AS types, "
        "count(s) AS spaces, count(DISTINCT s.Number) AS distinct_numbers",
        "room-type list and total space count",
    ),
    6: (
        "MATCH (d:IfcDoor) WHERE d.Width > 1.79 AND d.Width < 1.85 "
        "RETURN count(d) AS wide_doors, collect(DISTINCT d.Width)[0..3] AS widths",
        "the 1.8 m value exists; 'main entrance' is not a modelled designation",
    ),
    7: (
        "MATCH (s:IfcSpace) RETURN count(s) AS rooms, count(DISTINCT s.Number) AS distinct_numbers",
        "total room count — the headline dedup check",
    ),
    8: (
        "MATCH (s:IfcSpace) WHERE s.Number IN ['2C12','2C10'] "
        "RETURN sum(s.Area) AS total_area, collect([s.Number, s.Area]) AS rooms",
        "area sum over two named rooms",
    ),
    9: (None, "no occupancy or capacity property exists on IfcSpace"),
    10: (
        "MATCH (s:IfcSpace) WHERE s.Category_Description CONTAINS 'Storage' "
        "RETURN count(s) AS storage_rooms, round(sum(s.Area)*100)/100 AS total_area",
        "storage area total and count",
    ),
    11: (None, "no clinical / non-clinical classification is authored in the model"),
    12: (
        "MATCH (s:IfcSpace) WHERE toUpper(coalesce(s.LongName,'')) CONTAINS 'INTERACTION' "
        "RETURN count(s) AS rooms, avg(s.Unbounded_Height) AS avg_height",
        "average height over a named room class",
    ),
    13: (None, "no geometry, no route network; the fire-exit flag was junk and is now stripped"),
    14: (None, "same as Q13"),
    15: (
        "MATCH (st:IfcBuildingStorey)-[:DECOMPOSES]->(s:IfcSpace) "
        "WHERE toUpper(coalesce(s.LongName,'')) CONTAINS 'COMPUTER' "
        "RETURN s.Number AS room, s.LongName AS name, st.Name AS storey",
        "'server room' requires reinterpreting as 'computer room'",
    ),
    16: (None, "structural reinforcement for equipment is not modelled; the GT finds none either"),
    17: (
        "MATCH (c:IfcCovering) RETURN count(c) AS coverings, "
        "count(DISTINCT c.Name) AS distinct_names",
        "covering finish fields were self-named junk and are now stripped",
    ),
    18: (
        "MATCH (s:IfcSpace {Number:'1D34'})-[:BOUNDED_BY]->(w)-[:HAS_MATERIAL]->(m:IfcMaterial) "
        "RETURN DISTINCT m.Name AS material",
        "room -> bounding wall -> material, via BOUNDED_BY",
    ),
    19: (
        "MATCH (c:IfcCovering) WHERE c.Name CONTAINS 'ACT' RETURN DISTINCT c.Name AS ceiling",
        "ceiling system present; scoping it to offices needs a space->covering edge",
    ),
    20: (
        "MATCH (s:IfcSpace) WHERE toUpper(coalesce(s.LongName,'')) CONTAINS 'LAB' "
        "RETURN count(s) AS labs, round(sum(s.Area)*100)/100 AS total_area",
        "lab area total",
    ),
    21: (None, "no acoustic / STC property exists; the GT falls back to external values"),
    22: (
        "MATCH (w:IfcWall) WHERE w.IsExternal = false "
        "RETURN w.LoadBearing AS load_bearing, count(*) AS walls",
        "interior wall load-bearing flag",
    ),
    23: (
        "MATCH (w:IfcWindow) WHERE w.Window_Inset IS NOT NULL "
        "RETURN count(w) AS windows, collect(DISTINCT w.Window_Inset) AS insets",
        "glazing must be inferred from inset, exactly as the GT does",
    ),
    24: (
        "MATCH (s:IfcSpace) WHERE s.Category_Description = 'Office' "
        "RETURN count(s) AS offices, round(sum(s.Area)*100)/100 AS total_area",
        "office area total — was duplication-inflated",
    ),
    25: (
        "MATCH (st:IfcBuildingStorey {Name:'Second Floor'})-[:DECOMPOSES]->(s:IfcSpace) "
        "WHERE s.Category_Description = 'Restroom' RETURN count(s) AS toilets",
        "toilets on the second floor — was 16 raw, GT says 8",
    ),
    26: (
        "MATCH (st:IfcBuildingStorey) WHERE st.Name IN ['First Floor','Second Floor'] "
        "RETURN st.Name AS storey, st.Elevation AS elevation ORDER BY elevation",
        "floor-to-floor height from storey elevations",
    ),
    27: (
        "MATCH (w:IfcWindow {GlobalId:'0otfaO0qPDAhynjJ6DmgH8'}) "
        "RETURN w.Width AS width, w.Height AS height",
        "dimension lookup by GlobalId",
    ),
    28: (
        "MATCH (a:IfcSpace {Number:'1B07'})-[:BOUNDED_BY]->(w:IfcWall)-[:HAS_OPENING]->(o)<-[:FILLS]-(:IfcDoor) "
        "MATCH (b:IfcSpace)-[:BOUNDED_BY]->(w) WHERE b.Number <> '1B07' "
        "RETURN collect(DISTINCT b.Number) AS connected_rooms",
        "door connectivity — recorded as over-returning",
    ),
    29: (
        "MATCH (a:IfcSpace {Number:'1B07'})-[:BOUNDED_BY]->(w)<-[:BOUNDED_BY]-(b:IfcSpace) "
        "WHERE b.Number <> '1B07' RETURN collect(DISTINCT b.Number) AS adjacent_rooms",
        "shared-bounding-wall adjacency — recorded as a superset",
    ),
    30: (
        "MATCH (s:IfcSpace {Number:'2D18'}) RETURN s.Name AS name, s.Area AS area",
        "area of a named room",
    ),
    31: ("MATCH (w:IfcWindow) RETURN count(w) AS windows", "window count"),
    32: ("MATCH (d:IfcDoor) RETURN count(d) AS doors", "door count"),
    33: (
        "MATCH (s:IfcSpace) WHERE s.Category_Description = 'Scanning/Imaging Room' "
        "RETURN count(s) AS rooms, collect(s.Number) AS room_numbers",
        "scanning rooms — was 12 raw, GT says 6",
    ),
    34: (
        "MATCH (r:IfcRoof)-[:HAS_MATERIAL]->(m:IfcMaterial) RETURN DISTINCT m.Name AS material",
        "roof material present; lifespan is external knowledge",
    ),
    35: (
        "MATCH (s:IfcSpace) RETURN round(sum(s.GSA_BIM_Area)*100)/100 AS gsa_area_sum, "
        "round(sum(s.Area)*100)/100 AS area_sum",
        "gross floor area by summation — was duplication-corrupted",
    ),
    36: (
        "MATCH (st:IfcBuildingStorey)-[:DECOMPOSES]->(s:IfcSpace) "
        "WHERE st.Name IN ['First Floor','Second Floor'] "
        "RETURN round(sum(s.Area)*100)/100 AS occupied_floor_area",
        "no UFA property; closest surrogate is summed space area on occupied storeys",
    ),
    37: (
        "MATCH (st:IfcBuildingStorey)-[:DECOMPOSES]->(s:IfcSpace) "
        "RETURN st.Name AS storey, round(sum(s.Area)*100)/100 AS area ORDER BY area DESC",
        "GFA per storey — no GFA property, this is a summation surrogate",
    ),
    38: (None, "derived from Q36/Q37, both of which are surrogates rather than authored values"),
    39: (
        "MATCH (s:IfcSpace) WHERE toUpper(coalesce(s.LongName,'')) CONTAINS 'TRMT' "
        "WITH collect(s.Number) AS trmt_rooms "
        "MATCH (w:IfcWall) WHERE w.FireRating IS NOT NULL "
        "WITH trmt_rooms, w.FireRating AS rating, count(*) AS walls "
        "RETURN trmt_rooms, collect([rating, walls]) AS rated_walls",
        "we may be able to answer this where the GT could not",
    ),
    40: (
        "MATCH (c:IfcColumn)-[:IS_OF_TYPE]->(t) "
        "RETURN t.Name AS section, count(*) AS columns ORDER BY columns DESC",
        "steel column sections",
    ),
    41: (None, "narrative synthesis, not a query — the evidence is present but the answer is prose"),
    42: (
        "MATCH (f:IfcFooting) WHERE f.Material IS NOT NULL "
        "RETURN f.Material AS material, count(*) AS footings ORDER BY footings DESC",
        "foundation material",
    ),
    43: (
        "MATCH (f:IfcFooting) RETURN round(sum(f.Volume)*100)/100 AS concrete_volume, count(f) AS footings",
        "footing volume sum — recorded as mismatching the GT value",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ids", nargs="*", type=int, help="question numbers (default: all)")
    parser.add_argument("--csv", type=Path, help="also write the probe results to this CSV")
    args = parser.parse_args()

    with CSV_PATH.open() as fh:
        questions = {int(row["#"]): row for row in csv.DictReader(fh)}

    wanted = args.ids or sorted(questions)
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )

    out_rows: List[Dict[str, Any]] = []
    try:
        for qid in wanted:
            row = questions.get(qid)
            if row is None:
                continue
            cypher, note = PROBES.get(qid, (None, "no probe defined"))

            if cypher is None:
                live = "— no probe possible"
            else:
                try:
                    with driver.session() as session:
                        records = [r.data() for r in session.run(cypher)]
                    live = "∅ (no rows)" if not records else "; ".join(
                        str(r) for r in records[:4]
                    ) + (f" … +{len(records) - 4} rows" if len(records) > 4 else "")
                except Exception as exc:  # noqa: BLE001
                    live = f"ERROR: {str(exc).splitlines()[0][:110]}"

            print(f"{qid:>3} [was {row['Matches_Our_Model']:<7}] {row['Question']}")
            print(f"      GT   : {row['Ground_Truth (condensed)']}")
            print(f"      LIVE : {live}")
            print(f"      probe: {note}")
            print()

            out_rows.append(
                {
                    "#": qid,
                    "Question": row["Question"],
                    "Ground_Truth": row["Ground_Truth (condensed)"],
                    "Old_Verdict": row["Matches_Our_Model"],
                    "Live_Result": live,
                    "Probe_Note": note,
                    "Probe_Cypher": cypher or "",
                }
            )

        if args.csv:
            with args.csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
                writer.writeheader()
                writer.writerows(out_rows)
            print(f"wrote {args.csv}")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
