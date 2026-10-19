"""Bind the deployment's sensor-to-room mapping to the IfcSpace nodes of the graph.

The deployment maps each IEQ sensor serial to an institutional room code (`ifc_room_id`) in its configuration
(nbll `dt/config/building_config.json`). The code is the space's own IFC coding: the building prefix
(`MIA0206` = PBSs_A_POLO + PBSs_B_SEDE + PBSs_C_CAMPUS + PBSs_I_EDIFICIO) followed by the floor
(`PBSs_III_PIANO`, three characters) and the room (`PBSs_IV_VANO`). This script materialises the mapping as one static
property per instrumented space, `SensorSerial`, matched on floor + room (never on `Numero`, which repeats across
floors). It is a static attribute of the designed building: it records whether, and by which device, a space is
instrumented; it does not bring telemetry into the graph.

    # offline: check every binding resolves to exactly one IfcSpace of a bundle's model dump
    python scripts/bind_sensors.py CONFIG --check data/bundle_b15_ids.json
    # against the running graph: write the property, then run the case-study queries
    python scripts/bind_sensors.py CONFIG --apply --query results/building15/sensor_join
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

PREFIX = "MIA0206"
PROPERTY = "SensorSerial"

COUNT_QUERY = """
UNWIND $rows AS r
OPTIONAL MATCH (s:IfcSpace) WHERE s.PBSs_III_PIANO = r.piano AND s.PBSs_IV_VANO = r.vano
RETURN r.serial AS serial, count(s) AS n
"""
APPLY_QUERY = f"""
UNWIND $rows AS r
MATCH (s:IfcSpace) WHERE s.PBSs_III_PIANO = r.piano AND s.PBSs_IV_VANO = r.vano
SET s.{PROPERTY} = r.serial
RETURN r.serial AS serial, r.room_code AS room_code, s.GlobalId AS global_id, s.Numero AS numero,
       s.Livello AS livello, s.Nome AS nome
ORDER BY room_code
"""
# The case-study question (Ch. 9): one property condition, one pattern predicate, three columns, i.e. inside the
# compiled grammar profile of the case-study deployment.
DEMO_QUERY = (f"MATCH (s:IfcSpace) WHERE s.{PROPERTY} IS NOT NULL AND NOT (s)-[:BOUNDED_BY]->(:IfcWindow) "
              f"RETURN s.Numero, s.Livello, s.{PROPERTY}")
LEVELS_QUERY = (f"MATCH (s:IfcSpace) WHERE s.{PROPERTY} IS NOT NULL "
                f"RETURN s.Livello AS livello, count(s) AS instrumented ORDER BY livello")


def parse_room_code(code: str, prefix: str = PREFIX) -> tuple[str, str]:
    """`MIA0206000017b` -> ('000', '017b')."""
    if not code or not code.startswith(prefix):
        raise ValueError(f"room code {code!r} does not start with {prefix}")
    rest = code[len(prefix):]
    piano, vano = rest[:3], rest[3:]
    if len(piano) != 3 or not vano:
        raise ValueError(f"room code {code!r} has no floor+room part")
    return piano, vano


def load_bindings(config_path: Path) -> list[dict]:
    """Rows {serial, airthings_room_id, room_code, piano, vano} from the deployment configuration."""
    rows = []
    for building in json.loads(Path(config_path).read_text()):
        for room in building.get("rooms", []):
            serial, code = room.get("sensor_id"), room.get("ifc_room_id")
            if not serial or not code:
                raise ValueError(f"incomplete room entry: {room}")
            piano, vano = parse_room_code(code)
            rows.append({"serial": str(serial), "airthings_room_id": room.get("airthings_room_id"),
                         "room_code": code, "piano": piano, "vano": vano})
    serials = [r["serial"] for r in rows]
    codes = [r["room_code"] for r in rows]
    if len(set(serials)) != len(serials) or len(set(codes)) != len(codes):
        raise ValueError("a serial or a room code occurs twice in the configuration")
    return sorted(rows, key=lambda r: r["room_code"])


def match_spaces(rows: Iterable[dict], spaces: Iterable[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Match rows to space records on (PBSs_III_PIANO, PBSs_IV_VANO). Returns (matched, unmatched, ambiguous)."""
    index: dict[tuple, list[dict]] = {}
    for s in spaces:
        index.setdefault((s.get("PBSs_III_PIANO"), s.get("PBSs_IV_VANO")), []).append(s)
    matched, unmatched, ambiguous = [], [], []
    for r in rows:
        hits = index.get((r["piano"], r["vano"]), [])
        if len(hits) == 1:
            s = hits[0]
            matched.append({**r, "global_id": s.get("GlobalId"), "numero": s.get("Numero"),
                            "livello": s.get("Livello"), "nome": s.get("Nome"),
                            "category": s.get("SBSm_CATEGORIA_DESCRIZIONE")})
        elif not hits:
            unmatched.append(r)
        else:
            ambiguous.append(r)
    return matched, unmatched, ambiguous


def spaces_from_bundle(bundle_path: Path) -> list[dict]:
    bundle = json.loads(Path(bundle_path).read_text())
    return [x for x in bundle["model_dump"] if x.get("type") == "IfcSpace"]


def write_csv(path: Path, header: list[str], records: Iterable[Iterable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(records)


def apply_and_query(rows: list[dict], query_dir: Path | None) -> list[dict]:
    from neo4j import GraphDatabase  # imported here so the offline check needs no driver

    from src.config import get_settings

    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password_value))
    try:
        with driver.session() as session:
            census = {  # the graph as loaded, before the binding adds its property
                "graph_property_keys": session.run("CALL db.propertyKeys() YIELD propertyKey "
                                                   "RETURN count(propertyKey) AS n").single()["n"],
                "graph_node_labels": session.run("CALL db.labels() YIELD label RETURN count(label) AS n").single()["n"],
                "graph_relationship_signatures": session.run(
                    "MATCH (a)-[r]->(b) WITH DISTINCT labels(a) AS la, type(r) AS t, labels(b) AS lb "
                    "RETURN count(*) AS n").single()["n"],
            }
            counts = {r["serial"]: r["n"] for r in session.run(COUNT_QUERY, rows=rows)}
            bad = {k: v for k, v in counts.items() if v != 1}
            if bad:
                raise SystemExit(f"bindings that do not resolve to exactly one IfcSpace: {bad}")
            bound = [dict(r) for r in session.run(APPLY_QUERY, rows=rows)]
            if query_dir is not None:
                demo = [list(r.values()) for r in session.run(DEMO_QUERY)]
                demo.sort(key=lambda x: (str(x[1]), str(x[0])))
                write_csv(query_dir / "rooms.csv", ["Numero", "Livello", PROPERTY], demo)
                levels = [[r["livello"], r["instrumented"]] for r in session.run(LEVELS_QUERY)]
                write_csv(query_dir / "levels.csv", ["Livello", "instrumented"], levels)
                totals = session.run("MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
                                     "RETURN nodes, count(r) AS rels").single()
                spaces = session.run("MATCH (s:IfcSpace) RETURN count(s) AS n").single()["n"]
                meta = {"query": DEMO_QUERY, "levels_query": LEVELS_QUERY, "property": PROPERTY,
                        "bound_spaces": len(bound), "graph_nodes": totals["nodes"], "graph_relationships": totals["rels"],
                        "ifc_spaces": spaces, "rows_returned": len(demo), **census}
                (query_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    finally:
        driver.close()
    return bound


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path, help="deployment building_config.json")
    ap.add_argument("--check", type=Path, metavar="BUNDLE", help="offline check against a bundle's model dump")
    ap.add_argument("--apply", action="store_true", help="write the property on the running graph")
    ap.add_argument("--query", type=Path, metavar="DIR", help="with --apply: run the case-study queries into DIR")
    ap.add_argument("--binding-out", type=Path, default=Path("data/sensor_binding_b15.json"))
    args = ap.parse_args(argv)

    rows = load_bindings(args.config)
    print(f"{len(rows)} bindings in {args.config}")
    if args.check:
        matched, unmatched, ambiguous = match_spaces(rows, spaces_from_bundle(args.check))
        print(f"offline: {len(matched)} matched, {len(unmatched)} unmatched, {len(ambiguous)} ambiguous")
        if unmatched or ambiguous:
            print("unmatched:", [r["room_code"] for r in unmatched], "ambiguous:", [r["room_code"] for r in ambiguous])
            return 1
        args.binding_out.write_text(json.dumps(matched, indent=1, ensure_ascii=False))
        print(f"wrote {args.binding_out}")
    if args.apply:
        bound = apply_and_query(rows, args.query)
        print(f"graph: {PROPERTY} set on {len(bound)} IfcSpace nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
