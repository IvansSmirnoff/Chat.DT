#!/usr/bin/env python3
"""Collapse the duplication a federated IFC merge leaves in the graph.

Why this exists
---------------
``data/merged_optimized.ifc`` is not one model. It is three discipline models
(architecture, structure, MEP) merged into one file, and the merge kept all
three spatial structures side by side:

    SITE 3eM8...  BUILDING 3eM8...   ARC   TOF Footing / First Floor / Second Floor / Roof - Main
    SITE 050A...  BUILDING 050A...   STR   TOF Footing / First Floor / Second Floor / Roof - Main
    SITE 1VB9...  BUILDING 1VB9...   MEP   TOF Footing / Level 1     / Level 2      / Roof - Main

Three sites, three buildings, twelve storeys at four distinct elevations, and —
the part that breaks every count — **two independent sets of rooms**. ARC holds
269 spaces; MEP holds 526, of which 264 repeat an ARC room number verbatim, 259
repeat one with a ``-M`` suffix, and 3 repeat one with different punctuation
(``2D04-A`` for ARC's ``2D04A``, ``E1M`` for ``E1``). So "how many rooms are
there" answers 795 against a ground truth of 269, and every area or category
aggregate double-counts.

None of that is a fault of the query method, and none of it is visible from the
question. It has to be fixed in the graph, once, and declared.

The passes
----------
Each pass is separately selectable and separately reported, because they are
different claims and only some of them generalise:

``orphans``     Delete IfcPropertyDefinition nodes with no relationships.
                *Generic* — a loader defect, since fixed at source
                (``_get_ifc_root_elements``). This pass exists to repair graphs
                loaded before that fix. 135,569 nodes here, 82.5% of the graph.

``discipline``  Tag every node with the spatial tree it came from
                (``n.discipline`` = ARC / STR / MEP). *Generic* for federated
                models. Non-destructive, and a prerequisite for the rest.

``spaces``      Keep the ARC rooms as canonical, re-point MEP element
                containment onto them, delete the MEP duplicates.
                *Model-specific* — the rule "architecture owns the rooms" is a
                judgement about this file, and must be stated as one.

``storeys``     Merge the three storey systems by elevation, keeping the ARC
                name and recording the others in ``NameAliases``. Collapse the
                three sites and buildings into one. *Model-specific* for the
                same reason.

``props``       Drop ``Foo_1`` where it is identical to ``Foo`` on every node.
                The suffix is the loader's collision marker for two psets
                writing the same property name; when the values agree it is
                pure noise. Where they disagree the property is kept — it is a
                real second property, and dropping it would lose data.

``junk``        Drop properties whose value equals their own key on every node
                (``FloorCovering = 'FloorCovering'``) — unfilled COBie fields
                that Revit exports as their own label. 760 (label, property)
                pairs here. They are decodable, they pass SCR, they reach the
                prompt, and they can never answer anything.

Dry-run is the default. Nothing is written without ``--apply``.

Usage:
    python scripts/dedup_graph.py                      # report only
    python scripts/dedup_graph.py --apply              # run every pass
    python scripts/dedup_graph.py --apply --pass spaces --pass storeys
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import Driver, GraphDatabase

from src.config import get_settings

logger = logging.getLogger("dedup")

ALL_PASSES = ("orphans", "discipline", "spaces", "storeys", "props", "junk")

# Labels that carry no discriminating information when reporting which leaf
# class a property belongs to — every product has them.
GENERIC_LABELS = {
    "IfcRoot",
    "IfcObject",
    "IfcObjectDefinition",
    "IfcProduct",
    "IfcElement",
    "IfcBuildingElement",
    "IfcPropertyDefinition",
    "IfcPropertySetDefinition",
    "IfcDistributionElement",
    "IfcDistributionFlowElement",
    "IfcSpatialStructureElement",
    "IfcTypeObject",
    "IfcTypeProduct",
    "IfcElementType",
    "IfcFeatureElement",
    "IfcFeatureElementSubtraction",
}

# A discipline is guessed from what its tree actually contains, not from a name
# — the merged file names all three sites "Default" and leaves every building
# name null, so there is nothing to read.
DISCIPLINE_MARKERS = (
    ("MEP", ("IfcFlowSegment", "IfcFlowFitting", "IfcFlowTerminal")),
    ("STR", ("IfcBeam", "IfcColumn", "IfcFooting")),
    ("ARC", ("IfcSpace", "IfcDoor", "IfcWindow", "IfcCovering")),
)


# =============================================================================
# Helpers
# =============================================================================


def normalise_room_number(number: Optional[str]) -> Optional[str]:
    """Room number reduced to a form that survives the merge's punctuation.

    Upper-cased, with hyphens and spaces removed. ``'2D04-A'`` and ``'2D04A'``
    are the same room; ``'2C10-M'`` becomes ``'2C10M'``, which the mirror rule
    below then strips down to ``'2C10'``.
    """
    if not number:
        return None
    return re.sub(r"[\s\-]", "", str(number)).upper()


def mirror_base(normalised: str) -> Optional[str]:
    """The room a ``-M`` mirror mirrors, or None if this is not a mirror.

    Applied only after an exact match has already failed, so a room genuinely
    named ``...M`` is never mangled into a different room that exists.
    """
    if len(normalised) > 1 and normalised.endswith("M"):
        return normalised[:-1]
    return None


def leaf_label(labels: List[str]) -> Optional[str]:
    """The most specific IFC class among a node's label chain."""
    specific = [x for x in labels if x not in GENERIC_LABELS]
    return specific[0] if specific else None


class Runner:
    """Executes reads always, writes only when ``apply`` is set."""

    def __init__(self, driver: Driver, apply: bool) -> None:
        self.driver = driver
        self.apply = apply
        self.changes: List[str] = []

    def read(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, **params)]

    def write(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        if not self.apply:
            return []
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, **params)]

    def record(self, message: str) -> None:
        self.changes.append(message)
        logger.info(("APPLIED  " if self.apply else "would    ") + message)


# =============================================================================
# Pass 1 — orphan property-definition nodes
# =============================================================================


def pass_orphans(run: Runner) -> None:
    logger.info("--- pass: orphans ---")
    total = run.read(
        "MATCH (n:IfcPropertyDefinition) WHERE NOT (n)--() RETURN count(n) AS c"
    )[0]["c"]
    if total == 0:
        logger.info("no orphan IfcPropertyDefinition nodes — loader fix is in effect")
        return

    by_label: Dict[str, int] = {}
    for row in run.read(
        """MATCH (n:IfcPropertyDefinition) WHERE NOT (n)--()
           UNWIND labels(n) AS l WITH l, count(*) AS c
           WHERE NOT l IN ['IfcRoot', 'IfcPropertyDefinition', 'IfcPropertySetDefinition']
           RETURN l, c ORDER BY c DESC"""
    ):
        by_label[row["l"]] = row["c"]

    run.record(f"delete {total} orphan IfcPropertyDefinition nodes {by_label}")
    run.write(
        """MATCH (n:IfcPropertyDefinition) WHERE NOT (n)--()
           CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"""
    )


# =============================================================================
# Pass 2 — discipline tagging
# =============================================================================


def _tree_disciplines(run: Runner) -> Dict[str, str]:
    """Map building GlobalId -> discipline, inferred from tree contents."""
    counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in run.read(
        """MATCH (b:IfcBuilding)-[:DECOMPOSES]->(st:IfcBuildingStorey)
           OPTIONAL MATCH (st)-[:DECOMPOSES|CONTAINS]->(x)
           UNWIND labels(x) AS l
           RETURN b.GlobalId AS bld, l, count(*) AS c"""
    ):
        counts[row["bld"]][row["l"]] = row["c"]

    disciplines: Dict[str, str] = {}
    for bld, label_counts in counts.items():
        best, best_score = "UNK", 0
        for name, markers in DISCIPLINE_MARKERS:
            score = sum(label_counts.get(m, 0) for m in markers)
            if score > best_score:
                best, best_score = name, score
        disciplines[bld] = best
    return disciplines


def pass_discipline(run: Runner) -> Dict[str, str]:
    logger.info("--- pass: discipline ---")
    # Not idempotent by nature: the inference reads what each tree contains, and
    # once `spaces`/`storeys` have merged the trees there is only one tree left,
    # holding everything. Re-running would relabel the whole graph with whichever
    # discipline has the most elements (MEP, by a wide margin). Refuse instead.
    already = run.read("MATCH (n) WHERE n.discipline IS NOT NULL RETURN count(n) AS c")[0]["c"]
    if already:
        logger.info(
            "%d nodes already carry `discipline` — this graph has been tagged. "
            "Skipping (re-run the ETL for a clean pass).",
            already,
        )
        return _tree_disciplines(run)

    disciplines = _tree_disciplines(run)
    for bld, disc in sorted(disciplines.items(), key=lambda kv: kv[1]):
        n = run.read(
            """MATCH (b:IfcBuilding {GlobalId: $bld})-[:DECOMPOSES]->(st:IfcBuildingStorey)
               OPTIONAL MATCH (st)-[:DECOMPOSES|CONTAINS]->(x)
               RETURN count(DISTINCT st) + count(DISTINCT x) AS c""",
            bld=bld,
        )[0]["c"]
        run.record(f"tag {n} nodes under building {bld} as discipline={disc}")
        run.write(
            """MATCH (b:IfcBuilding {GlobalId: $bld})
               SET b.discipline = $disc
               WITH b
               OPTIONAL MATCH (site:IfcSite)-[:DECOMPOSES]->(b)
               FOREACH (_ IN CASE WHEN site IS NULL THEN [] ELSE [1] END |
                   SET site.discipline = $disc)
               WITH b
               MATCH (b)-[:DECOMPOSES]->(st:IfcBuildingStorey)
               SET st.discipline = $disc
               WITH DISTINCT st
               OPTIONAL MATCH (st)-[:DECOMPOSES|CONTAINS]->(x)
               SET x.discipline = $disc""",
            bld=bld,
            disc=disc,
        )
    return disciplines


# =============================================================================
# Pass 3 — space deduplication
# =============================================================================


def pass_spaces(run: Runner, disciplines: Dict[str, str]) -> None:
    logger.info("--- pass: spaces ---")
    arc = [b for b, d in disciplines.items() if d == "ARC"]
    if not arc:
        logger.warning("no ARC tree found — skipping space dedup")
        return
    arc_bld = arc[0]

    rows = run.read(
        """MATCH (b:IfcBuilding)-[:DECOMPOSES]->(:IfcBuildingStorey)-[:DECOMPOSES]->(s:IfcSpace)
           RETURN b.GlobalId AS bld, s.GlobalId AS gid, s.Number AS number"""
    )
    canonical: Dict[str, str] = {}
    others: List[Tuple[str, Optional[str], str]] = []
    for row in rows:
        key = normalise_room_number(row["number"])
        if row["bld"] == arc_bld:
            if key:
                canonical[key] = row["gid"]
        else:
            others.append((row["gid"], key, row["bld"]))

    exact: List[Tuple[str, str]] = []
    mirrored: List[Tuple[str, str]] = []
    unmatched: List[Tuple[str, Optional[str]]] = []
    for gid, key, _bld in others:
        if key and key in canonical:
            exact.append((gid, canonical[key]))
            continue
        base = mirror_base(key) if key else None
        if base and base in canonical:
            mirrored.append((gid, canonical[base]))
        else:
            unmatched.append((gid, key))

    # Second round: a mirror of a room the ARC tree simply does not have still
    # folds onto its own base, which by now is a surviving node. Without this
    # the graph keeps both '1ES2' and '1ES2-M' — the mirror survives only
    # because its original is missing from architecture, which is backwards.
    survivors = {key: gid for gid, key in unmatched if key and not mirror_base(key) in canonical}
    still_unmatched: List[Tuple[str, Optional[str]]] = []
    for gid, key in unmatched:
        base = mirror_base(key) if key else None
        if base and base in survivors and survivors[base] != gid:
            mirrored.append((gid, survivors[base]))
        else:
            still_unmatched.append((gid, key))
    unmatched = [(g, k) for g, k in still_unmatched if g not in {m[0] for m in mirrored}]

    logger.info(
        "canonical ARC rooms: %d | duplicates: %d exact + %d mirrored (-M) | unmatched: %d",
        len(canonical),
        len(exact),
        len(mirrored),
        len(unmatched),
    )
    if unmatched:
        numbers = run.read(
            "MATCH (s:IfcSpace) WHERE s.GlobalId IN $gids RETURN collect(s.Number) AS n",
            gids=[g for g, _ in unmatched],
        )[0]["n"]
        logger.info(
            "  kept, present in this tree only (not a duplicate of any ARC room): %s",
            sorted(map(str, numbers)),
        )

    pairs = [{"dup": d, "keep": k} for d, k in exact + mirrored]
    if not pairs:
        return

    moved = run.read(
        """UNWIND $pairs AS p
           MATCH (dup:IfcSpace {GlobalId: p.dup})-[c:CONTAINS]->(e)
           RETURN count(*) AS c""",
        pairs=pairs,
    )[0]["c"]
    run.record(
        f"re-point {moved} CONTAINS edges from {len(pairs)} duplicate spaces "
        f"onto their canonical ARC room"
    )
    run.write(
        """UNWIND $pairs AS p
           MATCH (dup:IfcSpace {GlobalId: p.dup})-[c:CONTAINS]->(e)
           MATCH (keep:IfcSpace {GlobalId: p.keep})
           MERGE (keep)-[n:CONTAINS]->(e)
           SET n.derived = c.derived, n.repointed_from_merge = true""",
        pairs=pairs,
    )

    # Anything else pointing at a duplicate would be silently lost on delete.
    residual = run.read(
        """UNWIND $pairs AS p
           MATCH (dup:IfcSpace {GlobalId: p.dup})-[r]-()
           WHERE NOT (type(r) = 'CONTAINS' AND startNode(r) = dup)
             AND NOT type(r) = 'DECOMPOSES'
           RETURN type(r) AS t, count(*) AS c ORDER BY c DESC""",
        pairs=pairs,
    )
    if residual:
        logger.warning("duplicate spaces also carry: %s — review before deleting", residual)

    run.record(f"delete {len(pairs)} duplicate IfcSpace nodes")
    run.write(
        """UNWIND $pairs AS p
           MATCH (dup:IfcSpace {GlobalId: p.dup})
           DETACH DELETE dup""",
        pairs=pairs,
    )


# =============================================================================
# Pass 4 — storey / site / building unification
# =============================================================================


def pass_storeys(run: Runner, disciplines: Dict[str, str]) -> None:
    logger.info("--- pass: storeys ---")
    arc = [b for b, d in disciplines.items() if d == "ARC"]
    if not arc:
        logger.warning("no ARC tree found — skipping storey merge")
        return
    arc_bld = arc[0]

    rows = run.read(
        """MATCH (b:IfcBuilding)-[:DECOMPOSES]->(st:IfcBuildingStorey)
           RETURN b.GlobalId AS bld, st.GlobalId AS gid, st.Name AS name,
                  st.Elevation AS elev"""
    )
    # Elevations differ in the 13th decimal between exports; 3 dp is far below
    # any real storey separation and far above that noise.
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[round(float(row["elev"] or 0.0), 3)].append(row)

    for elev, members in sorted(groups.items()):
        keep = next((m for m in members if m["bld"] == arc_bld), members[0])
        drop = [m for m in members if m["gid"] != keep["gid"]]
        if not drop:
            continue
        aliases = sorted({str(m["name"]) for m in drop if m["name"]})
        run.record(
            f"merge {len(drop)} storey(s) at elevation {elev} into "
            f"{keep['name']!r} (aliases {aliases})"
        )
        run.write(
            """MATCH (keep:IfcBuildingStorey {GlobalId: $keep})
               SET keep.NameAliases = $aliases
               WITH keep
               UNWIND $drop AS dgid
               MATCH (d:IfcBuildingStorey {GlobalId: dgid})
               OPTIONAL MATCH (d)-[c:CONTAINS]->(e)
               FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
                   MERGE (keep)-[n:CONTAINS]->(e)
                   SET n.derived = c.derived, n.repointed_from_merge = true)
               WITH DISTINCT keep, d
               OPTIONAL MATCH (d)-[:DECOMPOSES]->(s)
               FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                   MERGE (keep)-[:DECOMPOSES]->(s))
               WITH DISTINCT d
               DETACH DELETE d""",
            keep=keep["gid"],
            drop=[m["gid"] for m in drop],
            aliases=aliases,
        )

    # One site, one building. Storeys have already been merged onto the ARC
    # building's, so the others are empty shells. Drive this off the inferred
    # discipline map, not off the stored `discipline` property — in a dry run
    # nothing has been tagged yet, and reading the property would count the ARC
    # tree as deletable too.
    drop_buildings = [b for b, d in disciplines.items() if d != "ARC"]
    # Resolve the sites first: after the buildings go, the edge that identifies
    # them is gone too.
    drop_sites = [
        row["gid"]
        for row in run.read(
            """MATCH (s:IfcSite)-[:DECOMPOSES]->(b:IfcBuilding)
               WHERE b.GlobalId IN $gids RETURN DISTINCT s.GlobalId AS gid""",
            gids=drop_buildings,
        )
    ]
    if drop_buildings:
        run.record(
            f"delete {len(drop_buildings)} non-ARC IfcBuilding node(s) left empty by the merge"
        )
        run.write(
            "MATCH (b:IfcBuilding) WHERE b.GlobalId IN $gids DETACH DELETE b",
            gids=drop_buildings,
        )

    if drop_sites:
        run.record(f"delete {len(drop_sites)} non-ARC IfcSite node(s) left empty by the merge")
        run.write(
            "MATCH (s:IfcSite) WHERE s.GlobalId IN $gids DETACH DELETE s",
            gids=drop_sites,
        )


# =============================================================================
# Pass 5 / 6 — property-level cleanup
# =============================================================================


def _label_properties(run: Runner) -> Dict[str, Set[str]]:
    by_label: Dict[str, Set[str]] = defaultdict(set)
    for row in run.read(
        """CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName
           RETURN nodeLabels AS labels, propertyName AS prop"""
    ):
        leaf = leaf_label(row["labels"])
        if leaf:
            by_label[leaf].add(row["prop"])
    return by_label


def pass_props(run: Runner) -> None:
    logger.info("--- pass: props ---")
    by_label = _label_properties(run)
    redundant: List[Tuple[str, str, str, int]] = []
    kept: List[Tuple[str, str, str]] = []

    for label, props in sorted(by_label.items()):
        for prop in sorted(props):
            match = re.match(r"^(.*)_(\d+)$", prop)
            if not match or match.group(1) not in props:
                continue
            base = match.group(1)
            row = run.read(
                # Direct `=`, not toString(): list-valued properties (NameAliases,
                # written by the storey merge) make toString() raise a TypeError,
                # and Cypher equality already handles every type we store.
                f"""MATCH (n:`{label}`)
                    WHERE n.`{base}` IS NOT NULL AND n.`{prop}` IS NOT NULL
                    RETURN count(*) AS both,
                           sum(CASE WHEN n.`{base}` = n.`{prop}` THEN 1 ELSE 0 END) AS eq"""
            )[0]
            if not row["both"]:
                continue
            if row["eq"] == row["both"]:
                redundant.append((label, base, prop, row["both"]))
            else:
                kept.append((label, base, prop))

    logger.info(
        "collision-suffixed keys: %d redundant (identical to base), %d genuinely distinct (kept)",
        len(redundant),
        len(kept),
    )
    if kept:
        logger.info(
            "  kept — a name that genuinely ends in a number (Revit authors "
            "'Angle 1', 'Border 1 Type') and holds different data from its base. "
            "These are authored names, not loader artifacts: since the property "
            "provenance fix the ETL no longer resolves collisions with a '_N' "
            "suffix, it writes <PropertySet>_<Property> instead."
        )
        for label, base, prop in kept[:10]:
            logger.info("    %s.%s vs .%s", label, base, prop)

    for label, base, prop, n in redundant:
        run.record(f"remove {label}.{prop} (identical to .{base} on all {n} nodes)")
        run.write(f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL REMOVE n.`{prop}`")

    _report_qualified_duplicates(run, by_label)


def _report_qualified_duplicates(run: Runner, by_label: Dict[str, Set[str]]) -> None:
    """Report `<PropertySet>_<Property>` keys that duplicate their bare alias.

    Reported, never removed. The qualified key is the *only* record of which
    property set a value came from -- the bare alias is whichever pset the
    loader saw first -- so deleting the duplicate would undo exactly the
    provenance the ETL change added. This is a size figure for the paper, not a
    cleanup target.
    """
    duplicates = 0
    distinct = 0
    for label, props in sorted(by_label.items()):
        for prop in sorted(props):
            base = next(
                (b for b in props if b != prop and prop.endswith("_" + b)), None
            )
            if base is None:
                continue
            row = run.read(
                f"""MATCH (n:`{label}`)
                    WHERE n.`{base}` IS NOT NULL AND n.`{prop}` IS NOT NULL
                    RETURN count(*) AS both,
                           sum(CASE WHEN n.`{base}` = n.`{prop}` THEN 1 ELSE 0 END) AS eq"""
            )[0]
            if not row["both"]:
                continue
            if row["eq"] == row["both"]:
                duplicates += 1
            else:
                distinct += 1

    logger.info(
        "pset-qualified keys: %d duplicate their bare alias (the first writer), "
        "%d carry a different value (a second pset that would previously have "
        "been renamed to '_N' or lost). Neither is removed — the qualified name "
        "is the provenance record.",
        duplicates,
        distinct,
    )


def pass_junk(run: Runner) -> None:
    logger.info("--- pass: junk ---")
    by_label = _label_properties(run)
    junk: List[Tuple[str, str, int]] = []

    for label, props in sorted(by_label.items()):
        for prop in sorted(props):
            row = run.read(
                f"""MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL
                    RETURN count(*) AS c,
                           sum(CASE WHEN n.`{prop}` = $prop THEN 1 ELSE 0 END) AS eq""",
                prop=prop,
            )[0]
            if row["c"] and row["eq"] == row["c"]:
                junk.append((label, prop, row["c"]))

    affected = sorted({p for _, p, _ in junk})
    logger.info(
        "%d (label, property) pairs carry their own key as their value, over %d distinct keys",
        len(junk),
        len(affected),
    )
    run.record(f"remove {len(junk)} self-named (label, property) pairs")
    for label, prop, _n in junk:
        run.write(f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL REMOVE n.`{prop}`")


# =============================================================================
# Reporting
# =============================================================================


def report(run: Runner, title: str) -> Dict[str, int]:
    stats = {
        "nodes": run.read("MATCH (n) RETURN count(n) AS c")[0]["c"],
        "rels": run.read("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"],
        "labels": len(run.read("CALL db.labels()")),
        # db.propertyKeys() lists the token store, which never shrinks — removing
        # a property leaves its key token behind, so that count goes *up* after a
        # cleanup pass. nodeTypeProperties() reports keys actually in use.
        "liveKeys": len(
            run.read(
                """CALL db.schema.nodeTypeProperties() YIELD propertyName
                   RETURN DISTINCT propertyName"""
            )
        ),
        "spaces": run.read("MATCH (s:IfcSpace) RETURN count(s) AS c")[0]["c"],
        "storeys": run.read("MATCH (s:IfcBuildingStorey) RETURN count(s) AS c")[0]["c"],
        "buildings": run.read("MATCH (b:IfcBuilding) RETURN count(b) AS c")[0]["c"],
        "sites": run.read("MATCH (s:IfcSite) RETURN count(s) AS c")[0]["c"],
    }
    logger.info("%s: %s", title, stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--apply", action="store_true", help="write changes (default is dry-run)"
    )
    parser.add_argument(
        "--pass",
        dest="passes",
        action="append",
        choices=ALL_PASSES,
        help="run only this pass (repeatable); default is all passes in order",
    )
    args = parser.parse_args()
    passes = args.passes or list(ALL_PASSES)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    run = Runner(driver, apply=args.apply)

    try:
        before = report(run, "before")
        disciplines: Dict[str, str] = {}

        if "orphans" in passes:
            pass_orphans(run)
        if "discipline" in passes or "spaces" in passes or "storeys" in passes:
            disciplines = pass_discipline(run) if "discipline" in passes else _tree_disciplines(run)
        if "spaces" in passes:
            pass_spaces(run, disciplines)
        if "storeys" in passes:
            pass_storeys(run, disciplines)
        if "props" in passes:
            pass_props(run)
        if "junk" in passes:
            pass_junk(run)

        after = report(run, "after ")
        logger.info("")
        if args.apply:
            for key in before:
                if before[key] != after[key]:
                    logger.info("  %-13s %8d -> %8d", key, before[key], after[key])
        else:
            logger.info("DRY RUN — %d change(s) planned, nothing written.", len(run.changes))
            logger.info("Re-run with --apply to execute.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
