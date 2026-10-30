# Authored multi-hop questions

`dental_clinic__arc.jsonl` holds 22 questions we wrote ourselves for the IFC-Bench
dental clinic architectural model (`projects/dental_clinic/arc.ifc`). They are
not part of IFC-Bench. The questions, answers and gold queries are our own text;
the facts in them (room numbers, room names, door types, wall layers, areas) are
read from the model.

## Why

The clinic's IFC-Bench train split has only two questions about relations
between rooms (Q120 "connected to 1B07", Q121 "adjacent to 1B07"), so the
topology enrichment (`src/etl/topology.py`) has almost nothing to act on. These
questions ask about rooms reachable through doors, neighbouring rooms, stairs
and the floors they serve, routes between rooms (number of doors, sequence of
spaces), and combinations with properties (door types, areas, wall layers).
They were written without looking at the clinic's test split and do not
duplicate Q120/Q121.

## Format

Same keys as `../testsets/*.jsonl` (`id`, `question`, `ground_truth`,
`category`, `split`, `project`, `ifc_model`), so `src.agent.run` can read the
file, plus:

| key | meaning |
|---|---|
| `id` | 9001 onwards, clear of IFC-Bench ids |
| `split` | always `train`; nothing here is held out |
| `source` | `authored` |
| `kind` | `door_connectivity`, `adjacency`, `vertical`, `path_route`, `exterior`, `combined` |
| `hops` | relationship traversals in the gold query (below) |
| `needs` | `base`: the base graph answers it exactly; `geometry`: it needs the topology enrichment |
| `answer` | structured answer the gold queries are checked against |
| `gold_cypher` | `{"base": query or null, "topology": query}` |
| `verification` | which independent routes were compared, and what they gave |

`ground_truth` is a full sentence answer, written the way IFC-Bench writes it,
for the LLM judge (`src/eval/judge.py`). `answer` is for machines: lists of room
numbers, door counts, routes, `[room, corridor]` pairs.

`category` follows IFC-Bench: 3 (geometric/spatial) for questions about relations
between spaces, as IFC-Bench labels Q120/Q121, and 2 for the area sum.

**Hops.** The number of relationship patterns the gold query traverses: the
`base` query when `needs` is `base`, otherwise the `topology` query. Each
`-[]-` counts once, including those inside `COUNT {}` / `EXISTS {}` filters,
and a variable-length segment counts the edges on the answer's path. In the
base graph, going through a door is two traversals (room -> door <- room); in
the enriched graph it is one `CONNECTED_VIA`. Distribution: 6 questions with 2
hops, 5 with 3, 11 with 4 or more (up to 10).

**Base vs geometry.** 12 questions are `base`: door connections through a shared
door (`BOUNDED_BY`), door types, wall layers, routes made only of doors. 10 are
`geometry`: real adjacency (a shared wall element is not adjacency; base
over-returns on all three adjacency questions), open passages between spaces
without a door (the base graph does not see them, so its routes are too long or
missing), and which floors a stair serves.

## How answers were checked

Every answer agreed on at least two independent routes before it went in:

1. **Base graph** (`dental_clinic__arc.base` snapshot): `gold_cypher.base`.
2. **Topology graph** (`dental_clinic__arc.topology` snapshot):
   `gold_cypher.topology`, which uses the enrichment edges (`CONNECTED_VIA`,
   `OPEN_TO`, `ADJACENT_TO`, `VERTICAL_ACCESS`, `is_exterior_door`, `geo_*`)
   wherever they apply. Routes walk `CONNECTED_VIA|OPEN_TO` (topology rev 2;
   rev 1 had no `OPEN_TO` and walked `CONNECTED_VIA` plus `ADJACENT_TO`
   `method: 'open'`, which caught 23 of the 56 open passages).
3. **IFC directly** (ifcopenshell + shapely, no Neo4j and no code from
   `src/etl`): door-to-room by door geometry (rooms whose footprint touches the
   door leaf, independent of `IfcRelSpaceBoundary`; it agrees for all 236
   bounded doors); open passages from the virtual space boundaries; adjacency
   by re-implementing the enrichment's rule (same storey, footprint gap at most
   0.5 m, at least 0.3 m of shared edge); stairs by overlapping the stair
   flights' footprint with the stairwell spaces and their height range with the
   storey elevations; walls from `IfcMaterialLayerSetUsage`. Routes were
   searched with doors costing 1 and open passages 0, and every route answer was
   checked to be the only route with that door count.

`needs: base` questions agree on all three routes; `needs: geometry` questions
on routes 2 and 3. The `verification` field records the specifics.

Pitfalls the questions avoid:

* the 4 doors bounding three spaces (1A10/1A16/1AC4, 1A15/1AC2/1AS1,
  1B14/1BC2/1BC3, 2C12/2CC3/2CC4), where the base graph keeps all three pairs
  and `CONNECTED_VIA` the two across the door leaf (rev 1: the two rooms
  nearest the door, which for 1A10 were 1A16/1AC4, cutting 1A10 off);
* the 18 doors with no space boundary (8 toilet-cubicle partitions and 10
  curtain-wall doors: the main entrance in 1AC1, two interior storefront doors
  inside 1DC1 and the two gates of the mechanical yard 1F01, each exported
  twice). From rev 2 the enrichment places them by geometry: the 6 door nodes
  of the entrance and the gates are `is_exterior_door`, the cubicle and
  storefront doors have one room on both sides and join nothing. None of them
  has a `BOUNDED_BY` edge, so a query reaches them only through the door node
  (the IFC check above counted 1DC1 as an exit; it is not, and no answer
  depends on it);
* derived `CONTAINS` edges (no question uses `CONTAINS`);
* adjacency close to the thresholds: every included neighbour shares at least
  0.6 m of edge, and every excluded room sharing a wall is either more than
  1.5 m away or touches only at a corner (at most 0.15 m of edge);
* duplicate names: room numbers are unique in this model; generic names
  (CORRIDOR, JAN. CL., TOILET) are always given with the number.

The geometry questions were chosen where the obvious base-graph reading gives a
different answer. For example, reading "adjoins" as "shares a wall element"
gives 12 rooms without a door into 1D16 instead of 7, and a door-only route
from 2B3 to 2D11 has 8 doors instead of 6.

## Re-running the check

```bash
docker compose up -d neo4j app
docker compose exec -T app python scripts/ifcbench/verify_authored.py
```

The script restores the base snapshot, runs every non-null `gold_cypher.base`,
restores the topology snapshot, runs every `gold_cypher.topology`, compares each
single result row with `answer` (top-level lists as sets, nested lists in order,
numbers within 0.1%), prints a table and exits non-zero on any mismatch or
query error. It leaves the base snapshot loaded (`--leave` changes that). It
wipes and restores the database, so do not run it while a run is using Neo4j.
The snapshots are built by `python -m src.agent.run ... --snapshots-only`
(`--state topology` for the enriched one) if they are missing.
