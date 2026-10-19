#!/usr/bin/env python3
"""Re-score a results CSV with a projection-independent metric.

Why this exists
---------------
``EA`` compares the *values a query returns*. That penalises a prediction which
identifies exactly the right entities but projects a different column. On the
Building 15 demo, "which offices have no window" returned the identical ten
rooms as the gold query and scored 0.0, because the gold projects
``s.Numero, s.Livello`` while the model projected ``s.Name`` — and in this model
``Name`` and ``Numero`` hold the same values.

That is a test-design artefact, not a system failure, so it is reported as a
second metric rather than by loosening EA.

The rule (fixed in advance, applied uniformly — not a per-question rescue)
-------------------------------------------------------------------------
A prediction is ENTITY-CORRECT when, re-executed against the same graph:

* aggregate golds (a single row holding a single scalar): the prediction
  returns one row and that scalar appears among its values — i.e. exactly EA;
* set-valued golds: the prediction returns the same number of rows AND every
  value of **every** gold column appears somewhere in the prediction's values.

Requiring every column, not just the first, matters: an earlier draft checked
only the identifying column and therefore credited a prediction that found the
right two meeting rooms but reported the wrong floor for each — it happened to
project the identifier column, while an equally wrong prediction that projected
a different column was marked failed. Row-count equality stops the rule being
gamed by returning a wider projection or every entity of a label.

Usage:
    python scripts/adjudicate.py results/<run>_<setting>_results.csv [...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings


def _run(driver, cypher: str) -> Tuple[Optional[List[Tuple[Any, ...]]], str]:
    """Execute and return rows as tuples, or (None, error)."""
    if not cypher or not cypher.strip():
        return None, "empty query"
    try:
        with driver.session() as session:
            return [tuple(r.values()) for r in session.run(cypher)], ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc).split("\n")[0][:120]


def _flatten(rows: List[Tuple[Any, ...]]) -> Set[str]:
    return {str(v) for row in rows for v in row if v is not None}


def _is_aggregate(rows: List[Tuple[Any, ...]]) -> bool:
    return len(rows) == 1 and len(rows[0]) == 1 and isinstance(rows[0][0], (int, float))


def entity_correct(
    gold_rows: List[Tuple[Any, ...]], pred_rows: Optional[List[Tuple[Any, ...]]]
) -> bool:
    if pred_rows is None:
        return False
    if _is_aggregate(gold_rows):
        return len(pred_rows) == 1 and str(gold_rows[0][0]) in _flatten(pred_rows)
    if len(pred_rows) != len(gold_rows):
        return False
    required = _flatten(gold_rows)
    return bool(required) and required <= _flatten(pred_rows)


def adjudicate(paths: List[Path]) -> int:
    from neo4j import GraphDatabase

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password_value),
    )
    driver.verify_connectivity()

    exit_code = 0
    try:
        for path in paths:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            ea_hits = ent_hits = 0
            print("=" * 82)
            print(path.name)
            print("=" * 82)
            for r in rows:
                gold_rows, gold_err = _run(driver, r["gold_cypher"])
                pred_rows, pred_err = _run(driver, r["generated_cypher"])
                if gold_rows is None:
                    print(f"[{r['index']}] GOLD FAILED: {gold_err}")
                    exit_code = 1
                    continue

                ea = float(r.get("ea", 0.0) or 0.0) == 1.0
                ent = entity_correct(gold_rows, pred_rows)
                ea_hits += ea
                ent_hits += ent

                verdict = "OK " if ent else "BAD"
                note = "" if ea == ent else ("  <- EA misses it" if ent else "  <- EA credits it")
                print(f"[{r['index']}] EA={'1' if ea else '0'} ENTITY={'1' if ent else '0'} {verdict}"
                      f"{note}  {r['question'][:52]}")
                if not ent and pred_err:
                    print(f"      error: {pred_err}")
            n = len(rows)
            print(f"\n  EA            {ea_hits}/{n}")
            print(f"  ENTITY-CORRECT {ent_hits}/{n}\n")
    finally:
        driver.close()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="results CSV(s) to adjudicate")
    args = parser.parse_args()
    return adjudicate(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
