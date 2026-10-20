"""Read-only Cypher execution for the agent.

The agent writes arbitrary Cypher, so execution is the safety boundary:

  * a READ-access session -- Neo4j rejects CREATE / SET / DELETE / MERGE;
  * a per-query timeout -- an unbounded ``[*]`` traversal is cancelled, not
    left to hang the run;
  * a row cap -- the observation shows the first ``max_rows`` rows and the
    true total, so a 5,000-row answer costs a few hundred tokens and the model
    still learns that it was 5,000.

Values are rendered as JSON-ish text with long strings clipped; nodes and
relationships (``RETURN n``) are reduced to their labels and properties.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neo4j import READ_ACCESS, Query
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node, Path, Relationship

MAX_ROWS = 50
TIMEOUT_S = 10.0
MAX_VALUE_CHARS = 200


@dataclass
class QueryResult:
    query: str
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    total_rows: int = 0
    error: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def truncated(self) -> bool:
        return self.total_rows > len(self.rows)

    def render(self) -> str:
        """The observation text the model sees."""
        if self.error:
            return f"ERROR: {self.error}"
        if not self.total_rows:
            return f"0 rows (columns: {', '.join(self.columns) or '-'})"
        lines = [f"{self.total_rows} row(s); columns: {', '.join(self.columns)}"]
        lines += [json.dumps(r, ensure_ascii=False, default=str) for r in self.rows]
        if self.truncated:
            lines.append(f"... {self.total_rows - len(self.rows)} more row(s) not shown")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "columns": self.columns, "rows": self.rows,
                "total_rows": self.total_rows, "error": self.error,
                "elapsed_s": round(self.elapsed_s, 3)}


def _plain(value: Any) -> Any:
    """Neo4j values -> JSON-friendly primitives, long strings clipped."""
    if isinstance(value, Node):
        return {"labels": sorted(value.labels), **{k: _plain(v) for k, v in value.items()}}
    if isinstance(value, Relationship):
        return {"type": value.type, **{k: _plain(v) for k, v in value.items()}}
    if isinstance(value, Path):
        return {"path": [_plain(n) for n in value.nodes]}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + "…"
    if isinstance(value, float):
        return round(value, 6)
    return value


class GraphTools:
    def __init__(self, driver, max_rows: int = MAX_ROWS, timeout_s: float = TIMEOUT_S):
        self.driver = driver
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def run_cypher(self, query: str) -> QueryResult:
        result = QueryResult(query=query)
        start = time.monotonic()
        try:
            with self.driver.session(default_access_mode=READ_ACCESS) as session:
                records = session.run(Query(query, timeout=self.timeout_s))
                result.columns = list(records.keys())
                for i, record in enumerate(records):
                    if i < self.max_rows:
                        result.rows.append([_plain(v) for v in record.values()])
                    result.total_rows = i + 1
        except Neo4jError as exc:
            result.error = f"{exc.code}: {exc.message}" if getattr(exc, "code", None) else str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as an observation
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.monotonic() - start
        return result
