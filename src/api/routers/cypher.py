"""Cypher execution and validation endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from src.api.auth import require_bearer_token
from src.api.deps import ApiState, get_state
from src.api.schemas import (
    CypherRequest,
    CypherResponse,
    ValidateRequest,
    ValidateResponse,
)
from src.eval.neo4j_exec import calculate_svr_cypher, execute_and_get_ids

router = APIRouter(
    prefix="/cypher",
    tags=["cypher"],
    dependencies=[Depends(require_bearer_token)],
)


def _collect_rows(driver, query: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with driver.session() as session:
        for record in session.run(query):
            rows.append({k: _jsonify(v) for k, v in record.items()})
    return rows


def _jsonify(value: Any) -> Any:
    """Best-effort conversion of Neo4j values to JSON-friendly primitives."""
    if hasattr(value, "items"):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@router.post("/execute", response_model=CypherResponse)
def execute(req: CypherRequest, state: ApiState = Depends(get_state)) -> CypherResponse:
    """Execute a Cypher query. Returns extracted IDs (and optionally raw rows)."""
    ids, error = execute_and_get_ids(req.query, state.driver, req.id_property)
    rows = None
    if req.include_rows and error is None:
        try:
            rows = _collect_rows(state.driver, req.query)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    sorted_ids = sorted(ids)
    return CypherResponse(ids=sorted_ids, count=len(sorted_ids), rows=rows, error=error)


@router.post("/validate", response_model=ValidateResponse)
def validate(
    req: ValidateRequest, state: ApiState = Depends(get_state)
) -> ValidateResponse:
    """EXPLAIN-based syntax validation — no execution."""
    svr, is_valid, error = calculate_svr_cypher(req.query, driver=state.driver)
    return ValidateResponse(svr=svr, is_valid=is_valid, error=error)
