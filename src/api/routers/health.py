"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends

from src.api.auth import require_bearer_token
from src.api.deps import ApiState, get_state
from src.api.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. No auth — meant for uptime checks."""
    return HealthResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    dependencies=[Depends(require_bearer_token)],
)
def ready(state: ApiState = Depends(get_state)) -> ReadinessResponse:
    """Readiness probe. Verifies Neo4j is reachable."""
    try:
        with state.driver.session() as session:
            session.run("RETURN 1").consume()
        return ReadinessResponse(status="ok", neo4j=True)
    except Exception as exc:  # noqa: BLE001
        return ReadinessResponse(status="degraded", neo4j=False, detail=str(exc))
