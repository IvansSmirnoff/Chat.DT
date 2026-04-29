"""Shared FastAPI dependencies: Neo4j driver + cached schema state.

State is owned by the FastAPI app's ``app.state`` (populated in the lifespan).
The getters here are thin accessors so routers/tests can depend on them and
swap them via ``app.dependency_overrides`` without poking at ``app.state``
from every handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from src.constraints.vocabulary_merger import CombinedVocabulary


@dataclass
class ApiState:
    """Process-wide state built once during app startup."""

    driver: Any  # neo4j.Driver
    test_cases: List[Dict[str, str]]
    gold_id_sets: Dict[int, Set[str]]
    vocabulary: Optional["CombinedVocabulary"]
    valid_labels: Set[str]
    valid_properties: Set[str]
    test_set_path: Optional[Path]


def get_state(request: Request) -> ApiState:
    state: Optional[ApiState] = getattr(request.app.state, "api_state", None)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API state is not initialized.",
        )
    return state


def get_driver(request: Request):
    return get_state(request).driver
