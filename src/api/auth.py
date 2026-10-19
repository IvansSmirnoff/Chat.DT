"""Bearer-token auth dependencies for the FastAPI app.

Two shared secrets. ``settings.api_bearer_token`` guards every ordinary
endpoint; ``settings.api_admin_token`` guards the state-changing ones (graph
activation wipes and restores Neo4j), so the token handed to a Colab notebook
cannot wipe the database. If either token is empty in ``.env``, the calls it
guards return 503 — an unconfigured deployment fails closed rather than open.
"""

from secrets import compare_digest
from typing import Optional

from fastapi import Header, HTTPException, status

from src.config import settings

_BEARER_PREFIX = "Bearer "


def _require(expected: str, authorization: Optional[str], what: str) -> None:
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{what} is not configured on the server.",
        )

    if not authorization or not authorization.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided = authorization[len(_BEARER_PREFIX):]
    if not compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid {what}.",
        )


def require_bearer_token(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency that validates ``Authorization: Bearer <token>``.

    Raises:
        HTTPException 503: if no API token is configured server-side.
        HTTPException 401: if the header is missing or malformed.
        HTTPException 403: if the token does not match.
    """
    _require(settings.api_bearer_token_value, authorization, "API bearer token")


def require_admin_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Same contract as :func:`require_bearer_token`, against the admin token."""
    _require(settings.api_admin_token_value, authorization, "admin token")
