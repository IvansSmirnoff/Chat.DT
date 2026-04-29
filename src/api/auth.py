"""Bearer-token auth dependency for the FastAPI app.

A single shared secret (``settings.api_bearer_token``) is the only credential.
If the token is empty in ``.env``, every authenticated call returns 503 — this
makes an unconfigured deployment fail closed rather than silently open.
"""

from secrets import compare_digest
from typing import Optional

from fastapi import Header, HTTPException, status

from src.config import settings

_BEARER_PREFIX = "Bearer "


def require_bearer_token(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency that validates ``Authorization: Bearer <token>``.

    Raises:
        HTTPException 503: if no API token is configured server-side.
        HTTPException 401: if the header is missing or malformed.
        HTTPException 403: if the token does not match.
    """
    expected = settings.api_bearer_token_value
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API bearer token is not configured on the server.",
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
            detail="Invalid bearer token.",
        )
