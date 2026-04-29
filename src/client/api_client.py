"""HTTP client for the Chat.DT FastAPI service.

Used by the Colab notebook to drive the test loop against a remote ``api``
container without ever touching Bolt directly. All authenticated calls send
``Authorization: Bearer <token>`` and raise ``ApiClientError`` on non-2xx
responses (the body is included in the message so 401/403/503 are obvious).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class ApiClientError(RuntimeError):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str, url: str):
        super().__init__(f"HTTP {status_code} from {url}: {body}")
        self.status_code = status_code
        self.body = body
        self.url = url


class ApiClient:
    """Thin wrapper over ``httpx.Client`` with bearer auth + JSON helpers."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 60.0,
        verify: bool = True,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not token:
            raise ValueError("token is required (the API returns 503 if unset server-side)")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify,
            headers={"Authorization": f"Bearer {token}"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> Any:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise ApiClientError(resp.status_code, resp.text, str(resp.request.url))
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Unauthenticated liveness probe."""
        # /health is auth-free, but reusing the same client is fine.
        return self._request("GET", "/health")

    def health_ready(self) -> Dict[str, Any]:
        """Readiness probe — confirms Neo4j connectivity server-side."""
        return self._request("GET", "/health/ready")

    def get_test_set(self) -> List[Dict[str, Any]]:
        """Return the server-side test cases (with ``index`` per row)."""
        body = self._request("GET", "/test-set")
        return body.get("cases", [])

    def get_gold(self, index: int) -> Dict[str, Any]:
        """Return the pre-executed gold IDs for a given test case index."""
        return self._request("GET", f"/gold/{index}")

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST a generated output for full SVR/SCR/EA scoring."""
        return self._request("POST", "/evaluate", json=payload)

    def execute_cypher(
        self, query: str, *, id_property: str = "GlobalId", include_rows: bool = False
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/cypher/execute",
            json={"query": query, "id_property": id_property, "include_rows": include_rows},
        )

    def validate_cypher(self, query: str) -> Dict[str, Any]:
        return self._request("POST", "/cypher/validate", json={"query": query})

    def get_vocabulary(self) -> Dict[str, Any]:
        return self._request("GET", "/schema/vocabulary")

    def get_valid_labels(self) -> List[str]:
        return (self._request("GET", "/schema/valid-labels") or {}).get("labels", [])

    def get_valid_properties(self) -> List[str]:
        return (self._request("GET", "/schema/valid-properties") or {}).get("properties", [])
