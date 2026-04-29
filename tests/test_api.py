"""Tests for the FastAPI proxy.

The Neo4j driver is stubbed out via ``app.dependency_overrides`` and a fake
``ApiState``, so these run without a live database.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.deps import ApiState, get_state
from src.api.main import create_app
from src.config import settings


TEST_TOKEN = "test-token-please-ignore"


class FakeSession:
    def __init__(self, records=None, raises=None):
        self._records = records or []
        self._raises = raises

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        if self._raises is not None:
            raise self._raises
        return FakeResult(self._records)


class FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def consume(self):
        return None


class FakeRecord(dict):
    def items(self):
        return super().items()

    def values(self):
        return super().values()


class FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session

    def close(self):
        return None


def _build_state(driver=None) -> ApiState:
    if driver is None:
        driver = FakeDriver(FakeSession([FakeRecord({"n.GlobalId": "abc-123"})]))
    test_cases = [
        {
            "question": "All walls",
            "gold_cypher": "MATCH (n:IfcWall) RETURN n.GlobalId",
            "category": "basic",
            "difficulty": "easy",
        }
    ]
    return ApiState(
        driver=driver,
        test_cases=test_cases,
        gold_id_sets={0: {"abc-123"}},
        vocabulary=None,
        valid_labels={"IfcWall", "IfcDoor"},
        valid_properties={"GlobalId", "Name", "IsExternal"},
        test_set_path=None,
    )


@pytest.fixture
def client(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "api_bearer_token", SecretStr(TEST_TOKEN))

    app = create_app()
    app.dependency_overrides[get_state] = lambda: _build_state()
    # TestClient(app) without `with` does NOT fire the lifespan — avoids the real
    # Neo4j connection during unit tests.
    tc = TestClient(app, raise_server_exceptions=True)
    app.state.api_state = _build_state()
    yield tc


# -----------------------------------------------------------------------------
# /health
# -----------------------------------------------------------------------------

def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_requires_auth(client):
    r = client.get("/health/ready")
    assert r.status_code == 401


def test_ready_with_token(client):
    r = client.get("/health/ready", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 200
    assert r.json()["neo4j"] is True


# -----------------------------------------------------------------------------
# Auth error paths
# -----------------------------------------------------------------------------

def test_missing_token_rejected(client):
    r = client.post("/cypher/execute", json={"query": "MATCH (n) RETURN n"})
    assert r.status_code == 401


def test_wrong_token_rejected(client):
    r = client.post(
        "/cypher/execute",
        json={"query": "MATCH (n) RETURN n"},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 403


# -----------------------------------------------------------------------------
# /cypher/execute
# -----------------------------------------------------------------------------

def test_cypher_execute_returns_ids(client):
    r = client.post(
        "/cypher/execute",
        json={"query": "MATCH (n:IfcWall) RETURN n.GlobalId"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ids"] == ["abc-123"]
    assert body["count"] == 1
    assert body["rows"] is None


# -----------------------------------------------------------------------------
# /test-set and /gold
# -----------------------------------------------------------------------------

def test_test_set_listed(client):
    r = client.get("/test-set", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["cases"][0]["question"] == "All walls"


def test_gold_index(client):
    r = client.get("/gold/0", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 200
    assert r.json()["ids"] == ["abc-123"]


def test_gold_out_of_range(client):
    r = client.get("/gold/99", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert r.status_code == 404


# -----------------------------------------------------------------------------
# /schema
# -----------------------------------------------------------------------------

def test_valid_labels(client):
    r = client.get(
        "/schema/valid-labels", headers={"Authorization": f"Bearer {TEST_TOKEN}"}
    )
    assert r.status_code == 200
    assert set(r.json()["labels"]) == {"IfcWall", "IfcDoor"}


# -----------------------------------------------------------------------------
# 503 when no token configured
# -----------------------------------------------------------------------------

def test_no_token_configured_returns_503(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "api_bearer_token", SecretStr(""))
    app = create_app()
    app.dependency_overrides[get_state] = lambda: _build_state()
    tc = TestClient(app)
    app.state.api_state = _build_state()
    r = tc.get("/health/ready", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 503
