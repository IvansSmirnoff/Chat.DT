"""Shared pytest fixtures.

Fixtures point at the real data files mounted into the container at
``/app/data``. Tests that need Neo4j are skipped automatically when the
database is unreachable so the suite runs in a Colab-only environment too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


DATA_DIR = Path(os.environ.get("NL2C_DATA_DIR", "/app/data"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def barcelona_ifc(data_dir: Path) -> Path:
    path = data_dir / "Barcelona.ifc"
    if not path.exists():
        pytest.skip(f"Barcelona.ifc not available at {path}")
    return path


@pytest.fixture(scope="session")
def requirements_ids(data_dir: Path) -> Path:
    path = data_dir / "requirements.ids"
    if not path.exists():
        pytest.skip(f"requirements.ids not available at {path}")
    return path


@pytest.fixture(scope="session")
def neo4j_driver():
    """Live Neo4j driver or skip the test."""
    try:
        from neo4j import GraphDatabase
        from src.config import get_settings
    except ImportError as exc:
        pytest.skip(f"neo4j/config unavailable: {exc}")

    settings = get_settings()
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password_value),
        )
        driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j unreachable: {exc}")
        return None
    yield driver
    driver.close()
