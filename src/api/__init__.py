"""FastAPI app that proxies authenticated query/scoring operations to Neo4j."""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
