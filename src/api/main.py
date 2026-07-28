"""FastAPI entry point.

Run with:

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

The lifespan opens a single Neo4j driver for the process, builds the combined
IFC+IDS vocabulary, and pre-executes gold queries for the default test set.
All routers (except ``/health``) require the bearer token from
``settings.api_bearer_token``.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from src.api.deps import ApiState
from src.config import settings

logger = logging.getLogger(__name__)


DEFAULT_TEST_SET_CANDIDATES = (
    Path("/app/data/test_set.csv"),
    Path("/app/data/test_set.json"),
)


def _resolve_default_test_set() -> Optional[Path]:
    """Explicit TEST_SET_PATH wins; otherwise fall back to the conventional names."""
    # An unset compose variable arrives as the empty string, and Path("") is
    # Path(".") — a directory that exists. Without the strip()/is_file() guard
    # the API would "resolve" the test set to the working directory.
    configured = settings.test_set_path
    if configured is not None and str(configured).strip():
        configured = Path(str(configured).strip())
        if configured.is_file():
            return configured
        logger.warning(
            "TEST_SET_PATH is set to %s but that file does not exist; "
            "falling back to the default candidates.",
            configured,
        )
    for candidate in DEFAULT_TEST_SET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _build_state() -> ApiState:
    """Open Neo4j driver and build cached schema/test-set state.

    Heavy imports (ifcopenshell via schema_scanner, the full llm_engine) are
    deferred here so merely importing ``src.api.main`` stays lightweight — the
    test suite and tooling can load the FastAPI app without the full research
    stack installed.
    """
    from neo4j import GraphDatabase

    from src.constraints.ids_parser import IDSParser, IDSSchema
    from src.constraints.vocabulary_merger import (
        build_combined_vocabulary,
        merge_graph_stats_into_vocabulary,
    )
    from src.eval.neo4j_exec import collect_graph_stats, execute_gold_queries
    from src.eval.runner import load_test_set

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password_value),
    )
    driver.verify_connectivity()
    logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    # Vocabulary (best-effort: missing IFC/IDS files should not kill the API)
    vocabulary = None
    ids_path = settings.ids_file_path
    ifc_path = settings.ifc_file_path
    try:
        ids_arg = ids_path if Path(ids_path).exists() else None
        ifc_arg = ifc_path if Path(ifc_path).exists() else None
        if ids_arg or ifc_arg:
            vocabulary = build_combined_vocabulary(ids_arg, ifc_arg)
            # Same merge the bundle builder applies. Without it the API scores
            # SCR against the IFC scan's 23 labels while the client decodes
            # against the graph's 76, so a perfectly valid `IfcMaterial` query
            # is counted as a schema violation.
            merge_graph_stats_into_vocabulary(vocabulary, collect_graph_stats(driver))
            logger.info(
                "Loaded vocabulary: %d entities, %d properties",
                len(vocabulary.get_entity_names()),
                len(vocabulary.get_all_property_names()),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vocabulary build failed, SCR will be limited: %s", exc)

    valid_labels: set = set()
    valid_properties: set = set()
    if vocabulary:
        valid_labels = set(vocabulary.get_entity_names())
        valid_properties = set(vocabulary.get_all_property_names())
    elif Path(ids_path).exists():
        schema: IDSSchema = IDSParser().parse(ids_path)
        valid_labels = schema.get_neo4j_labels()
        valid_properties = set(schema.properties)
    valid_properties.update({"GlobalId", "Name", "Description", "ObjectType"})

    # Test set + gold executions (best-effort)
    test_cases: list = []
    gold_id_sets: dict = {}
    test_set_path: Optional[Path] = _resolve_default_test_set()
    if test_set_path:
        logger.info("Resolved test-set path: %s", test_set_path)
        try:
            test_cases = load_test_set(test_set_path)
            gold_id_sets = execute_gold_queries(test_cases, driver)
            logger.info(
                "Loaded %d test cases and pre-executed gold queries", len(test_cases)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Test set / gold execution failed: %s", exc)
    else:
        logger.warning(
            "No test-set file found in %s",
            [str(p) for p in DEFAULT_TEST_SET_CANDIDATES],
        )

    return ApiState(
        driver=driver,
        test_cases=test_cases,
        gold_id_sets=gold_id_sets,
        vocabulary=vocabulary,
        valid_labels=valid_labels,
        valid_properties=valid_properties,
        test_set_path=test_set_path,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level.upper())
    state = _build_state()
    app.state.api_state = state
    try:
        yield
    finally:
        try:
            state.driver.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error closing Neo4j driver")


def create_app() -> FastAPI:
    from src.api.routers import cypher, evaluation, health, schema_router

    app = FastAPI(
        title="Chat.DT API",
        description=(
            "Authenticated proxy between Colab notebooks (LLM Cypher generation) "
            "and Neo4j. Exposes query execution, scoring, and schema retrieval."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(cypher.router)
    app.include_router(evaluation.router)
    app.include_router(schema_router.router)

    return app


app = create_app()
