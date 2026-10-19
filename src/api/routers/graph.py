"""Which model is loaded, and switching it.

``GET /graph/active`` lets a client check that the graph it is about to query is
the one its bundle was built against (the fingerprint). ``POST /graph/activate``
wipes Neo4j, restores a cached model and rebuilds the API's vocabulary for it,
so it takes the separate admin token.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import require_admin_token, require_bearer_token
from src.api.deps import ApiState, get_state
from src.api.schemas import ActivateRequest, ActivateResponse, ActiveGraphResponse
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

# One restore at a time: two concurrent activations would interleave wipes and
# writes into a graph that matches neither fingerprint.
_ACTIVATION_LOCK = threading.Lock()


@router.get(
    "/active",
    response_model=ActiveGraphResponse,
    dependencies=[Depends(require_bearer_token)],
)
def active(state: ApiState = Depends(get_state)) -> ActiveGraphResponse:
    from src.etl.graph_cache import fingerprint, graph_summary, read_meta

    meta = read_meta(state.driver)
    live = fingerprint(graph_summary(state.driver))
    return ActiveGraphResponse(
        meta=meta,
        live_fingerprint=live,
        matches_meta=bool(meta) and meta.get("fingerprint") == live,
    )


def _data_file(rel: Optional[str], data_root: Path) -> Optional[Path]:
    """Resolve a client-supplied path, refusing anything outside the data dir."""
    if not rel:
        return None
    path = Path(rel)
    path = (path if path.is_absolute() else data_root.parent / path).resolve()
    if data_root.resolve() not in path.parents:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{rel} is outside {data_root}")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{rel} does not exist")
    return path


@router.post(
    "/activate",
    response_model=ActivateResponse,
    dependencies=[Depends(require_admin_token)],
)
def activate(req: ActivateRequest, request: Request) -> ActivateResponse:
    from src.api.main import _build_state
    from src.etl.graph_cache import restore_graph
    from src.etl.manifest import cache_path, load_manifest, model_entry, resolve

    current: ApiState = get_state(request)
    manifest = load_manifest(settings.ifcbench_manifest_path)
    try:
        entry = model_entry(manifest, req.model_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    snapshot = cache_path(manifest, entry, req.state)
    if not snapshot.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No graph cache for {req.model_id}/{req.state} at {snapshot}; "
            "export one with `python -m src.etl.graph_cache export`.",
        )
    data_root = manifest["_root"] / "data"
    ids_path = _data_file(req.ids_path, data_root)
    test_set_path = _data_file(req.test_set_path, data_root)

    if not _ACTIVATION_LOCK.acquire(blocking=False):
        raise HTTPException(status.HTTP_409_CONFLICT, "Another activation is in progress.")
    try:
        header = restore_graph(current.driver, snapshot)
        new_state = _build_state(
            ifc_path=resolve(manifest, entry["paths"]["ifc"]),
            ids_path=ids_path,
            test_set_path=test_set_path,
            resolve_default_test_set=False,
            driver=current.driver,
        )
        request.app.state.api_state = new_state
    finally:
        _ACTIVATION_LOCK.release()

    vocab = new_state.vocabulary
    logger.info("Activated %s/%s (%s)", req.model_id, req.state, header["fingerprint"][:12])
    return ActivateResponse(
        model_id=req.model_id,
        state=req.state,
        fingerprint=header["fingerprint"],
        totals=header["summary"]["totals"],
        vocabulary_entities=len(vocab.get_entity_names()) if vocab else 0,
        vocabulary_properties=len(vocab.get_all_property_names()) if vocab else 0,
        test_cases=len(new_state.test_cases),
    )
