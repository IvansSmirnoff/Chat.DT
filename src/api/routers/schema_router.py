"""Schema / vocabulary endpoints — what Colab needs to build prompts or do
client-side SCR."""

from fastapi import APIRouter, Depends

from src.api.auth import require_bearer_token
from src.api.deps import ApiState, get_state
from src.api.schemas import LabelsResponse, PropertiesResponse, VocabularyResponse

router = APIRouter(
    prefix="/schema",
    tags=["schema"],
    dependencies=[Depends(require_bearer_token)],
)


@router.get("/valid-labels", response_model=LabelsResponse)
def valid_labels(state: ApiState = Depends(get_state)) -> LabelsResponse:
    labels = sorted(state.valid_labels)
    return LabelsResponse(count=len(labels), labels=labels)


@router.get("/valid-properties", response_model=PropertiesResponse)
def valid_properties(state: ApiState = Depends(get_state)) -> PropertiesResponse:
    props = sorted(state.valid_properties)
    return PropertiesResponse(count=len(props), properties=props)


@router.get("/vocabulary", response_model=VocabularyResponse)
def vocabulary(state: ApiState = Depends(get_state)) -> VocabularyResponse:
    vocab = state.vocabulary
    if vocab is None:
        return VocabularyResponse(
            entities=sorted(state.valid_labels),
            properties=sorted(state.valid_properties),
            strict_properties=[],
            relations=[],
            ids_loaded=False,
            ifc_loaded=False,
        )
    return VocabularyResponse(
        entities=sorted(vocab.get_entity_names()),
        properties=sorted(vocab.get_all_property_names()),
        strict_properties=sorted(vocab.strict_properties),
        relations=sorted(vocab.all_relations),
        ids_loaded=vocab.ids_schema is not None,
        ifc_loaded=vocab.ifc_schema is not None,
    )
