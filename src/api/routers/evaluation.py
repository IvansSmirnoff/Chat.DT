"""Test-set, gold-query, and end-to-end scoring endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import require_bearer_token
from src.api.deps import ApiState, get_state
from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    GoldResponse,
    TestCase,
    TestSetResponse,
)
from src.eval.aggregate import evaluate_output

router = APIRouter(
    tags=["evaluation"],
    dependencies=[Depends(require_bearer_token)],
)


@router.get("/test-set", response_model=TestSetResponse)
def get_test_set(state: ApiState = Depends(get_state)) -> TestSetResponse:
    """Return the server-side test set (Question, Gold_Cypher, ...)."""
    cases = [
        TestCase(
            index=i,
            question=case.get("question", ""),
            gold_cypher=case.get("gold_cypher", ""),
            category=case.get("category", ""),
            difficulty=case.get("difficulty", ""),
        )
        for i, case in enumerate(state.test_cases)
    ]
    return TestSetResponse(count=len(cases), cases=cases)


@router.get("/gold/{index}", response_model=GoldResponse)
def get_gold(index: int, state: ApiState = Depends(get_state)) -> GoldResponse:
    """Return the pre-executed gold IDs for a given test case index."""
    if index < 0 or index >= len(state.test_cases):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case {index} not found (have {len(state.test_cases)}).",
        )
    case = state.test_cases[index]
    ids = sorted(state.gold_id_sets.get(index, set()))
    return GoldResponse(
        index=index,
        gold_cypher=case.get("gold_cypher", ""),
        ids=ids,
        count=len(ids),
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, state: ApiState = Depends(get_state)) -> EvaluateResponse:
    """Full scoring pipeline (SVR + SCR + EA) for a generated output."""
    if req.gold_index is not None:
        if req.gold_index < 0 or req.gold_index >= len(state.test_cases):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"gold_index {req.gold_index} not in test set.",
            )
        gold_ids = state.gold_id_sets.get(req.gold_index, set())
        gold_cypher = req.gold_cypher or state.test_cases[req.gold_index].get("gold_cypher")
    elif req.gold_ids is not None:
        gold_ids = set(req.gold_ids)
        gold_cypher = req.gold_cypher
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either gold_index or gold_ids must be supplied.",
        )

    result = evaluate_output(
        question=req.question,
        output=req.output,
        output_type=req.output_type,
        gold_ids=gold_ids,
        driver=state.driver,
        valid_labels=state.valid_labels,
        valid_properties=state.valid_properties,
        experiment_setting=req.experiment_setting,
        gold_cypher=gold_cypher,
    )

    return EvaluateResponse(
        question=result.question,
        output_type=result.output_type,
        experiment_setting=result.experiment_setting,
        generated_cypher=result.generated_cypher,
        gold_cypher=result.gold_cypher,
        generated_ids=sorted(result.generated_ids),
        gold_ids=sorted(result.gold_ids),
        svr=result.svr,
        scr=result.scr,
        ea=result.ea,
        precision=result._calculate_precision(),
        recall=result._calculate_recall(),
        f1=result._calculate_f1(),
        is_valid_syntax=result.is_valid_syntax,
        invalid_labels=sorted(result.invalid_labels),
        invalid_properties=sorted(result.invalid_properties),
        error=result.error,
        metadata=result.metadata,
    )
