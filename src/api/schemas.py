"""Pydantic request/response models for the API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import ExperimentSetting
from src.eval.scoring import OutputType


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str
    neo4j: bool
    detail: Optional[str] = None


# -----------------------------------------------------------------------------
# Cypher
# -----------------------------------------------------------------------------

class CypherRequest(BaseModel):
    query: str = Field(..., description="Cypher query to execute.")
    id_property: str = Field(
        default="GlobalId",
        description="Node property used to extract IDs from results.",
    )
    include_rows: bool = Field(
        default=False,
        description="If true, also return raw rows (list of dicts). Large result "
                    "sets can be expensive — default is IDs only.",
    )


class CypherResponse(BaseModel):
    ids: List[str]
    count: int
    rows: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ValidateRequest(BaseModel):
    query: str


class ValidateResponse(BaseModel):
    svr: float
    is_valid: bool
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

class TestCase(BaseModel):
    index: int
    question: str
    gold_cypher: str
    category: str = ""
    difficulty: str = ""


class TestSetResponse(BaseModel):
    count: int
    cases: List[TestCase]


class GoldResponse(BaseModel):
    index: int
    gold_cypher: str
    ids: List[str]
    count: int


class EvaluateRequest(BaseModel):
    question: str
    output: str = Field(..., description="Generated Cypher query OR JSON ID list.")
    output_type: OutputType
    gold_index: Optional[int] = Field(
        default=None,
        description="Index into the server-side test set (preferred).",
    )
    gold_ids: Optional[List[str]] = Field(
        default=None,
        description="Explicit gold IDs — used if gold_index is not provided.",
    )
    gold_cypher: Optional[str] = None
    experiment_setting: Optional[ExperimentSetting] = None


class EvaluateResponse(BaseModel):
    question: str
    output_type: OutputType
    experiment_setting: Optional[ExperimentSetting]
    generated_cypher: Optional[str]
    gold_cypher: Optional[str]
    generated_ids: List[str]
    gold_ids: List[str]
    svr: float
    scr: float
    ea: float
    precision: float
    recall: float
    f1: float
    is_valid_syntax: bool
    invalid_labels: List[str]
    invalid_properties: List[str]
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

class LabelsResponse(BaseModel):
    count: int
    labels: List[str]


class PropertiesResponse(BaseModel):
    count: int
    properties: List[str]


class VocabularyResponse(BaseModel):
    entities: List[str]
    properties: List[str]
    strict_properties: List[str]
    relations: List[str]
    ids_loaded: bool
    ifc_loaded: bool
