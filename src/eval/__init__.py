"""
Evaluation module for NL2Cypher comparative experiments.

Supports 4 Experimental Settings:
1. DIRECT_QA_BASELINE: Direct QA without IDS grounding
2. DIRECT_QA_GROUNDED: Direct QA with IDS grounding
3. CYPHER_SOFT: Cypher generation with soft constraints
4. CYPHER_STRICT: Cypher generation with strict constraints

Module layout:
- ``scoring``    — pure SVR/SCR/EA primitives + EvaluationResult
- ``neo4j_exec`` — Cypher-execution helpers that require a Neo4j driver
- ``aggregate``  — evaluate/aggregate batch orchestration
- ``runner``     — ExperimentRunner + ExperimentConfig
- ``cli``        — ``python -m src.eval.cli`` argparse entry point
"""

from .scoring import (
    EvaluationResult,
    OutputType,
    calculate_ea_direct,
    calculate_ea_from_ids,
    calculate_scr,
    calculate_scr_cypher,
    calculate_svr_json,
)
from .neo4j_exec import (
    calculate_ea,
    calculate_ea_cypher,
    calculate_svr,
    calculate_svr_cypher,
    execute_and_get_ids,
    execute_gold_queries,
)
from .aggregate import (
    aggregate_by_category,
    aggregate_by_difficulty,
    aggregate_by_setting,
    aggregate_results,
    evaluate_batch,
    evaluate_cypher_output,
    evaluate_direct_qa_output,
    evaluate_output,
)
from .runner import (
    ExperimentConfig,
    ExperimentRunner,
    load_test_set,
    save_results_csv,
    save_summary,
)

__all__ = [
    # Metrics
    "calculate_svr",
    "calculate_svr_cypher",
    "calculate_svr_json",
    "calculate_scr",
    "calculate_scr_cypher",
    "calculate_ea",
    "calculate_ea_cypher",
    "calculate_ea_direct",
    "calculate_ea_from_ids",
    "execute_and_get_ids",
    # Evaluation
    "evaluate_output",
    "evaluate_cypher_output",
    "evaluate_direct_qa_output",
    "evaluate_batch",
    "aggregate_results",
    "aggregate_by_setting",
    "aggregate_by_difficulty",
    "aggregate_by_category",
    "execute_gold_queries",
    # Data classes
    "EvaluationResult",
    "OutputType",
    # Experiment runner
    "ExperimentConfig",
    "ExperimentRunner",
    "load_test_set",
    "save_results_csv",
    "save_summary",
]
