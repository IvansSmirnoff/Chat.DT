"""
Orchestration and aggregation for evaluation results.

Ties the pure scoring primitives (``scoring.py``) and the Neo4j-backed
execution primitives (``neo4j_exec.py``) together into high-level
evaluate/aggregate helpers used by ``runner.py`` and ``cli.py``.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

from src.config import ExperimentSetting
from src.eval.scoring import (
    EvaluationResult,
    OutputType,
    calculate_ea_from_ids,
    calculate_scr_cypher,
    calculate_svr_json,
)
from src.eval.neo4j_exec import (
    calculate_ea_cypher,
    calculate_svr_cypher,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Combined evaluation
# =============================================================================

def evaluate_cypher_output(
    question: str,
    generated_cypher: str,
    gold_ids: Set[str],
    driver,
    valid_labels: Set[str],
    valid_properties: Set[str],
    experiment_setting: Optional[ExperimentSetting] = None,
    gold_cypher: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate a generated Cypher query (Settings 3 & 4).
    
    Args:
        question: Original question
        generated_cypher: Generated Cypher query
        gold_ids: Gold standard result IDs
        driver: Neo4j driver
        valid_labels: Valid entity labels
        valid_properties: Valid property names
        experiment_setting: Experimental setting used
        gold_cypher: Gold Cypher query (for reference)
        
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult(
        question=question,
        experiment_setting=experiment_setting,
        output_type=OutputType.CYPHER,
        generated_cypher=generated_cypher,
        gold_cypher=gold_cypher,
        gold_ids=gold_ids,
        raw_output=generated_cypher,
    )
    
    # Calculate SVR
    svr, is_valid, svr_error = calculate_svr_cypher(generated_cypher, driver=driver)
    result.svr = svr
    result.is_valid_syntax = is_valid
    if svr_error:
        result.error = svr_error
    
    # Calculate SCR
    scr, invalid_labels, invalid_properties = calculate_scr_cypher(
        generated_cypher, valid_labels, valid_properties
    )
    result.scr = scr
    result.invalid_labels = invalid_labels
    result.invalid_properties = invalid_properties
    
    # Calculate EA (only if syntax is valid)
    if is_valid:
        ea, gen_ids, ea_error = calculate_ea_cypher(
            generated_cypher, gold_ids, driver
        )
        result.ea = ea
        result.generated_ids = gen_ids
        if ea_error:
            result.error = (result.error or "") + f"; {ea_error}"
    else:
        result.ea = 0.0
        result.metadata["ea_skipped"] = "Invalid syntax"
    
    return result


def evaluate_direct_qa_output(
    question: str,
    raw_output: str,
    gold_ids: Set[str],
    experiment_setting: Optional[ExperimentSetting] = None,
) -> EvaluationResult:
    """
    Evaluate a Direct QA output (Settings 1 & 2).
    
    Args:
        question: Original question
        raw_output: Raw LLM output (should be JSON array)
        gold_ids: Gold standard result IDs
        experiment_setting: Experimental setting used
        
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult(
        question=question,
        experiment_setting=experiment_setting,
        output_type=OutputType.DIRECT_QA,
        generated_cypher=None,
        gold_ids=gold_ids,
        raw_output=raw_output,
    )
    
    # Calculate SVR (JSON validation)
    svr, is_valid, svr_error, parsed_list = calculate_svr_json(raw_output)
    result.svr = svr
    result.is_valid_syntax = is_valid
    if svr_error:
        result.error = svr_error
    
    # SCR is not applicable for Direct QA
    result.scr = 1.0  # N/A, set to 1.0
    result.metadata["scr_note"] = "Not applicable for Direct QA"
    
    # Calculate EA
    if is_valid and parsed_list is not None:
        result.generated_ids = set(parsed_list)
        result.ea = calculate_ea_from_ids(result.generated_ids, gold_ids)
    else:
        result.ea = 0.0
        result.generated_ids = set()
        result.metadata["ea_skipped"] = "Invalid JSON output"
    
    return result


def evaluate_output(
    question: str,
    output: str,
    output_type: OutputType,
    gold_ids: Set[str],
    driver=None,
    valid_labels: Optional[Set[str]] = None,
    valid_properties: Optional[Set[str]] = None,
    experiment_setting: Optional[ExperimentSetting] = None,
    gold_cypher: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate generated output based on type.
    
    Unified evaluation function that delegates to the appropriate
    type-specific evaluator.
    
    Args:
        question: Original question
        output: Generated output (Cypher or JSON)
        output_type: Type of output
        gold_ids: Gold standard result IDs
        driver: Neo4j driver (for Cypher)
        valid_labels: Valid entity labels (for Cypher SCR)
        valid_properties: Valid property names (for Cypher SCR)
        experiment_setting: Experimental setting used
        gold_cypher: Gold Cypher query (for reference)
        
    Returns:
        EvaluationResult with all metrics
    """
    if output_type == OutputType.CYPHER:
        return evaluate_cypher_output(
            question=question,
            generated_cypher=output,
            gold_ids=gold_ids,
            driver=driver,
            valid_labels=valid_labels or set(),
            valid_properties=valid_properties or set(),
            experiment_setting=experiment_setting,
            gold_cypher=gold_cypher,
        )
    else:
        return evaluate_direct_qa_output(
            question=question,
            raw_output=output,
            gold_ids=gold_ids,
            experiment_setting=experiment_setting,
        )


# =============================================================================
# Batch evaluation
# =============================================================================

def evaluate_batch(
    test_cases: List[Dict[str, Any]],
    gold_id_sets: Dict[int, Set[str]],
    output_type: OutputType,
    driver=None,
    valid_labels: Optional[Set[str]] = None,
    valid_properties: Optional[Set[str]] = None,
    experiment_setting: Optional[ExperimentSetting] = None,
    generate_fn=None,
) -> List[EvaluationResult]:
    """
    Evaluate a batch of test cases.
    
    Args:
        test_cases: List of test case dicts
        gold_id_sets: Pre-computed gold ID sets (from execute_gold_queries)
        output_type: Type of output expected
        driver: Neo4j driver (for Cypher)
        valid_labels: Valid entity labels
        valid_properties: Valid property names
        experiment_setting: Experimental setting used
        generate_fn: Optional function to generate output from question
                    
    Returns:
        List of EvaluationResult
    """
    results = []
    
    for i, case in enumerate(test_cases):
        question = case.get("question", case.get("Question", ""))
        gold_ids = gold_id_sets.get(i, set())
        gold_cypher = case.get("gold_cypher", case.get("Gold_Cypher"))
        
        # Get generated output
        if generate_fn:
            try:
                gen_result = generate_fn(question)
                if output_type == OutputType.CYPHER:
                    output = gen_result.query if hasattr(gen_result, 'query') else str(gen_result)
                else:
                    if hasattr(gen_result, 'raw_output'):
                        output = gen_result.raw_output
                    elif hasattr(gen_result, 'direct_answer'):
                        output = json.dumps(gen_result.direct_answer or [])
                    else:
                        output = str(gen_result)
            except Exception as e:
                logger.error(f"Generation failed for case {i}: {e}")
                result = EvaluationResult(
                    question=question,
                    experiment_setting=experiment_setting,
                    output_type=output_type,
                    gold_ids=gold_ids,
                    error=f"Generation failed: {e}",
                )
                results.append(result)
                continue
        else:
            # Expect output to be in test case
            if output_type == OutputType.CYPHER:
                output = case.get("generated_cypher", case.get("Generated_Cypher", ""))
            else:
                output = case.get("generated_output", case.get("raw_output", ""))
        
        # Evaluate
        result = evaluate_output(
            question=question,
            output=output,
            output_type=output_type,
            gold_ids=gold_ids,
            driver=driver,
            valid_labels=valid_labels,
            valid_properties=valid_properties,
            experiment_setting=experiment_setting,
            gold_cypher=gold_cypher,
        )
        results.append(result)
        
        logger.info(
            f"Evaluated {i+1}/{len(test_cases)}: "
            f"SVR={result.svr:.0f}, SCR={result.scr:.2f}, EA={result.ea:.2f}"
        )
    
    return results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate evaluation results into summary statistics.
    
    Args:
        results: List of EvaluationResult
        
    Returns:
        Dict with aggregated metrics
    """
    if not results:
        return {
            "count": 0,
            "svr_mean": 0.0,
            "scr_mean": 0.0,
            "ea_mean": 0.0,
        }
    
    n = len(results)
    
    # Separate by output type
    cypher_results = [r for r in results if r.output_type == OutputType.CYPHER]
    direct_qa_results = [r for r in results if r.output_type == OutputType.DIRECT_QA]
    
    summary = {
        "count": n,
        "cypher_count": len(cypher_results),
        "direct_qa_count": len(direct_qa_results),
        
        # Overall metrics
        "svr_mean": sum(r.svr for r in results) / n,
        "svr_sum": sum(r.svr for r in results),
        "scr_mean": sum(r.scr for r in results) / n,
        "ea_mean": sum(r.ea for r in results) / n,
        "ea_min": min(r.ea for r in results),
        "ea_max": max(r.ea for r in results),
        
        # Precision/Recall/F1
        "precision_mean": sum(r._calculate_precision() for r in results) / n,
        "recall_mean": sum(r._calculate_recall() for r in results) / n,
        "f1_mean": sum(r._calculate_f1() for r in results) / n,
        
        # Counts
        "syntax_valid_count": sum(1 for r in results if r.is_valid_syntax),
        "perfect_score_count": sum(
            1 for r in results if r.svr == 1.0 and r.scr == 1.0 and r.ea == 1.0
        ),
    }
    
    # Add type-specific metrics if applicable
    if cypher_results:
        summary["cypher_svr_mean"] = sum(r.svr for r in cypher_results) / len(cypher_results)
        summary["cypher_scr_mean"] = sum(r.scr for r in cypher_results) / len(cypher_results)
        summary["cypher_ea_mean"] = sum(r.ea for r in cypher_results) / len(cypher_results)
    
    if direct_qa_results:
        summary["direct_qa_svr_mean"] = sum(r.svr for r in direct_qa_results) / len(direct_qa_results)
        summary["direct_qa_ea_mean"] = sum(r.ea for r in direct_qa_results) / len(direct_qa_results)
    
    # Add metrics by difficulty level (Complexity Analysis)
    summary["metrics_by_difficulty"] = aggregate_by_difficulty(results)
    
    return summary


def aggregate_by_difficulty(results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results grouped by difficulty level.
    
    This enables "Complexity vs. Performance" analysis by grouping
    results by Easy/Medium/Hard difficulty levels from the test set.
    
    Args:
        results: List of EvaluationResult
        
    Returns:
        Dict mapping difficulty level to aggregated metrics
    """
    # Group results by difficulty level
    grouped: Dict[str, List[EvaluationResult]] = {}
    
    for result in results:
        # Get difficulty from metadata (added during evaluation)
        difficulty = result.metadata.get("difficulty", "").strip()
        
        # Normalize difficulty levels
        if not difficulty:
            difficulty = "unknown"
        else:
            difficulty = difficulty.lower()
            # Map common aliases
            if difficulty in ("level 1", "l1", "1"):
                difficulty = "easy"
            elif difficulty in ("level 2", "l2", "2"):
                difficulty = "medium"
            elif difficulty in ("level 3", "l3", "3"):
                difficulty = "hard"
        
        if difficulty not in grouped:
            grouped[difficulty] = []
        grouped[difficulty].append(result)
    
    # Calculate metrics for each difficulty level
    difficulty_metrics = {}
    for difficulty, difficulty_results in grouped.items():
        n = len(difficulty_results)
        if n == 0:
            continue
            
        difficulty_metrics[difficulty] = {
            "count": n,
            "ea_mean": sum(r.ea for r in difficulty_results) / n,
            "ea_min": min(r.ea for r in difficulty_results),
            "ea_max": max(r.ea for r in difficulty_results),
            "svr_mean": sum(r.svr for r in difficulty_results) / n,
            "scr_mean": sum(r.scr for r in difficulty_results) / n,
            "f1_mean": sum(r._calculate_f1() for r in difficulty_results) / n,
            "precision_mean": sum(r._calculate_precision() for r in difficulty_results) / n,
            "recall_mean": sum(r._calculate_recall() for r in difficulty_results) / n,
            "syntax_valid_count": sum(1 for r in difficulty_results if r.is_valid_syntax),
        }
    
    return difficulty_metrics


def aggregate_by_category(results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results grouped by category (e.g., basic_query, property_filter, relationship).
    
    Args:
        results: List of EvaluationResult
        
    Returns:
        Dict mapping category to aggregated metrics
    """
    # Group results by category
    grouped: Dict[str, List[EvaluationResult]] = {}
    
    for result in results:
        category = result.metadata.get("category", "").strip()
        if not category:
            category = "unknown"
        
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(result)
    
    # Calculate metrics for each category
    category_metrics = {}
    for category, category_results in grouped.items():
        n = len(category_results)
        if n == 0:
            continue
            
        category_metrics[category] = {
            "count": n,
            "ea_mean": sum(r.ea for r in category_results) / n,
            "svr_mean": sum(r.svr for r in category_results) / n,
            "scr_mean": sum(r.scr for r in category_results) / n,
            "f1_mean": sum(r._calculate_f1() for r in category_results) / n,
        }
    
    return category_metrics


def aggregate_by_setting(
    results: List[EvaluationResult]
) -> Dict[ExperimentSetting, Dict[str, Any]]:
    """
    Aggregate results grouped by experimental setting.
    
    Args:
        results: List of EvaluationResult
        
    Returns:
        Dict mapping ExperimentSetting to aggregated metrics
    """
    grouped = {}
    for result in results:
        setting = result.experiment_setting
        if setting not in grouped:
            grouped[setting] = []
        grouped[setting].append(result)
    
    return {
        setting: aggregate_results(setting_results)
        for setting, setting_results in grouped.items()
    }

