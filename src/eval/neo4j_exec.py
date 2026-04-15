"""
Neo4j-backed execution primitives for evaluation.

These functions take a live Neo4j driver and touch the database. Kept
separate from ``scoring.py`` so the pure scoring primitives can be imported
in environments without a running Neo4j instance (e.g. a Colab notebook
re-scoring pre-computed results).
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, Neo4jError

from src.eval.scoring import (
    OutputType,
    _extract_global_id,
    calculate_ea_direct,
    calculate_ea_from_ids,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SVR: Cypher syntax validation (requires Neo4j driver)
# =============================================================================

def calculate_svr_cypher(
    query: str,
    driver=None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> Tuple[float, bool, Optional[str]]:
    """
    Calculate Syntactic Validity Rate for Cypher queries.
    
    Uses Neo4j's EXPLAIN to validate syntax without execution.
    
    Args:
        query: Cypher query to validate
        driver: Optional Neo4j driver
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        Tuple of (svr_score, is_valid, error_message)
    """
    should_close_driver = False
    if driver is None:
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            return 0.0, False, "No database connection provided"
        try:
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            should_close_driver = True
        except Exception as e:
            return 0.0, False, f"Connection failed: {e}"
    
    try:
        with driver.session() as session:
            explain_query = f"EXPLAIN {query}"
            session.run(explain_query).consume()
            return 1.0, True, None
            
    except CypherSyntaxError as e:
        error_msg = str(e.message) if hasattr(e, 'message') else str(e)
        logger.debug(f"Syntax error in query: {error_msg}")
        return 0.0, False, f"Syntax error: {error_msg}"
        
    except Neo4jError as e:
        error_msg = str(e.message) if hasattr(e, 'message') else str(e)
        logger.debug(f"Neo4j error: {error_msg}")
        return 0.0, False, f"Neo4j error: {error_msg}"
        
    except Exception as e:
        logger.error(f"Unexpected error validating query: {e}")
        return 0.0, False, f"Unexpected error: {e}"
        
    finally:
        if should_close_driver and driver:
            driver.close()


def calculate_svr(
    output: str,
    output_type: OutputType,
    driver=None,
    **kwargs,
) -> Tuple[float, bool, Optional[str]]:
    """
    Calculate Syntactic Validity Rate based on output type.

    Dispatches to ``calculate_svr_cypher`` (DB) or ``calculate_svr_json`` (pure).
    """
    from src.eval.scoring import calculate_svr_json

    if output_type == OutputType.CYPHER:
        return calculate_svr_cypher(output, driver=driver, **kwargs)
    svr, is_valid, error, _ = calculate_svr_json(output)
    return svr, is_valid, error


# =============================================================================
# EA: Cypher execution (requires Neo4j driver)
# =============================================================================

def execute_and_get_ids(
    query: str,
    driver,
    id_property: str = "GlobalId",
) -> Tuple[Set[str], Optional[str]]:
    """
    Execute a Cypher query and extract node IDs from results.
    
    This function is robust to different RETURN formats:
    - RETURN n (full node)
    - RETURN n.Name, n.GlobalId (specific properties)
    - RETURN n.GlobalId AS id (aliased)
    
    Args:
        query: Cypher query to execute
        driver: Neo4j driver
        id_property: Property to use as ID (default: GlobalId)
        
    Returns:
        Tuple of (set_of_ids, error_message)
    """
    ids = set()
    
    try:
        with driver.session() as session:
            result = session.run(query)
            
            for record in result:
                for value in record.values():
                    extracted_id = _extract_global_id(value, id_property)
                    if extracted_id:
                        ids.add(extracted_id)
                        
    except Exception as e:
        logger.warning(f"Cypher execution failed: {e} | Query: {query[:200]}")
        return set(), str(e)

    return ids, None


def calculate_ea_cypher(
    generated_query: str,
    gold_ids: Set[str],
    driver,
    id_property: str = "GlobalId",
) -> Tuple[float, Set[str], Optional[str]]:
    """
    Calculate Execution Accuracy for Cypher queries.
    
    Executes the generated query and compares results to gold IDs.
    
    Args:
        generated_query: Generated Cypher query
        gold_ids: Set of GlobalIds from gold standard
        driver: Neo4j driver
        id_property: Property to use as ID
        
    Returns:
        Tuple of (ea_score, generated_ids, error_message)
    """
    generated_ids, error = execute_and_get_ids(generated_query, driver, id_property)
    if error:
        return 0.0, set(), f"Query execution error: {error}"
    
    ea = calculate_ea_from_ids(generated_ids, gold_ids)
    
    logger.debug(
        f"EA (Cypher): |generated|={len(generated_ids)}, "
        f"|gold|={len(gold_ids)}, EA={ea:.3f}"
    )
    
    return ea, generated_ids, None


def calculate_ea(
    output: str,
    output_type: OutputType,
    gold_ids: Set[str],
    driver=None,
    id_property: str = "GlobalId",
) -> Tuple[float, Set[str], Optional[str]]:
    """
    Calculate Execution Accuracy based on output type.
    
    Args:
        output: Generated output (Cypher query or JSON list)
        output_type: Type of output
        gold_ids: Set of GlobalIds from gold standard
        driver: Neo4j driver (required for Cypher)
        id_property: Property to use as ID
        
    Returns:
        Tuple of (ea_score, generated_ids, error_message)
    """
    if output_type == OutputType.CYPHER:
        if driver is None:
            return 0.0, set(), "Neo4j driver required for Cypher execution"
        return calculate_ea_cypher(output, gold_ids, driver, id_property)
    else:
        return calculate_ea_direct(output, gold_ids)


# =============================================================================
# Gold standard preprocessing
# =============================================================================

def execute_gold_queries(
    test_cases: List[Dict[str, str]],
    driver,
    id_property: str = "GlobalId",
) -> Dict[int, Set[str]]:
    """
    Pre-execute all gold Cypher queries to establish gold ID sets.
    
    This is CRUCIAL for the experiment: we need the true result sets
    from executing the gold queries to compare against both:
    - Direct QA outputs
    - Generated Cypher query results
    
    Args:
        test_cases: List of test cases with 'gold_cypher' field
        driver: Neo4j driver
        id_property: Property to use as ID
        
    Returns:
        Dict mapping test case index to set of gold IDs
    """
    gold_results = {}
    
    logger.info(f"Pre-executing {len(test_cases)} gold queries...")
    
    for i, case in enumerate(test_cases):
        gold_cypher = case.get("gold_cypher", "")
        
        if not gold_cypher:
            logger.warning(f"Test case {i}: No gold Cypher provided")
            gold_results[i] = set()
            continue
        
        ids, error = execute_and_get_ids(gold_cypher, driver, id_property)
        
        if error:
            logger.warning(f"Test case {i}: Gold query failed - {error}")
            gold_results[i] = set()
        else:
            gold_results[i] = ids
            logger.debug(f"Test case {i}: Gold query returned {len(ids)} IDs")
    
    logger.info(f"Gold query execution complete. "
                f"Total IDs: {sum(len(ids) for ids in gold_results.values())}")
    
    return gold_results

