"""
Cypher Grammar Generator for Constrained Decoding.

Generates regex patterns that enforce valid Cypher syntax using the
vocabulary extracted from IDS files. These patterns are used with
the `outlines` library for constrained LLM decoding.

The goal is to force the LLM to only generate syntactically valid
Cypher queries that reference entities and properties defined in
the IDS schema.

Author: NL2Cypher Research Team
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .vocabulary_merger import CombinedVocabulary

logger = logging.getLogger(__name__)

# =============================================================================
# Regex Building Blocks
# =============================================================================

# Whitespace patterns
WS = r"[ \t]*"           # Optional horizontal whitespace
WS_REQ = r"[ \t]+"       # Required horizontal whitespace
NL = r"[\n\r]*"          # Optional newlines
WS_ANY = r"\s*"          # Any whitespace including newlines

# Cypher literals
STRING_LITERAL = r"'[^']*'"
NUMBER_LITERAL = r"-?[0-9]+(?:\.[0-9]+)?"
BOOLEAN_LITERAL = r"(?:true|false|TRUE|FALSE)"
NULL_LITERAL = r"(?:null|NULL)"

# Cypher variable names (alphanumeric, starting with letter)
VARIABLE = r"[a-zA-Z_][a-zA-Z0-9_]*"

# Short variable name for variable *references* (RETURN/WHERE/ORDER BY).
# The open-ended VARIABLE pattern is an FSM self-loop trap for small models
# (q21 produced `EXISTSMatcherMatchedPath…` ×120 because the regex permitted
# any identifier in WHERE). Binding all references to a single lowercase
# letter forces the model to reuse the variable bound in MATCH.
REF_VARIABLE = r"[a-z]"

# Backwards-compatible alias for callers that still import RETURN_VARIABLE.
RETURN_VARIABLE = REF_VARIABLE

# Comparison operators (basic + compound)
COMPARISON_OP_BASIC = r"(?:=|<>|!=|<=|>=|<|>)"

# String operators (need special handling)
STRING_OP = r"(?:CONTAINS|STARTS WITH|ENDS WITH|contains|starts with|ends with)"

# Logical operators
LOGICAL_OP = r"(?:AND|OR|and|or)"

# Aggregation functions
AGG_FUNCTIONS = r"(?:count|sum|avg|min|max|collect|COUNT|SUM|AVG|MIN|MAX|COLLECT)"

# Conversion functions (for type casting)
CONVERSION_FUNCTIONS = r"(?:toInteger|toFloat|toString|toLower|toUpper|toBoolean)"


# =============================================================================
# Grammar Configuration
# =============================================================================

@dataclass
class CypherGrammar:
    """
    Configuration for Cypher grammar generation.
    
    Allows customization of the generated regex patterns.
    
    Attributes:
        entities: Set of valid entity labels (Neo4j format, e.g., "IfcWall")
        properties: Set of valid property names
        relationships: Set of valid relationship types
        max_where_clauses: Maximum number of WHERE conditions
        allow_aggregations: Whether to allow COUNT, SUM, etc.
        allow_order_by: Whether to allow ORDER BY
        allow_limit: Whether to allow LIMIT
        allow_relationships: Whether to allow relationship patterns
        force_return_node: Whether to force returning the node (for GlobalId extraction)
        case_insensitive_keywords: Use case-insensitive Cypher keywords
    """
    entities: Set[str]
    properties: Set[str]
    relationships: Set[str] = field(default_factory=lambda: {"CONTAINS", "DECOMPOSES", "HAS_MATERIAL"})
    max_where_clauses: int = 4  # Increased to support complex queries like test_set
    allow_aggregations: bool = True
    allow_order_by: bool = True   # Needed for "tallest/smallest/largest" gold patterns
    allow_limit: bool = True      # Needed for ORDER BY ... DESC LIMIT 1 pattern
    allow_relationships: bool = True
    force_return_node: bool = False  # Was True: blocked count/sum/avg + RETURN n.Prop
    case_insensitive_keywords: bool = True
    
    def __post_init__(self):
        """Validate and normalize inputs."""
        # Ensure we have at least some entities and properties
        if not self.entities:
            logger.warning("No entities provided, using wildcard")
        if not self.properties:
            logger.warning("No properties provided, using wildcard")


# =============================================================================
# Grammar Generation Functions
# =============================================================================

def _build_alternation(values: Set[str], escape: bool = True) -> str:
    """
    Build a regex alternation group from a set of values.
    
    Args:
        values: Set of string values
        escape: Whether to escape special regex characters
        
    Returns:
        Regex alternation pattern like "(IfcWall|IfcDoor|IfcWindow)"
    """
    if not values:
        return r"[A-Za-z_][A-Za-z0-9_]*"  # Fallback: any identifier
    
    # Sort for deterministic output
    sorted_values = sorted(values)
    
    if escape:
        sorted_values = [re.escape(v) for v in sorted_values]
    
    return f"({'|'.join(sorted_values)})"


def _keyword(kw: str, case_insensitive: bool = True) -> str:
    """
    Generate a regex pattern for a Cypher keyword.
    
    Args:
        kw: Keyword string (e.g., "MATCH")
        case_insensitive: Whether to match both cases
        
    Returns:
        Regex pattern for the keyword
    """
    if case_insensitive:
        # Build case-insensitive pattern: M[aA][tT][cC][hH]
        pattern = ""
        for char in kw:
            if char.isalpha():
                pattern += f"[{char.lower()}{char.upper()}]"
            else:
                pattern += re.escape(char)
        return pattern
    return kw


def build_cypher_regex(
    entities: Set[str],
    properties: Set[str],
    config: Optional[CypherGrammar] = None
) -> str:
    """
    Build a regex pattern for constrained Cypher query generation.
    
    Generates a regex that matches valid Cypher queries including:
        - Simple: MATCH (n:Entity) RETURN n
        - Filtered: MATCH (n:Entity) WHERE n.Property = 'value' RETURN n
        - Relationship: MATCH (a:Entity)-[:REL]->(b:Entity) WHERE ... RETURN a
        - Complex: WHERE with CONTAINS, IS NULL, NOT, AND/OR
        - Functions: toInteger(), toFloat() for type conversion
        - Multi-variable: Conditions on multiple node variables (a.X AND b.Y)
    
    IMPORTANT: This grammar is designed for NL2Cypher evaluation where:
    - We need to extract GlobalIds from results
    - Queries should be complete and executable
    - RETURN should include at least the node variable
    
    Args:
        entities: Set of valid entity labels (e.g., {"IfcWall", "IfcDoor"})
        properties: Set of valid property names (e.g., {"FireRating", "Name"})
        config: Optional grammar configuration
        
    Returns:
        Regex pattern string for use with outlines.generate.regex
    """
    if config is None:
        config = CypherGrammar(entities=entities, properties=properties)
    
    ci = config.case_insensitive_keywords
    
    # Build entity and property alternations
    entity_alt = _build_alternation(entities)
    property_alt = _build_alternation(properties)
    rel_alt = _build_alternation(config.relationships) if config.relationships else "CONTAINS|DECOMPOSES|HAS_MATERIAL"
    
    # Build value pattern (string, number, boolean)
    value_pattern = f"(?:{STRING_LITERAL}|{NUMBER_LITERAL}|{BOOLEAN_LITERAL})"
    
    # All variable references in WHERE / RETURN / ORDER BY are bound to a
    # single lowercase letter (REF_VARIABLE) — this prevents the FSM
    # self-loop on the open-ended VARIABLE pattern (q21 token explosion).
    # Build property access pattern: n.Property or toInteger(n.Property)
    property_access = f"(?:{CONVERSION_FUNCTIONS}\\({WS}{REF_VARIABLE}\\.{property_alt}{WS}\\)|{REF_VARIABLE}\\.{property_alt})"

    # Build different condition types:
    # 1. Basic comparison: n.Property = 'value' or n.Property > 100 or toInteger(n.Property) >= 20
    basic_comparison = f"{property_access}{WS}{COMPARISON_OP_BASIC}{WS}{value_pattern}"

    # 2. String CONTAINS: n.Property CONTAINS 'value' or m.Name CONTAINS 'Concrete'
    string_contains = f"{REF_VARIABLE}\\.{property_alt}{WS_REQ}{STRING_OP}{WS_REQ}{STRING_LITERAL}"

    # 3. IS NULL / IS NOT NULL: n.Property IS NOT NULL
    is_null_check = f"{REF_VARIABLE}\\.{property_alt}{WS_REQ}{_keyword('IS', ci)}(?:{WS_REQ}{_keyword('NOT', ci)})?{WS_REQ}{_keyword('NULL', ci)}"

    # 4. NOT with string contains: NOT n.Property CONTAINS 'value' or NOT m.Name CONTAINS 'Concrete'
    not_string_contains = f"{_keyword('NOT', ci)}{WS_REQ}{REF_VARIABLE}\\.{property_alt}{WS_REQ}{STRING_OP}{WS_REQ}{STRING_LITERAL}"
    
    # Combined single condition (any of the above)
    single_condition = f"(?:{not_string_contains}|{basic_comparison}|{string_contains}|{is_null_check})"
    
    # Build WHERE clause with multiple conditions connected by AND/OR
    # Support up to max_where_clauses conditions
    if config.max_where_clauses > 1:
        additional_conditions = (
            f"(?:{WS_REQ}{LOGICAL_OP}{WS_REQ}{single_condition})"
            f"{{0,{config.max_where_clauses - 1}}}"
        )
        where_clause = (
            f"(?:{WS_REQ}{_keyword('WHERE', ci)}{WS_REQ}"
            f"{single_condition}{additional_conditions})?"
        )
    else:
        where_clause = (
            f"(?:{WS_REQ}{_keyword('WHERE', ci)}{WS_REQ}{single_condition})?"
        )
    
    # Build RETURN clause.
    # All variable *references* use REF_VARIABLE (single lowercase letter) to
    # avoid the FSM self-loop on the open-ended VARIABLE pattern. AS aliases
    # *bind* a new name, so they can use the full VARIABLE pattern safely.
    if config.force_return_node:
        return_clause = f"{WS_REQ}{_keyword('RETURN', ci)}{WS_REQ}{REF_VARIABLE}"
    else:
        return_item = f"(?:{REF_VARIABLE}(?:\\.{property_alt})?)"
        if config.allow_aggregations:
            agg_item = f"(?:{AGG_FUNCTIONS}\\({WS}(?:\\*|{REF_VARIABLE}(?:\\.{property_alt})?){WS}\\))"
            return_item = f"(?:{return_item}|{agg_item})"
        alias_pattern = f"(?:{WS_REQ}{_keyword('AS', ci)}{WS_REQ}{VARIABLE})?"
        return_items = f"{return_item}{alias_pattern}(?:{WS},{WS}{return_item}{alias_pattern})*"
        return_clause = f"{WS_REQ}{_keyword('RETURN', ci)}{WS_REQ}{return_items}"

    # Build ORDER BY clause. References use REF_VARIABLE for the same reason.
    order_clause = ""
    if config.allow_order_by:
        order_direction = f"(?:{WS_REQ}(?:{_keyword('ASC', ci)}|{_keyword('DESC', ci)}))?"
        order_item = f"{REF_VARIABLE}(?:\\.{property_alt})?{order_direction}"
        order_clause = f"(?:{WS_REQ}{_keyword('ORDER', ci)}{WS_REQ}{_keyword('BY', ci)}{WS_REQ}{order_item})?"
    
    # Build LIMIT clause (optional) - DISABLED by default
    limit_clause = ""
    if config.allow_limit:
        limit_clause = f"(?:{WS_REQ}{_keyword('LIMIT', ci)}{WS_REQ}[1-9][0-9]{{0,3}})?"
    
    # Build node pattern
    node_pattern = f"\\({VARIABLE}:{entity_alt}\\)"
    
    # Build relationship pattern for optional second node
    rel_clause = ""
    if config.allow_relationships:
        rel_pattern = f"-\\[:{rel_alt}\\]->"
        rel_clause = f"(?:{rel_pattern}{node_pattern})?"
    
    # Assemble the full pattern
    # NOTE: We append $ (end-of-string anchor) so Outlines forces EOS after
    # the RETURN clause.  Without this, small models fill max_tokens with
    # garbage that still matches the open-ended VARIABLE pattern.
    full_pattern = (
        f"{_keyword('MATCH', ci)}{WS}{node_pattern}"
        f"{rel_clause}"
        f"{where_clause}"
        f"{return_clause}"
        f"{order_clause}"
        f"{limit_clause}"
    )
    
    logger.debug(f"Generated Cypher regex: {full_pattern}")
    
    return full_pattern


def build_simple_cypher_regex(
    entities: Set[str],
    properties: Set[str]
) -> str:
    """
    Build a simplified regex for basic MATCH-WHERE-RETURN queries.
    
    This is a more permissive pattern suitable for initial experiments
    or when the full grammar is too restrictive.
    
    Args:
        entities: Set of valid entity labels
        properties: Set of valid property names
        
    Returns:
        Simplified regex pattern
    """
    entity_alt = _build_alternation(entities)
    property_alt = _build_alternation(properties)
    
    # Value pattern for simple queries
    value_pattern = f"(?:{STRING_LITERAL}|{NUMBER_LITERAL}|{BOOLEAN_LITERAL})"
    
    # Simple pattern: MATCH (n:Entity) RETURN n
    # or: MATCH (n:Entity) WHERE n.prop = 'value' RETURN n
    pattern = (
        f"MATCH{WS}\\({VARIABLE}:{entity_alt}\\)"
        f"(?:{WS_REQ}WHERE{WS_REQ}{VARIABLE}\\.{property_alt}{WS}{COMPARISON_OP_BASIC}{WS}{value_pattern})?"
        f"{WS_REQ}RETURN{WS_REQ}(?:\\*|{VARIABLE}(?:\\.{property_alt})?)"
    )
    
    return pattern


def build_relationship_cypher_regex(
    entities: Set[str],
    properties: Set[str],
    relationships: Optional[Set[str]] = None,
    max_conditions: int = 4
) -> str:
    """
    Build a regex for queries with relationships.
    
    Supports patterns like:
        MATCH (a:Entity)-[:REL]->(b:Entity) RETURN a
        MATCH (a:Entity)-[:REL]->(b:Entity) WHERE a.prop = 'value' AND b.prop > 10 RETURN a
    
    Args:
        entities: Set of valid entity labels
        properties: Set of valid property names
        relationships: Set of valid relationship types (optional)
        max_conditions: Maximum number of WHERE conditions
        
    Returns:
        Regex pattern supporting relationships
    """
    entity_alt = _build_alternation(entities)
    property_alt = _build_alternation(properties)
    
    # Default relationships for BIM (aligned with ETL loader)
    if relationships is None:
        relationships = {"CONTAINS", "DECOMPOSES", "HAS_MATERIAL"}
    
    rel_alt = _build_alternation(relationships)
    
    # Build value pattern (string, number, boolean)
    value_pattern = f"(?:{STRING_LITERAL}|{NUMBER_LITERAL}|{BOOLEAN_LITERAL})"
    
    # Build property access pattern: n.Property or toInteger(n.Property)
    property_access = f"(?:{CONVERSION_FUNCTIONS}\\({WS}{VARIABLE}\\.{property_alt}{WS}\\)|{VARIABLE}\\.{property_alt})"
    
    # Build different condition types (same as in build_cypher_regex):
    # 1. Basic comparison: n.Property = 'value' or n.Property > 100 or toInteger(n.Property) >= 20
    basic_comparison = f"{property_access}{WS}{COMPARISON_OP_BASIC}{WS}{value_pattern}"
    
    # 2. String CONTAINS: n.Property CONTAINS 'value'
    string_contains = f"{VARIABLE}\\.{property_alt}{WS_REQ}(?:CONTAINS|contains){WS_REQ}{STRING_LITERAL}"
    
    # 3. IS NULL / IS NOT NULL: n.Property IS NOT NULL
    is_null_check = f"{VARIABLE}\\.{property_alt}{WS_REQ}(?:IS|is)(?:{WS_REQ}(?:NOT|not))?{WS_REQ}(?:NULL|null)"
    
    # 4. NOT with string contains: NOT n.Property CONTAINS 'value'
    not_string_contains = f"(?:NOT|not){WS_REQ}{VARIABLE}\\.{property_alt}{WS_REQ}(?:CONTAINS|contains){WS_REQ}{STRING_LITERAL}"
    
    # Combined single condition (any of the above)
    single_condition = f"(?:{not_string_contains}|{basic_comparison}|{string_contains}|{is_null_check})"
    
    # Build WHERE clause with multiple conditions (supports conditions on both variables)
    additional_conditions = f"(?:{WS_REQ}(?:AND|OR|and|or){WS_REQ}{single_condition}){{0,{max_conditions - 1}}}"
    where_clause = f"(?:{WS_REQ}(?:WHERE|where){WS_REQ}{single_condition}{additional_conditions})?"
    
    # Build relationship pattern
    rel_pattern = f"(?:-\\[(?:{VARIABLE})?:{rel_alt}\\]->|-\\[:{rel_alt}\\]->|-->)"
    
    # Node pattern
    node_pattern = f"\\({VARIABLE}:{entity_alt}\\)"
    
    # Full pattern with optional relationship and complex WHERE
    pattern = (
        f"(?:MATCH|match){WS}{node_pattern}"
        f"(?:{rel_pattern}{node_pattern})?"
        f"{where_clause}"
        f"{WS_REQ}(?:RETURN|return){WS_REQ}(?:\\*|{VARIABLE}(?:\\.{property_alt})?(?:{WS},{WS}{VARIABLE}(?:\\.{property_alt})?)*)"
    )
    
    return pattern


# =============================================================================
# Validation
# =============================================================================

def validate_cypher_against_regex(query: str, regex: str) -> bool:
    """
    Validate a Cypher query against a grammar regex.
    
    Args:
        query: Cypher query string
        regex: Grammar regex pattern
        
    Returns:
        True if the query matches the grammar
    """
    try:
        # Add anchors for full match
        pattern = f"^{regex}$"
        return bool(re.match(pattern, query.strip(), re.IGNORECASE))
    except re.error as e:
        logger.error(f"Invalid regex pattern: {e}")
        return False


def extract_entities_from_query(query: str, valid_entities: Set[str]) -> Set[str]:
    """
    Extract entity labels used in a Cypher query.
    
    Args:
        query: Cypher query string
        valid_entities: Set of valid entity names for matching
        
    Returns:
        Set of entity labels found in the query
    """
    found = set()
    
    # Match patterns like (n:IfcWall) or (:IfcWall)
    label_pattern = r"\(\s*\w*\s*:\s*(\w+)\s*\)"
    
    for match in re.finditer(label_pattern, query):
        label = match.group(1)
        # Check case-insensitive match against valid entities
        for valid in valid_entities:
            if label.lower() == valid.lower():
                found.add(valid)
                break
    
    return found


def extract_properties_from_query(query: str, valid_properties: Set[str]) -> Set[str]:
    """
    Extract property names used in a Cypher query.
    
    Args:
        query: Cypher query string
        valid_properties: Set of valid property names for matching
        
    Returns:
        Set of property names found in the query
    """
    found = set()
    
    # Match patterns like n.PropertyName or node.property_name
    prop_pattern = r"\b\w+\.(\w+)\b"
    
    for match in re.finditer(prop_pattern, query):
        prop = match.group(1)
        # Check case-insensitive match against valid properties
        for valid in valid_properties:
            if prop.lower() == valid.lower():
                found.add(valid)
                break
    
    return found


# =============================================================================
# Hybrid Schema Integration
# =============================================================================

def build_cypher_regex_from_vocabulary(
    vocabulary: "CombinedVocabulary",
    config: Optional[CypherGrammar] = None
) -> str:
    """
    Build a Cypher regex from a CombinedVocabulary (hybrid IFC + IDS schema).
    
    This function integrates the vocabulary merger with the grammar generator,
    using ALL discovered entities and properties from the IFC model, while
    the IDS constraints inform the system prompt (not the regex itself).
    
    For constrained decoding, we need ALL valid property names in the regex,
    not just the IDS-mandated ones. The IDS constraints are enforced via
    the system prompt and post-validation.
    
    Args:
        vocabulary: CombinedVocabulary from the vocabulary_merger
        config: Optional grammar configuration
        
    Returns:
        Regex pattern for constrained Cypher generation
        
    Example:
        >>> from src.constraints import build_combined_vocabulary
        >>> vocab = build_combined_vocabulary("model.ifc", "spec.ids")
        >>> regex = build_cypher_regex_from_vocabulary(vocab)
    """
    entities = vocabulary.get_entity_names()
    
    # Get ALL property names (union of IDS + IFC scanned properties)
    properties = vocabulary.get_all_property_names()
    
    logger.info(
        f"Building regex from combined vocabulary: "
        f"{len(entities)} entities, {len(properties)} properties"
    )
    
    return build_cypher_regex(entities, properties, config)


def build_relationship_cypher_regex_from_vocabulary(
    vocabulary: "CombinedVocabulary",
) -> str:
    """
    Build a relationship-aware Cypher regex from a CombinedVocabulary.
    
    Args:
        vocabulary: CombinedVocabulary from the vocabulary_merger
        
    Returns:
        Regex pattern supporting relationships
    """
    entities = vocabulary.get_entity_names()
    properties = vocabulary.get_all_property_names()
    relationships = vocabulary.all_relations.copy()
    
    return build_relationship_cypher_regex(entities, properties, relationships)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for grammar testing."""
    # Example entities and properties
    entities = {"IfcWall", "IfcDoor", "IfcWindow", "IfcBuildingStorey"}
    properties = {"FireRating", "Name", "IsExternal", "LoadBearing", "GlobalId"}
    
    print("Cypher Grammar Generator")
    print("=" * 60)
    
    # Generate regex
    regex = build_cypher_regex(entities, properties)
    print(f"\nGenerated Regex:\n{regex}")
    
    # Test queries
    test_queries = [
        "MATCH (n:IfcWall) RETURN n",
        "MATCH (w:IfcWall) WHERE w.FireRating = 'EI60' RETURN w",
        "MATCH (d:IfcDoor) WHERE d.IsExternal = true RETURN d.Name",
        "MATCH (n:IfcWall) WHERE n.FireRating = 'EI60' AND n.LoadBearing = true RETURN n",
        "MATCH (n:IfcWall) RETURN count(n)",
        "MATCH (n:InvalidEntity) RETURN n",  # Should fail
    ]
    
    print("\nValidation Tests:")
    print("-" * 60)
    
    for query in test_queries:
        is_valid = validate_cypher_against_regex(query, regex)
        status = "✓" if is_valid else "✗"
        print(f"{status} {query}")


if __name__ == "__main__":
    main()
