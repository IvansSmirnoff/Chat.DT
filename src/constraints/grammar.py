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

# ---------------------------------------------------------------------------
# EVERY quantifier below is bounded. An unbounded `*` or `+` is an FSM
# self-loop: the decoder can stay in one state indefinitely and burn the whole
# max_new_tokens budget on a single token class, returning a truncated query
# that neither full-matches this regex nor executes. All five original loops
# (WS, WS_REQ, VARIABLE, STRING_LITERAL, NUMBER_LITERAL) are closed here.
# ---------------------------------------------------------------------------

# Whitespace patterns
WS = r"[ \t]{0,2}"       # Optional horizontal whitespace
WS_REQ = r"[ \t]{1,2}"   # Required horizontal whitespace
NL = r"[\n\r]{0,2}"      # Optional newlines
WS_ANY = r"\s{0,2}"      # Any whitespace including newlines

# Cypher literals. The 60-char string cap is comfortably above the longest
# real value in the B15 model ("LABORATORIO DIDATTICO/ATTREZZATURA LEGGERA",
# 42 chars) while closing the `'[^']*'` loop.
STRING_LITERAL = r"'[^']{0,60}'"
NUMBER_LITERAL = r"-?[0-9]{1,9}(?:\.[0-9]{1,4})?"
BOOLEAN_LITERAL = r"(?:true|false|TRUE|FALSE)"
NULL_LITERAL = r"(?:null|NULL)"

# ONE bounded identifier for both MATCH *bindings* and later *references*.
#
# The previous split — unbounded VARIABLE for bindings, single-letter
# REF_VARIABLE for references — had two defects. (a) VARIABLE was a self-loop
# trap. (b) The two patterns disagreed, so `MATCH (space:IfcSpace) WHERE s.X`
# full-matched the regex and then failed in Neo4j with "variable s not
# defined". A single 1-3 char token closes the loop, forces bind and reference
# to draw from the same language, and admits the two-letter bindings the gold
# queries use (`st` for storey).
IDENT = r"[a-z][a-z0-9]{0,2}"

# Bounded identifier for AS-alias names. Aliases bind a fresh name (not a
# reference back into MATCH), so they need >1 char, but unbounded length is
# an FSM self-loop trap: model emitted `door_countMATCHrelsrelsrels…`×200 as
# a single alias token. 20-char cap closes the trap while keeping aliases
# readable.
ALIAS_NAME = r"[a-zA-Z_][a-zA-Z0-9_]{0,19}"

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

# Last-resort relationship set, used ONLY when no vocabulary is available.
# Real runs must pass the graph's own relationship types (see
# build_cypher_regex_from_vocabulary) — a hardcoded list silently makes any
# other relationship, e.g. BOUNDED_BY, undecodable.
DEFAULT_RELATIONSHIPS = {"CONTAINS", "DECOMPOSES", "HAS_MATERIAL"}


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
        allow_distinct: Whether to allow RETURN DISTINCT and count(DISTINCT x)
        allow_pattern_predicates: Whether to allow `(a)-[:REL]->(:Label)` and its
            negation as a WHERE condition (existence / non-existence of a path)
        force_return_node: Whether to force returning the node (for GlobalId extraction)
        case_insensitive_keywords: Use case-insensitive Cypher keywords
    """
    entities: Set[str]
    properties: Set[str]
    relationships: Set[str] = field(default_factory=lambda: set(DEFAULT_RELATIONSHIPS))
    max_where_clauses: int = 4  # Increased to support complex queries like test_set
    max_return_items: int = 8   # Columns allowed in RETURN (dup-column guard)
    allow_aggregations: bool = True
    allow_order_by: bool = True   # Needed for "tallest/smallest/largest" gold patterns
    allow_limit: bool = True      # Needed for ORDER BY ... DESC LIMIT 1 pattern
    allow_relationships: bool = True
    allow_distinct: bool = True   # Needed for DISTINCT categories / count(DISTINCT n)
    allow_pattern_predicates: bool = True  # Needed for "offices with no window"
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
    rel_alt = (
        _build_alternation(config.relationships)
        if config.relationships
        else _build_alternation(DEFAULT_RELATIONSHIPS)
    )

    # Build value pattern (string, number, boolean)
    value_pattern = f"(?:{STRING_LITERAL}|{NUMBER_LITERAL}|{BOOLEAN_LITERAL})"

    # Node pattern. All three forms Cypher allows are legal here:
    #   (s:IfcSpace)  bound + labelled
    #   (e)           bound, unlabelled  — "how many elements bound room 021"
    #   (:IfcWindow)  anonymous, labelled — the target of a pattern predicate
    node_pattern = f"\\((?:{IDENT}:{entity_alt}|{IDENT}|:{entity_alt})\\)"

    # Directed relationship hop, e.g. -[:BOUNDED_BY]->
    rel_pattern = f"-\\[:{rel_alt}\\]->"

    # Variables are a single bounded token (IDENT) in bindings AND references,
    # so a name introduced in MATCH is drawn from the same language as its
    # later uses in WHERE / RETURN / ORDER BY.
    # Build property access pattern: n.Property or toInteger(n.Property)
    property_access = f"(?:{CONVERSION_FUNCTIONS}\\({WS}{IDENT}\\.{property_alt}{WS}\\)|{IDENT}\\.{property_alt})"

    # Build different condition types:
    # 1. Basic comparison: n.Property = 'value' or n.Property > 100 or toInteger(n.Property) >= 20
    basic_comparison = f"{property_access}{WS}{COMPARISON_OP_BASIC}{WS}{value_pattern}"

    # 2. String CONTAINS: n.Property CONTAINS 'value' or m.Name CONTAINS 'Concrete'
    string_contains = f"{IDENT}\\.{property_alt}{WS_REQ}{STRING_OP}{WS_REQ}{STRING_LITERAL}"

    # 3. IS NULL / IS NOT NULL: n.Property IS NOT NULL
    is_null_check = f"{IDENT}\\.{property_alt}{WS_REQ}{_keyword('IS', ci)}(?:{WS_REQ}{_keyword('NOT', ci)})?{WS_REQ}{_keyword('NULL', ci)}"

    # 4. NOT with string contains: NOT n.Property CONTAINS 'value' or NOT m.Name CONTAINS 'Concrete'
    not_string_contains = f"{_keyword('NOT', ci)}{WS_REQ}{IDENT}\\.{property_alt}{WS_REQ}{STRING_OP}{WS_REQ}{STRING_LITERAL}"

    # 5. Pattern predicate: (s)-[:BOUNDED_BY]->(:IfcWindow), optionally negated.
    #    This is existence / non-existence of a path, and it is the one
    #    construct that cannot be answered from a serialized property dump:
    #    "which offices have no window" is an absence in the topology, not a
    #    property value.
    #
    #    It is deliberately NOT one of the alternatives inside the repeated
    #    condition list, but the reason is expressiveness, not FSM size.
    #    max_where_clauses is pinned to 1 (see the profile note in
    #    build_cypher_regex_from_vocabulary), and a single condition slot
    #    cannot hold both a comparison and a predicate. Hoisting is what lets
    #    "offices with no window" keep
    #        WHERE s.Cat = 'UFFICI' AND NOT (s)-[:BOUNDED_BY]->(:IfcWindow)
    #    under that cap. At most one predicate; no gold query needs two.
    conditions = [not_string_contains, basic_comparison, string_contains, is_null_check]

    # Combined single condition (property-level comparisons only)
    single_condition = f"(?:{'|'.join(conditions)})"

    pattern_predicate = None
    if config.allow_pattern_predicates and config.allow_relationships:
        # The target must be ANONYMOUS. Cypher rejects a pattern predicate that
        # introduces a new variable:
        #   WHERE (w)-[:HAS_OPENING]->(d:IfcDoor)
        #   -> "PatternExpressions are not allowed to introduce new variables: 'd'"
        # The decoder produced exactly that on the first Building 15 run, so the
        # general node_pattern (which permits `(d:IfcDoor)`) cannot be reused here.
        anon_node = f"\\((?::{entity_alt})?\\)"
        pattern_predicate = (
            f"(?:{_keyword('NOT', ci)}{WS_REQ})?"
            f"\\({IDENT}\\){WS}{rel_pattern}{WS}{anon_node}"
        )

    # DISTINCT is legal in two places: after RETURN, and as the first token
    # inside an aggregation — count(DISTINCT s).
    distinct_kw = f"(?:{_keyword('DISTINCT', ci)}{WS_REQ})?" if config.allow_distinct else ""

    # Build WHERE clause with multiple conditions connected by AND/OR
    # Support up to max_where_clauses conditions
    if config.max_where_clauses > 1:
        additional_conditions = (
            f"(?:{WS_REQ}{LOGICAL_OP}{WS_REQ}{single_condition})"
            f"{{0,{config.max_where_clauses - 1}}}"
        )
        comparison_body = f"{single_condition}{additional_conditions}"
    else:
        comparison_body = single_condition

    if pattern_predicate is not None:
        # Either a comparison chain optionally closed by ONE pattern predicate
        # (`WHERE s.Cat = 'UFFICI' AND NOT (s)-[:BOUNDED_BY]->(:IfcWindow)`),
        # or a bare pattern predicate. Two copies of the sub-pattern, neither
        # of them inside a repetition.
        where_body = (
            f"(?:{comparison_body}"
            f"(?:{WS_REQ}{LOGICAL_OP}{WS_REQ}{pattern_predicate})?"
            f"|{pattern_predicate})"
        )
    else:
        where_body = comparison_body

    where_clause = f"(?:{WS_REQ}{_keyword('WHERE', ci)}{WS_REQ}{where_body})?"


    # Build RETURN clause.
    # Variable references use IDENT, the same bounded token MATCH binds with.
    # AS aliases *bind* a fresh name, so they use the wider ALIAS_NAME.
    if config.force_return_node:
        return_clause = f"{WS_REQ}{_keyword('RETURN', ci)}{WS_REQ}{IDENT}"
    else:
        return_item = f"(?:{IDENT}(?:\\.{property_alt})?)"
        if config.allow_aggregations:
            agg_item = (
                f"(?:{AGG_FUNCTIONS}\\({WS}{distinct_kw}"
                f"(?:\\*|{IDENT}(?:\\.{property_alt})?){WS}\\))"
            )
            return_item = f"(?:{return_item}|{agg_item})"
        alias_pattern = f"(?:{WS_REQ}{_keyword('AS', ci)}{WS_REQ}{ALIAS_NAME})?"
        # Cap the RETURN list. Unbounded `*` let the model spam
        # `n.Window_Board_Extension`×24 in q10, hitting Cypher's
        # duplicate-column-name error.
        #
        # This cap is also an FSM *size* knob, not just a sanity guard: each
        # extra permitted column is another copy of the return-item sub-NFA in
        # the determinized automaton, so the index cost grows with it.
        max_extra_return_items = max(0, config.max_return_items - 1)
        return_items = (
            f"{return_item}{alias_pattern}"
            f"(?:{WS},{WS}{return_item}{alias_pattern}){{0,{max_extra_return_items}}}"
        )
        return_clause = (
            f"{WS_REQ}{_keyword('RETURN', ci)}{WS_REQ}{distinct_kw}{return_items}"
        )

    # Build ORDER BY clause. References use IDENT for the same reason.
    order_clause = ""
    if config.allow_order_by:
        order_direction = f"(?:{WS_REQ}(?:{_keyword('ASC', ci)}|{_keyword('DESC', ci)}))?"
        order_item = f"{IDENT}(?:\\.{property_alt})?{order_direction}"
        order_clause = f"(?:{WS_REQ}{_keyword('ORDER', ci)}{WS_REQ}{_keyword('BY', ci)}{WS_REQ}{order_item})?"

    # Build LIMIT clause (optional) - DISABLED by default
    limit_clause = ""
    if config.allow_limit:
        limit_clause = f"(?:{WS_REQ}{_keyword('LIMIT', ci)}{WS_REQ}[1-9][0-9]{{0,3}})?"

    # Optional second node reached by one directed hop
    rel_clause = f"(?:{WS}{rel_pattern}{WS}{node_pattern})?" if config.allow_relationships else ""

    # Assemble the full pattern.
    #
    # No end-of-string anchor is appended: Outlines derives EOS from the FSM's
    # accepting states, and the state after RETURN (with the trailing ORDER BY
    # / LIMIT groups optional) is accepting, so EOS is permitted there. What
    # actually stops runaway generation is that every quantifier above is
    # bounded — see the note at the top of this module.
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


# =============================================================================
# Validation
# =============================================================================

def validate_cypher_against_regex(query: str, regex: str) -> bool:
    """Validate a Cypher query against a grammar regex (full-match).

    Deliberately case-SENSITIVE. Cypher keywords are already case-insensitive
    inside the pattern (see ``_keyword``), but node labels and property names
    are case-sensitive in Neo4j. Matching with ``re.IGNORECASE`` made this
    check accept `MATCH (n:ifcspace) RETURN n`, which the decoder can never
    emit and the database rejects — i.e. it scored syntactic validity higher
    than the run actually achieved.
    """
    try:
        pattern = f"^{regex}$"
        return bool(re.match(pattern, query.strip()))
    except re.error as e:
        logger.error(f"Invalid regex pattern: {e}")
        return False


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

    # Relationship types MUST come from the vocabulary. Letting CypherGrammar's
    # default win here made every relationship outside {CONTAINS, DECOMPOSES,
    # HAS_MATERIAL} undecodable — BOUNDED_BY among them, which is what the
    # space/window and space/door queries traverse.
    relationships = set(getattr(vocabulary, "all_relations", None) or ())
    if not relationships:
        relationships = set(DEFAULT_RELATIONSHIPS)
        logger.warning(
            "Vocabulary carries no relationship types; falling back to %s. "
            "Queries traversing any other relationship cannot be decoded.",
            sorted(relationships),
        )

    if config is None:
        # FSM-safe profile. max_where_clauses is the load-bearing cap: it is
        # what keeps outlines-core from raising "Failed to build DFA number of
        # DFA states exceeds limit of 2147483647". Measured on Building 15
        # (230 properties, 76 labels, 8 relationship types) against the
        # Qwen2.5-14B-Instruct vocabulary (151,652 tokens):
        #
        #   where=4, ret=8, predicate inside  -> FAIL (~2 min)
        #   where=4, ret=8, no predicate      -> FAIL (~4 min)
        #   where=4, ret=3, predicate inside  -> FAIL (~2 min)
        #   where=4, ret=8, predicate hoisted -> FAIL (~11 min)
        #   where=2, ret=8, predicate inside  -> OK  67,244 states, 1.6 GB
        #   where=2, ret=3, predicate inside  -> OK  41,424 states, 3.0 GB
        #   where=1, ret=3, predicate hoisted -> OK  31,892 states, 868 MB
        #
        # where=4 fails in every combination tried, including with pattern
        # predicates removed entirely; where<=2 succeeds even with them inside
        # the repetition. So the driver is the repeated condition list, not the
        # predicate. where=1 is chosen over where=2 for the much smaller
        # footprint, and costs nothing: no gold query uses two comparisons.
        # ret=3 matches the widest gold RETURN (s.Numero, s.Nome, s.Livello).
        config = CypherGrammar(
            entities=entities,
            properties=properties,
            relationships=relationships,
            max_where_clauses=1,
            max_return_items=3,
        )
    elif not config.relationships:
        config.relationships = relationships

    logger.info(
        f"Building regex from combined vocabulary: "
        f"{len(entities)} entities, {len(properties)} properties, "
        f"{len(config.relationships)} relationship types"
    )

    return build_cypher_regex(entities, properties, config)


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
