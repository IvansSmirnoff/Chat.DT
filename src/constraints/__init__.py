"""
Constraints module for NL2Cypher.

Provides IDS parsing, IFC schema scanning, vocabulary merging,
and grammar generation for constrained LLM decoding.
"""

from .ids_parser import IDSParser, IDSSchema
from .grammar import (
    build_cypher_regex,
    build_cypher_regex_from_vocabulary,
    CypherGrammar,
)
from .schema_scanner import IFCSchemaScanner, IFCModelSchema, scan_ifc_schema
from .vocabulary_merger import (
    VocabularyMerger,
    CombinedVocabulary,
    PropertyVocabulary,
    EntityVocabulary,
    PropertyType,
    build_combined_vocabulary,
    get_system_schema_string,
)
from .context_builder import (
    ContextBuilder,
    PromptContext,
    build_system_prompt,
    build_full_prompt,
    DEFAULT_FEW_SHOT_EXAMPLES,
)

__all__ = [
    # IDS Parser
    "IDSParser",
    "IDSSchema",
    # IFC Schema Scanner
    "IFCSchemaScanner",
    "IFCModelSchema",
    "scan_ifc_schema",
    # Vocabulary Merger
    "VocabularyMerger",
    "CombinedVocabulary",
    "PropertyVocabulary",
    "EntityVocabulary",
    "PropertyType",
    "build_combined_vocabulary",
    "get_system_schema_string",
    # Context Builder
    "ContextBuilder",
    "PromptContext",
    "build_system_prompt",
    "build_full_prompt",
    "DEFAULT_FEW_SHOT_EXAMPLES",
    # Grammar
    "build_cypher_regex",
    "build_cypher_regex_from_vocabulary",
    "CypherGrammar",
]
