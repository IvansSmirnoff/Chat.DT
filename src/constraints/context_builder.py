"""
Context Builder for NL2Cypher LLM Prompts.

Builds system prompts and context that include:
- The merged IFC + IDS schema information
- Strict constraints from IDS
- Open vocabulary from IFC scan
- Examples for few-shot learning

This module bridges the vocabulary_merger with the LLM engine,
creating prompts that leverage both hard (regex) and soft (prompt) constraints.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

from .vocabulary_merger import CombinedVocabulary, PropertyType, get_system_schema_string


logger = logging.getLogger(__name__)


# =============================================================================
# Few-Shot Examples
# =============================================================================
#
# Few-shot examples are no longer hardcoded — they are sampled from the actual
# graph at bundle-build time (see ``src/constraints/value_sampler.py``). When
# no examples are supplied, the prompt simply omits the ``## Examples`` block
# rather than risk seeding the model with values that don't exist in the data.

DEFAULT_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = []


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a Neo4j Cypher query generator for Building Information Modeling (BIM) data.

Your task is to convert natural language questions into valid Cypher queries that can be executed against a Neo4j database containing IFC building data.

{schema_section}
{known_values_section}
## Query Guidelines

1. **MATCH clause**: Use `MATCH (n:EntityLabel)` where EntityLabel is one of the available entities
2. **WHERE clause**: Filter using property comparisons: `n.PropertyName = 'value'` or `n.PropertyName = true/false`
3. **RETURN clause**: Return the node `n`, specific properties `n.PropertyName`, or aggregations `count(n)`
4. **Case sensitivity**: Entity labels are case-sensitive (use exactly as listed)
5. **Property values**: Use single quotes for strings, no quotes for booleans and numbers
6. **Do not add a hop you do not need**: if the answer is already a property of the
   node, filter or return it directly. A room's floor, number and name are
   properties of the `IfcSpace` itself, so reaching for the storey node is both
   unnecessary and likely to pick the wrong relationship
7. **Only traverse the exact patterns listed under RELATIONSHIPS**: source label,
   relationship type and target label must all match one listed line. Any other
   combination is not reported as an error — it silently returns no rows
8. **For "which X have no Y", never MATCH the Y**: match X alone and exclude the
   pattern in WHERE, e.g. `MATCH (s:IfcSpace) WHERE NOT (s)-[:BOUNDED_BY]->(:IfcWindow)`.
   Matching `(s)-[:BOUNDED_BY]->(w:IfcWindow)` and *then* excluding the same pattern
   is self-contradictory and always returns nothing

## Constraint Rules

{constraint_rules}

## Output Format

Respond with ONLY a valid Cypher query. Do not include explanations, markdown code blocks, or any other text.

{examples_section}"""


CONSTRAINT_RULES_STRICT = """For properties marked as **STRICT**, you MUST use exactly one of the allowed values.
For example, if FireRating allows ['EI30', 'EI60', 'EI90'], do not use 'EI-60' or 'ei60'.

For **BOOLEAN** properties, use `true` or `false` (no quotes).
For **NUMERIC** properties, use numbers without quotes.
For **STRING** properties, use any string value in single quotes.
For **OPEN** properties, use appropriate values based on the question context."""


CONSTRAINT_RULES_SOFT = """The schema lists available properties. Use them appropriately:
- Fire ratings, material types, and similar categorical data have specific allowed values
- Boolean properties like IsExternal, LoadBearing use true/false
- Numeric properties like Height, Width, Area use numbers
- Text properties like Name, Description use quoted strings

When in doubt, use the property names exactly as listed in the schema."""


# =============================================================================
# Context Builder
# =============================================================================

@dataclass
class PromptContext:
    """
    Complete context for LLM prompt generation.
    
    Attributes:
        system_prompt: Full system prompt with schema and examples
        schema_string: Just the schema section (for debugging)
        constraint_mode: 'strict' or 'soft'
        entity_count: Number of available entities
        property_count: Number of available properties
    """
    system_prompt: str
    schema_string: str
    constraint_mode: str
    entity_count: int
    property_count: int


class ContextBuilder:
    """
    Builds LLM prompts with merged IFC + IDS schema information.
    
    This class creates system prompts that include:
    - Available entities and their descriptions
    - Property vocabulary with type information
    - Strict constraints from IDS
    - Few-shot examples for better generation
    
    Example:
        >>> from src.constraints import build_combined_vocabulary
        >>> vocab = build_combined_vocabulary("model.ifc", "spec.ids")
        >>> builder = ContextBuilder(vocab)
        >>> context = builder.build_context(constraint_mode='strict')
        >>> print(context.system_prompt)
    """
    
    def __init__(
        self,
        vocabulary: CombinedVocabulary,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        value_enumerations: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the context builder.

        Args:
            vocabulary: Combined vocabulary from vocabulary_merger
            few_shot_examples: Optional Q/A pairs sampled from the graph
                (see ``value_sampler.generate_few_shot_examples``). If
                omitted, no ``## Examples`` block is emitted — better than
                seeding the model with values that don't exist in the data.
            value_enumerations: Optional ``{"Label.prop": [v1, v2, ...]}``
                mapping (see ``value_sampler.sample_value_enumerations``).
                Rendered as a ``## Known Values`` block so the model picks
                from the literal labels the graph holds.
        """
        self.vocabulary = vocabulary
        self.few_shot_examples = few_shot_examples or DEFAULT_FEW_SHOT_EXAMPLES
        self.value_enumerations = value_enumerations or {}
    
    def build_context(
        self,
        constraint_mode: str = "strict",
        include_examples: bool = True,
        max_examples: int = 5
    ) -> PromptContext:
        """
        Build the complete prompt context.
        
        Args:
            constraint_mode: 'strict' for hard constraints, 'soft' for flexible
            include_examples: Whether to include few-shot examples
            max_examples: Maximum number of examples to include
            
        Returns:
            PromptContext with the full system prompt and metadata
        """
        # Build schema section
        schema_string = get_system_schema_string(self.vocabulary)
        schema_section = f"## Available Schema\n\n{schema_string}"

        # Build constraint rules
        if constraint_mode == "strict":
            constraint_rules = CONSTRAINT_RULES_STRICT
        else:
            constraint_rules = CONSTRAINT_RULES_SOFT

        # Known values block (graph-sampled categorical enums).
        known_values_section = self._build_known_values_section()

        # Build examples section
        examples_section = ""
        if include_examples:
            examples_section = self._build_examples_section(max_examples)

        # Assemble full prompt
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            schema_section=schema_section,
            known_values_section=known_values_section,
            constraint_rules=constraint_rules,
            examples_section=examples_section,
        )
        
        return PromptContext(
            system_prompt=system_prompt,
            schema_string=schema_string,
            constraint_mode=constraint_mode,
            entity_count=len(self.vocabulary.get_entity_names()),
            property_count=len(self.vocabulary.get_all_property_names())
        )
    
    def _build_examples_section(self, max_examples: int) -> str:
        """Build the few-shot examples section."""
        examples = self.few_shot_examples[:max_examples]

        if not examples:
            return ""

        lines = ["## Examples", ""]

        for i, ex in enumerate(examples, 1):
            lines.append(f"**Q{i}**: {ex['question']}")
            lines.append(f"**A{i}**: {ex['cypher']}")
            lines.append("")

        return "\n".join(lines)

    def _build_known_values_section(self) -> str:
        """Render graph-sampled categorical enumerations as a prompt block.

        Returns an empty string when no enumerations are available so the
        prompt template collapses cleanly to ``schema_section`` + rules.
        """
        if not self.value_enumerations:
            return ""

        lines = [
            "",
            "## Known Values",
            "",
            "These literal values exist in the graph. When a question implies one of "
            "these properties, pick the matching value verbatim — do not invent labels.",
            "",
        ]
        for key in sorted(self.value_enumerations.keys()):
            values = self.value_enumerations[key]
            rendered = ", ".join(f"'{v}'" for v in values)
            lines.append(f"- `{key}`: [{rendered}]")
        lines.append("")
        return "\n".join(lines)
    
    def build_user_prompt(self, question: str) -> str:
        """
        Build the user prompt for a question.
        
        Args:
            question: Natural language question
            
        Returns:
            Formatted user prompt
        """
        return f"Convert this question to a Cypher query: {question}"
    
    def get_schema_summary(self) -> Dict:
        """
        Get a summary of the schema for debugging/logging.
        
        Returns:
            Dictionary with schema statistics
        """
        strict_props = [
            p for p in self.vocabulary.all_properties.values()
            if p.property_type == PropertyType.STRICT
        ]
        
        return {
            "total_entities": len(self.vocabulary.get_entity_names()),
            "total_properties": len(self.vocabulary.get_all_property_names()),
            "strict_properties": len(strict_props),
            "relationships": list(self.vocabulary.all_relations),
            "source_ifc": self.vocabulary.ifc_schema.source_file if self.vocabulary.ifc_schema else None,
            "source_ids": str(self.vocabulary.ids_schema) if self.vocabulary.ids_schema else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def build_system_prompt(
    vocabulary: CombinedVocabulary,
    constraint_mode: str = "strict",
    include_examples: bool = True
) -> str:
    """
    Convenience function to build a system prompt from a vocabulary.
    
    Args:
        vocabulary: Combined vocabulary from vocabulary_merger
        constraint_mode: 'strict' or 'soft'
        include_examples: Whether to include few-shot examples
        
    Returns:
        Complete system prompt string
    """
    builder = ContextBuilder(vocabulary)
    context = builder.build_context(
        constraint_mode=constraint_mode,
        include_examples=include_examples
    )
    return context.system_prompt


def build_full_prompt(
    vocabulary: CombinedVocabulary,
    question: str,
    constraint_mode: str = "strict"
) -> Dict[str, str]:
    """
    Build complete system and user prompts for a question.
    
    Args:
        vocabulary: Combined vocabulary from vocabulary_merger
        question: Natural language question
        constraint_mode: 'strict' or 'soft'
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    builder = ContextBuilder(vocabulary)
    context = builder.build_context(constraint_mode=constraint_mode)
    
    return {
        "system": context.system_prompt,
        "user": builder.build_user_prompt(question)
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Demo the context builder with sample data."""
    from .vocabulary_merger import (
        CombinedVocabulary,
        EntityVocabulary,
        PropertyVocabulary,
        PropertyType
    )
    
    # Create sample vocabulary
    vocab = CombinedVocabulary(
        entities={
            "IfcWall": EntityVocabulary(
                name="IfcWall",
                source="ifc",
                count=42,
                description="Wall elements"
            ),
            "IfcDoor": EntityVocabulary(
                name="IfcDoor",
                source="ifc",
                count=15,
                description="Door elements"
            ),
        },
        properties={
            "FireRating": PropertyVocabulary(
                name="FireRating",
                property_type=PropertyType.STRICT,
                allowed_values={"EI30", "EI60", "EI90"},
                source="ids",
                applicable_entities={"IfcWall", "IfcDoor"}
            ),
            "Name": PropertyVocabulary(
                name="Name",
                property_type=PropertyType.STRING,
                source="ifc"
            ),
            "Height": PropertyVocabulary(
                name="Height",
                property_type=PropertyType.NUMERIC,
                source="ifc"
            ),
        },
        relationships={"CONTAINS", "DECOMPOSES"},
        source_ifc_file="sample.ifc",
        source_ids_file="sample.ids"
    )
    
    # Build context
    builder = ContextBuilder(vocab)
    context = builder.build_context(constraint_mode="strict")
    
    print("=" * 60)
    print("CONTEXT BUILDER DEMO")
    print("=" * 60)
    print(f"\nEntities: {context.entity_count}")
    print(f"Properties: {context.property_count}")
    print(f"Mode: {context.constraint_mode}")
    print("\n" + "=" * 60)
    print("SYSTEM PROMPT:")
    print("=" * 60)
    print(context.system_prompt)


if __name__ == "__main__":
    main()
