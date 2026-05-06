"""
LLM Engine for Constrained Cypher Generation and Direct QA.

This module provides a unified interface for LLM-based query generation
supporting both:
- Direct QA Mode: Text-based question answering returning GlobalIDs
- Cypher Generation Mode: Graph-based query generation with optional constraints

The 4 Experimental Settings (2x2 Matrix):
1. DIRECT_QA_BASELINE: Text dump + Question → JSON list of GlobalIDs
2. DIRECT_QA_GROUNDED: Text dump + IDS specs + Question → JSON list of GlobalIDs
3. CYPHER_SOFT: Schema + Question → Cypher (soft constraints via prompting)
4. CYPHER_STRICT: Schema + IDS + Question → Cypher (strict constraints via grammar)

Author: NL2Cypher Research Team
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import get_settings, Settings, LLMProvider, ExperimentSetting
from src.constraints.ids_parser import IDSParser, IDSSchema
from src.constraints.grammar import (
    build_cypher_regex,
    build_cypher_regex_from_vocabulary,
    validate_cypher_against_regex,
)
from src.constraints.vocabulary_merger import CombinedVocabulary
from src.constraints.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GenerationResult:
    """
    Result of an LLM generation (either Cypher or Direct QA).
    
    Attributes:
        query: Generated Cypher query (None for Direct QA mode)
        direct_answer: JSON string or parsed list of GlobalIDs (for Direct QA mode)
        is_valid: Whether the output passed validation
        is_constrained: Whether constrained decoding was used
        model_name: Name of the model used
        prompt: The prompt sent to the model
        raw_output: Raw model output (before post-processing)
        experiment_setting: The experimental setting used
        metadata: Additional metadata (tokens, latency, etc.)
    """
    query: Optional[str] = None
    direct_answer: Optional[List[str]] = None
    is_valid: bool = False
    is_constrained: bool = False
    model_name: str = ""
    prompt: str = ""
    raw_output: str = ""
    experiment_setting: Optional[ExperimentSetting] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_cypher_mode(self) -> bool:
        """Check if this result is from Cypher generation mode."""
        return self.query is not None
    
    @property
    def is_direct_qa_mode(self) -> bool:
        """Check if this result is from Direct QA mode."""
        return self.direct_answer is not None
    
    def get_global_ids(self) -> Set[str]:
        """
        Get the GlobalIDs from the result.
        
        For Direct QA mode, returns the parsed answer.
        For Cypher mode, this returns empty set (IDs come from DB execution).
        """
        if self.direct_answer:
            return set(self.direct_answer)
        return set()


# =============================================================================
# System Prompts
# =============================================================================

# Cypher Generation Prompts
CYPHER_SYSTEM_PROMPT_TEMPLATE = """You are a Cypher query generator for Neo4j databases containing Building Information Model (BIM) data from IFC files.

TASK: Convert natural language questions about buildings into valid Cypher queries.

SCHEMA CONSTRAINTS:
You MUST only use the following entity labels and properties:

VALID ENTITY LABELS (use exactly as shown):
{entities}

VALID PROPERTY NAMES (use exactly as shown):
{properties}

RELATIONSHIPS:
- CONTAINS: Spatial containment (e.g., IfcBuildingStorey CONTAINS IfcSpace, IfcBuildingStorey CONTAINS IfcWall)
- DECOMPOSES: Hierarchical decomposition (e.g., IfcBuilding DECOMPOSES to IfcBuildingStorey)
- HAS_MATERIAL: Material association (e.g., IfcColumn HAS_MATERIAL IfcMaterial)

CRITICAL RULES:
1. Generate ONLY valid Cypher syntax - the query MUST be executable
2. Use ONLY the entity labels and properties listed above
3. Use single quotes for string literals (e.g., 'Concrete')
4. Boolean values are: true or false (lowercase, no quotes)
5. ALWAYS end with RETURN followed by the main node variable (e.g., RETURN n)
6. For checking if a property exists, use: WHERE n.PropertyName IS NOT NULL
7. For partial string matching, use: WHERE n.PropertyName CONTAINS 'value'
8. For negation with CONTAINS, use: WHERE NOT n.PropertyName CONTAINS 'value'
9. DO NOT use ORDER BY or LIMIT
10. Numeric values should NOT have quotes (e.g., WHERE n.Width > 200)

OUTPUT FORMAT:
Return ONLY the Cypher query. No explanations, no markdown, no code blocks.
"""
# NOTE: This fallback template intentionally has no inline few-shots. When a
# CombinedVocabulary is loaded (the standard path), prompts are built by
# ``ContextBuilder``, which sources examples and known values from the live
# graph — see ``src/constraints/value_sampler.py``.

CYPHER_SOFT_CONSTRAINT_SUFFIX = """

CRITICAL REMINDER: 
- Output ONLY the Cypher query starting with MATCH
- End with RETURN followed by the node variable (e.g., RETURN n)
- Do NOT include any text before or after the query
- Do NOT use markdown formatting or code blocks
"""

# Direct QA Prompts
# Direct QA prompts - LLM answers WITHOUT data access (tests if model can "imagine" answers)
# Expected result: EA ≈ 0 (model cannot guess actual GlobalIds)
# This demonstrates the necessity of graph-based data access

DIRECT_QA_SYSTEM_PROMPT_BASELINE = """You are a BIM (Building Information Model) expert. Answer questions about building elements.

IMPORTANT: You do NOT have access to the actual building data. Answer based on your general knowledge of BIM and IFC standards. If you cannot determine specific element IDs, explain what types of elements would typically match the query.

OUTPUT FORMAT:
Respond with a JSON array of GlobalId strings if you can identify specific elements.
Otherwise respond with an empty array: []

Note: Without access to the actual model, you likely cannot provide real GlobalIds.
"""

DIRECT_QA_SYSTEM_PROMPT_GROUNDED = """You are a BIM (Building Information Model) expert. Answer questions about building elements.

You have access to the following IDS (Information Delivery Specification) schema that describes what types and properties SHOULD exist in a compliant model:

{ids_specs}

IMPORTANT: You do NOT have access to the actual building data - only the schema above. Use this schema to understand what element types and property values are valid, but you cannot see the actual elements.

OUTPUT FORMAT:
Respond with a JSON array of GlobalId strings if you can identify specific elements.
Otherwise respond with an empty array: []

Note: Without access to the actual model data, you likely cannot provide real GlobalIds.
"""

DIRECT_QA_OUTPUT_FORMAT_REMINDER = "\nRespond with ONLY a JSON array: [\"id1\", \"id2\"] or []"


# Cloud Direct QA Prompts (with actual data access via Gemini context cache)
# These are used in --cloud-direct mode where the full model_dump.json is
# uploaded to Gemini's context cache, giving the LLM real data access.

CLOUD_DIRECT_QA_SYSTEM_PROMPT = """You are a BIM (Building Information Model) query engine. You have access to the complete building model data provided in the context.

Your task is to find building elements that match the user's query by searching through the provided model data.

CRITICAL RULES:
1. Search the provided model data thoroughly for matching elements
2. Return ONLY the GlobalId values of elements that match the query
3. Be precise - only include elements that truly match the query criteria
4. If a query asks about properties (e.g., IsExternal, FireRating), check the actual property values in the data

OUTPUT FORMAT:
Respond with ONLY a JSON array of GlobalId strings: ["globalid1", "globalid2", ...]
If no elements match, respond with: []
Do NOT include any explanation or text outside the JSON array."""

CLOUD_DIRECT_QA_IDS_SUPPLEMENT = """
Additionally, use the following IDS (Information Delivery Specification) constraints to understand property semantics and valid values:

{ids_specs}
"""


# =============================================================================
# Base LLM Interface
# =============================================================================

class BaseLLMEngine(ABC):
    """
    Abstract base class for LLM engines.
    
    Supports both Direct QA and Cypher Generation modes based on
    the experimental setting.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema: Optional[IDSSchema] = None,
        vocabulary: Optional[CombinedVocabulary] = None,
        model_dump: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        value_enumerations: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the LLM engine.

        Args:
            settings: Application settings
            schema: IDS schema for constraints (legacy mode)
            vocabulary: Combined IFC + IDS vocabulary (preferred mode)
            model_dump: Pre-loaded model data for Direct QA mode
            few_shot_examples: Graph-sourced Q/A pairs for the Cypher prompt
                (see ``src.constraints.value_sampler``).
            value_enumerations: Graph-sampled categorical property values
                rendered as the prompt's ``Known Values`` block.
        """
        self.settings = settings or get_settings()
        self.schema = schema
        self.vocabulary = vocabulary
        self.model_dump = model_dump
        self.few_shot_examples = few_shot_examples
        self.value_enumerations = value_enumerations
        self.model = None
        self._initialized = False
        self._context_builder: Optional[ContextBuilder] = None

        # Initialize context builder if vocabulary provided
        if self.vocabulary:
            self._context_builder = ContextBuilder(
                self.vocabulary,
                few_shot_examples=self.few_shot_examples,
                value_enumerations=self.value_enumerations,
            )
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model. Must be called before generate()."""
        pass
    
    def generate(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
        model_context: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate output based on the experimental setting.
        
        Args:
            user_query: Natural language question about the BIM model
            experiment_setting: Which of the 4 experimental settings to use
            model_context: Optional model data context for Direct QA mode
            
        Returns:
            GenerationResult with either cypher query or direct answer
        """
        if experiment_setting.is_direct_qa:
            return self._generate_direct_qa(user_query, experiment_setting, model_context)
        else:
            return self._generate_cypher(user_query, experiment_setting)
    
    @abstractmethod
    def _generate_direct_qa(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
        model_context: Optional[str] = None,
    ) -> GenerationResult:
        """Generate direct answer (list of GlobalIDs) for Direct QA mode."""
        pass
    
    @abstractmethod
    def _generate_cypher(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
    ) -> GenerationResult:
        """Generate Cypher query for Cypher Generation mode."""
        pass
    
    def _build_cypher_system_prompt(self, use_ids: bool = False) -> str:
        """
        Build the system prompt for Cypher generation mode.
        
        Args:
            use_ids: Whether to include IDS constraints
        """
        # Prefer CombinedVocabulary for richer context
        if self._context_builder:
            constraint_mode = "strict" if use_ids else "soft"
            context = self._context_builder.build_context(
                constraint_mode=constraint_mode,
                include_examples=True
            )
            return context.system_prompt
        
        # Fallback to legacy IDSSchema mode
        if self.schema:
            entities = "\n".join(f"  - {e}" for e in sorted(self.schema.get_neo4j_labels()))
            properties = "\n".join(f"  - {p}" for p in sorted(self.schema.properties))
        else:
            entities = "  (No schema loaded - use any valid IFC entity)"
            properties = "  (No schema loaded - use any valid IFC property)"
        
        return CYPHER_SYSTEM_PROMPT_TEMPLATE.format(
            entities=entities,
            properties=properties,
        )
    
    def _build_direct_qa_system_prompt(
        self,
        model_context: str,
        use_ids: bool = False,
    ) -> str:
        """
        Build the system prompt for Direct QA mode.
        
        Args:
            model_context: Serialized model data
            use_ids: Whether to include IDS constraints
        """
        if use_ids:
            # Setting 2: Include IDS specs as schema guidance (but no actual data)
            ids_specs = self._format_ids_specs()
            return DIRECT_QA_SYSTEM_PROMPT_GROUNDED.format(ids_specs=ids_specs)
        else:
            # Setting 1: Pure baseline - no data, no schema
            return DIRECT_QA_SYSTEM_PROMPT_BASELINE
    
    def _format_ids_specs(self) -> str:
        """Format IDS specifications for the grounded Direct QA prompt."""
        specs_parts = []
        
        if self.vocabulary:
            # Entity types
            entities = sorted(self.vocabulary.get_entity_names())
            specs_parts.append(f"VALID ENTITY TYPES:\\n  {', '.join(entities)}")
            
            # Properties with constraints
            prop_lines = []
            for prop_name, prop_vocab in sorted(self.vocabulary.all_properties.items()):
                if prop_vocab.property_type.value == "strict" and prop_vocab.allowed_values:
                    vals = sorted(prop_vocab.allowed_values)
                    prop_lines.append(f"  - {prop_name}: STRICT values = {vals}")
                elif prop_vocab.property_type.value == "boolean":
                    prop_lines.append(f"  - {prop_name}: BOOLEAN (true/false)")
                elif prop_vocab.property_type.value == "numeric":
                    prop_lines.append(f"  - {prop_name}: NUMERIC")
            if prop_lines:
                specs_parts.append(f"PROPERTIES:\\n" + "\\n".join(prop_lines))
            
        elif self.schema:
            entities = sorted(self.schema.get_neo4j_labels())
            specs_parts.append(f"VALID ENTITY TYPES:\\n  {', '.join(entities)}")
            
            prop_lines = []
            for prop_name in sorted(self.schema.properties):
                if prop_name in self.schema.property_values:
                    vals = sorted(self.schema.property_values[prop_name])
                    prop_lines.append(f"  - {prop_name}: STRICT values = {vals}")
                else:
                    prop_lines.append(f"  - {prop_name}")
            if prop_lines:
                specs_parts.append(f"PROPERTIES:\\n" + "\\n".join(prop_lines))
        
        return "\\n\\n".join(specs_parts) if specs_parts else "(No schema available)"
    
    def _validate_cypher(self, query: str) -> bool:
        """Validate a generated Cypher query against the schema."""
        cached = getattr(self, "_regex_pattern", None)
        if cached:
            return validate_cypher_against_regex(query, cached)

        if self.vocabulary:
            regex = build_cypher_regex_from_vocabulary(self.vocabulary)
            return validate_cypher_against_regex(query, regex)

        if self.schema:
            regex = build_cypher_regex(
                self.schema.get_neo4j_labels(),
                self.schema.properties,
            )
            return validate_cypher_against_regex(query, regex)

        return True  # No schema to validate against
    
    def _validate_json_list(self, output: str) -> tuple[bool, Optional[List[str]]]:
        """
        Validate that the output is a valid JSON list of strings.
        
        Returns:
            Tuple of (is_valid, parsed_list)
        """
        try:
            # Try to extract JSON from the output
            output = output.strip()
            
            # Handle markdown code blocks
            if "```json" in output:
                match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
                if match:
                    output = match.group(1)
            elif "```" in output:
                match = re.search(r"```\s*(.*?)\s*```", output, re.DOTALL)
                if match:
                    output = match.group(1)
            
            # Find JSON array in output
            start_idx = output.find("[")
            end_idx = output.rfind("]")
            if start_idx != -1 and end_idx != -1:
                output = output[start_idx:end_idx + 1]
            
            parsed = json.loads(output)
            
            if not isinstance(parsed, list):
                return False, None
            
            # Ensure all elements are strings
            result = [str(item) for item in parsed]
            return True, result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON validation failed: {e}")
            return False, None
    
    def _get_regex_pattern(self) -> Optional[str]:
        """Get the regex pattern for constrained generation."""
        if self.vocabulary:
            return build_cypher_regex_from_vocabulary(self.vocabulary)
        elif self.schema:
            return build_cypher_regex(
                self.schema.get_neo4j_labels(),
                self.schema.properties,
            )
        return None
    
    def _clean_cypher_query(self, query: str) -> str:
        """
        Clean up generated Cypher query text.
        
        Handles various output formats:
        - Markdown code blocks (```cypher ... ```)
        - Plain text with extra content before/after
        - Multi-line queries
        
        Returns:
            Cleaned Cypher query string, or empty string if no valid query found
        """
        if not query:
            return ""

        original = query
        query = query.strip()

        # Remove markdown code blocks if present.
        # Only accept fence extraction if the content looks like Cypher
        # (contains MATCH). Otherwise the fence regex can match between
        # adjacent ``` ``` pairs and discard the actual query.
        if "```cypher" in query.lower():
            match = re.search(r"```cypher\s*(.*?)\s*```", query, re.DOTALL | re.IGNORECASE)
            if match and "MATCH" in match.group(1).upper():
                query = match.group(1).strip()
        elif "```" in query:
            match = re.search(r"```\s*(.*?)\s*```", query, re.DOTALL)
            if match and "MATCH" in match.group(1).upper():
                query = match.group(1).strip()

        # Strip any remaining backtick characters (common LLM artifact,
        # never valid in our Cypher grammar subset)
        query = query.replace('`', '').strip()
        
        # Try to extract a complete MATCH...RETURN query
        # Pattern: MATCH ... RETURN ... (possibly with WHERE in between)
        cypher_pattern = re.search(
            r"(MATCH\s*\([^)]+(?::[^\)]+)?\)(?:\s*-\[[^\]]*\]->\s*\([^)]+\))?\s*(?:WHERE\s+.+?)?\s*RETURN\s+[^\n;]+)",
            query,
            re.IGNORECASE | re.DOTALL
        )
        if cypher_pattern:
            return cypher_pattern.group(1).strip()
        
        # Fallback: Find MATCH and collect until RETURN is complete
        match_start = re.search(r"\bMATCH\b", query, re.IGNORECASE)
        if match_start:
            query_part = query[match_start.start():]
            
            # Find RETURN and include everything up to end of that clause
            return_match = re.search(r"\bRETURN\b\s+\S+", query_part, re.IGNORECASE)
            if return_match:
                end_pos = return_match.end()
                # Extend to include any following content on same line (like ORDER BY, LIMIT)
                remaining = query_part[end_pos:]
                line_end = remaining.find('\n')
                if line_end > 0:
                    end_pos += line_end
                else:
                    end_pos = len(query_part)
                
                result = query_part[:end_pos].strip()
                # Clean up any trailing incomplete clauses
                result = re.sub(r'\s+(ORDER BY|LIMIT)\s*$', '', result, flags=re.IGNORECASE)
                return result
        
        # Last resort: if it starts with MATCH, return as-is (even without RETURN)
        if query.upper().startswith("MATCH"):
            logger.warning(f"Query starts with MATCH but no RETURN found: {query[:100]}")
            return query
        
        logger.warning(f"Could not extract Cypher from: {original[:200]}")
        return ""
    
    def load_schema_from_ids(self, ids_path: Path | str) -> None:
        """Load schema constraints from an IDS file."""
        parser = IDSParser()
        self.schema = parser.parse(ids_path)
        logger.info(
            f"Loaded IDS schema: {len(self.schema.entities)} entities, "
            f"{len(self.schema.properties)} properties"
        )
    
    def load_combined_vocabulary(
        self,
        ifc_path: Path | str,
        ids_path: Optional[Path | str] = None
    ) -> None:
        """Load combined IFC + IDS vocabulary (hybrid schema)."""
        from src.constraints.vocabulary_merger import build_combined_vocabulary
        
        self.vocabulary = build_combined_vocabulary(ifc_path, ids_path)
        self._context_builder = ContextBuilder(self.vocabulary)
        
        logger.info(
            f"Loaded combined vocabulary: "
            f"{len(self.vocabulary.get_entity_names())} entities, "
            f"{len(self.vocabulary.get_all_property_names())} properties"
        )
    
    def load_model_dump(self, dump_path: Path | str) -> None:
        """Load model dump for Direct QA mode."""
        with open(dump_path, 'r') as f:
            self.model_dump = json.load(f)
        logger.info(f"Loaded model dump from {dump_path}")
    
    def set_model_dump(self, model_dump: Dict[str, Any]) -> None:
        """Set model dump directly for Direct QA mode."""
        self.model_dump = model_dump


# =============================================================================
# Local LLM Engine (Outlines - Constrained Decoding)
# =============================================================================

class LocalLLMEngine(BaseLLMEngine):
    """
    Local LLM engine using Outlines for constrained decoding.
    
    This is the PREFERRED implementation for CYPHER_STRICT setting
    as it provides true constrained decoding.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema: Optional[IDSSchema] = None,
        vocabulary: Optional[CombinedVocabulary] = None,
        model_dump: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        value_enumerations: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            settings, schema, vocabulary, model_dump,
            few_shot_examples=few_shot_examples,
            value_enumerations=value_enumerations,
        )
        self.generator = None
        self._regex_pattern = None
        self._strict_generator = None  # cached Outlines Generator (FSM)
    
    def initialize(self) -> None:
        """Initialize the local model with Outlines."""
        if self._initialized:
            return
        
        try:
            import outlines
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"Outlines import failed: {e}. "
                "Ensure outlines, torch, transformers, and accelerate are installed."
            )
        
        model_path = self.settings.llm_model_name
        logger.info(f"Loading local model: {model_path}")
        
        if model_path.endswith(".gguf") or "/gguf/" in model_path.lower():
            logger.info("Using llama.cpp backend")
            self._backend = "llamacpp"
            try:
                from llama_cpp import Llama
                llm = Llama.from_pretrained(
                    repo_id=model_path.rsplit("/", 1)[0] if "/" in model_path else model_path,
                    filename=model_path.rsplit("/", 1)[-1] if "/" in model_path else None,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                )
                self.model = outlines.from_llamacpp(llm)
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
                )
        else:
            logger.info("Using Transformers backend")
            self._backend = "transformers"
            import torch
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = outlines.from_transformers(hf_model, hf_tokenizer)

        self._regex_pattern = self._get_regex_pattern()
        if self._regex_pattern:
            logger.info("Built constraint regex from schema")
            try:
                from outlines import Generator
                from outlines.types import Regex
                logger.info("Compiling Outlines FSM (one-time)…")
                self._strict_generator = Generator(self.model, Regex(self._regex_pattern))
                logger.info("Outlines FSM cached")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to pre-compile Outlines Generator: %s", exc)
                self._strict_generator = None

        self._initialized = True
        logger.info("Local LLM engine initialized")
    
    def _truncate_model_context(self, model_context: str, max_chars: int = 50000) -> str:
        """
        Truncate model context to fit within context window.
        
        For Direct QA experiments, we truncate the model dump if it's too large.
        This may cause the model to miss relevant elements, demonstrating the
        limitation of Direct QA approaches on large BIM models.
        
        Args:
            model_context: The full model dump as JSON string
            max_chars: Maximum characters to keep (default ~12K tokens, fits in 32K with prompt)
            
        Returns:
            Truncated model context with warning appended
        """
        if len(model_context) <= max_chars:
            return model_context
        
        # Try to truncate at a valid JSON boundary (end of an object)
        truncated = model_context[:max_chars]
        
        # Find last complete JSON object
        last_brace = truncated.rfind('},')
        if last_brace > 0:
            truncated = truncated[:last_brace + 1] + "\n]"
        else:
            # Just truncate and close the array
            truncated = truncated[:max_chars] + "...(TRUNCATED)...]"
        
        original_len = len(model_context)
        kept_pct = (len(truncated) / original_len) * 100
        logger.warning(
            f"Model dump truncated from {original_len:,} to {len(truncated):,} chars "
            f"({kept_pct:.1f}% kept). Direct QA may miss relevant elements."
        )
        
        return truncated
    
    def _generate_direct_qa(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
        model_context: Optional[str] = None,  # Ignored - Direct QA has no data access
    ) -> GenerationResult:
        """
        Generate direct answer using local model WITHOUT data access.
        
        The LLM must answer from "imagination" / prior knowledge only.
        Setting 1 (BASELINE): No grounding at all
        Setting 2 (GROUNDED): IDS schema as guidance, but no actual data
        
        Expected result: EA ≈ 0 (model cannot guess real GlobalIds)
        This demonstrates the necessity of graph-based data access.
        """
        if not self._initialized:
            self.initialize()
        
        # Build prompt (no model data - that's the point of this experiment)
        use_ids = experiment_setting.uses_ids_grounding
        system_prompt = self._build_direct_qa_system_prompt(None, use_ids)
        prompt = f"{system_prompt}\n\nQuestion: {user_query}{DIRECT_QA_OUTPUT_FORMAT_REMINDER}"
        
        # Generate (unconstrained for speed)
        direct_qa_kwargs = {"max_tokens": 512} if self._backend == "llamacpp" else {"max_new_tokens": 512}
        raw_output = self.model(prompt, **direct_qa_kwargs)
        
        # Validate output
        is_valid, parsed_list = self._validate_json_list(raw_output)
        
        return GenerationResult(
            query=None,
            direct_answer=parsed_list if is_valid else [],
            is_valid=is_valid,
            is_constrained=False,  # No regex constraint for Direct QA
            model_name=self.settings.llm_model_name,
            prompt=prompt,
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "direct_qa",
                "uses_ids": use_ids,
            },
        )
    
    def _generate_cypher(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
    ) -> GenerationResult:
        """Generate Cypher query with optional constrained decoding."""
        if not self._initialized:
            self.initialize()
        
        from outlines import Generator
        from outlines.types import Regex
        
        use_strict = experiment_setting.uses_strict_constraints
        use_ids = experiment_setting.uses_ids_grounding
        
        # Build prompt
        system_prompt = self._build_cypher_system_prompt(use_ids=use_ids)
        if not use_strict:
            system_prompt += CYPHER_SOFT_CONSTRAINT_SUFFIX
        
        prompt = f"{system_prompt}\n\nQuestion: {user_query}\n\nCypher:"
        
        # Generate based on constraint mode
        # llama.cpp uses max_tokens; transformers uses max_new_tokens
        gen_kwargs = {"max_tokens": 256} if self._backend == "llamacpp" else {"max_new_tokens": 256}

        if use_strict and self._regex_pattern:
            logger.debug("Using STRICT constrained generation")
            if self._strict_generator is None:
                logger.info("Compiling Outlines FSM (lazy)…")
                self._strict_generator = Generator(self.model, Regex(self._regex_pattern))
            raw_output = self._strict_generator(prompt, **gen_kwargs)
            is_constrained = True
        else:
            logger.debug("Using unconstrained generation")
            raw_output = self.model(prompt, **gen_kwargs)
            is_constrained = False
        
        # Log raw output for debugging
        logger.debug(f"Raw model output: {raw_output[:500] if raw_output else '(empty)'}")
        
        query = self._clean_cypher_query(raw_output)
        
        # If cleaning failed, try using raw output directly if it looks like Cypher
        if not query and raw_output:
            raw_stripped = raw_output.replace('`', '').strip()
            if raw_stripped.upper().startswith("MATCH"):
                logger.warning(f"Cleaning failed but raw looks like Cypher, using raw: {raw_stripped[:100]}")
                query = raw_stripped
        
        return GenerationResult(
            query=query,
            direct_answer=None,
            is_valid=self._validate_cypher(query),
            is_constrained=is_constrained,
            model_name=self.settings.llm_model_name,
            prompt=prompt,
            raw_output=raw_output,  # Store actual raw output, not cleaned
            experiment_setting=experiment_setting,
            metadata={
                "mode": "cypher",
                "uses_strict": use_strict,
                "uses_ids": use_ids,
            },
        )


# =============================================================================
# Gemini LLM Engine (Soft Constraints)
# =============================================================================

class GeminiLLMEngine(BaseLLMEngine):
    """
    Google Gemini LLM engine with soft constraints.

    Supports both Direct QA and Cypher generation modes.
    Does NOT support true constrained decoding - use local engine
    for CYPHER_STRICT setting.

    With ``use_context_cache=True``, uploads the model_dump JSON to
    Gemini's context cache so Direct QA settings can query real model
    data without re-sending the full context per request. Falls back to
    inline per-request context injection if cache creation fails.
    """

    # Rate limiting: tracks token usage to stay under TPM quota.
    # Shared across instances so separate engines for different settings
    # don't independently blow through the same API quota.
    _tpm_limit: int = 1_000_000
    _rpm_limit: int = 1_000
    _request_log: List = []  # list of (timestamp, token_count) tuples

    # Leave headroom for system prompt + per-query prompt + response
    _CACHE_TOKEN_BUDGET = 900_000

    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema: Optional[IDSSchema] = None,
        vocabulary: Optional[CombinedVocabulary] = None,
        model_dump: Optional[Dict[str, Any]] = None,
        use_context_cache: bool = False,
        model_dump_json: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        value_enumerations: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            settings, schema, vocabulary, model_dump,
            few_shot_examples=few_shot_examples,
            value_enumerations=value_enumerations,
        )
        self.client = None
        self.use_context_cache = use_context_cache
        self._model_dump_json = model_dump_json
        self._cache = None
        self._cached_model = None
        self._cache_available = False

    def _call_gemini(self, model, content, generation_config=None, max_retries=5):
        """
        Call model.generate_content with proactive rate limiting and 429 retry.

        Rate limiting: before each call, checks whether the TPM/RPM budget
        has been exhausted in the current 60-second window and sleeps if needed.

        Retry: if a 429 is returned, parses the retry_delay from the error
        and waits before retrying (up to max_retries).
        """
        import time

        self._wait_for_rate_limit()

        kwargs = {}
        if generation_config is not None:
            kwargs["generation_config"] = generation_config

        for attempt in range(max_retries + 1):
            try:
                response = model.generate_content(content, **kwargs)

                # Record actual token usage for future rate limiting
                try:
                    um = response.usage_metadata
                    tokens_used = um.prompt_token_count + um.candidates_token_count
                except (AttributeError, TypeError):
                    tokens_used = len(str(content)) // 4  # rough estimate

                GeminiLLMEngine._request_log.append((time.time(), tokens_used))
                return response

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "quota" in error_str.lower()
                if not is_rate_limit or attempt == max_retries:
                    raise

                wait = self._parse_retry_delay(error_str) or (2 ** attempt * 10 + 5)
                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)
                # Proactive check again after sleeping
                self._wait_for_rate_limit()

    @classmethod
    def _wait_for_rate_limit(cls):
        """Sleep if the rolling 60-second token/request budget is exhausted."""
        import time

        if not cls._request_log:
            return

        now = time.time()
        window_start = now - 60.0

        # Prune entries older than the window
        cls._request_log = [
            (ts, tok) for ts, tok in cls._request_log if ts > window_start
        ]

        if not cls._request_log:
            return

        # Check RPM
        if len(cls._request_log) >= cls._rpm_limit:
            oldest_ts = cls._request_log[0][0]
            wait = 60.0 - (now - oldest_ts) + 1.0
            if wait > 0:
                logger.info(f"RPM limit approached. Waiting {wait:.0f}s...")
                time.sleep(wait)

        # Check TPM
        tokens_in_window = sum(tok for _, tok in cls._request_log)
        if tokens_in_window >= cls._tpm_limit * 0.90:  # 90% threshold
            oldest_ts = cls._request_log[0][0]
            wait = 60.0 - (now - oldest_ts) + 2.0
            if wait > 0:
                logger.info(
                    f"TPM limit approached ({tokens_in_window:,} tokens in window). "
                    f"Waiting {wait:.0f}s..."
                )
                time.sleep(wait)

    @staticmethod
    def _parse_retry_delay(error_str: str) -> Optional[float]:
        """Parse retry_delay seconds from a Gemini 429 error message."""
        match = re.search(r"retry.*?(\d+\.?\d*)\s*s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2  # 2s safety buffer
        return None
    
    def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._initialized:
            return
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        api_key = self.settings.llm_api_key_value
        if not api_key:
            raise ValueError("LLM_API_KEY not set for Gemini provider")
        
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            self.settings.llm_model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        if self.use_context_cache:
            # Resolve model dump JSON for cache upload
            if self._model_dump_json is None and self.model_dump is not None:
                self._model_dump_json = json.dumps(self.model_dump, indent=2)
            if not self._model_dump_json:
                raise ValueError(
                    "GeminiLLMEngine(use_context_cache=True) requires "
                    "model_dump or model_dump_json."
                )
            self._truncate_for_cache(genai)
            self._setup_context_cache(genai)

        self._initialized = True
        if self.use_context_cache:
            mode = "context cache" if self._cache_available else "per-request context"
            logger.info(
                f"Gemini engine initialized ({mode}): {self.settings.llm_model_name}"
            )
        else:
            logger.info(f"Gemini engine initialized: {self.settings.llm_model_name}")
    
    def _generate_direct_qa(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
        model_context: Optional[str] = None,
    ) -> GenerationResult:
        """Generate direct answer using Gemini."""
        if not self._initialized:
            self.initialize()

        if self.use_context_cache:
            return self._generate_direct_qa_cached(user_query, experiment_setting)

        # Standard path: no real data access; prompt may include IDS specs but
        # the model has to guess IDs (baseline experiment).
        if model_context is None:
            if self.model_dump:
                model_context = json.dumps(self.model_dump, indent=2)
            else:
                model_context = "(Model data not provided - placeholder for actual IFC data)"

        use_ids = experiment_setting.uses_ids_grounding
        system_prompt = self._build_direct_qa_system_prompt(model_context, use_ids)
        prompt = f"{system_prompt}\n\nQuestion: {user_query}{DIRECT_QA_OUTPUT_FORMAT_REMINDER}"

        response = self._call_gemini(self.model, prompt)
        raw_output = response.text

        is_valid, parsed_list = self._validate_json_list(raw_output)

        return GenerationResult(
            query=None,
            direct_answer=parsed_list if is_valid else [],
            is_valid=is_valid,
            is_constrained=False,
            model_name=self.settings.llm_model_name,
            prompt=prompt,
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "direct_qa",
                "uses_ids": use_ids,
                "provider": "gemini",
            },
        )

    def _generate_direct_qa_cached(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
    ) -> GenerationResult:
        """
        Direct QA with real model data via Gemini context cache.

        Uses the context cache if available; otherwise falls back to
        sending the (truncated) model dump inline with each request.
        """
        import google.generativeai as genai

        use_ids = experiment_setting.uses_ids_grounding

        if use_ids:
            ids_specs = self._format_ids_specs()
            ids_supplement = CLOUD_DIRECT_QA_IDS_SUPPLEMENT.format(ids_specs=ids_specs)
            query_prompt = (
                f"{ids_supplement}\n\n"
                f"Question: {user_query}"
                f"{DIRECT_QA_OUTPUT_FORMAT_REMINDER}"
            )
        else:
            query_prompt = (
                f"Question: {user_query}"
                f"{DIRECT_QA_OUTPUT_FORMAT_REMINDER}"
            )

        gen_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
        )

        if self._cache_available and self._cached_model:
            response = self._call_gemini(self._cached_model, query_prompt, gen_config)
            context_mode = "cached"
        else:
            full_prompt = (
                f"{CLOUD_DIRECT_QA_SYSTEM_PROMPT}\n\n"
                f"MODEL DATA:\n{self._model_dump_json}\n\n"
                f"{query_prompt}"
            )
            response = self._call_gemini(self.model, full_prompt, gen_config)
            context_mode = "inline"

        raw_output = response.text

        usage = {}
        try:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "cached_tokens": getattr(um, "cached_content_token_count", None),
            }
        except (AttributeError, TypeError):
            pass

        is_valid, parsed_list = self._validate_json_list(raw_output)

        return GenerationResult(
            query=None,
            direct_answer=parsed_list if is_valid else [],
            is_valid=is_valid,
            is_constrained=False,
            model_name=self.settings.llm_model_name,
            prompt=(
                f"[CONTEXT ({context_mode}): "
                f"{len(self._model_dump_json):,} chars]\n{query_prompt}"
            ),
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "direct_qa",
                "uses_ids": use_ids,
                "provider": "gemini_cloud",
                "context_cached": self._cache_available,
                "context_mode": context_mode,
                "cache_name": self._cache.display_name if self._cache else None,
                "usage": usage,
            },
        )

    def _truncate_for_cache(self, genai) -> None:
        """
        Truncate model dump to fit within the 1M token context window.

        Uses count_tokens API for exact measurement when available,
        falls back to character-based estimation otherwise.
        """
        counter = genai.GenerativeModel(self.settings.llm_model_name)
        try:
            token_count = counter.count_tokens(self._model_dump_json).total_tokens
        except Exception as e:
            token_count = len(self._model_dump_json) // 4
            logger.warning(
                f"count_tokens failed ({e}), "
                f"estimating ~{token_count:,} tokens from char count"
            )

        if token_count <= self._CACHE_TOKEN_BUDGET:
            logger.info(
                f"Model dump fits in context: {token_count:,} tokens "
                f"(budget: {self._CACHE_TOKEN_BUDGET:,})"
            )
            return

        logger.warning(
            f"Model dump too large: {token_count:,} tokens "
            f"(budget: {self._CACHE_TOKEN_BUDGET:,}). Truncating..."
        )

        keep_ratio = (self._CACHE_TOKEN_BUDGET / token_count) * 0.92
        try:
            data = json.loads(self._model_dump_json)
            if isinstance(data, list) and len(data) > 0:
                original_count = len(data)
                keep_count = max(1, int(original_count * keep_ratio))
                self._model_dump_json = json.dumps(data[:keep_count], indent=2)
                logger.warning(
                    f"Truncated model dump: kept {keep_count}/{original_count} "
                    f"elements (~{int(token_count * keep_ratio):,} tokens est.)"
                )
                return
        except (json.JSONDecodeError, TypeError):
            pass

        char_budget = int(len(self._model_dump_json) * keep_ratio)
        truncated = self._model_dump_json[:char_budget]
        last_obj_end = truncated.rfind('},')
        if last_obj_end > 0:
            self._model_dump_json = truncated[:last_obj_end + 1] + "\n]"
        else:
            self._model_dump_json = truncated
        logger.warning(
            f"Raw-truncated model dump to {len(self._model_dump_json):,} chars"
        )

    def _setup_context_cache(self, genai) -> None:
        """
        Set up or reuse Gemini context cache for the model dump.

        Uses SHA-256 hash of the (possibly truncated) content to identify
        caches. If cache creation fails (e.g. deprecated SDK doesn't support
        the model), sets _cache_available=False so the generate path falls
        back to sending the dump inline with each request.
        """
        import datetime
        import hashlib
        from google.generativeai import caching

        content_hash = hashlib.sha256(
            self._model_dump_json.encode("utf-8")
        ).hexdigest()[:16]
        cache_display_name = f"bim_model_{content_hash}"

        try:
            for cached in caching.CachedContent.list():
                if cached.display_name == cache_display_name:
                    logger.info(
                        f"Reusing existing context cache: {cache_display_name}"
                    )
                    self._cache = cached
                    self._cached_model = (
                        genai.GenerativeModel.from_cached_content(
                            cached_content=self._cache
                        )
                    )
                    self._cache_available = True
                    return
        except Exception as e:
            logger.warning(f"Could not check existing caches: {e}")

        model_name = self.settings.llm_model_name
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        logger.info(
            f"Creating context cache '{cache_display_name}' for model "
            f"{model_name} ({len(self._model_dump_json):,} chars, TTL: 60 min)"
        )

        try:
            self._cache = caching.CachedContent.create(
                model=model_name,
                display_name=cache_display_name,
                system_instruction=CLOUD_DIRECT_QA_SYSTEM_PROMPT,
                contents=[self._model_dump_json],
                ttl=datetime.timedelta(minutes=60),
            )
            self._cached_model = genai.GenerativeModel.from_cached_content(
                cached_content=self._cache
            )
            self._cache_available = True
            logger.info("Context cache created successfully (TTL: 60 min)")
        except Exception as e:
            logger.warning(
                f"Context cache creation failed: {e}. "
                f"Falling back to per-request context injection."
            )
            self._cache_available = False
            self._cached_model = None

    def _generate_cypher(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
    ) -> GenerationResult:
        """Generate Cypher query using Gemini (soft constraints only)."""
        if not self._initialized:
            self.initialize()

        use_ids = experiment_setting.uses_ids_grounding

        # Build prompt
        system_prompt = self._build_cypher_system_prompt(use_ids=use_ids)
        system_prompt += CYPHER_SOFT_CONSTRAINT_SUFFIX
        prompt = f"{system_prompt}\n\nQuestion: {user_query}\n\nCypher:"

        # Generate
        response = self._call_gemini(self.model, prompt)
        raw_output = response.text
        
        query = self._clean_cypher_query(raw_output)
        
        return GenerationResult(
            query=query,
            direct_answer=None,
            is_valid=self._validate_cypher(query),
            is_constrained=False,
            model_name=self.settings.llm_model_name,
            prompt=prompt,
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "cypher",
                "uses_ids": use_ids,
                "provider": "gemini",
            },
        )


# =============================================================================
# OpenAI LLM Engine (Soft Constraints)
# =============================================================================

class OpenAILLMEngine(BaseLLMEngine):
    """
    OpenAI LLM engine with soft constraints.
    
    Supports both Direct QA and Cypher generation modes.
    Does NOT support true constrained decoding.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        schema: Optional[IDSSchema] = None,
        vocabulary: Optional[CombinedVocabulary] = None,
        model_dump: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        value_enumerations: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            settings, schema, vocabulary, model_dump,
            few_shot_examples=few_shot_examples,
            value_enumerations=value_enumerations,
        )
        self.client = None

    def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )
        
        api_key = self.settings.llm_api_key_value
        if not api_key:
            raise ValueError("LLM_API_KEY not set for OpenAI provider")
        
        client_kwargs = {"api_key": api_key}
        if self.settings.llm_base_url:
            client_kwargs["base_url"] = self.settings.llm_base_url
        self.client = OpenAI(**client_kwargs)
        self._initialized = True
        logger.info(f"OpenAI engine initialized: {self.settings.llm_model_name}")
    
    def _generate_direct_qa(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
        model_context: Optional[str] = None,
    ) -> GenerationResult:
        """Generate direct answer using OpenAI."""
        if not self._initialized:
            self.initialize()
        
        # Get model context
        if model_context is None:
            if self.model_dump:
                model_context = json.dumps(self.model_dump, indent=2)
            else:
                model_context = "(Model data not provided - placeholder for actual IFC data)"
        
        # Build prompt
        use_ids = experiment_setting.uses_ids_grounding
        system_prompt = self._build_direct_qa_system_prompt(model_context, use_ids)
        user_prompt = f"{user_query}{DIRECT_QA_OUTPUT_FORMAT_REMINDER}"
        
        # Use JSON mode for more reliable output
        response = self.client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"} if "gpt-4" in self.settings.llm_model_name else None,
        )
        
        raw_output = response.choices[0].message.content
        
        # Validate output
        is_valid, parsed_list = self._validate_json_list(raw_output)
        
        return GenerationResult(
            query=None,
            direct_answer=parsed_list if is_valid else [],
            is_valid=is_valid,
            is_constrained=False,
            model_name=self.settings.llm_model_name,
            prompt=f"System: {system_prompt}\nUser: {user_prompt}",
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "direct_qa",
                "uses_ids": use_ids,
                "provider": "openai",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        )
    
    def _generate_cypher(
        self,
        user_query: str,
        experiment_setting: ExperimentSetting,
    ) -> GenerationResult:
        """Generate Cypher query using OpenAI."""
        if not self._initialized:
            self.initialize()
        
        use_ids = experiment_setting.uses_ids_grounding
        
        # Build prompt
        system_prompt = self._build_cypher_system_prompt(use_ids=use_ids)
        system_prompt += CYPHER_SOFT_CONSTRAINT_SUFFIX
        
        response = self.client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        
        raw_output = response.choices[0].message.content
        query = self._clean_cypher_query(raw_output)
        
        return GenerationResult(
            query=query,
            direct_answer=None,
            is_valid=self._validate_cypher(query),
            is_constrained=False,
            model_name=self.settings.llm_model_name,
            prompt=f"System: {system_prompt}\nUser: {user_query}",
            raw_output=raw_output,
            experiment_setting=experiment_setting,
            metadata={
                "mode": "cypher",
                "uses_ids": use_ids,
                "provider": "openai",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_llm_engine(
    provider: Optional[LLMProvider] = None,
    settings: Optional[Settings] = None,
    schema: Optional[IDSSchema] = None,
    vocabulary: Optional[CombinedVocabulary] = None,
    model_dump: Optional[Dict[str, Any]] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    value_enumerations: Optional[Dict[str, List[str]]] = None,
) -> BaseLLMEngine:
    """
    Factory function to create the appropriate LLM engine.

    Args:
        provider: LLM provider (local, gemini, openai). Uses config if not specified.
        settings: Application settings
        schema: IDS schema for constraints (legacy mode)
        vocabulary: Combined IFC + IDS vocabulary (preferred mode)
        model_dump: Pre-loaded model data for Direct QA mode
        few_shot_examples: Graph-sourced Q/A pairs for the Cypher prompt.
        value_enumerations: Graph-sampled categorical property values
            rendered as the prompt's ``Known Values`` block.

    Returns:
        Configured LLM engine instance
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider

    logger.info(f"Creating LLM engine for provider: {provider.value}")

    engine_map = {
        LLMProvider.LOCAL: LocalLLMEngine,
        LLMProvider.GEMINI: GeminiLLMEngine,
        LLMProvider.OPENAI: OpenAILLMEngine,
    }

    engine_class = engine_map.get(provider)
    if engine_class is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return engine_class(
        settings=settings,
        schema=schema,
        vocabulary=vocabulary,
        model_dump=model_dump,
        few_shot_examples=few_shot_examples,
        value_enumerations=value_enumerations,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_for_experiment(
    user_query: str,
    experiment_setting: ExperimentSetting,
    ifc_path: Optional[Path | str] = None,
    ids_path: Optional[Path | str] = None,
    model_dump_path: Optional[Path | str] = None,
    model_context: Optional[str] = None,
) -> GenerationResult:
    """
    Convenience function for generating output based on experimental setting.
    
    Args:
        user_query: Natural language question
        experiment_setting: Which of the 4 experimental settings to use
        ifc_path: Path to IFC file (for vocabulary)
        ids_path: Path to IDS file for schema constraints
        model_dump_path: Path to model dump JSON (for Direct QA mode)
        model_context: Optional model data context string (for Direct QA mode)
        
    Returns:
        GenerationResult with the generated output
    """
    settings = get_settings()
    
    vocabulary = None
    schema = None
    model_dump = None
    
    # Determine paths from settings if not provided
    if ifc_path is None:
        ifc_path = settings.ifc_file_path
    if ids_path is None:
        ids_path = settings.ids_file_path
    if model_dump_path is None:
        model_dump_path = settings.model_dump_path
    
    # Load vocabulary for Cypher modes
    if experiment_setting.is_cypher_gen:
        if ifc_path and Path(ifc_path).exists():
            from src.constraints.vocabulary_merger import build_combined_vocabulary
            ids_arg = ids_path if ids_path and Path(ids_path).exists() else None
            vocabulary = build_combined_vocabulary(ifc_path, ids_arg)
        elif ids_path and Path(ids_path).exists():
            parser = IDSParser()
            schema = parser.parse(ids_path)
    
    # Load model dump for Direct QA modes
    if experiment_setting.is_direct_qa:
        if model_dump_path and Path(model_dump_path).exists():
            with open(model_dump_path, 'r') as f:
                model_dump = json.load(f)
    
    # Create and initialize engine
    engine = create_llm_engine(
        schema=schema,
        vocabulary=vocabulary,
        model_dump=model_dump,
    )
    engine.initialize()
    
    return engine.generate(user_query, experiment_setting, model_context)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for LLM engine testing."""
    import sys
    
    print("NL2Cypher LLM Engine (Refactored)")
    print("=" * 60)
    print("Supports 4 Experimental Settings:")
    for setting in ExperimentSetting:
        print(f"  - {setting.value}")
    print("=" * 60)
    
    settings = get_settings()
    print(f"\nProvider: {settings.llm_provider.value}")
    print(f"Model: {settings.llm_model_name}")
    
    # Try to load combined vocabulary
    vocabulary = None
    schema = None
    
    if settings.ifc_file_path.exists():
        try:
            from src.constraints.vocabulary_merger import build_combined_vocabulary
            ids_path = settings.ids_file_path if settings.ids_file_path.exists() else None
            vocabulary = build_combined_vocabulary(settings.ifc_file_path, ids_path)
            print(f"Vocabulary: {len(vocabulary.get_entity_names())} entities, "
                  f"{len(vocabulary.get_all_property_names())} properties (hybrid)")
        except Exception as e:
            logger.warning(f"Could not load combined vocabulary: {e}")
    
    if vocabulary is None and settings.ids_file_path.exists():
        parser = IDSParser()
        schema = parser.parse(settings.ids_file_path)
        print(f"Schema: {len(schema.entities)} entities, {len(schema.properties)} properties (IDS-only)")
    
    # Interactive mode
    print("\n" + "-" * 60)
    print("Enter queries. Format: <setting_number> <question>")
    print("Settings: 1=DirectQA, 2=DirectQA+IDS, 3=Cypher, 4=Cypher+Strict")
    print("Example: 3 Find all external walls")
    print("-" * 60)
    
    setting_map = {
        "1": ExperimentSetting.DIRECT_QA_BASELINE,
        "2": ExperimentSetting.DIRECT_QA_GROUNDED,
        "3": ExperimentSetting.CYPHER_SOFT,
        "4": ExperimentSetting.CYPHER_STRICT,
    }
    
    try:
        engine = create_llm_engine(schema=schema, vocabulary=vocabulary)
        engine.initialize()
        
        while True:
            try:
                user_input = input("\nInput: ").strip()
                if not user_input:
                    continue
                
                # Parse setting and query
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print("Format: <setting_number> <question>")
                    continue
                
                setting_key = parts[0]
                query = parts[1]
                
                if setting_key not in setting_map:
                    print(f"Invalid setting. Use 1, 2, 3, or 4.")
                    continue
                
                setting = setting_map[setting_key]
                result = engine.generate(query, setting)
                
                print(f"\nSetting: {setting.value}")
                if result.is_cypher_mode:
                    print(f"Cypher: {result.query}")
                else:
                    print(f"Answer (GlobalIDs): {result.direct_answer}")
                print(f"Valid: {result.is_valid}")
                print(f"Constrained: {result.is_constrained}")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
                
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
