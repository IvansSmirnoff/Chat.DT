"""
Pure scoring primitives for NL2Cypher evaluation.

No Neo4j driver, no I/O. Safe to import in a Colab notebook that only needs
to re-score an already-collected results CSV, or to unit-test with pytest.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import ExperimentSetting

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class OutputType(str, Enum):
    """Type of LLM output being evaluated."""
    CYPHER = "cypher"       # Settings 3 & 4: Cypher query
    DIRECT_QA = "direct_qa"  # Settings 1 & 2: JSON list of GlobalIDs


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a generated output.
    
    Supports both Cypher query evaluation (Settings 3 & 4) and
    Direct QA evaluation (Settings 1 & 2).
    
    Attributes:
        question: Original natural language question
        experiment_setting: Which of the 4 settings was used
        output_type: Type of output (CYPHER or DIRECT_QA)
        
        # Cypher-specific (Settings 3 & 4)
        generated_cypher: Generated Cypher query (None for Direct QA)
        gold_cypher: Ground truth Cypher query (optional)
        
        # Direct QA-specific (Settings 1 & 2)
        generated_ids: Set of GlobalIds from generated output
        raw_output: Raw LLM output before parsing
        
        # Common fields
        gold_ids: Set of GlobalIds from gold standard (required)
        
        # Metrics
        svr: Syntactic Validity Rate (0 or 1)
        scr: Semantic Compliance Rate (0.0 to 1.0)
        ea: Execution Accuracy (0.0 to 1.0)
        
        # Validation details
        is_valid_syntax: Whether output has valid syntax
        invalid_labels: Set of labels not in schema (Cypher only)
        invalid_properties: Set of properties not in schema (Cypher only)
        
        # Error tracking
        error: Error message if evaluation failed
        metadata: Additional metadata
    """
    question: str
    experiment_setting: Optional[ExperimentSetting] = None
    output_type: OutputType = OutputType.CYPHER
    
    # Cypher-specific
    generated_cypher: Optional[str] = None
    gold_cypher: Optional[str] = None
    
    # Output IDs
    generated_ids: Set[str] = field(default_factory=set)
    gold_ids: Set[str] = field(default_factory=set)
    raw_output: str = ""
    
    # Metrics
    svr: float = 0.0
    scr: float = 0.0
    ea: float = 0.0
    
    # Validation details
    is_valid_syntax: bool = False
    invalid_labels: Set[str] = field(default_factory=set)
    invalid_properties: Set[str] = field(default_factory=set)
    
    # Error tracking
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            "question": self.question,
            "experiment_setting": self.experiment_setting.value if self.experiment_setting else "",
            "output_type": self.output_type.value,
            "generated_cypher": self.generated_cypher or "",
            "gold_cypher": self.gold_cypher or "",
            "raw_output": self.raw_output[:500] if self.raw_output else "",  # Truncate
            "svr": self.svr,
            "scr": self.scr,
            "ea": self.ea,
            "is_valid_syntax": self.is_valid_syntax,
            "invalid_labels": ",".join(sorted(self.invalid_labels)),
            "invalid_properties": ",".join(sorted(self.invalid_properties)),
            "num_generated_results": len(self.generated_ids),
            "num_gold_results": len(self.gold_ids),
            "precision": self._calculate_precision(),
            "recall": self._calculate_recall(),
            "f1": self._calculate_f1(),
            "error": self.error or "",
        }
    
    def _calculate_precision(self) -> float:
        """Calculate precision: |intersection| / |generated|"""
        if not self.generated_ids:
            return 0.0
        return len(self.generated_ids & self.gold_ids) / len(self.generated_ids)
    
    def _calculate_recall(self) -> float:
        """Calculate recall: |intersection| / |gold|"""
        if not self.gold_ids:
            return 0.0
        return len(self.generated_ids & self.gold_ids) / len(self.gold_ids)
    
    def _calculate_f1(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# SVR: JSON validation (pure)
# =============================================================================

def calculate_svr_json(output: str) -> Tuple[float, bool, Optional[str], Optional[List[Any]]]:
    """
    Calculate Syntactic Validity Rate for JSON array output (Direct QA).

    Validates that the output is a valid JSON array. Elements are returned
    in their parsed form (str/int/float/bool) so downstream scoring can match
    a numeric-only gold (``RETURN count(n)``) against a numeric JSON answer
    without losing the type.

    Args:
        output: Raw LLM output to validate

    Returns:
        Tuple of (svr_score, is_valid, error_message, parsed_list)
    """
    try:
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
            return 0.0, False, "Output is not a JSON array", None

        return 1.0, True, None, list(parsed)

    except json.JSONDecodeError as e:
        return 0.0, False, f"JSON parse error: {e}", None
    except Exception as e:
        return 0.0, False, f"Validation error: {e}", None


_NUMERIC_STRING_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _coerce_direct_qa_token(value: Any) -> str:
    """Map a JSON-parsed Direct-QA element to the same token namespace used
    when tokenizing Cypher gold results.

    Numeric / bool elements become ``num:X`` / ``bool:X`` so a scalar gold
    (``RETURN count(n)``) matches a Direct-QA answer like ``[5]`` or ``["5"]``.
    Strings that aren't numeric pass through unchanged so GlobalId answers
    keep working.
    """
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"num:{value}"
    if isinstance(value, float):
        return f"num:{round(value, 6)}"
    if isinstance(value, str):
        candidate = value.strip()
        if _NUMERIC_STRING_RE.match(candidate):
            try:
                if (
                    "." in candidate
                    or "e" in candidate.lower()
                ):
                    return f"num:{round(float(candidate), 6)}"
                return f"num:{int(candidate)}"
            except (ValueError, OverflowError):
                return value
        return value
    return str(value)


# =============================================================================
# SCR: Schema compliance (pure)
# =============================================================================

def extract_labels_from_cypher(query: str) -> Set[str]:
    """Extract node labels from a Cypher query."""
    labels = set()
    pattern = r'\(\s*\w*\s*:([\w:]+)\s*\)'
    
    for match in re.finditer(pattern, query, re.IGNORECASE):
        label_str = match.group(1)
        for label in label_str.split(':'):
            if label:
                labels.add(label)
    
    return labels


def extract_properties_from_cypher(query: str) -> Set[str]:
    """Extract property names from a Cypher query."""
    properties = set()
    pattern = r'\b\w+\.(\w+)\b'
    
    for match in re.finditer(pattern, query):
        prop = match.group(1)
        if prop not in {'id', 'labels', 'type'}:
            properties.add(prop)
    
    return properties


def calculate_scr_cypher(
    query: str,
    valid_labels: Set[str],
    valid_properties: Set[str],
) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate Semantic Compliance Rate for Cypher queries.

    Args:
        query: Cypher query to analyze
        valid_labels: Set of valid entity labels
        valid_properties: Set of valid property names

    Returns:
        Tuple of (scr_score, invalid_labels, invalid_properties)
    """
    # Empty/whitespace predicted output extracts zero labels + zero properties.
    # Without this guard the function returns 1.0 (vacuous compliance) and the
    # batch summary's scr_mean inflates on runs where the model emits nothing.
    if not query or not query.strip():
        return 0.0, set(), set()

    query_labels = extract_labels_from_cypher(query)
    query_properties = extract_properties_from_cypher(query)
    
    valid_labels_lower = {l.lower() for l in valid_labels}
    valid_props_lower = {p.lower() for p in valid_properties}
    
    invalid_labels = set()
    valid_label_count = 0
    for label in query_labels:
        if label.lower() in valid_labels_lower:
            valid_label_count += 1
        else:
            invalid_labels.add(label)
    
    invalid_properties = set()
    valid_prop_count = 0
    for prop in query_properties:
        if prop.lower() in valid_props_lower:
            valid_prop_count += 1
        else:
            invalid_properties.add(prop)
    
    total_tokens = len(query_labels) + len(query_properties)
    valid_tokens = valid_label_count + valid_prop_count
    
    if total_tokens == 0:
        scr = 1.0
    else:
        scr = valid_tokens / total_tokens
    
    return scr, invalid_labels, invalid_properties


def calculate_scr(
    output: str,
    output_type: OutputType,
    valid_labels: Set[str],
    valid_properties: Set[str],
) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate Semantic Compliance Rate based on output type.
    
    For Direct QA (Settings 1 & 2), SCR is not applicable and returns 1.0.
    
    Args:
        output: Generated output
        output_type: Type of output
        valid_labels: Set of valid entity labels
        valid_properties: Set of valid property names
        
    Returns:
        Tuple of (scr_score, invalid_labels, invalid_properties)
    """
    if output_type == OutputType.CYPHER:
        return calculate_scr_cypher(output, valid_labels, valid_properties)
    else:
        # SCR is not applicable for Direct QA
        return 1.0, set(), set()


# =============================================================================
# EA: ID-based scoring (pure)
# =============================================================================

# Separators chosen so they cannot appear inside scalar tokens or IFC GlobalIds.
# RECORD_SEP joins the cells of a single Neo4j record into one composite token,
# preserving column order so group-by results like ``RETURN s.Name, count(d)``
# do not collide when the model returns the same names + counts mis-paired.
# LIST_SEP joins the (sorted) elements of a ``collect()``-style cell.
RECORD_SEP = "\x1e"
LIST_SEP = "\x1f"


def _scalar_token(value: Any) -> Optional[str]:
    """Tokenize aggregation-style scalars (count/sum/min/max/avg, bools)
    into a distinct namespace so set-based EA still works for scalar gold.

    Returns ``None`` for non-scalar values so callers can dispatch further.
    """
    if value is None:
        return None
    # bool is a subclass of int — check it first.
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"num:{value}"
    if isinstance(value, float):
        # Round to dampen float jitter from sums/avgs.
        return f"num:{round(value, 6)}"
    return None


def _extract_global_id(value: Any, id_property: str = "GlobalId") -> Optional[str]:
    """
    Tokenize a single cell of a Neo4j record into a comparable string.

    Handles, in order:
    - Neo4j Node / dict-like objects → look up ``id_property``.
    - Lists / tuples (e.g. from ``collect()``) → sorted, LIST_SEP-joined
      bag of cell tokens (order-insensitive, like the SQL semantics).
    - Direct string values → passthrough.
    - Numeric / boolean scalars → ``num:``/``bool:`` namespace via
      :func:`_scalar_token`.
    """
    if value is None:
        return None

    # Handle Neo4j Node objects (have _properties or items() or get())
    # neo4j Node objects support dict-like access
    if hasattr(value, '_properties'):
        props = dict(value._properties)
        if id_property in props:
            return str(props[id_property])
        # Fall back to any property that looks like a GlobalId
        for k, v in props.items():
            if k.lower() == 'globalid' or k.lower() == 'global_id':
                return str(v)
        return None

    # Handle neo4j Node that supports item access via .get() but not _properties
    # This covers nodes returned directly from session.run()
    if hasattr(value, 'keys') and callable(getattr(value, 'keys', None)):
        try:
            # Try direct property access
            keys = list(value.keys())
            for key in keys:
                if key.lower() == id_property.lower():
                    return str(value[key])
                if key.lower() == 'globalid' or key.lower() == 'global_id':
                    return str(value[key])
        except (TypeError, KeyError):
            pass

    # Handle dict-like objects (from node.get or similar). Strings also expose
    # .get via attribute lookup on some platforms — exclude them explicitly so
    # this branch only catches actual mappings.
    if (
        not isinstance(value, (str, bytes, list, tuple))
        and hasattr(value, 'get')
        and callable(value.get)
    ):
        try:
            result = value.get(id_property)
        except TypeError:
            result = None
        if result is not None:
            return str(result)
        # Try case-insensitive lookup via items()
        if hasattr(value, 'items') and callable(value.items):
            try:
                for k, v in value.items():
                    if k.lower() == id_property.lower():
                        return str(v)
            except (TypeError, AttributeError):
                pass
        return None

    # Handle dict directly
    if isinstance(value, dict):
        if id_property in value:
            return str(value[id_property])
        for k, v in value.items():
            if k.lower() == id_property.lower():
                return str(v)
        return None

    # collect() / list-valued cells. Tokenize each element, drop misses,
    # sort to make the bag order-insensitive, and join with LIST_SEP.
    if isinstance(value, (list, tuple)):
        parts = [
            tok for tok in (_extract_global_id(v, id_property) for v in value)
            if tok
        ]
        if not parts:
            return None
        return LIST_SEP.join(sorted(parts))

    # Handle direct string values that look like GlobalIds
    # IFC GlobalIds are 22 characters with specific charset
    if isinstance(value, str) and value:
        # Check if it looks like an IFC GlobalId (22 chars, alphanumeric + special)
        if len(value) == 22 and all(c.isalnum() or c in '_$' for c in value):
            return value
        # Coerce numeric strings to the same ``num:`` namespace used by the
        # Direct-QA scorer. Without this, a Cypher gold like ``RETURN d.FireRating``
        # that returns the string ``"0"`` doesn't match a Direct-QA answer of
        # ``[0]`` / ``["0"]`` which the coercer turns into ``num:0``.
        candidate = value.strip()
        if _NUMERIC_STRING_RE.match(candidate):
            try:
                if "." in candidate or "e" in candidate.lower():
                    return f"num:{round(float(candidate), 6)}"
                return f"num:{int(candidate)}"
            except (ValueError, OverflowError):
                pass
        return value

    scalar = _scalar_token(value)
    if scalar is not None:
        return scalar

    return None


def tokenize_record(record_values: Any, id_property: str = "GlobalId") -> Optional[str]:
    """Hash a whole Neo4j record into one composite token.

    Cells are tokenized via :func:`_extract_global_id` and joined with
    ``RECORD_SEP`` in column order. Single-cell records collapse to the cell's
    own token (preserves the legacy ``Set[str]`` of GlobalIds behaviour).
    Returns ``None`` if every cell tokenizes to ``None`` (whole-row drop).

    Accepts either an iterable of values (e.g. ``record.values()``) or anything
    with a ``.values()`` method (the neo4j Record itself).
    """
    if hasattr(record_values, "values") and callable(record_values.values):
        try:
            values_iter = list(record_values.values())
        except TypeError:
            values_iter = list(record_values)
    else:
        values_iter = list(record_values)

    cells = [_extract_global_id(v, id_property) for v in values_iter]
    if all(c is None for c in cells):
        return None
    # Single-cell case is the common Cypher pattern (RETURN n / RETURN n.GlobalId
    # / RETURN count(n)) — collapse to the cell token to stay 1:1 with the
    # pre-refactor result-set contents.
    if len(cells) == 1:
        return cells[0]
    return RECORD_SEP.join("" if c is None else c for c in cells)


def calculate_ea_from_ids(
    generated_ids: Set[str],
    gold_ids: Set[str],
) -> float:
    """
    Calculate Execution Accuracy using Jaccard similarity.
    
    EA = |Generated ∩ Gold| / |Generated ∪ Gold|
    
    Args:
        generated_ids: Set of IDs from generated output
        gold_ids: Set of IDs from gold standard
        
    Returns:
        EA score (0.0 to 1.0)
    """
    if not generated_ids and not gold_ids:
        # Both empty - consider it a match
        return 1.0
    elif not generated_ids or not gold_ids:
        # One empty, one not - no overlap
        return 0.0
    else:
        intersection = generated_ids & gold_ids
        union = generated_ids | gold_ids
        return len(intersection) / len(union)


def calculate_ea_direct(
    raw_output: str,
    gold_ids: Set[str],
) -> Tuple[float, Set[str], Optional[str]]:
    """
    Calculate Execution Accuracy for Direct QA output.
    
    Parses the JSON list and compares to gold IDs.
    
    Args:
        raw_output: Raw LLM output (JSON array)
        gold_ids: Set of GlobalIds from gold standard
        
    Returns:
        Tuple of (ea_score, generated_ids, error_message)
    """
    svr, is_valid, error, parsed_list = calculate_svr_json(raw_output)

    if not is_valid or parsed_list is None:
        return 0.0, set(), f"Could not parse output: {error}"

    generated_ids = {_coerce_direct_qa_token(item) for item in parsed_list}
    ea = calculate_ea_from_ids(generated_ids, gold_ids)
    
    logger.debug(
        f"EA (Direct QA): |generated|={len(generated_ids)}, "
        f"|gold|={len(gold_ids)}, EA={ea:.3f}"
    )
    
    return ea, generated_ids, None

