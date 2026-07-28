"""Graph-sourced value enumerations and few-shot examples.

Both prompt-context inputs are sampled from the live Neo4j graph at
bundle-build time (or runner setup time), so the system prompt never
hardcodes IFC author conventions. The same code generalises across
storey naming (`Level 2` vs `Storey 02` vs `P2`), space types, fire
ratings, materials, etc.

Design rules
------------
* Code owns *query shapes*; concrete labels, property names, and string
  values are sampled live.
* Per ``(label, property)`` pair, run a single
  ``MATCH (n:Label) RETURN DISTINCT n.prop LIMIT K+1``. Keep iff the
  count is ``<= max_cardinality`` *and* values are short categorical
  strings.
* Few-shot generation iterates a small set of templates whose
  preconditions are checked against the current vocabulary +
  enumerations + (optional) graph stats. Variable bindings are always
  single lowercase letters reused verbatim in WHERE/RETURN. The grammar's
  ``IDENT`` now admits 1-3 chars, so this is no longer a hard requirement —
  but examples must still stay inside ``IDENT``, and reusing the bound name
  verbatim is what teaches the model to do the same.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from .vocabulary_merger import CombinedVocabulary, PropertyType

logger = logging.getLogger(__name__)


# Properties that should never be enumerated (identity, free-text, geometry).
EXCLUDE_PROPS: Set[str] = {
    "GlobalId",
    "Tag",
    "OwnerHistory",
    "ObjectPlacement",
    "Description",
    "LongDescription",
    "Representation",
}

# Maximum average value length (in characters) before we treat a property
# as free-text and skip it.
MAX_VALUE_LENGTH = 80

# Cap how many properties we probe per entity. Distributing the budget
# this way prevents one metadata-rich entity (IfcBuilding can carry 50+
# Revit psets) from starving every other entity of its sampling budget —
# without that, alphabetically-later entities like IfcBuildingStorey
# would never get their Name property enumerated.
MAX_PROPS_PER_ENTITY = 12


# =============================================================================
# Value enumerations
# =============================================================================


def _is_categorical(values: List[Any]) -> bool:
    """Decide if sampled values look like a categorical string enum."""
    if not values:
        return False
    if not all(isinstance(v, str) for v in values):
        return False
    if not all(v.strip() for v in values):
        return False
    avg_len = sum(len(v) for v in values) / len(values)
    if avg_len > MAX_VALUE_LENGTH:
        return False
    return True


def sample_value_enumerations(
    driver,
    vocab: CombinedVocabulary,
    max_cardinality: int = 50,
) -> Dict[str, List[str]]:
    """Sample low-cardinality categorical property values from Neo4j.

    Returns a mapping like ``{"IfcBuildingStorey.Name": ["Level 0", ...]}``
    sorted by key. A pair is included only when:

    * the property is not in ``EXCLUDE_PROPS``,
    * the distinct count is ``<= max_cardinality``,
    * all values are non-empty strings averaging less than
      ``MAX_VALUE_LENGTH`` chars.
    """
    enumerations: Dict[str, List[str]] = {}
    pairs_examined = 0

    with driver.session() as session:
        # Pre-fetch the real (label, property) pairs the graph actually
        # carries so we never probe a key the loader didn't materialise.
        # Without this, the vocabulary (which is a superset of the graph,
        # since it includes every IFC pset key) triggers a Neo4j
        # ``UnknownPropertyKeyWarning`` notification per dead probe.
        known_keys = _fetch_known_property_keys(session)

        for entity_name in sorted(vocab.entities.keys()):
            if not entity_name:
                continue
            entity = vocab.entities[entity_name]
            entity_probes = 0
            # Prioritise the properties the model is most likely to need
            # (Name / LongName) so they always make it through the cap.
            ordered_props = _ordered_properties(entity.properties.keys())
            for prop_name in ordered_props:
                if entity_probes >= MAX_PROPS_PER_ENTITY:
                    break
                if prop_name in EXCLUDE_PROPS:
                    continue

                # Strict properties already carry their allowed values
                # from IDS; sampling them again would just duplicate the
                # information the prompt already exposes.
                prop_vocab = entity.properties[prop_name]
                if prop_vocab.property_type == PropertyType.STRICT:
                    continue

                # Skip probes for properties the loader never wrote on
                # this label — they would emit dead-key warnings.
                if known_keys and prop_name not in known_keys.get(entity_name, set()):
                    continue

                entity_probes += 1
                pairs_examined += 1
                cypher = (
                    f"MATCH (n:`{entity_name}`) "
                    f"WHERE n.`{prop_name}` IS NOT NULL "
                    f"RETURN DISTINCT n.`{prop_name}` AS v "
                    f"LIMIT $cap"
                )
                try:
                    rows = session.run(cypher, cap=max_cardinality + 1).data()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Sampling skipped for %s.%s: %s",
                        entity_name, prop_name, exc,
                    )
                    continue

                values = [r["v"] for r in rows]
                if len(values) > max_cardinality:
                    continue
                if not _is_categorical(values):
                    continue

                key = f"{entity_name}.{prop_name}"
                enumerations[key] = sorted(values)

    logger.info(
        "Sampled %d value enumerations (examined %d pairs, cap=%d)",
        len(enumerations), pairs_examined, max_cardinality,
    )
    return enumerations


# Properties to push to the front of the per-entity sampling order so they
# always survive the MAX_PROPS_PER_ENTITY cap. These are the categorical
# fields the model most often needs literal values for.
_PRIORITY_PROPERTIES = (
    "Name",
    "LongName",
    "ObjectType",
    "PredefinedType",
    "FireRating",
    "Material",
    "Category",
)


def _fetch_known_property_keys(session) -> Dict[str, Set[str]]:
    """Map each Neo4j label to the property keys the graph actually carries.

    Uses the ``db.schema.nodeTypeProperties`` procedure (Neo4j 4.x+). Returns
    an empty dict if the procedure is unavailable, in which case the sampler
    falls back to probing every vocabulary pair (and tolerates the resulting
    notifications).
    """
    keys: Dict[str, Set[str]] = {}
    try:
        rows = session.run(
            "CALL db.schema.nodeTypeProperties() "
            "YIELD nodeLabels, propertyName "
            "RETURN nodeLabels, propertyName"
        ).data()
    except Exception as exc:  # noqa: BLE001
        logger.debug("db.schema.nodeTypeProperties unavailable: %s", exc)
        return {}

    for row in rows:
        prop = row.get("propertyName")
        if not prop:
            continue
        for label in row.get("nodeLabels") or []:
            keys.setdefault(label, set()).add(prop)
    return keys


def _ordered_properties(names) -> List[str]:
    """Sort property names so priority fields come first, others alphabetic."""
    rest = sorted(name for name in names if name not in _PRIORITY_PROPERTIES)
    head = [name for name in _PRIORITY_PROPERTIES if name in names]
    return head + rest


# =============================================================================
# Few-shot generation
# =============================================================================


@dataclass
class _TemplateContext:
    """Pre-computed lookups passed to template applicability checks."""

    vocab: CombinedVocabulary
    enumerations: Dict[str, List[str]]
    label_counts: Dict[str, int]
    relationships: Set[str]

    def entity_with_count(self, name: str) -> bool:
        if name not in self.vocab.entities:
            return False
        if self.label_counts and self.label_counts.get(name, 0) <= 0:
            return False
        return True

    def first_enum(self, label: str, prop: str) -> Optional[str]:
        values = self.enumerations.get(f"{label}.{prop}")
        if not values:
            return None
        return values[0]

    def numeric_property_for(self, label: str) -> Optional[str]:
        entity = self.vocab.entities.get(label)
        if entity is None:
            return None
        for name, prop in sorted(entity.properties.items()):
            if name in EXCLUDE_PROPS:
                continue
            if prop.property_type == PropertyType.NUMERIC:
                return name
        return None

    def has_relationship(self, rel: str) -> bool:
        return rel in self.relationships

    def relation_between(self, src: str, dst: str) -> Optional[str]:
        """Which relationship actually connects ``src`` to ``dst``, if any.

        Examples must never assert a traversal the graph does not have. A
        template that hardcoded ``(IfcBuildingStorey)-[:CONTAINS]->(...)``
        taught the model to reach spaces that way; on Building 15 spaces hang
        off the storey by DECOMPOSES, so every query it inspired returned
        nothing.
        """
        for s, rel, d in self.vocab.relationship_signatures:
            if s == src and d == dst:
                return rel
        return None


def _humanise(label: str) -> str:
    """``IfcBuildingStorey`` -> ``building storey``."""
    if label.startswith("Ifc"):
        label = label[3:]
    parts = re.findall(r"[A-Z][a-z]*|[0-9]+", label) or [label]
    return " ".join(p.lower() for p in parts) if parts else label.lower()


def _pick_entity_with_enum(
    ctx: _TemplateContext, prop: str
) -> Optional[tuple[str, str]]:
    """Return (label, value) for the first entity with an enum on ``prop``."""
    for key, values in ctx.enumerations.items():
        label, sampled_prop = key.rsplit(".", 1)
        if sampled_prop != prop:
            continue
        if not ctx.entity_with_count(label):
            continue
        if not values:
            continue
        return label, values[0]
    return None


def _t_count_by_label(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    candidates = [
        "IfcWall", "IfcDoor", "IfcWindow", "IfcSpace", "IfcColumn",
        "IfcSlab", "IfcBeam",
    ]
    for label in candidates:
        if ctx.entity_with_count(label):
            return {
                "question": f"How many {_humanise(label)} elements are in the model?",
                "cypher": f"MATCH (n:{label}) RETURN count(n)",
            }
    return None


_FILTER_PREFERRED_LABELS = (
    "IfcWall", "IfcDoor", "IfcWindow", "IfcSpace", "IfcColumn",
    "IfcSlab", "IfcBeam", "IfcStair", "IfcRamp",
)


def _t_filter_by_categorical(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    """Prefer queryable element types (walls/doors/...) over building-level
    metadata so the example doesn't anchor the model on Revit project fields."""

    def _emit(label: str, prop: str, value: str) -> Dict[str, str]:
        return {
            "question": (
                f"Find {_humanise(label)} elements where {prop} is '{value}'"
            ),
            "cypher": (
                f"MATCH (n:{label}) WHERE n.{prop} = '{value}' RETURN n"
            ),
        }

    for preferred in _FILTER_PREFERRED_LABELS:
        for key, values in sorted(ctx.enumerations.items()):
            label, prop = key.rsplit(".", 1)
            if label != preferred:
                continue
            if not ctx.entity_with_count(label):
                continue
            if not values:
                continue
            return _emit(label, prop, values[0])

    # Fallback: any non-storey entity with at least one categorical value.
    for key, values in sorted(ctx.enumerations.items()):
        label, prop = key.rsplit(".", 1)
        if label == "IfcBuildingStorey":
            continue
        if not ctx.entity_with_count(label):
            continue
        if not values:
            continue
        return _emit(label, prop, values[0])
    return None


def _t_storey_contains(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    """Storey -> element. Uses whichever relationship the graph really has."""
    storey_value = ctx.first_enum("IfcBuildingStorey", "Name")
    if storey_value is None:
        return None
    for child in ("IfcDoor", "IfcWindow", "IfcWall"):
        if not ctx.entity_with_count(child):
            continue
        rel = ctx.relation_between("IfcBuildingStorey", child)
        if rel is None:
            continue
        return {
            "question": (
                f"How many {_humanise(child)} elements are on "
                f"'{storey_value}'?"
            ),
            "cypher": (
                f"MATCH (s:IfcBuildingStorey)-[:{rel}]->(d:{child}) "
                f"WHERE s.Name = '{storey_value}' "
                f"RETURN count(d)"
            ),
        }
    return None


def _t_storey_spaces(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    """Storey -> space, kept separate from storey -> element on purpose.

    Spatial containment and element containment use *different* relationships
    in this ETL (DECOMPOSES vs CONTAINS). One example covering only elements
    let the model assume a single rule, which is the error that made every
    per-floor query on Building 15 return zero rows.
    """
    storey_value = ctx.first_enum("IfcBuildingStorey", "Name")
    if storey_value is None or not ctx.entity_with_count("IfcSpace"):
        return None
    rel = ctx.relation_between("IfcBuildingStorey", "IfcSpace")
    if rel is None:
        return None
    return {
        "question": f"How many rooms are on '{storey_value}'?",
        "cypher": (
            f"MATCH (s:IfcBuildingStorey)-[:{rel}]->(sp:IfcSpace) "
            f"WHERE s.Name = '{storey_value}' "
            f"RETURN count(sp)"
        ),
    }


def _t_aggregate_numeric(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    for label in ("IfcWindow", "IfcDoor", "IfcWall", "IfcSpace", "IfcColumn"):
        if not ctx.entity_with_count(label):
            continue
        prop = ctx.numeric_property_for(label)
        if prop is None:
            continue
        return {
            "question": (
                f"What is the total {prop} of all {_humanise(label)} "
                f"elements?"
            ),
            "cypher": f"MATCH (n:{label}) RETURN sum(n.{prop})",
        }
    return None


def _t_top_one_desc(ctx: _TemplateContext) -> Optional[Dict[str, str]]:
    for label in ("IfcColumn", "IfcWall", "IfcWindow", "IfcDoor", "IfcSpace"):
        if not ctx.entity_with_count(label):
            continue
        prop = ctx.numeric_property_for(label)
        if prop is None:
            continue
        return {
            "question": (
                f"Which {_humanise(label)} element has the largest {prop}?"
            ),
            "cypher": (
                f"MATCH (n:{label}) "
                f"RETURN n.Name ORDER BY n.{prop} DESC LIMIT 1"
            ),
        }
    return None


_TEMPLATES: List[Callable[[_TemplateContext], Optional[Dict[str, str]]]] = [
    _t_count_by_label,
    _t_filter_by_categorical,
    _t_storey_contains,
    _t_storey_spaces,
    _t_aggregate_numeric,
    _t_top_one_desc,
]


def generate_few_shot_examples(
    driver,
    vocab: CombinedVocabulary,
    enumerations: Optional[Dict[str, List[str]]] = None,
    n: int = 5,
) -> List[Dict[str, str]]:
    """Build up to ``n`` graph-sourced few-shot Q/A pairs.

    Each template only fires when the underlying entity, property, or
    relationship actually exists in this graph; values are read from
    ``enumerations`` (sampled separately) or sourced just-in-time from
    the driver. Examples that fail to execute are dropped — the prompt
    must never advertise a query that breaks.
    """
    enumerations = enumerations or {}

    label_counts: Dict[str, int] = {}
    relationships: Set[str] = set()
    with driver.session() as session:
        for record in session.run(
            "CALL db.labels() YIELD label RETURN label"
        ):
            label = record["label"]
            count_row = session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS c"
            ).single()
            if count_row is not None:
                label_counts[label] = count_row["c"]
        for record in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType"
        ):
            relationships.add(record["relationshipType"])

    ctx = _TemplateContext(
        vocab=vocab,
        enumerations=enumerations,
        label_counts=label_counts,
        relationships=relationships,
    )

    examples: List[Dict[str, str]] = []
    seen_cyphers: Set[str] = set()
    for template in _TEMPLATES:
        if len(examples) >= n:
            break
        try:
            example = template(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Template %s failed: %s", template.__name__, exc)
            continue
        if example is None:
            continue
        if example["cypher"] in seen_cyphers:
            continue

        # Smoke-execute so we never advertise a broken example.
        if not _example_runs(driver, example["cypher"]):
            logger.debug(
                "Dropping non-executing few-shot from %s: %s",
                template.__name__, example["cypher"],
            )
            continue

        examples.append(example)
        seen_cyphers.add(example["cypher"])

    logger.info("Generated %d graph-sourced few-shot examples", len(examples))
    return examples


def _example_runs(driver, cypher: str) -> bool:
    """Return True iff ``cypher`` executes against the live graph."""
    try:
        with driver.session() as session:
            session.run(cypher).consume()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Few-shot smoke test failed for %r: %s", cypher, exc)
        return False
