"""
Vocabulary Merger for Hybrid Schema.

Merges the "Available Vocabulary" from IFC model scanning with the
"Enforced Vocabulary" from IDS constraints to create a Super-Schema.

The Super-Schema tells the LLM:
- ALL valid entity names and property names (from IFC scan)
- Which properties have STRICT enumerated values (from IDS)
- Which properties are OPEN (any value allowed)

Author: NL2Cypher Research Team
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ids_parser import IDSParser, IDSSchema
from .schema_scanner import IFCSchemaScanner, IFCModelSchema

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

class PropertyType(str, Enum):
    """Type classification for properties."""
    STRICT = "strict"      # Has enumerated values from IDS
    BOOLEAN = "boolean"    # Boolean type
    NUMERIC = "numeric"    # Numeric type (int, float)
    STRING = "string"      # String/text type
    OPEN = "open"          # Any value allowed


@dataclass
class PropertyVocabulary:
    """
    Vocabulary definition for a single property.

    Attributes:
        name: Property name
        property_type: Classification (STRICT, BOOLEAN, NUMERIC, STRING, OPEN)
        allowed_values: Set of allowed values (for STRICT properties)
        data_type: IFC data type if known (e.g., IFCLABEL, IFCREAL)
        source: Where this property came from ("ids", "ifc", "both")
        description: Optional description
    """
    name: str
    property_type: PropertyType = PropertyType.OPEN
    allowed_values: Set[str] = field(default_factory=set)
    data_type: Optional[str] = None
    source: str = "ifc"
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "PropertyVocabulary":
        prop_type = data.get("property_type", PropertyType.OPEN.value)
        if isinstance(prop_type, PropertyType):
            pt = prop_type
        else:
            pt = PropertyType(prop_type)
        return cls(
            name=data.get("name", ""),
            property_type=pt,
            allowed_values=set(data.get("allowed_values") or []),
            data_type=data.get("data_type"),
            source=data.get("source", "ifc"),
            description=data.get("description"),
        )

    def is_strict(self) -> bool:
        """Check if this property has strict constraints."""
        return self.property_type == PropertyType.STRICT and bool(self.allowed_values)
    
    def format_for_prompt(self) -> str:
        """Format this property for inclusion in LLM prompt."""
        if self.property_type == PropertyType.STRICT and self.allowed_values:
            values_str = ", ".join(f"'{v}'" for v in sorted(self.allowed_values))
            return f"- {self.name} (STRICT: Must be one of [{values_str}])"
        elif self.property_type == PropertyType.BOOLEAN:
            return f"- {self.name} (Boolean: true/false)"
        elif self.property_type == PropertyType.NUMERIC:
            return f"- {self.name} (Numeric)"
        else:
            return f"- {self.name}"


@dataclass
class EntityVocabulary:
    """
    Complete vocabulary for a single entity type.
    
    Attributes:
        name: Entity name (e.g., "IfcWall")
        properties: Dict mapping property names to PropertyVocabulary
        relations: Set of valid relationship types
        count: Number of instances in the model
        from_ids: Whether this entity is defined in IDS
        from_ifc: Whether this entity exists in IFC model
    """
    name: str
    properties: Dict[str, PropertyVocabulary] = field(default_factory=dict)
    relations: Set[str] = field(default_factory=set)
    count: int = 0
    from_ids: bool = False
    from_ifc: bool = False
    
    def get_strict_properties(self) -> List[PropertyVocabulary]:
        """Get properties with strict constraints."""
        return [p for p in self.properties.values() if p.is_strict()]
    
    def get_open_properties(self) -> List[PropertyVocabulary]:
        """Get properties without strict constraints."""
        return [p for p in self.properties.values() if not p.is_strict()]
    
    def get_all_property_names(self) -> Set[str]:
        """Get all property names for this entity."""
        return set(self.properties.keys())

    @classmethod
    def from_dict(cls, data: Dict) -> "EntityVocabulary":
        return cls(
            name=data.get("name", ""),
            properties={
                k: PropertyVocabulary.from_dict(v)
                for k, v in (data.get("properties") or {}).items()
            },
            relations=set(data.get("relations") or []),
            count=data.get("count", 0),
            from_ids=data.get("from_ids", False),
            from_ifc=data.get("from_ifc", False),
        )


@dataclass
class CombinedVocabulary:
    """
    The Super-Schema combining IDS constraints and IFC vocabulary.
    
    Attributes:
        entities: Dict mapping entity names to EntityVocabulary
        all_properties: Dict mapping property names to PropertyVocabulary
        strict_properties: Set of property names with strict constraints
        open_properties: Set of property names without constraints
        all_relations: Set of all relationship types
        ids_schema: Original IDS schema (if loaded)
        ifc_schema: Original IFC schema (if loaded)
    """
    entities: Dict[str, EntityVocabulary] = field(default_factory=dict)
    all_properties: Dict[str, PropertyVocabulary] = field(default_factory=dict)
    strict_properties: Set[str] = field(default_factory=set)
    open_properties: Set[str] = field(default_factory=set)
    all_relations: Set[str] = field(default_factory=set)
    # (source_label, rel_type, target_label) triples read from the loaded graph.
    # Relationship *names* alone are not enough to write a correct traversal —
    # see merge_graph_stats_into_vocabulary.
    relationship_signatures: List[Tuple[str, str, str]] = field(default_factory=list)
    ids_schema: Optional[IDSSchema] = None
    ifc_schema: Optional[IFCModelSchema] = None
    
    def get_entity_names(self) -> Set[str]:
        """Get all entity names (excludes empty strings)."""
        return {name for name in self.entities.keys() if name}
    
    def get_all_property_names(self) -> Set[str]:
        """Get all property names."""
        return set(self.all_properties.keys())
    
    def get_property(self, name: str) -> Optional[PropertyVocabulary]:
        """Get property vocabulary by name (case-insensitive)."""
        # Try exact match first
        if name in self.all_properties:
            return self.all_properties[name]
        
        # Try case-insensitive match
        name_lower = name.lower()
        for prop_name, prop in self.all_properties.items():
            if prop_name.lower() == name_lower:
                return prop
        
        return None
    
    def get_properties_for_entity(self, entity_name: str) -> Dict[str, PropertyVocabulary]:
        """Get all properties for a specific entity."""
        if entity_name in self.entities:
            return self.entities[entity_name].properties
        return {}

    @classmethod
    def from_dict(cls, data: Dict) -> "CombinedVocabulary":
        """Rehydrate from a bundle JSON dict (produced via ``dataclasses.asdict``).

        ``ifc_schema`` is intentionally dropped on the way back in — the heavy
        IFCModelSchema is only used during merging, not during prompt building
        or grammar generation, so the client side never needs it. ``ids_schema``
        is rehydrated via ``IDSSchema.from_dict`` so downstream code that probes
        ``vocab.ids_schema`` still works.
        """
        ids_blob = data.get("ids_schema")
        ids_schema = IDSSchema.from_dict(ids_blob) if isinstance(ids_blob, dict) else None

        return cls(
            entities={
                k: EntityVocabulary.from_dict(v)
                for k, v in (data.get("entities") or {}).items()
            },
            all_properties={
                k: PropertyVocabulary.from_dict(v)
                for k, v in (data.get("all_properties") or {}).items()
            },
            strict_properties=set(data.get("strict_properties") or []),
            open_properties=set(data.get("open_properties") or []),
            all_relations=set(data.get("all_relations") or []),
            relationship_signatures=[
                tuple(sig) for sig in (data.get("relationship_signatures") or [])
            ],
            ids_schema=ids_schema,
            ifc_schema=None,
        )


# =============================================================================
# Vocabulary Merger
# =============================================================================

def merge_graph_stats_into_vocabulary(
    vocab: CombinedVocabulary,
    graph_stats: Dict[str, Any],
) -> CombinedVocabulary:
    """Fold the loaded graph's own labels and relationship types into a vocabulary.

    The IFC scan sees products; the loaded Neo4j graph sees more. On Building 15
    the scan yields 23 entities and 4 relationship types while the graph holds 76
    labels (``IfcMaterial``, the IFC supertype chain, type objects) and 8
    relationship types.

    Anything missing here is undecodable under CYPHER_STRICT, invisible to the
    prompt under CYPHER_SOFT, and counted as a schema violation by SCR — which is
    how ``BOUNDED_BY`` came to be absent from all three. Both the bundle builder
    and the API must call this so client-side decoding and server-side scoring
    agree on one vocabulary.

    Mutates and returns ``vocab``.
    """
    graph_rels = set(graph_stats.get("relationships") or {})
    new_rels = graph_rels - set(vocab.all_relations)
    vocab.all_relations |= graph_rels

    graph_labels = graph_stats.get("labels") or {}
    new_labels = [lbl for lbl in graph_labels if lbl and lbl not in vocab.entities]
    for label in new_labels:
        vocab.entities[label] = EntityVocabulary(
            name=label,
            count=graph_labels[label],
            from_ids=False,
            from_ifc=False,
        )

    vocab.relationship_signatures = [
        tuple(sig) for sig in (graph_stats.get("relationship_signatures") or [])
    ]

    logger.info(
        "Merged graph stats into vocabulary: +%d relationship types %s, "
        "+%d labels, %d relationship signatures (now %d rels, %d entities)",
        len(new_rels), sorted(new_rels), len(new_labels),
        len(vocab.relationship_signatures),
        len(vocab.all_relations), len(vocab.entities),
    )
    return vocab


class VocabularyMerger:
    """
    Merges IDS constraints with IFC vocabulary to create a Super-Schema.
    
    Example:
        ```python
        merger = VocabularyMerger()
        vocab = merger.merge(
            ids_path="/path/to/requirements.ids",
            ifc_path="/path/to/model.ifc"
        )
        
        print(vocab.get_entity_names())
        print(vocab.strict_properties)
        ```
    """
    
    # Common property type mappings
    BOOLEAN_PROPERTIES = {
        "isexternal", "loadbearing", "isclosed", "islandmark",
        "compartmentation", "status",
    }
    
    NUMERIC_PROPERTIES = {
        "height", "width", "length", "depth", "thickness",
        "area", "volume", "netarea", "grossarea", "netvolume", "grossvolume",
        "perimeter", "slope", "rise", "run",
        "nominalheight", "nominalwidth", "nominallength",
        "thermaltransmittance",
    }
    
    def __init__(self):
        """Initialize the merger."""
        self._ids_parser = IDSParser()
        self._ifc_scanner = IFCSchemaScanner()
    
    def merge(
        self,
        ids_path: Optional[Path | str] = None,
        ifc_path: Optional[Path | str] = None,
        ids_schema: Optional[IDSSchema] = None,
        ifc_schema: Optional[IFCModelSchema] = None,
    ) -> CombinedVocabulary:
        """
        Merge IDS constraints with IFC vocabulary.
        
        Args:
            ids_path: Path to IDS file (optional if ids_schema provided)
            ifc_path: Path to IFC file (optional if ifc_schema provided)
            ids_schema: Pre-loaded IDS schema (optional)
            ifc_schema: Pre-loaded IFC schema (optional)
            
        Returns:
            CombinedVocabulary with merged schema
        """
        vocab = CombinedVocabulary()
        
        # Load IDS schema
        if ids_schema:
            vocab.ids_schema = ids_schema
        elif ids_path and Path(ids_path).exists():
            logger.info(f"Loading IDS schema from: {ids_path}")
            vocab.ids_schema = self._ids_parser.parse(ids_path)
        
        # Load IFC schema
        if ifc_schema:
            vocab.ifc_schema = ifc_schema
        elif ifc_path and Path(ifc_path).exists():
            logger.info(f"Scanning IFC model: {ifc_path}")
            vocab.ifc_schema = self._ifc_scanner.scan(ifc_path)
        
        # Merge entities and properties
        self._merge_entities(vocab)
        self._merge_properties(vocab)
        self._merge_relations(vocab)
        self._classify_properties(vocab)
        
        logger.info(
            f"Merged vocabulary: {len(vocab.entities)} entities, "
            f"{len(vocab.all_properties)} properties "
            f"({len(vocab.strict_properties)} strict, {len(vocab.open_properties)} open)"
        )
        
        return vocab
    
    def _merge_entities(self, vocab: CombinedVocabulary) -> None:
        """Merge entity definitions from IDS and IFC."""
        # Add IFC entities first (the "available" vocabulary)
        if vocab.ifc_schema:
            for entity_name, entity_schema in vocab.ifc_schema.entities.items():
                entity_vocab = EntityVocabulary(
                    name=entity_name,
                    count=entity_schema.count,
                    relations=entity_schema.relations.copy(),
                    from_ifc=True,
                )
                vocab.entities[entity_name] = entity_vocab
        
        # Mark entities that are also in IDS (the "enforced" vocabulary)
        if vocab.ids_schema:
            for ifc_entity in vocab.ids_schema.entities:
                # Convert IDS entity name to Neo4j format
                neo4j_name = self._ids_to_neo4j_name(ifc_entity)
                
                if neo4j_name in vocab.entities:
                    vocab.entities[neo4j_name].from_ids = True
                else:
                    # Entity in IDS but not in IFC model - add it anyway
                    vocab.entities[neo4j_name] = EntityVocabulary(
                        name=neo4j_name,
                        from_ids=True,
                        from_ifc=False,
                    )
    
    def _merge_properties(self, vocab: CombinedVocabulary) -> None:
        """Merge property definitions from IDS and IFC."""
        # First, add all properties from IFC scan (open by default)
        if vocab.ifc_schema:
            for prop_name in vocab.ifc_schema.get_all_property_names():
                prop_vocab = PropertyVocabulary(
                    name=prop_name,
                    property_type=PropertyType.OPEN,
                    source="ifc",
                )
                vocab.all_properties[prop_name] = prop_vocab
            
            # Add properties to their entities
            for entity_name, entity_schema in vocab.ifc_schema.entities.items():
                if entity_name in vocab.entities:
                    for prop_name in entity_schema.all_properties():
                        if prop_name in vocab.all_properties:
                            vocab.entities[entity_name].properties[prop_name] = \
                                vocab.all_properties[prop_name]
        
        # Then, add/update with IDS constraints
        if vocab.ids_schema:
            # Add properties from IDS
            for prop_name in vocab.ids_schema.properties:
                if prop_name in vocab.all_properties:
                    vocab.all_properties[prop_name].source = "both"
                else:
                    vocab.all_properties[prop_name] = PropertyVocabulary(
                        name=prop_name,
                        property_type=PropertyType.OPEN,
                        source="ids",
                    )
            
            # Apply enumerated value constraints from IDS
            for prop_name, allowed_values in vocab.ids_schema.property_values.items():
                if allowed_values:
                    if prop_name in vocab.all_properties:
                        vocab.all_properties[prop_name].property_type = PropertyType.STRICT
                        vocab.all_properties[prop_name].allowed_values = allowed_values
                    else:
                        vocab.all_properties[prop_name] = PropertyVocabulary(
                            name=prop_name,
                            property_type=PropertyType.STRICT,
                            allowed_values=allowed_values,
                            source="ids",
                        )
            
            # Apply constraints from entity-specific properties
            for entity_name, entity_constraint in vocab.ids_schema.entity_constraints.items():
                neo4j_name = self._ids_to_neo4j_name(entity_name)
                
                for prop_constraint in entity_constraint.properties:
                    prop_name = prop_constraint.name
                    
                    # Update global property
                    if prop_name in vocab.all_properties:
                        if prop_constraint.allowed_values:
                            vocab.all_properties[prop_name].property_type = PropertyType.STRICT
                            vocab.all_properties[prop_name].allowed_values.update(
                                prop_constraint.allowed_values
                            )
                        if prop_constraint.data_type:
                            vocab.all_properties[prop_name].data_type = prop_constraint.data_type
                    
                    # Add to entity if exists
                    if neo4j_name in vocab.entities:
                        if prop_name not in vocab.entities[neo4j_name].properties:
                            vocab.entities[neo4j_name].properties[prop_name] = \
                                vocab.all_properties.get(prop_name, PropertyVocabulary(name=prop_name))
    
    def _merge_relations(self, vocab: CombinedVocabulary) -> None:
        """Merge relationship types."""
        if vocab.ifc_schema:
            vocab.all_relations = vocab.ifc_schema.all_relations.copy()
        
        # Add standard relationships
        vocab.all_relations.update({"CONTAINS", "DECOMPOSES"})
    
    def _classify_properties(self, vocab: CombinedVocabulary) -> None:
        """Classify properties by type (boolean, numeric, etc.)."""
        for prop_name, prop in vocab.all_properties.items():
            # Skip already-classified strict properties
            if prop.property_type == PropertyType.STRICT:
                vocab.strict_properties.add(prop_name)
                continue
            
            # Infer type from name
            name_lower = prop_name.lower()
            
            if name_lower in self.BOOLEAN_PROPERTIES or name_lower.startswith("is"):
                prop.property_type = PropertyType.BOOLEAN
            elif name_lower in self.NUMERIC_PROPERTIES:
                prop.property_type = PropertyType.NUMERIC
            elif prop.data_type:
                # Infer from IFC data type
                if "BOOLEAN" in prop.data_type.upper():
                    prop.property_type = PropertyType.BOOLEAN
                elif any(t in prop.data_type.upper() for t in ["REAL", "INTEGER", "LENGTH", "AREA", "VOLUME"]):
                    prop.property_type = PropertyType.NUMERIC
                elif "LABEL" in prop.data_type.upper() or "TEXT" in prop.data_type.upper():
                    prop.property_type = PropertyType.STRING
            
            vocab.open_properties.add(prop_name)
    
    def _ids_to_neo4j_name(self, ifc_name: str) -> str:
        """
        Convert IDS entity name to Neo4j label format.

        Uses ifcopenshell's schema to get the canonical casing that
        matches element.is_a() (and thus Neo4j labels from the ETL).

        IFCWALL -> IfcWall
        IFCBUILDINGSTOREY -> IfcBuildingStorey
        """
        try:
            import ifcopenshell
            ifc_schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name("IFC4")
            decl = ifc_schema.declaration_by_name(ifc_name)
            return decl.name()
        except Exception:
            pass
        # Fallback
        if ifc_name.upper().startswith("IFC"):
            return "Ifc" + ifc_name[3:].title().replace("_", "")
        return ifc_name.title().replace("_", "")


# =============================================================================
# Convenience Functions
# =============================================================================

def build_combined_vocabulary(
    ids_path: Optional[Path | str] = None,
    ifc_path: Optional[Path | str] = None,
) -> CombinedVocabulary:
    """
    Build combined vocabulary from IDS and IFC files.
    
    Args:
        ids_path: Path to IDS file (optional)
        ifc_path: Path to IFC file (optional)
        
    Returns:
        CombinedVocabulary with merged schema
    """
    merger = VocabularyMerger()
    return merger.merge(ids_path=ids_path, ifc_path=ifc_path)


def get_schema_for_grammar(
    ids_path: Optional[Path | str] = None,
    ifc_path: Optional[Path | str] = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Get entity and property sets for grammar generation.
    
    Args:
        ids_path: Path to IDS file (optional)
        ifc_path: Path to IFC file (optional)
        
    Returns:
        Tuple of (entity_names, property_names)
    """
    vocab = build_combined_vocabulary(ids_path, ifc_path)
    return vocab.get_entity_names(), vocab.get_all_property_names()


# =============================================================================
# System Prompt Generation
# =============================================================================

def get_system_schema_string(vocab: CombinedVocabulary) -> str:
    """
    Generate a formatted schema string for the LLM system prompt.
    
    Args:
        vocab: Combined vocabulary
        
    Returns:
        Formatted schema string
    """
    lines = [
        "You are a Neo4j Cypher expert for BIM data. Use ONLY the following schema:",
        "",
        "=" * 50,
        "ENTITIES (Valid Node Labels):",
        "=" * 50,
    ]
    
    # List entities
    entity_names = sorted(vocab.get_entity_names())
    lines.append(", ".join(entity_names))
    
    lines.extend([
        "",
        "=" * 50,
        "PROPERTIES BY ENTITY:",
        "=" * 50,
    ])
    
    # Group properties by entity
    for entity_name in sorted(vocab.entities.keys()):
        entity = vocab.entities[entity_name]
        if not entity.properties:
            continue
        
        lines.append(f"\n{entity_name}:")
        
        # Strict properties first
        strict_props = entity.get_strict_properties()
        if strict_props:
            for prop in sorted(strict_props, key=lambda p: p.name):
                lines.append(f"  {prop.format_for_prompt()}")
        
        # Then open properties
        open_props = entity.get_open_properties()
        for prop in sorted(open_props, key=lambda p: p.name)[:15]:  # Limit for prompt size
            lines.append(f"  {prop.format_for_prompt()}")
        
        if len(open_props) > 15:
            lines.append(f"  ... and {len(open_props) - 15} more properties")
    
    # Relationships. Prefer typed signatures: a bare list of names does not say
    # which relationship connects which labels, and the model guesses wrong —
    # CONTAINS reaches doors from a storey but spaces need DECOMPOSES, and every
    # "per floor" query written with CONTAINS returns zero rows.
    if vocab.relationship_signatures:
        lines.extend([
            "",
            "=" * 50,
            "RELATIONSHIPS (exact patterns that exist in the graph —",
            "any combination not listed here returns no rows):",
            "=" * 50,
        ])
        by_rel: Dict[str, List[Tuple[str, str]]] = {}
        for src, rel, dst in vocab.relationship_signatures:
            by_rel.setdefault(rel, []).append((src, dst))
        for rel in sorted(by_rel):
            lines.append(f"  {rel}:")
            for src, dst in sorted(by_rel[rel]):
                lines.append(f"    ({src})-[:{rel}]->({dst})")
    elif vocab.all_relations:
        lines.extend([
            "",
            "=" * 50,
            "RELATIONSHIPS:",
            "=" * 50,
        ])
        for rel in sorted(vocab.all_relations):
            lines.append(f"  - {rel}")
    
    # Strict property summary
    if vocab.strict_properties:
        lines.extend([
            "",
            "=" * 50,
            "STRICT PROPERTIES (Must use exact values):",
            "=" * 50,
        ])
        for prop_name in sorted(vocab.strict_properties):
            prop = vocab.all_properties[prop_name]
            if prop.allowed_values:
                values = ", ".join(f"'{v}'" for v in sorted(prop.allowed_values))
                lines.append(f"  {prop_name}: [{values}]")
    
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for vocabulary merging."""
    import sys
    
    # Default paths
    ids_path = Path("data/requirements.ids")
    ifc_path = Path("data/model.ifc")
    
    # Override from command line
    if len(sys.argv) >= 2:
        ifc_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        ids_path = Path(sys.argv[2])
    
    print(f"IFC Path: {ifc_path}")
    print(f"IDS Path: {ids_path}")
    print()
    
    vocab = build_combined_vocabulary(
        ids_path=ids_path if ids_path.exists() else None,
        ifc_path=ifc_path if ifc_path.exists() else None,
    )
    
    print("=" * 60)
    print("COMBINED VOCABULARY SUMMARY")
    print("=" * 60)
    print(f"\nEntities: {len(vocab.entities)}")
    print(f"Total Properties: {len(vocab.all_properties)}")
    print(f"Strict Properties: {len(vocab.strict_properties)}")
    print(f"Open Properties: {len(vocab.open_properties)}")
    print(f"Relationships: {len(vocab.all_relations)}")
    
    print("\n" + "=" * 60)
    print("STRICT PROPERTIES (with enumerated values):")
    print("=" * 60)
    for prop_name in sorted(vocab.strict_properties):
        prop = vocab.all_properties[prop_name]
        print(f"  {prop_name}: {sorted(prop.allowed_values)}")
    
    print("\n" + "=" * 60)
    print("SYSTEM PROMPT SCHEMA:")
    print("=" * 60)
    print(get_system_schema_string(vocab))


if __name__ == "__main__":
    main()
