"""
IFC Schema Scanner.

Scans an IFC model file to extract ALL available entities and properties,
building a complete vocabulary of what's actually present in the model.

This "Available Vocabulary" is then merged with the IDS "Enforced Vocabulary"
to create a Super-Schema that the LLM can use for constrained generation.

Author: NL2Cypher Research Team
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import ifcopenshell
import ifcopenshell.util.element

from src.utils.ifc_attributes import storable_attributes
from src.utils.ifc_edges import EDGE_SPECS
from src.utils.property_names import canonical_property_name

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EntitySchema:
    """
    Schema for a single IFC entity type.
    
    Attributes:
        name: IFC class name (e.g., "IfcWall")
        properties: Set of property names found on this entity type
        quantity_properties: Set of quantity property names (from Qto_* psets)
        relations: Set of relationship types this entity participates in
        count: Number of instances found in the model
    """
    name: str
    properties: Set[str] = field(default_factory=set)
    quantity_properties: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    count: int = 0
    
    def all_properties(self) -> Set[str]:
        """Get all properties including quantities."""
        return self.properties | self.quantity_properties


@dataclass
class IFCModelSchema:
    """
    Complete schema extracted from an IFC model.
    
    Attributes:
        entities: Dict mapping entity names to EntitySchema
        all_properties: Set of all unique property names across all entities
        all_quantity_properties: Set of all quantity property names
        all_relations: Set of all relationship types
        property_sets: Dict mapping pset names to their properties
        model_info: Basic model metadata
    """
    entities: Dict[str, EntitySchema] = field(default_factory=dict)
    all_properties: Set[str] = field(default_factory=set)
    all_quantity_properties: Set[str] = field(default_factory=set)
    all_relations: Set[str] = field(default_factory=set)
    property_sets: Dict[str, Set[str]] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_neo4j_labels(self) -> Set[str]:
        """Get entity names in Neo4j label format."""
        return set(self.entities.keys())
    
    def get_all_property_names(self) -> Set[str]:
        """Get all property names (regular + quantities)."""
        return self.all_properties | self.all_quantity_properties
    
    def get_properties_for_entity(self, entity_name: str) -> Set[str]:
        """Get all properties for a specific entity type."""
        if entity_name in self.entities:
            return self.entities[entity_name].all_properties()
        return set()


# =============================================================================
# Schema Scanner
# =============================================================================

class IFCSchemaScanner:
    """
    Scans an IFC model to extract the complete vocabulary of entities,
    properties, and relationships.
    
    Example:
        ```python
        scanner = IFCSchemaScanner()
        schema = scanner.scan("/path/to/model.ifc")
        
        print(f"Entities: {schema.get_neo4j_labels()}")
        print(f"Properties: {schema.get_all_property_names()}")
        ```
    """
    
    # Which IFC direct attributes an entity has is derived from the schema at
    # scan time (see ``src.utils.ifc_attributes``), not listed here. The list
    # that used to live at this spot held eight names and was unioned onto
    # *every* entity regardless of whether the entity declared them -- and
    # three of them (``LongName``, ``PredefinedType``, ``Tag``) were not
    # written by the loader at all, so they reached the prompt, passed SCR,
    # decoded fine under STRICT, and returned nulls.

    # Property-set keys ifcopenshell injects as bookkeeping, plus inverse
    # attribute names that are relationships rather than properties. These are
    # structural: they are never authored property names.
    SKIP_PROPERTIES = {
        "id", "type", "OwnerHistory", "ObjectPlacement", 
        "Representation", "HasAssignments", "Nests", "IsNestedBy",
        "HasContext", "IsDecomposedBy", "Decomposes", "HasAssociations",
    }
    
    def __init__(self):
        """Initialize the scanner."""
        self._ifc_file = None
    
    def scan(self, file_path: Path | str) -> IFCModelSchema:
        """
        Scan an IFC file and extract the complete schema.
        
        Args:
            file_path: Path to the IFC file
            
        Returns:
            IFCModelSchema with all entities, properties, and relations
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"IFC file not found: {path}")
        
        logger.info(f"Scanning IFC schema from: {path}")
        
        # Open IFC file
        self._ifc_file = ifcopenshell.open(str(path))
        
        # Initialize schema
        schema = IFCModelSchema()
        
        # Extract model info
        schema.model_info = self._extract_model_info()
        
        # Scan all products for entities and properties
        self._scan_products(schema)
        
        # Scan relationships
        self._scan_relationships(schema)
        
        
        logger.info(
            f"Schema scan complete: {len(schema.entities)} entities, "
            f"{len(schema.get_all_property_names())} unique properties"
        )
        
        return schema
    
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract basic model metadata."""
        info = {
            "schema": self._ifc_file.schema,
            "project_name": None,
            "site_name": None,
            "building_name": None,
        }
        
        # Get project info
        projects = self._ifc_file.by_type("IfcProject")
        if projects:
            info["project_name"] = projects[0].Name
        
        # Get site info
        sites = self._ifc_file.by_type("IfcSite")
        if sites:
            info["site_name"] = sites[0].Name
        
        # Get building info
        buildings = self._ifc_file.by_type("IfcBuilding")
        if buildings:
            info["building_name"] = buildings[0].Name
        
        return info
    
    def _scan_products(self, schema: IFCModelSchema) -> None:
        """
        Scan all IfcProduct elements for properties.
        
        Iterates through all products (walls, doors, windows, etc.)
        and extracts their property sets.
        """
        # Get all products (physical building elements)
        products = self._ifc_file.by_type("IfcProduct")
        
        logger.info(f"Scanning {len(products)} IfcProduct elements...")
        
        for product in products:
            entity_name = product.is_a()
            
            # Convert to Neo4j label format (already in correct case from ifcopenshell)
            # e.g., "IfcWall" stays "IfcWall"
            
            # Initialize entity schema if needed
            if entity_name not in schema.entities:
                schema.entities[entity_name] = EntitySchema(name=entity_name)
                # Direct IFC attributes, derived from this class's schema
                # declaration. These are the names an IDS speaks
                # (IfcSpace.LongName, IfcDoor.OverallWidth), as opposed to the
                # exporter's property-set vocabulary that get_psets returns.
                attributes = set(storable_attributes(self._ifc_file.schema, entity_name))
                schema.entities[entity_name].properties.update(attributes)
                schema.all_properties.update(attributes)
            
            entity_schema = schema.entities[entity_name]
            entity_schema.count += 1
            
            # Extract properties from property sets
            try:
                psets = ifcopenshell.util.element.get_psets(product, psets_only=False)
                
                for pset_name, pset_props in psets.items():
                    if not isinstance(pset_props, dict):
                        continue
                    
                    # Track pset -> properties mapping
                    if pset_name not in schema.property_sets:
                        schema.property_sets[pset_name] = set()
                    
                    for prop_name, prop_value in pset_props.items():
                        # Skip internal properties
                        if prop_name in self.SKIP_PROPERTIES:
                            continue
                        
                        # Sanitize property name for Neo4j compatibility
                        clean_name = canonical_property_name(prop_name)
                        
                        # Categorize as quantity or regular property
                        if pset_name.startswith("Qto_"):
                            entity_schema.quantity_properties.add(clean_name)
                            schema.all_quantity_properties.add(clean_name)
                        else:
                            entity_schema.properties.add(clean_name)
                            schema.all_properties.add(clean_name)
                        
                        schema.property_sets[pset_name].add(clean_name)
                        
            except Exception as e:
                logger.debug(f"Error extracting psets from {entity_name}: {e}")
        
        # Also scan IfcTypeProduct for type-level properties
        self._scan_type_products(schema)
    
    def _scan_type_products(self, schema: IFCModelSchema) -> None:
        """Scan IfcTypeProduct elements for type-level properties."""
        type_products = self._ifc_file.by_type("IfcTypeProduct")
        
        seen_types: Set[str] = set()
        for type_product in type_products:
            type_name = type_product.is_a()
            if type_name not in seen_types:
                seen_types.add(type_name)
                schema.all_properties.update(
                    storable_attributes(self._ifc_file.schema, type_name)
                )
            try:
                psets = ifcopenshell.util.element.get_psets(type_product, psets_only=False)
                
                for pset_name, pset_props in psets.items():
                    if not isinstance(pset_props, dict):
                        continue
                    
                    if pset_name not in schema.property_sets:
                        schema.property_sets[pset_name] = set()
                    
                    for prop_name in pset_props.keys():
                        if prop_name in self.SKIP_PROPERTIES:
                            continue
                        
                        clean_name = canonical_property_name(prop_name)
                        
                        if pset_name.startswith("Qto_"):
                            schema.all_quantity_properties.add(clean_name)
                        else:
                            schema.all_properties.add(clean_name)
                        
                        schema.property_sets[pset_name].add(clean_name)
                        
            except Exception as e:
                logger.debug(f"Error extracting psets from type: {e}")
    
    def _scan_relationships(self, schema: IFCModelSchema) -> None:
        """
        Scan relationships to understand entity connections.

        Driven by ``src.utils.ifc_edges.EDGE_SPECS`` -- the same table the ETL
        loader materialises from -- so the advertised relationship vocabulary
        cannot drift from what is actually in the graph. This method previously
        knew about four relationship types while the loader wrote eight; the
        four it did not know about were undecodable under STRICT, invisible
        under SOFT, and scored as violations by SCR.

        Material and classification associations are not here because their far
        side is not an ``IfcProduct``; they arrive through the live graph stats
        merge instead.
        """
        for spec in EDGE_SPECS:
            for rel in self._ifc_file.by_type(spec.ifc_class):
                sources = self._entities_of(getattr(rel, spec.source_attr, None))
                targets = self._entities_of(getattr(rel, spec.target_attr, None))
                if not sources or not targets:
                    continue

                schema.all_relations.add(spec.rel_type)
                for entity in sources:
                    name = entity.is_a()
                    if name in schema.entities:
                        schema.entities[name].relations.add(spec.rel_type)

    @staticmethod
    def _entities_of(value: Any) -> Tuple[Any, ...]:
        """Normalise a relationship attribute to a tuple of entities."""
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(v for v in value if hasattr(v, "is_a"))
        return (value,) if hasattr(value, "is_a") else ()


# =============================================================================
# Convenience Functions
# =============================================================================

def scan_ifc_schema(ifc_file_path: Path | str) -> IFCModelSchema:
    """
    Convenience function to scan an IFC file.
    
    Args:
        ifc_file_path: Path to the IFC file
        
    Returns:
        IFCModelSchema with complete vocabulary
    """
    scanner = IFCSchemaScanner()
    return scanner.scan(ifc_file_path)


def get_ifc_vocabulary(ifc_file_path: Path | str) -> Tuple[Set[str], Set[str]]:
    """
    Get entity and property vocabulary from an IFC file.
    
    Args:
        ifc_file_path: Path to the IFC file
        
    Returns:
        Tuple of (entity_names, property_names)
    """
    schema = scan_ifc_schema(ifc_file_path)
    return schema.get_neo4j_labels(), schema.get_all_property_names()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for schema scanning."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.constraints.schema_scanner <ifc_file>")
        sys.exit(1)
    
    scanner = IFCSchemaScanner()
    schema = scanner.scan(sys.argv[1])
    
    print(f"\nIFC Model Schema Summary")
    print("=" * 60)
    print(f"Schema: {schema.model_info.get('schema')}")
    print(f"Project: {schema.model_info.get('project_name')}")
    
    print(f"\nEntities ({len(schema.entities)}):")
    print("-" * 40)
    for name, entity in sorted(schema.entities.items()):
        print(f"  {name} ({entity.count} instances)")
        print(f"    Properties: {len(entity.properties)}")
        print(f"    Quantities: {len(entity.quantity_properties)}")
        if entity.relations:
            print(f"    Relations: {entity.relations}")
    
    print(f"\nAll Properties ({len(schema.all_properties)}):")
    print("-" * 40)
    for prop in sorted(schema.all_properties)[:30]:
        print(f"  - {prop}")
    if len(schema.all_properties) > 30:
        print(f"  ... and {len(schema.all_properties) - 30} more")
    
    print(f"\nQuantity Properties ({len(schema.all_quantity_properties)}):")
    print("-" * 40)
    for prop in sorted(schema.all_quantity_properties)[:20]:
        print(f"  - {prop}")
    if len(schema.all_quantity_properties) > 20:
        print(f"  ... and {len(schema.all_quantity_properties) - 20} more")
    
    print(f"\nRelationship Types ({len(schema.all_relations)}):")
    print("-" * 40)
    for rel in sorted(schema.all_relations):
        print(f"  - {rel}")


if __name__ == "__main__":
    main()
