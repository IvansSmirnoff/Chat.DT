"""
IDS (Information Delivery Specification) Parser.

Parses buildingSMART IDS XML files to extract valid entity names and property
names that define the allowed vocabulary for Cypher query generation.

IDS Specification: https://technical.buildingsmart.org/projects/information-delivery-specification-ids/

Author: NL2Cypher Research Team
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# =============================================================================
# IDS XML Namespaces
# =============================================================================

# Standard IDS namespace
IDS_NS = "http://standards.buildingsmart.org/IDS"

# Namespace map for XPath queries
NAMESPACES = {
    "ids": IDS_NS,
    "xs": "http://www.w3.org/2001/XMLSchema",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PropertyConstraint:
    """
    Represents a property constraint from IDS.
    
    Attributes:
        name: Property name (e.g., "FireRating")
        property_set: Property set name (e.g., "Pset_WallCommon")
        data_type: IFC data type (e.g., "IFCLABEL")
        allowed_values: Set of allowed values (if enumerated)
        cardinality: "required", "optional", or "prohibited"
    """
    name: str
    property_set: Optional[str] = None
    data_type: Optional[str] = None
    allowed_values: Set[str] = field(default_factory=set)
    cardinality: str = "optional"


@dataclass
class EntityConstraint:
    """
    Represents an entity constraint from IDS.
    
    Attributes:
        name: IFC entity name (e.g., "IFCWALL")
        predefined_type: Optional predefined type filter
        properties: List of property constraints for this entity
    """
    name: str
    predefined_type: Optional[str] = None
    properties: List[PropertyConstraint] = field(default_factory=list)


@dataclass
class IDSSchema:
    """
    Complete schema extracted from an IDS file.
    
    Provides sets of valid entities and properties for grammar generation,
    plus detailed constraint information for validation.
    
    Attributes:
        title: IDS document title
        version: IDS document version
        entities: Set of valid entity names (uppercase, e.g., "IFCWALL")
        properties: Set of valid property names (e.g., "FireRating")
        property_sets: Set of valid property set names
        entity_constraints: Detailed entity constraints by name
        property_values: Dict mapping property names to allowed values
    """
    title: str = ""
    version: str = ""
    entities: Set[str] = field(default_factory=set)
    properties: Set[str] = field(default_factory=set)
    property_sets: Set[str] = field(default_factory=set)
    entity_constraints: Dict[str, EntityConstraint] = field(default_factory=dict)
    property_values: Dict[str, Set[str]] = field(default_factory=dict)
    
    def get_neo4j_labels(self) -> Set[str]:
        """
        Convert IFC entity names to Neo4j label format.

        Neo4j labels use PascalCase (e.g., "IfcWall" not "IFCWALL").
        Uses ifcopenshell's schema to get the canonical casing, which
        matches what the ETL loader writes to Neo4j via element.is_a().

        Returns:
            Set of entity names in Neo4j label format
        """
        labels = set()
        try:
            import ifcopenshell
            ifc_schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name("IFC4")
        except Exception:
            ifc_schema = None

        for entity in self.entities:
            if ifc_schema is not None:
                try:
                    # Get canonical casing from ifcopenshell schema
                    decl = ifc_schema.declaration_by_name(entity)
                    labels.add(decl.name())
                    continue
                except Exception:
                    pass
            # Fallback: simple conversion (IFCWALL -> IfcWall)
            if entity.startswith("IFC"):
                label = "Ifc" + entity[3:].title().replace("_", "")
            else:
                label = entity.title().replace("_", "")
            labels.add(label)
        return labels
    
    def get_properties_for_entity(self, entity: str) -> Set[str]:
        """
        Get all valid properties for a specific entity.
        
        Args:
            entity: Entity name (case-insensitive)
            
        Returns:
            Set of property names applicable to this entity
        """
        entity_upper = entity.upper()
        if entity_upper in self.entity_constraints:
            return {p.name for p in self.entity_constraints[entity_upper].properties}
        return self.properties  # Fallback to all properties


# =============================================================================
# IDS Parser
# =============================================================================

class IDSParser:
    """
    Parser for IDS (Information Delivery Specification) XML files.
    
    Extracts entity names and property names that define the valid
    vocabulary for constrained Cypher query generation.
    
    Example:
        ```python
        parser = IDSParser()
        schema = parser.parse("/path/to/requirements.ids")
        
        print(f"Entities: {schema.entities}")
        print(f"Properties: {schema.properties}")
        ```
    """
    
    def __init__(self):
        """Initialize the IDS parser."""
        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None
    
    def parse(self, file_path: Path | str) -> IDSSchema:
        """
        Parse an IDS XML file and extract the schema.
        
        Args:
            file_path: Path to the IDS XML file
            
        Returns:
            IDSSchema containing entities, properties, and constraints
            
        Raises:
            FileNotFoundError: If the IDS file doesn't exist
            ET.ParseError: If the XML is malformed
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"IDS file not found: {path}")
        
        logger.info(f"Parsing IDS file: {path}")
        
        # Parse XML
        self._tree = ET.parse(path)
        self._root = self._tree.getroot()
        
        # Initialize schema
        schema = IDSSchema()
        
        # Extract document info
        self._parse_info(schema)
        
        # Extract specifications
        self._parse_specifications(schema)
        
        # Add common IFC entities if schema is empty (fallback)
        if not schema.entities:
            logger.warning("No entities found in IDS, adding common IFC entities")
            schema.entities = self._get_common_ifc_entities()
        
        # Add common properties if schema is empty (fallback)
        if not schema.properties:
            logger.warning("No properties found in IDS, adding common properties")
            schema.properties = self._get_common_properties()
        
        logger.info(
            f"Parsed IDS: {len(schema.entities)} entities, "
            f"{len(schema.properties)} properties"
        )
        
        return schema
    
    def _parse_info(self, schema: IDSSchema) -> None:
        """Extract document info (title, version)."""
        if self._root is None:
            return
        
        info = self._root.find("ids:info", NAMESPACES)
        if info is not None:
            title_elem = info.find("ids:title", NAMESPACES)
            version_elem = info.find("ids:version", NAMESPACES)
            
            schema.title = title_elem.text if title_elem is not None else ""
            schema.version = version_elem.text if version_elem is not None else ""
    
    def _parse_specifications(self, schema: IDSSchema) -> None:
        """Extract all specifications from the IDS."""
        if self._root is None:
            return
        
        specs = self._root.find("ids:specifications", NAMESPACES)
        if specs is None:
            return
        
        for spec in specs.findall("ids:specification", NAMESPACES):
            self._parse_specification(spec, schema)
    
    def _parse_specification(
        self, spec: ET.Element, schema: IDSSchema
    ) -> None:
        """
        Parse a single specification element.
        
        A specification contains:
        - applicability: Which entities this applies to
        - requirements: What properties/constraints are required
        """
        # Parse applicability (entities)
        applicability = spec.find("ids:applicability", NAMESPACES)
        if applicability is not None:
            entities = self._extract_entities(applicability)
            schema.entities.update(entities)
            
            # Create entity constraints
            for entity_name in entities:
                if entity_name not in schema.entity_constraints:
                    schema.entity_constraints[entity_name] = EntityConstraint(
                        name=entity_name
                    )
        
        # Parse requirements (properties)
        requirements = spec.find("ids:requirements", NAMESPACES)
        if requirements is not None:
            # Get the entities this applies to (for linking properties)
            applicable_entities = (
                self._extract_entities(applicability) if applicability else set()
            )
            
            # Extract properties
            properties = self._extract_properties(requirements, schema)
            schema.properties.update(p.name for p in properties)
            
            # Link properties to entities
            for entity_name in applicable_entities:
                if entity_name in schema.entity_constraints:
                    schema.entity_constraints[entity_name].properties.extend(properties)
    
    def _extract_entities(self, parent: ET.Element) -> Set[str]:
        """
        Extract entity names from an applicability or requirements element.
        
        Looks for patterns like:
        <ids:entity>
            <ids:name>
                <ids:simpleValue>IFCWALL</ids:simpleValue>
            </ids:name>
        </ids:entity>
        """
        entities = set()
        
        for entity_elem in parent.findall(".//ids:entity", NAMESPACES):
            name_elem = entity_elem.find("ids:name", NAMESPACES)
            if name_elem is not None:
                # Try simpleValue first
                simple = name_elem.find("ids:simpleValue", NAMESPACES)
                if simple is not None and simple.text:
                    entities.add(simple.text.upper())
                else:
                    # Try direct text content
                    if name_elem.text:
                        entities.add(name_elem.text.strip().upper())
        
        return entities
    
    def _extract_properties(
        self, parent: ET.Element, schema: IDSSchema
    ) -> List[PropertyConstraint]:
        """
        Extract property constraints from a requirements element.
        
        Looks for patterns like:
        <ids:property dataType="IFCLABEL" cardinality="required">
            <ids:propertySet>
                <ids:simpleValue>Pset_WallCommon</ids:simpleValue>
            </ids:propertySet>
            <ids:baseName>
                <ids:simpleValue>FireRating</ids:simpleValue>
            </ids:baseName>
            <ids:value>
                <xs:restriction>
                    <xs:enumeration value="EI30"/>
                </xs:restriction>
            </ids:value>
        </ids:property>
        """
        properties = []
        
        for prop_elem in parent.findall(".//ids:property", NAMESPACES):
            prop = PropertyConstraint(name="")
            
            # Get attributes
            prop.data_type = prop_elem.get("dataType")
            prop.cardinality = prop_elem.get("cardinality", "optional")
            
            # Get property set name
            pset_elem = prop_elem.find("ids:propertySet", NAMESPACES)
            if pset_elem is not None:
                simple = pset_elem.find("ids:simpleValue", NAMESPACES)
                if simple is not None and simple.text:
                    prop.property_set = simple.text
                    schema.property_sets.add(simple.text)
            
            # Get property name (baseName or name)
            name_elem = (
                prop_elem.find("ids:baseName", NAMESPACES) or 
                prop_elem.find("ids:name", NAMESPACES)
            )
            if name_elem is not None:
                simple = name_elem.find("ids:simpleValue", NAMESPACES)
                if simple is not None and simple.text:
                    prop.name = simple.text
            
            # Skip if no name found
            if not prop.name:
                continue
            
            # Get allowed values (from xs:restriction/xs:enumeration)
            value_elem = prop_elem.find("ids:value", NAMESPACES)
            if value_elem is not None:
                allowed_values = self._extract_enum_values(value_elem)
                prop.allowed_values = allowed_values
                
                # Store in schema-level property values map
                if allowed_values:
                    if prop.name not in schema.property_values:
                        schema.property_values[prop.name] = set()
                    schema.property_values[prop.name].update(allowed_values)
            
            properties.append(prop)
        
        return properties
    
    def _extract_enum_values(self, value_elem: ET.Element) -> Set[str]:
        """Extract enumeration values from an xs:restriction."""
        values = set()
        
        # Look for xs:enumeration elements
        for enum in value_elem.findall(".//xs:enumeration", NAMESPACES):
            val = enum.get("value")
            if val:
                values.add(val)
        
        return values
    
    @staticmethod
    def _get_common_ifc_entities() -> Set[str]:
        """
        Return a set of common IFC entities as fallback.
        
        These are frequently used BIM entities that should be
        queryable even without an IDS file.
        """
        return {
            # Spatial structure
            "IFCPROJECT", "IFCSITE", "IFCBUILDING", "IFCBUILDINGSTOREY", "IFCSPACE",
            # Building elements
            "IFCWALL", "IFCWALLSTANDARDCASE", "IFCSLAB", "IFCROOF",
            "IFCCOLUMN", "IFCBEAM", "IFCMEMBER",
            "IFCDOOR", "IFCWINDOW",
            "IFCSTAIR", "IFCRAMP", "IFCRAILING",
            "IFCCURTAINWALL", "IFCPLATE",
            # MEP
            "IFCFLOWSEGMENT", "IFCFLOWTERMINAL", "IFCFLOWFITTING",
            "IFCPIPESEGMENT", "IFCDUCTSEGMENT",
            # Openings and voids
            "IFCOPENINGELEMENT", "IFCVOIDINGELEMENT",
            # Furnishing
            "IFCFURNISHINGELEMENT",
            # Generic
            "IFCBUILDINGELEMENTPROXY",
        }
    
    @staticmethod
    def _get_common_properties() -> Set[str]:
        """
        Return a set of common IFC properties as fallback.
        
        These are frequently used properties from standard Psets.
        """
        return {
            # Identity
            "GlobalId", "Name", "Description", "ObjectType",
            # Common properties
            "FireRating", "IsExternal", "LoadBearing", "Reference",
            "ThermalTransmittance", "AcousticRating",
            # Quantities
            "NetArea", "GrossArea", "NetVolume", "GrossVolume",
            "Length", "Width", "Height", "Perimeter",
            "NominalLength", "NominalWidth", "NominalHeight",
            # Status
            "Status", "Phase",
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_ids_file(file_path: Path | str) -> IDSSchema:
    """
    Convenience function to parse an IDS file.
    
    Args:
        file_path: Path to the IDS XML file
        
    Returns:
        IDSSchema containing entities and properties
    """
    parser = IDSParser()
    return parser.parse(file_path)


def extract_vocabulary_from_ids(file_path: Path | str) -> Tuple[Set[str], Set[str]]:
    """
    Extract entity and property vocabulary from an IDS file.
    
    Convenience function for grammar generation.
    
    Args:
        file_path: Path to the IDS XML file
        
    Returns:
        Tuple of (entities_set, properties_set)
    """
    schema = parse_ids_file(file_path)
    return schema.get_neo4j_labels(), schema.properties


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for IDS parsing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.constraints.ids_parser <ids_file>")
        sys.exit(1)
    
    parser = IDSParser()
    schema = parser.parse(sys.argv[1])
    
    print(f"\nIDS Document: {schema.title} v{schema.version}")
    print(f"\nEntities ({len(schema.entities)}):")
    for entity in sorted(schema.entities):
        print(f"  - {entity}")
    
    print(f"\nNeo4j Labels:")
    for label in sorted(schema.get_neo4j_labels()):
        print(f"  - :{label}")
    
    print(f"\nProperties ({len(schema.properties)}):")
    for prop in sorted(schema.properties):
        values = schema.property_values.get(prop, set())
        if values:
            print(f"  - {prop}: {sorted(values)}")
        else:
            print(f"  - {prop}")
    
    print(f"\nProperty Sets ({len(schema.property_sets)}):")
    for pset in sorted(schema.property_sets):
        print(f"  - {pset}")


if __name__ == "__main__":
    main()
