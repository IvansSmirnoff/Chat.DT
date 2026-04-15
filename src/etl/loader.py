"""
IFC to Neo4j ETL Loader.

This module transforms IFC (Industry Foundation Classes) BIM data into a Neo4j
Property Graph. It handles:
- Loading IFC files using ifcopenshell
- Flattening IFC property sets into node properties
- Creating relationships for spatial containment and decomposition
- Batch processing for memory efficiency on large models

Author: NL2Cypher Research Team
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import ifcopenshell
import ifcopenshell.util.element
from ifcopenshell import ifcopenshell_wrapper
from neo4j import GraphDatabase, Transaction

from src.config import get_settings, Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Batch size for Neo4j transactions (tune based on model size and memory)
BATCH_SIZE = 1000

# Properties to skip during flattening (internal IFC properties)
SKIP_PROPERTIES = {"id", "type", "GlobalId", "OwnerHistory"}

# Reserved Neo4j property names that need prefixing
RESERVED_NAMES = {"id", "labels", "type", "start", "end"}


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_property_name(name: str, existing_keys: Set[str]) -> str:
    """
    Sanitize a property name for Neo4j compatibility.
    
    - Replaces spaces and special characters with underscores
    - Handles collisions by adding numeric suffix
    - Prefixes reserved Neo4j names
    
    Args:
        name: Original property name from IFC
        existing_keys: Set of already used property names
        
    Returns:
        Sanitized property name safe for Neo4j
    """
    # Replace problematic characters
    sanitized = name.replace(" ", "_").replace(".", "_").replace("-", "_")
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    
    # Handle reserved names
    if sanitized.lower() in RESERVED_NAMES:
        sanitized = f"ifc_{sanitized}"
    
    # Handle collisions with numeric suffix
    original = sanitized
    counter = 1
    while sanitized in existing_keys:
        sanitized = f"{original}_{counter}"
        counter += 1
    
    return sanitized


def safe_value(value: Any) -> Any:
    """
    Convert IFC values to Neo4j-compatible types.
    
    Neo4j supports: bool, int, float, str, list (of primitives)
    
    Args:
        value: Value from IFC property
        
    Returns:
        Neo4j-compatible value or None
    """
    if value is None:
        return None
    
    # Handle basic types
    if isinstance(value, (bool, int, float, str)):
        return value
    
    # Handle ifcopenshell entity references (convert to GlobalId if available)
    if hasattr(value, "GlobalId"):
        return str(value.GlobalId)
    
    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        # Convert to list of safe values, filter out None
        safe_list = [safe_value(v) for v in value if safe_value(v) is not None]
        # Neo4j requires homogeneous lists
        if safe_list and all(isinstance(v, type(safe_list[0])) for v in safe_list):
            return safe_list
        # Convert mixed types to strings
        return [str(v) for v in safe_list] if safe_list else None
    
    # Fallback: convert to string
    try:
        return str(value)
    except Exception:
        return None


def flatten_psets(element: ifcopenshell.entity_instance) -> Dict[str, Any]:
    """
    Flatten all property sets of an IFC element into a flat dictionary.
    
    Uses ifcopenshell.util.element.get_psets() to retrieve all property sets
    and quantity sets, then flattens them for direct node property storage.
    
    Args:
        element: IFC element instance
        
    Returns:
        Dictionary of flattened properties with sanitized keys
    """
    properties: Dict[str, Any] = {}
    used_keys: Set[str] = set()
    
    try:
        # Get all property sets (including quantity sets)
        psets = ifcopenshell.util.element.get_psets(element, psets_only=False)
        
        for pset_name, pset_props in psets.items():
            if not isinstance(pset_props, dict):
                continue
                
            for prop_name, prop_value in pset_props.items():
                # Skip internal properties
                if prop_name in SKIP_PROPERTIES:
                    continue
                
                # Convert value to Neo4j-safe type
                safe_val = safe_value(prop_value)
                if safe_val is None:
                    continue
                
                # Sanitize the property name
                # For LLM simplicity, prefer flat names without pset prefix
                # Only add prefix if there's a collision
                key = sanitize_property_name(prop_name, used_keys)
                
                # If key already exists from another pset, prefix with pset name
                if key in properties:
                    prefixed_key = sanitize_property_name(
                        f"{pset_name}_{prop_name}", used_keys
                    )
                    key = prefixed_key
                
                properties[key] = safe_val
                used_keys.add(key)
                
    except Exception as e:
        logger.warning(f"Error flattening psets for {element}: {e}")
    
    return properties


# =============================================================================
# Main ETL Class
# =============================================================================

class IFCToNeo4jLoader:
    """
    ETL loader for transforming IFC data into Neo4j Property Graph.
    
    This class handles the complete ETL pipeline:
    1. Loading IFC files
    2. Wiping the existing database (for clean experiment state)
    3. Creating constraints/indexes
    4. Creating nodes from IFC entities with flattened properties
    5. Creating relationships (spatial containment, decomposition)
    
    Example:
        ```python
        loader = IFCToNeo4jLoader()
        loader.run()
        ```
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the loader with configuration.

        Args:
            settings: Optional Settings instance. If not provided,
                     loads from environment/config.
        """
        self.settings = settings or get_settings()
        self.driver = None
        self.ifc_file = None
        self._label_chain_cache: Dict[str, Tuple[str, ...]] = {}
        self._schema = None

        # Statistics tracking
        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_flattened": 0,
            "errors": 0,
        }

    def _label_chain(self, ifc_class: str) -> Tuple[str, ...]:
        """
        Return the IFC class plus all of its supertypes, as a label tuple.

        Walks the IFC schema declaration chain (e.g. IfcWallStandardCase ->
        IfcWall -> IfcBuildingElement -> IfcElement -> IfcProduct -> IfcObject
        -> IfcObjectDefinition -> IfcRoot). Cached per leaf class.
        """
        cached = self._label_chain_cache.get(ifc_class)
        if cached is not None:
            return cached
        chain: List[str] = []
        try:
            decl = self._schema.declaration_by_name(ifc_class)
            while decl is not None:
                chain.append(decl.name())
                decl = decl.supertype() if hasattr(decl, "supertype") else None
        except Exception as e:
            logger.debug(f"Could not resolve supertypes for {ifc_class}: {e}")
            chain = [ifc_class, "IfcRoot"]
        if "IfcRoot" not in chain:
            chain.append("IfcRoot")
        result = tuple(chain)
        self._label_chain_cache[ifc_class] = result
        return result
    
    def connect(self) -> None:
        """
        Establish connection to Neo4j database.
        
        Raises:
            Exception: If connection fails
        """
        logger.info(f"Connecting to Neo4j at {self.settings.neo4j_uri}")
        
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(
                self.settings.neo4j_user,
                self.settings.neo4j_password_value
            )
        )
        
        # Verify connectivity
        self.driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j")
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def load_ifc(self, file_path: Optional[Path] = None) -> None:
        """
        Load an IFC file using ifcopenshell.
        
        Args:
            file_path: Path to IFC file. If not provided, uses config.
            
        Raises:
            FileNotFoundError: If IFC file doesn't exist
            Exception: If IFC parsing fails
        """
        path = file_path or self.settings.ifc_file_path
        
        if not Path(path).exists():
            raise FileNotFoundError(f"IFC file not found: {path}")
        
        logger.info(f"Loading IFC file: {path}")
        self.ifc_file = ifcopenshell.open(str(path))

        # Log basic file info
        schema = self.ifc_file.schema
        project = self.ifc_file.by_type("IfcProject")
        project_name = project[0].Name if project else "Unknown"

        # Cache the schema declaration object for supertype walks
        self._schema = ifcopenshell_wrapper.schema_by_name(schema)
        self._label_chain_cache.clear()

        logger.info(f"IFC Schema: {schema}")
        logger.info(f"Project: {project_name}")
    
    def wipe_database(self) -> None:
        """
        Wipe all nodes and relationships from the database.
        
        CRITICAL: This ensures a clean state for the experiment.
        Use with caution in production environments.
        """
        logger.warning("Wiping entire Neo4j database...")
        
        with self.driver.session() as session:
            session.run("""
                MATCH (n)
                CALL { WITH n DETACH DELETE n }
                IN TRANSACTIONS OF 10000 ROWS
            """).consume()

        logger.info("Database wiped successfully")
    
    def create_constraints(self) -> None:
        """
        Create constraints and indexes for performance.
        
        Creates a uniqueness constraint on GlobalId which also
        creates an index for fast lookups.
        """
        logger.info("Creating database constraints and indexes...")
        
        with self.driver.session() as session:
            # Create constraint on GlobalId for all IFC entities
            # This ensures uniqueness and creates an index
            try:
                session.run("""
                    CREATE CONSTRAINT ifc_global_id IF NOT EXISTS
                    FOR (n:IfcRoot)
                    REQUIRE n.GlobalId IS UNIQUE
                """)
                logger.info("Created uniqueness constraint on GlobalId")
            except Exception as e:
                # Constraint might already exist or not be supported
                logger.warning(f"Could not create constraint: {e}")
                
                # Fallback: create index
                try:
                    session.run("""
                        CREATE INDEX ifc_global_id_index IF NOT EXISTS
                        FOR (n:IfcRoot)
                        ON (n.GlobalId)
                    """)
                    logger.info("Created index on GlobalId (fallback)")
                except Exception as e2:
                    logger.warning(f"Could not create index: {e2}")
    
    def _get_ifc_root_elements(self) -> Iterator[ifcopenshell.entity_instance]:
        """
        Iterate through all IfcRoot descendants in the IFC file, skipping
        IfcRelationship subclasses.

        IfcRel* entities are reifications of relationships and are
        materialised as graph edges (CONTAINS, DECOMPOSES, FILLS,
        HAS_OPENING, HAS_MATERIAL, IS_OF_TYPE) — keeping them as nodes
        would just duplicate that information in a less queryable form.

        Yields:
            IFC entity instances that inherit from IfcRoot, except
            IfcRelationship and its subclasses.
        """
        if not self.ifc_file:
            raise RuntimeError("IFC file not loaded. Call load_ifc() first.")

        for element in self.ifc_file.by_type("IfcRoot"):
            if element.is_a("IfcRelationship"):
                continue
            yield element
    
    def _extract_node_data(
        self, element: ifcopenshell.entity_instance
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract node data from an IFC element.
        
        Args:
            element: IFC entity instance
            
        Returns:
            Tuple of (label, global_id, properties_dict)
        """
        # Get IFC class name as label (e.g., "IfcWall", "IfcDoor")
        label = element.is_a()
        
        # Get GlobalId for identity
        global_id = str(element.GlobalId)
        
        # Build base properties
        properties = {
            "GlobalId": global_id,
            "Name": element.Name if hasattr(element, "Name") else None,
            "Description": element.Description if hasattr(element, "Description") else None,
            "ObjectType": element.ObjectType if hasattr(element, "ObjectType") else None,
        }
        
        # Filter out None values
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # Flatten property sets and merge
        flattened = flatten_psets(element)
        properties.update(flattened)
        
        self.stats["properties_flattened"] += len(flattened)
        
        return label, global_id, properties
    
    def _create_nodes_batch(
        self, tx: Transaction, nodes_data: List[Tuple[str, str, Dict[str, Any]]]
    ) -> None:
        """
        Create nodes in a single transaction batch.
        
        Uses UNWIND for efficient bulk insertion and applies
        multiple labels (specific class + IfcRoot for indexing).
        
        Args:
            tx: Neo4j transaction
            nodes_data: List of (label, global_id, properties) tuples
        """
        # Group by full label chain (leaf class + all supertypes) so each
        # group shares one CREATE statement.
        by_chain: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}

        for label, global_id, props in nodes_data:
            chain = self._label_chain(label)
            by_chain.setdefault(chain, []).append(props)

        for chain, props_list in by_chain.items():
            label_clause = ":".join(f"`{l}`" for l in chain)
            query = f"""
                UNWIND $props AS p
                CREATE (n:{label_clause})
                SET n = p
            """
            tx.run(query, props=props_list)
    
    def create_nodes(self) -> None:
        """
        Create Neo4j nodes from all IFC entities.
        
        Iterates through all IfcRoot descendants, extracts properties,
        flattens property sets, and creates nodes in batches.
        """
        logger.info("Creating nodes from IFC entities...")
        
        batch: List[Tuple[str, str, Dict[str, Any]]] = []
        total_count = 0
        
        with self.driver.session() as session:
            for element in self._get_ifc_root_elements():
                try:
                    node_data = self._extract_node_data(element)
                    batch.append(node_data)
                    
                    # Process batch when it reaches BATCH_SIZE
                    if len(batch) >= BATCH_SIZE:
                        session.execute_write(self._create_nodes_batch, batch)
                        total_count += len(batch)
                        logger.info(f"Processed {total_count} nodes...")
                        batch = []
                        
                except Exception as e:
                    logger.error(f"Error processing element {element}: {e}")
                    self.stats["errors"] += 1
            
            # Process remaining batch
            if batch:
                session.execute_write(self._create_nodes_batch, batch)
                total_count += len(batch)
        
        self.stats["nodes_created"] = total_count
        logger.info(f"Created {total_count} nodes")
    
    def _extract_spatial_containment(
        self
    ) -> Iterator[Tuple[str, str]]:
        """
        Extract spatial containment relationships from IFC.
        
        IfcRelContainedInSpatialStructure defines what elements
        are contained within a spatial element (building, storey, space).
        
        Yields:
            Tuples of (spatial_element_id, contained_element_id)
        """
        if not self.ifc_file:
            return
        
        for rel in self.ifc_file.by_type("IfcRelContainedInSpatialStructure"):
            spatial_element = rel.RelatingStructure
            if not spatial_element or not hasattr(spatial_element, "GlobalId"):
                continue
            
            spatial_id = str(spatial_element.GlobalId)
            
            for element in rel.RelatedElements:
                if hasattr(element, "GlobalId"):
                    yield (spatial_id, str(element.GlobalId))
    
    def _extract_decomposition(self) -> Iterator[Tuple[str, str]]:
        """
        Extract decomposition relationships from IFC.
        
        IfcRelAggregates defines hierarchical decomposition
        (e.g., Project -> Site -> Building -> Storey).
        
        Yields:
            Tuples of (parent_id, child_id)
        """
        if not self.ifc_file:
            return
        
        for rel in self.ifc_file.by_type("IfcRelAggregates"):
            parent = rel.RelatingObject
            if not parent or not hasattr(parent, "GlobalId"):
                continue
            
            parent_id = str(parent.GlobalId)
            
            for child in rel.RelatedObjects:
                if hasattr(child, "GlobalId"):
                    yield (parent_id, str(child.GlobalId))

    def _extract_type_relations(self) -> Iterator[Tuple[str, str]]:
        """
        IfcRelDefinesByType: instance -> its type definition.

        Lets queries hop from a wall instance to its IfcWallType (and from
        there to type-level psets, manufacturer, cost, etc.).

        Yields (instance_GlobalId, type_GlobalId).
        """
        if not self.ifc_file:
            return
        for rel in self.ifc_file.by_type("IfcRelDefinesByType"):
            t = getattr(rel, "RelatingType", None)
            if t is None or not hasattr(t, "GlobalId"):
                continue
            t_gid = str(t.GlobalId)
            for obj in rel.RelatedObjects:
                if hasattr(obj, "GlobalId"):
                    yield (str(obj.GlobalId), t_gid)

    def _extract_voids(self) -> Iterator[Tuple[str, str]]:
        """
        IfcRelVoidsElement: building element (wall/slab) -> opening it hosts.

        Yields (host_GlobalId, opening_GlobalId).
        """
        if not self.ifc_file:
            return
        for rel in self.ifc_file.by_type("IfcRelVoidsElement"):
            host = getattr(rel, "RelatingBuildingElement", None)
            opening = getattr(rel, "RelatedOpeningElement", None)
            if host is None or opening is None:
                continue
            if hasattr(host, "GlobalId") and hasattr(opening, "GlobalId"):
                yield (str(host.GlobalId), str(opening.GlobalId))

    def _extract_fills(self) -> Iterator[Tuple[str, str]]:
        """
        IfcRelFillsElement: door/window -> opening it fills.

        Yields (filler_GlobalId, opening_GlobalId).
        """
        if not self.ifc_file:
            return
        for rel in self.ifc_file.by_type("IfcRelFillsElement"):
            opening = getattr(rel, "RelatingOpeningElement", None)
            filler = getattr(rel, "RelatedBuildingElement", None)
            if opening is None or filler is None:
                continue
            if hasattr(filler, "GlobalId") and hasattr(opening, "GlobalId"):
                yield (str(filler.GlobalId), str(opening.GlobalId))

    def _extract_material_associations(self) -> Iterator[Tuple[str, str, str]]:
        """
        Extract material association relationships from IFC.
        
        IfcRelAssociatesMaterial defines which materials are associated
        with building elements.
        
        Yields:
            Tuples of (element_id, material_id, material_name)
        """
        if not self.ifc_file:
            return
        
        for rel in self.ifc_file.by_type("IfcRelAssociatesMaterial"):
            material = rel.RelatingMaterial
            if not material:
                continue
            
            # Handle different material types
            material_name = None
            material_id = None
            
            # IfcMaterial - simple material
            if material.is_a("IfcMaterial"):
                material_name = material.Name if hasattr(material, "Name") else None
                material_id = f"mat_{hash(material_name) if material_name else id(material)}"
            
            # IfcMaterialLayerSet - layered materials (walls, slabs)
            elif material.is_a("IfcMaterialLayerSet"):
                if hasattr(material, "MaterialLayers") and material.MaterialLayers:
                    # Use the first layer's material as representative
                    first_layer = material.MaterialLayers[0]
                    if hasattr(first_layer, "Material") and first_layer.Material:
                        material_name = first_layer.Material.Name
                        material_id = f"mat_{hash(material_name) if material_name else id(material)}"
            
            # IfcMaterialLayerSetUsage
            elif material.is_a("IfcMaterialLayerSetUsage"):
                layer_set = material.ForLayerSet
                if layer_set and hasattr(layer_set, "MaterialLayers") and layer_set.MaterialLayers:
                    first_layer = layer_set.MaterialLayers[0]
                    if hasattr(first_layer, "Material") and first_layer.Material:
                        material_name = first_layer.Material.Name
                        material_id = f"mat_{hash(material_name) if material_name else id(material)}"
            
            # IfcMaterialConstituentSet (IFC4)
            elif material.is_a("IfcMaterialConstituentSet"):
                if hasattr(material, "MaterialConstituents") and material.MaterialConstituents:
                    first_const = material.MaterialConstituents[0]
                    if hasattr(first_const, "Material") and first_const.Material:
                        material_name = first_const.Material.Name
                        material_id = f"mat_{hash(material_name) if material_name else id(material)}"
            
            if not material_name or not material_id:
                continue
            
            # Yield relationships for each related object
            for related_obj in rel.RelatedObjects:
                if hasattr(related_obj, "GlobalId"):
                    yield (str(related_obj.GlobalId), material_id, material_name)

    def _create_material_nodes_and_relationships(
        self,
        session,
        material_data: List[Tuple[str, str, str]]
    ) -> int:
        """
        Create material nodes and HAS_MATERIAL relationships.
        
        Args:
            session: Neo4j session
            material_data: List of (element_id, material_id, material_name) tuples
            
        Returns:
            Number of relationships created
        """
        # First, collect unique materials
        materials = {}
        for _, mat_id, mat_name in material_data:
            if mat_id not in materials:
                materials[mat_id] = mat_name
        
        # Create material nodes
        if materials:
            mat_props = [
                {"material_id": mid, "Name": mname}
                for mid, mname in materials.items()
            ]
            session.run("""
                UNWIND $mats AS m
                MERGE (mat:IfcMaterial {material_id: m.material_id})
                SET mat.Name = m.Name
            """, mats=mat_props)
            logger.info(f"Created/updated {len(materials)} material nodes")
        
        # Create HAS_MATERIAL relationships
        rel_data = [
            {"element_id": eid, "material_id": mid}
            for eid, mid, _ in material_data
        ]
        
        if rel_data:
            session.run("""
                UNWIND $rels AS r
                MATCH (e:IfcRoot {GlobalId: r.element_id})
                MATCH (m:IfcMaterial {material_id: r.material_id})
                MERGE (e)-[:HAS_MATERIAL]->(m)
            """, rels=rel_data)
        
        return len(rel_data)

    def _create_relationships_batch(
        self,
        tx: Transaction,
        relationships: List[Tuple[str, str]],
        rel_type: str
    ) -> None:
        """
        Create relationships in a single transaction batch.
        
        Args:
            tx: Neo4j transaction
            relationships: List of (source_id, target_id) tuples
            rel_type: Relationship type name
        """
        query = f"""
            UNWIND $rels AS r
            MATCH (a:IfcRoot {{GlobalId: r[0]}})
            MATCH (b:IfcRoot {{GlobalId: r[1]}})
            CREATE (a)-[:{rel_type}]->(b)
        """
        tx.run(query, rels=relationships)
    
    def create_relationships(self) -> None:
        """
        Create all relationships between nodes.
        
        Creates two types of relationships:
        - CONTAINS: Spatial containment (storey contains wall)
        - DECOMPOSES: Hierarchical decomposition (building has storeys)
        """
        logger.info("Creating relationships...")
        rel_count = 0
        
        with self.driver.session() as session:
            # Spatial Containment: (:IfcSpatialElement)-[:CONTAINS]->(:IfcElement)
            logger.info("Processing spatial containment relationships...")
            batch: List[Tuple[str, str]] = []
            
            for rel in self._extract_spatial_containment():
                batch.append(rel)
                
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(
                        self._create_relationships_batch, batch, "CONTAINS"
                    )
                    rel_count += len(batch)
                    batch = []
            
            if batch:
                session.execute_write(
                    self._create_relationships_batch, batch, "CONTAINS"
                )
                rel_count += len(batch)
            
            logger.info(f"Created {rel_count} CONTAINS relationships")
            
            # Decomposition: (:IfcObjectDefinition)-[:DECOMPOSES]->(:IfcObjectDefinition)
            logger.info("Processing decomposition relationships...")
            batch = []
            decomp_count = 0
            
            for rel in self._extract_decomposition():
                batch.append(rel)
                
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(
                        self._create_relationships_batch, batch, "DECOMPOSES"
                    )
                    decomp_count += len(batch)
                    batch = []
            
            if batch:
                session.execute_write(
                    self._create_relationships_batch, batch, "DECOMPOSES"
                )
                decomp_count += len(batch)
            
            logger.info(f"Created {decomp_count} DECOMPOSES relationships")
            rel_count += decomp_count

            # Openings: (:IfcElement)-[:HAS_OPENING]->(:IfcOpeningElement)
            logger.info("Processing void (opening host) relationships...")
            batch = []
            voids_count = 0
            for rel in self._extract_voids():
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(
                        self._create_relationships_batch, batch, "HAS_OPENING"
                    )
                    voids_count += len(batch)
                    batch = []
            if batch:
                session.execute_write(
                    self._create_relationships_batch, batch, "HAS_OPENING"
                )
                voids_count += len(batch)
            logger.info(f"Created {voids_count} HAS_OPENING relationships")
            rel_count += voids_count

            # Fills: (:IfcDoor|:IfcWindow)-[:FILLS]->(:IfcOpeningElement)
            logger.info("Processing fill relationships...")
            batch = []
            fills_count = 0
            for rel in self._extract_fills():
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(
                        self._create_relationships_batch, batch, "FILLS"
                    )
                    fills_count += len(batch)
                    batch = []
            if batch:
                session.execute_write(
                    self._create_relationships_batch, batch, "FILLS"
                )
                fills_count += len(batch)
            logger.info(f"Created {fills_count} FILLS relationships")
            rel_count += fills_count

            # Type definitions: (:IfcElement)-[:IS_OF_TYPE]->(:IfcTypeProduct)
            logger.info("Processing type-definition relationships...")
            batch = []
            type_count = 0
            for rel in self._extract_type_relations():
                batch.append(rel)
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(
                        self._create_relationships_batch, batch, "IS_OF_TYPE"
                    )
                    type_count += len(batch)
                    batch = []
            if batch:
                session.execute_write(
                    self._create_relationships_batch, batch, "IS_OF_TYPE"
                )
                type_count += len(batch)
            logger.info(f"Created {type_count} IS_OF_TYPE relationships")
            rel_count += type_count

            # Material Associations: (:IfcElement)-[:HAS_MATERIAL]->(:IfcMaterial)
            logger.info("Processing material association relationships...")
            material_data = list(self._extract_material_associations())
            if material_data:
                mat_rel_count = self._create_material_nodes_and_relationships(
                    session, material_data
                )
                logger.info(f"Created {mat_rel_count} HAS_MATERIAL relationships")
                rel_count += mat_rel_count
            else:
                logger.info("No material associations found in IFC file")

            # Derived: spatial containment is transitive through aggregation.
            # If a spatial element CONTAINS X, and X DECOMPOSES (transitively)
            # into Y, then Y is logically also contained in that spatial
            # element (curtain-wall doors, stair flights, ramp railings, ...).
            # Marked r.derived = true so it is distinguishable from the
            # literal IfcRelContainedInSpatialStructure edges.
            logger.info("Propagating containment through aggregation...")
            session.run("""
                MATCH (s)-[c:CONTAINS]->(parent)
                WHERE c.derived IS NULL
                MATCH (parent)-[:DECOMPOSES*1..]->(child)
                MERGE (s)-[d:CONTAINS]->(child)
                ON CREATE SET d.derived = true
            """)
            result = session.run("""
                MATCH ()-[d:CONTAINS]->()
                WHERE d.derived = true
                RETURN count(d) AS derived_edges
            """).single()
            derived_count = result["derived_edges"] if result else 0
            logger.info(f"Created {derived_count} derived CONTAINS edges")
            rel_count += derived_count

        self.stats["relationships_created"] = rel_count
    
    def run(self, ifc_path: Optional[Path] = None, wipe: bool = True) -> Dict[str, int]:
        """
        Execute the full ETL pipeline.
        
        Args:
            ifc_path: Optional path to IFC file (uses config if not provided)
            wipe: Whether to wipe the database before loading (default: True)
            
        Returns:
            Statistics dictionary with counts
        """
        logger.info("=" * 60)
        logger.info("Starting IFC to Neo4j ETL Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Connect to Neo4j
            self.connect()
            
            # Step 2: Load IFC file
            self.load_ifc(ifc_path)
            
            # Step 3: Wipe database (for clean experiment state)
            if wipe:
                self.wipe_database()
            
            # Step 4: Create constraints/indexes
            self.create_constraints()
            
            # Step 5: Create nodes
            self.create_nodes()
            
            # Step 6: Create relationships
            self.create_relationships()
            
            # Log summary
            logger.info("=" * 60)
            logger.info("ETL Pipeline Complete")
            logger.info(f"  Nodes created: {self.stats['nodes_created']}")
            logger.info(f"  Relationships created: {self.stats['relationships_created']}")
            logger.info(f"  Properties flattened: {self.stats['properties_flattened']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info("=" * 60)
            
            return self.stats
            
        finally:
            self.close()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """
    Command-line entry point for the ETL loader.
    
    Usage:
        python -m src.etl.loader
        
    Or with custom IFC path:
        python -m src.etl.loader /path/to/model.ifc
    """
    import sys
    
    # Optional: accept IFC path as argument
    ifc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    
    loader = IFCToNeo4jLoader()
    stats = loader.run(ifc_path=ifc_path)
    
    # Exit with error code if there were errors
    sys.exit(1 if stats["errors"] > 0 else 0)


if __name__ == "__main__":
    main()
