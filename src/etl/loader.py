"""
IFC to Neo4j ETL Loader.

This module transforms IFC (Industry Foundation Classes) BIM data into a Neo4j
Property Graph. It handles:
- Loading IFC files using ifcopenshell
- Writing every storable IFC direct attribute, derived from the schema
- Flattening IFC property sets into node properties, without silent overwrites
- Creating relationships for spatial containment, decomposition and the rest of
  the materialised ``IfcRel*`` hierarchy
- Reporting what it retained and what it discarded, so coverage is measurable
- Batch processing for memory efficiency on large models

Design rules this file is held to:

1. **Lose nothing.** Every IFC attribute, property-set property, quantity and
   material layer present in the source must be reachable in the graph.
2. **Hardcode nothing.** No hand-maintained lists of entity classes or attribute
   names. Attribute selection is derived from the ifcopenshell schema at runtime
   via :mod:`src.utils.ifc_attributes`; property names go through
   :mod:`src.utils.property_names`. A hardcoded list is how this loader once came
   to read exactly four of IFC's attributes.
3. **Never overwrite silently.** When two sources supply the same key, both
   survive and which is which is recoverable from the key name.

Author: NL2Cypher Research Team
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import ifcopenshell
import ifcopenshell.util.element
from ifcopenshell import ifcopenshell_wrapper
from neo4j import Driver, GraphDatabase, Transaction

from src.config import get_settings, Settings
from src.utils.ifc_attributes import storable_attributes
from src.utils.ifc_edges import (
    EDGE_SPECS,
    FLATTENED_RELATIONSHIPS,
    SYNTHETIC_EDGE_NOTES,
    EdgeSpec,
)
from src.utils.property_names import (
    assign_unique_property_name,
    canonical_property_name,
)

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

# Keys ``ifcopenshell.util.element.get_psets`` injects into each pset dict as
# bookkeeping. These are not IFC properties and must never become node keys.
# Structural, not an opinion.
IFCOPENSHELL_PSET_KEYS = {"id", "type"}

# The identity property. It carries the uniqueness constraint and every
# relationship MATCH, so nothing may take its key.
IDENTITY_PROPERTY = "GlobalId"

# Backwards-compatible alias. The old skip list additionally carried
# ``OwnerHistory``, which is now unnecessary: it is entity-valued, so the schema
# classifier in ``src.utils.ifc_attributes`` excludes it without being told.
SKIP_PROPERTIES = IFCOPENSHELL_PSET_KEYS | {IDENTITY_PROPERTY}


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_property_name(name: str, existing_keys: Set[str]) -> str:
    """Backwards-compatible wrapper around the shared canonicaliser."""
    return assign_unique_property_name(name, existing_keys)


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


def qualified_property_name(pset_name: str, prop_name: str) -> str:
    """Canonical ``<PropertySet>_<Property>`` key.

    This is the provenance-carrying form. It is deterministic (the pset name is
    authored in the file, not assigned at load time) and therefore reproducible
    by the schema scanner, which the old ``_1``/``_2`` collision suffix was not.
    """
    return canonical_property_name(f"{pset_name}_{prop_name}")


def scan_property_name_ambiguity(
    ifc_file: ifcopenshell.file,
) -> Dict[str, Set[str]]:
    """Map every canonical property name to the property sets that claim it.

    Reads the property-set *definitions* directly rather than walking every
    element, so this is cheap: one pass over ``IfcPropertySet`` and
    ``IfcElementQuantity`` instead of ``get_psets`` per product.

    Ambiguity is resolved **model-wide** on purpose. Deciding per element would
    give one node of a label a bare key and its neighbour a qualified one,
    depending on which psets each happened to carry; a query written against the
    label would then work on some nodes and silently miss others.
    """
    owners: Dict[str, Set[str]] = {}

    def record(pset_name: Optional[str], prop_name: Optional[str]) -> None:
        if not prop_name:
            return
        canonical = canonical_property_name(prop_name)
        if not canonical:
            return
        owners.setdefault(canonical, set()).add(pset_name or "")

    for pset in ifc_file.by_type("IfcPropertySet"):
        for prop in (pset.HasProperties or ()):
            record(pset.Name, getattr(prop, "Name", None))

    for quantity_set in ifc_file.by_type("IfcElementQuantity"):
        for quantity in (quantity_set.Quantities or ()):
            record(quantity_set.Name, getattr(quantity, "Name", None))

    return owners


def flatten_psets(
    element: ifcopenshell.entity_instance,
    ambiguous: Optional[Set[str]] = None,
    reserved: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Flatten all property sets of an IFC element into a flat dictionary.

    Uses ifcopenshell.util.element.get_psets() to retrieve all property sets
    and quantity sets, then flattens them for direct node property storage.

    Two names are written for a property whose canonical name is ambiguous
    model-wide (claimed by more than one property set): the bare name, which the
    first writer keeps, and the provenance-carrying ``<PropertySet>_<Property>``
    form. A property that cannot have the bare name -- because an IFC direct
    attribute owns it, or because another pset got there first -- is written
    under the qualified name only. Nothing is dropped and nothing is
    overwritten.

    Args:
        element: IFC element instance
        ambiguous: canonical names claimed by more than one property set,
            from :func:`scan_property_name_ambiguity`
        reserved: keys owned by IFC direct attributes, which psets may not take

    Returns:
        ``(properties, counters)`` -- the flattened properties with sanitised
        keys, and counts of ``qualified``, ``displaced`` and ``fallback`` keys.
        ``fallback`` counts uses of the ``_1``/``_2`` last resort and is expected
        to be zero; a non-zero value means two qualified names collided and
        should be investigated rather than ignored.
    """
    ambiguous = ambiguous or set()
    reserved = reserved or set()

    properties: Dict[str, Any] = {}
    counters = {"qualified": 0, "displaced": 0, "fallback": 0}

    def put(key: str, value: Any) -> str:
        """Write ``key``, falling back to a numeric suffix rather than clobbering."""
        if key in properties:
            key = assign_unique_property_name(key, set(properties))
            counters["fallback"] += 1
        properties[key] = value
        return key

    try:
        # Get all property sets (including quantity sets)
        psets = ifcopenshell.util.element.get_psets(element, psets_only=False)

        for pset_name, pset_props in psets.items():
            if not isinstance(pset_props, dict):
                continue

            for prop_name, prop_value in pset_props.items():
                # Skip ifcopenshell's own bookkeeping keys and the identity key
                if prop_name in SKIP_PROPERTIES:
                    continue

                # Convert value to Neo4j-safe type
                safe_val = safe_value(prop_value)
                if safe_val is None:
                    continue

                canonical = canonical_property_name(prop_name)
                if not canonical:
                    continue

                bare_is_free = canonical not in reserved and canonical not in properties

                # The provenance-carrying key, written whenever the bare name is
                # ambiguous model-wide or is not available to this property.
                if canonical in ambiguous or not bare_is_free:
                    put(qualified_property_name(pset_name, prop_name), safe_val)
                    counters["qualified"] += 1
                    if not bare_is_free:
                        counters["displaced"] += 1

                if bare_is_free:
                    properties[canonical] = safe_val

    except Exception as e:
        logger.warning(f"Error flattening psets for {element}: {e}")

    return properties, counters


def resolve_material(
    material: ifcopenshell.entity_instance,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Resolve any ``IfcMaterialSelect`` into material names and layer records.

    Replaces a four-branch ``is_a()`` chain that understood only ``IfcMaterial``,
    ``IfcMaterialLayerSet``, ``IfcMaterialLayerSetUsage`` and
    ``IfcMaterialConstituentSet`` -- and, for the two layered cases, kept only
    the **first** layer's name, discarding the rest of the set and every
    thickness with it. That single omission made insulation and finish thickness
    unanswerable.

    Dispatch is on which attributes the entity actually has rather than on a
    list of class names, so IFC2X3 and IFC4 material structures resolve through
    the same code.

    Returns:
        ``(material_names, layers)``. Each layer is a property map destined for
        an ``:IfcMaterialLayer`` node, carrying ``layer_id``, ``LayerIndex``,
        ``LayerThickness``, ``MaterialName`` and ``LayerSetName``.
    """
    names: List[str] = []
    layers: List[Dict[str, Any]] = []
    seen: Set[int] = set()

    def add_name(entity: Any) -> None:
        name = getattr(entity, "Name", None)
        if name:
            names.append(str(name))

    def walk(entity: Any, layer_set_name: Optional[str] = None) -> None:
        if entity is None or not hasattr(entity, "is_a"):
            return
        step_id = entity.id()
        if step_id and step_id in seen:
            return
        if step_id:
            seen.add(step_id)

        # Usages point at the set that actually holds the data.
        for usage_attr in ("ForLayerSet", "ForProfileSet"):
            referenced = getattr(entity, usage_attr, None)
            if referenced is not None:
                walk(referenced, layer_set_name)
                return

        layer_members = getattr(entity, "MaterialLayers", None)
        if layer_members:
            set_name = getattr(entity, "LayerSetName", None) or layer_set_name
            for index, layer in enumerate(layer_members):
                layer_material = getattr(layer, "Material", None)
                material_name = getattr(layer_material, "Name", None)
                record = {
                    "layer_id": f"layer_{layer.id()}",
                    "LayerIndex": index,
                    "MaterialName": str(material_name) if material_name else None,
                    "LayerSetName": str(set_name) if set_name else None,
                }
                # Everything else the schema declares on IfcMaterialLayer
                # (LayerThickness, IsVentilated, Name, Category, Priority, ...)
                for attribute in storable_attributes(_schema_of(layer), layer.is_a()):
                    value = safe_value(getattr(layer, attribute, None))
                    if value is not None:
                        record[attribute] = value
                layers.append(record)
                if layer_material is not None:
                    add_name(layer_material)
            return

        for collection_attr in ("Materials", "MaterialConstituents", "MaterialProfiles"):
            members = getattr(entity, collection_attr, None)
            if members:
                for member in members:
                    walk(member, layer_set_name)
                return

        nested = getattr(entity, "Material", None)
        if nested is not None:
            walk(nested, layer_set_name)
            return

        add_name(entity)

    walk(material)
    return names, layers


def _schema_of(entity: ifcopenshell.entity_instance) -> str:
    """IFC schema identifier of the file an entity belongs to."""
    try:
        return entity.wrapped_data.file.schema
    except Exception:
        return "IFC4"


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
    4. Creating nodes from IFC entities with attributes + flattened properties
    5. Creating relationships (spatial containment, decomposition, ...)
    6. Reporting coverage

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
        self.driver: Optional[Driver] = None
        self.ifc_file: Optional[ifcopenshell.file] = None
        self._label_chain_cache: Dict[str, Tuple[str, ...]] = {}
        self._schema: Optional[Any] = None
        self._schema_name: str = ""

        # Property-name provenance, computed once per file in load_ifc()
        self._property_owners: Dict[str, Set[str]] = {}
        self._ambiguous_properties: Set[str] = set()

        # Statistics tracking
        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_flattened": 0,
            "errors": 0,
            "attributes_written": 0,
            "qualified_keys_written": 0,
            "displaced_properties": 0,
            "fallback_suffix_keys": 0,
            "material_nodes": 0,
            "material_layer_nodes": 0,
            "classification_nodes": 0,
        }
        self.relationship_counts: Dict[str, int] = {}
        self.unmatched_endpoints: Dict[str, int] = {}
        self.relationship_census: Dict[str, Dict[str, Any]] = {}

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
        self._schema_name = schema
        self._label_chain_cache.clear()

        # Which property names more than one pset claims. Needed before any
        # element is flattened, so the key set is consistent across nodes.
        self._property_owners = scan_property_name_ambiguity(self.ifc_file)
        self._ambiguous_properties = {
            name for name, owners in self._property_owners.items() if len(owners) > 1
        }

        logger.info(f"IFC Schema: {schema}")
        logger.info(f"Project: {project_name}")
        logger.info(
            f"Property names: {len(self._property_owners)} canonical, "
            f"{len(self._ambiguous_properties)} claimed by more than one property set"
        )

    def wipe_database(self) -> None:
        """
        Wipe all nodes and relationships from the database.

        CRITICAL: This ensures a clean state for the experiment.
        Use with caution in production environments.
        """
        logger.warning("Wiping entire Neo4j database...")

        # Relationships first, then nodes: a DETACH DELETE batch of nodes also
        # deletes every edge of each node in the same transaction, and the
        # generic instance graph has hubs (owner history, shared placements)
        # with tens of thousands of edges, which exhausts the transaction
        # memory pool on a small heap.
        with self.driver.session() as session:
            session.run("""
                MATCH ()-[r]->()
                CALL { WITH r DELETE r }
                IN TRANSACTIONS OF 10000 ROWS
            """).consume()
            session.run("""
                MATCH (n)
                CALL { WITH n DELETE n }
                IN TRANSACTIONS OF 10000 ROWS
            """).consume()

        logger.info("Database wiped successfully")

    def create_constraints(self) -> None:
        """
        Create constraints and indexes for performance.

        Creates a uniqueness constraint on GlobalId which also
        creates an index for fast lookups, plus indexes on the synthetic
        node keys (materials, layers, classifications) that relationship
        writing MATCHes on.
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
                """).consume()
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
                    """).consume()
                    logger.info("Created index on GlobalId (fallback)")
                except Exception as e2:
                    logger.warning(f"Could not create index: {e2}")

            # Synthetic nodes are not IfcRoot and carry no GlobalId, so they
            # need their own indexes or every edge write is a label scan.
            for label, key, name in (
                ("IfcMaterial", "material_id", "ifc_material_id_index"),
                ("IfcMaterialLayer", "layer_id", "ifc_material_layer_id_index"),
                ("IfcClassificationReference", "classification_id", "ifc_classification_id_index"),
            ):
                try:
                    session.run(
                        f"CREATE INDEX {name} IF NOT EXISTS FOR (n:`{label}`) ON (n.`{key}`)"
                    ).consume()
                except Exception as e:
                    logger.warning(f"Could not create index {name}: {e}")

    def _get_ifc_root_elements(self) -> Iterator[ifcopenshell.entity_instance]:
        """
        Iterate through all IfcRoot descendants in the IFC file, skipping
        IfcRelationship and IfcPropertyDefinition subclasses.

        IfcRel* entities are reifications of relationships and are
        materialised as graph edges (CONTAINS, DECOMPOSES, FILLS,
        HAS_OPENING, HAS_MATERIAL, IS_OF_TYPE, ...) — keeping them as nodes
        would just duplicate that information in a less queryable form.

        IfcPropertyDefinition subclasses (IfcPropertySet, IfcElementQuantity,
        IfcDoorLiningProperties, ...) are the same story for properties: their
        contents are flattened onto the owning product by ``flatten_psets``,
        so materialising them as nodes duplicates that information in a less
        queryable form. They carry no relationship of their own — the
        IfcRelDefinesByProperties that binds them to a product is itself an
        IfcRelationship and therefore already skipped — so every one of them
        lands in the graph as an orphan. On the dental-clinic model that was
        135,569 orphan nodes, 82.5% of the graph.

        Yields:
            IFC entity instances that inherit from IfcRoot, except
            IfcRelationship and IfcPropertyDefinition and their subclasses.
        """
        if not self.ifc_file:
            raise RuntimeError("IFC file not loaded. Call load_ifc() first.")

        for element in self.ifc_file.by_type("IfcRoot"):
            if element.is_a("IfcRelationship") or element.is_a("IfcPropertyDefinition"):
                continue
            yield element

    def _extract_attributes(
        self, element: ifcopenshell.entity_instance
    ) -> Dict[str, Any]:
        """
        Read every storable IFC direct attribute declared for this class.

        Which attributes those are is derived from the ifcopenshell schema
        declaration (see :mod:`src.utils.ifc_attributes`), not from a list kept
        in this file. The list this replaced held four names, so
        ``IfcSpace.LongName``, ``IfcDoor.OverallWidth/OverallHeight`` and
        ``IfcBuildingStorey.Elevation`` were never written -- while an
        IDS-grounded prompt asked for them by those exact names and got nulls.
        """
        attributes: Dict[str, Any] = {}
        for name in storable_attributes(self._schema_name, element.is_a()):
            value = safe_value(getattr(element, name, None))
            if value is not None:
                attributes[name] = value
        # Identity is non-negotiable even if the schema walk came up empty.
        attributes[IDENTITY_PROPERTY] = str(element.GlobalId)
        return attributes

    def _extract_node_data(
        self, element: ifcopenshell.entity_instance
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract node data from an IFC element.

        IFC direct attributes win their own names; a property set carrying the
        same name is written under its qualified ``<PropertySet>_<Property>``
        key instead of overwriting the attribute. That is the one behaviour
        change visible in existing data: ``IfcSpace.Name`` is now the IFC value
        (the room number) and the exporter's copy lives beside it, with the room
        name reachable through ``LongName`` -- which is what an IDS asks for.

        Args:
            element: IFC entity instance

        Returns:
            Tuple of (label, global_id, properties_dict)
        """
        # Get IFC class name as label (e.g., "IfcWall", "IfcDoor")
        label = element.is_a()

        # Get GlobalId for identity
        global_id = str(element.GlobalId)

        properties = self._extract_attributes(element)
        self.stats["attributes_written"] += len(properties)

        flattened, counters = flatten_psets(
            element,
            ambiguous=self._ambiguous_properties,
            reserved=set(properties),
        )
        properties.update(flattened)

        self.stats["properties_flattened"] += len(flattened)
        self.stats["qualified_keys_written"] += counters["qualified"]
        self.stats["displaced_properties"] += counters["displaced"]
        self.stats["fallback_suffix_keys"] += counters["fallback"]

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
                except Exception as e:
                    logger.error(f"Error processing element {element}: {e}")
                    self.stats["errors"] += 1
                    continue

                batch.append(node_data)

                # Process batch when it reaches BATCH_SIZE
                if len(batch) >= BATCH_SIZE:
                    session.execute_write(self._create_nodes_batch, batch)
                    total_count += len(batch)
                    logger.info(f"Processed {total_count} nodes...")
                    batch = []

            # Process remaining batch
            if batch:
                session.execute_write(self._create_nodes_batch, batch)
                total_count += len(batch)

        self.stats["nodes_created"] = total_count
        logger.info(f"Created {total_count} nodes")

    # -------------------------------------------------------------------------
    # Relationship extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _as_entities(value: Any) -> Sequence[Any]:
        """Normalise a relationship attribute to a sequence of entities."""
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return value
        return (value,)

    def _extract_edges(self, spec: EdgeSpec) -> Iterator[Tuple[str, str]]:
        """
        Yield ``(source_GlobalId, target_GlobalId)`` pairs for one edge spec.

        Replaces eight near-identical extractor methods. Either side may be a
        single entity or a list; the cartesian product is emitted, which matches
        the IFC semantics of ``RelatedObjects``-style aggregates.

        Entities without a GlobalId (virtual space boundaries carry no
        RelatedBuildingElement at all, for instance) are skipped.
        """
        if not self.ifc_file:
            return
        for rel in self.ifc_file.by_type(spec.ifc_class):
            sources = [
                str(e.GlobalId)
                for e in self._as_entities(getattr(rel, spec.source_attr, None))
                if hasattr(e, "GlobalId")
            ]
            if not sources:
                continue
            targets = [
                str(e.GlobalId)
                for e in self._as_entities(getattr(rel, spec.target_attr, None))
                if hasattr(e, "GlobalId")
            ]
            for source in sources:
                for target in targets:
                    yield (source, target)

    def _extract_material_associations(
        self,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, str], List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
        """
        Extract material associations, including full layer sets.

        Returns:
            ``(material_edges, materials, layer_edges, layers)`` where
            ``material_edges`` are ``(element_GlobalId, material_id)`` pairs,
            ``materials`` maps ``material_id -> name``, ``layer_edges`` are
            ``(element_GlobalId, layer_id)`` pairs and ``layers`` maps
            ``layer_id`` to the layer's property map.
        """
        material_edges: List[Tuple[str, str]] = []
        materials: Dict[str, str] = {}
        layer_edges: List[Tuple[str, str]] = []
        layers: Dict[str, Dict[str, Any]] = {}

        if not self.ifc_file:
            return material_edges, materials, layer_edges, layers

        for rel in self.ifc_file.by_type("IfcRelAssociatesMaterial"):
            material = rel.RelatingMaterial
            if not material:
                continue

            names, layer_records = resolve_material(material)
            if not names and not layer_records:
                continue

            # Identity is the material name, so the same material authored twice
            # is one node. Deterministic across runs -- the previous hash()-based
            # id was salted by PYTHONHASHSEED and changed every load.
            material_ids = []
            for name in names:
                material_id = f"mat_{canonical_property_name(name) or name}"
                materials[material_id] = name
                material_ids.append(material_id)

            for record in layer_records:
                layers[record["layer_id"]] = record

            for related_obj in rel.RelatedObjects:
                if not hasattr(related_obj, "GlobalId"):
                    continue
                element_id = str(related_obj.GlobalId)
                for material_id in material_ids:
                    material_edges.append((element_id, material_id))
                for record in layer_records:
                    layer_edges.append((element_id, record["layer_id"]))

        return material_edges, materials, layer_edges, layers

    def _extract_classification_associations(
        self,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]]]:
        """
        Extract ``IfcRelAssociatesClassification`` into synthetic reference nodes.

        Classification assignments were previously dropped without record --
        the one omission EC³ Reviewer 2 named. The far side is an
        ``IfcClassificationReference`` (or ``IfcClassification``), neither of
        which is an ``IfcRoot``, so it gets a synthetic node keyed by its
        identification, the same pattern as ``:IfcMaterial``.

        Node properties come from the schema declaration, so IFC2X3's
        ``ItemReference`` and IFC4's ``Identification`` are both carried without
        naming either.
        """
        edges: List[Tuple[str, str]] = []
        references: Dict[str, Dict[str, Any]] = {}

        if not self.ifc_file:
            return edges, references

        for rel in self.ifc_file.by_type("IfcRelAssociatesClassification"):
            reference = getattr(rel, "RelatingClassification", None)
            if reference is None:
                continue

            props: Dict[str, Any] = {}
            for attribute in storable_attributes(self._schema_name, reference.is_a()):
                value = safe_value(getattr(reference, attribute, None))
                if value is not None:
                    props[attribute] = value
            if not props:
                continue

            identity = (
                props.get("Identification")
                or props.get("ItemReference")
                or props.get("Name")
                or f"ref_{reference.id()}"
            )
            classification_id = f"cls_{canonical_property_name(str(identity)) or reference.id()}"
            props["classification_id"] = classification_id
            references[classification_id] = props

            for related_obj in getattr(rel, "RelatedObjects", ()) or ():
                if hasattr(related_obj, "GlobalId"):
                    edges.append((str(related_obj.GlobalId), classification_id))

        return edges, references

    # -------------------------------------------------------------------------
    # Relationship writing
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_relationships_batch(
        tx: Transaction,
        relationships: List[Tuple[str, str]],
        rel_type: str
    ) -> int:
        """
        Create relationships in a single transaction batch.

        Returns the number of edges Neo4j actually created, not the number
        attempted. The two differ whenever an endpoint was never loaded, and
        counting attempts is how the old loader reported edges that do not
        exist.

        Args:
            tx: Neo4j transaction
            relationships: List of (source_id, target_id) tuples
            rel_type: Relationship type name
        """
        query = f"""
            UNWIND $rels AS r
            MATCH (a:IfcRoot {{GlobalId: r[0]}})
            MATCH (b:IfcRoot {{GlobalId: r[1]}})
            CREATE (a)-[:`{rel_type}`]->(b)
        """
        result = tx.run(query, rels=relationships)
        return result.consume().counters.relationships_created

    def _write_edges(
        self,
        session,
        pairs: Iterator[Tuple[str, str]],
        rel_type: str,
    ) -> int:
        """Batch and write ``(source, target)`` GlobalId pairs as one edge type."""
        batch: List[Tuple[str, str]] = []
        attempted = 0
        created = 0

        for pair in pairs:
            batch.append(pair)
            if len(batch) >= BATCH_SIZE:
                created += session.execute_write(
                    self._create_relationships_batch, batch, rel_type
                )
                attempted += len(batch)
                batch = []

        if batch:
            created += session.execute_write(
                self._create_relationships_batch, batch, rel_type
            )
            attempted += len(batch)

        if attempted:
            logger.info(f"Created {created} {rel_type} relationships")
        missing = attempted - created
        if missing:
            # Endpoints that were never loaded. Worth surfacing: it is the
            # symptom of an element class being filtered out upstream.
            self.unmatched_endpoints[rel_type] = (
                self.unmatched_endpoints.get(rel_type, 0) + missing
            )
            logger.warning(
                f"{rel_type}: {missing} of {attempted} pairs had an endpoint "
                f"that is not in the graph"
            )

        self.relationship_counts[rel_type] = (
            self.relationship_counts.get(rel_type, 0) + created
        )
        return created

    def _write_synthetic_nodes(
        self,
        session,
        label: str,
        key: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """MERGE synthetic (non-IfcRoot) nodes such as materials and layers."""
        if not records:
            return 0
        session.run(
            f"""
            UNWIND $records AS r
            MERGE (n:`{label}` {{`{key}`: r.`{key}`}})
            SET n += r
            """,
            records=records,
        ).consume()
        logger.info(f"Created/updated {len(records)} {label} nodes")
        return len(records)

    def _write_synthetic_edges(
        self,
        session,
        pairs: List[Tuple[str, str]],
        rel_type: str,
        label: str,
        key: str,
    ) -> int:
        """Write edges from an IfcRoot element to a synthetic node."""
        if not pairs:
            return 0
        created = 0
        for start in range(0, len(pairs), BATCH_SIZE):
            chunk = pairs[start:start + BATCH_SIZE]
            result = session.run(
                f"""
                UNWIND $rels AS r
                MATCH (e:IfcRoot {{GlobalId: r[0]}})
                MATCH (n:`{label}` {{`{key}`: r[1]}})
                MERGE (e)-[:`{rel_type}`]->(n)
                """,
                rels=chunk,
            )
            created += result.consume().counters.relationships_created
        logger.info(f"Created {created} {rel_type} relationships")
        self.relationship_counts[rel_type] = (
            self.relationship_counts.get(rel_type, 0) + created
        )
        return created

    def create_relationships(self) -> None:
        """
        Create all relationships between nodes.

        Nine IfcRoot-to-IfcRoot edge types come from :data:`EDGE_SPECS`;
        materials and classifications need synthetic nodes and are handled
        separately; spatial containment is then propagated through aggregation.
        """
        logger.info("Creating relationships...")
        rel_count = 0

        with self.driver.session() as session:
            for spec in EDGE_SPECS:
                logger.info(f"Processing {spec.ifc_class} -> {spec.rel_type}...")
                rel_count += self._write_edges(
                    session, self._extract_edges(spec), spec.rel_type
                )

            # Materials: (:IfcElement)-[:HAS_MATERIAL]->(:IfcMaterial) and,
            # for layered constructions, the full ordered layer set.
            logger.info("Processing material association relationships...")
            material_edges, materials, layer_edges, layers = (
                self._extract_material_associations()
            )
            self.stats["material_nodes"] = self._write_synthetic_nodes(
                session,
                "IfcMaterial",
                "material_id",
                [{"material_id": mid, "Name": name} for mid, name in materials.items()],
            )
            rel_count += self._write_synthetic_edges(
                session, material_edges, "HAS_MATERIAL", "IfcMaterial", "material_id"
            )
            self.stats["material_layer_nodes"] = self._write_synthetic_nodes(
                session, "IfcMaterialLayer", "layer_id", list(layers.values())
            )
            rel_count += self._write_synthetic_edges(
                session,
                layer_edges,
                "HAS_MATERIAL_LAYER",
                "IfcMaterialLayer",
                "layer_id",
            )

            # Classification assignments: (:IfcRoot)-[:HAS_CLASSIFICATION]->(:IfcClassificationReference)
            logger.info("Processing classification relationships...")
            classification_edges, references = self._extract_classification_associations()
            self.stats["classification_nodes"] = self._write_synthetic_nodes(
                session,
                "IfcClassificationReference",
                "classification_id",
                list(references.values()),
            )
            rel_count += self._write_synthetic_edges(
                session,
                classification_edges,
                "HAS_CLASSIFICATION",
                "IfcClassificationReference",
                "classification_id",
            )

            # Derived: spatial containment is transitive through aggregation.
            # If a spatial element CONTAINS X, and X DECOMPOSES (transitively)
            # into Y, then Y is logically also contained in that spatial
            # element (curtain-wall doors, stair flights, ramp railings, ...).
            # Marked r.derived = true so it is distinguishable from the
            # literal IfcRelContainedInSpatialStructure edges.
            logger.info("Propagating containment through aggregation...")
            counters = session.run("""
                MATCH (s)-[c:CONTAINS]->(parent)
                WHERE c.derived IS NULL
                MATCH (parent)-[:DECOMPOSES*1..]->(child)
                MERGE (s)-[d:CONTAINS]->(child)
                ON CREATE SET d.derived = true
            """).consume().counters
            derived_count = counters.relationships_created
            logger.info(f"Created {derived_count} derived CONTAINS edges")
            self.relationship_counts["CONTAINS"] = (
                self.relationship_counts.get("CONTAINS", 0) + derived_count
            )
            self.relationship_counts["CONTAINS (derived)"] = derived_count
            rel_count += derived_count

        self.stats["nodes_created"] += (
            self.stats["material_nodes"]
            + self.stats["material_layer_nodes"]
            + self.stats["classification_nodes"]
        )
        self.stats["relationships_created"] = rel_count

    # -------------------------------------------------------------------------
    # Coverage reporting
    # -------------------------------------------------------------------------

    def census_relationships(self) -> Dict[str, Dict[str, Any]]:
        """
        Count every ``IfcRelationship`` subtype in the file and say what became
        of it.

        The point is that the IFC-to-LPG coverage table can be *generated* from
        a run rather than hand-written and left to drift -- EC³ Reviewer 2 asked
        for exactly that table (which relationship types are retained, which are
        discarded, and what it costs).
        """
        census: Dict[str, Dict[str, Any]] = {}
        if not self.ifc_file:
            return census

        counts: Dict[str, int] = {}
        for rel in self.ifc_file.by_type("IfcRelationship"):
            name = rel.is_a()
            counts[name] = counts.get(name, 0) + 1

        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            disposition = "DISCARDED"
            mapped_to = None
            note = ""

            declaration_matches = [
                spec for spec in EDGE_SPECS
                if self._is_subtype_of(name, spec.ifc_class)
            ]
            if declaration_matches:
                spec = declaration_matches[0]
                disposition, mapped_to, note = "RETAINED", spec.rel_type, spec.note
            elif name in SYNTHETIC_EDGE_NOTES:
                disposition, (mapped_to, note) = "RETAINED", SYNTHETIC_EDGE_NOTES[name]
            elif name in FLATTENED_RELATIONSHIPS:
                disposition = "RETAINED"
                mapped_to = "node properties"
                note = FLATTENED_RELATIONSHIPS[name]

            census[name] = {
                "count": count,
                "disposition": disposition,
                "mapped_to": mapped_to,
                "note": note,
            }

        self.relationship_census = census
        return census

    def _is_subtype_of(self, ifc_class: str, ancestor: str) -> bool:
        """True if ``ifc_class`` is ``ancestor`` or inherits from it."""
        if ifc_class == ancestor:
            return True
        try:
            decl = self._schema.declaration_by_name(ifc_class)
        except Exception:
            return False
        while decl is not None:
            if decl.name() == ancestor:
                return True
            decl = decl.supertype() if hasattr(decl, "supertype") else None
        return False

    def _project_units(self) -> Dict[str, str]:
        """Unit assignment of the project, recorded so numbers are interpretable.

        The loader stores measures as authored, without conversion. That is a
        deliberate non-goal, but leaving the unit undocumented is how a length
        in millimetres gets compared against one in metres.
        """
        units: Dict[str, str] = {}
        try:
            for project in self.ifc_file.by_type("IfcProject"):
                assignment = getattr(project, "UnitsInContext", None)
                for unit in (getattr(assignment, "Units", None) or ()):
                    unit_type = getattr(unit, "UnitType", None)
                    if not unit_type:
                        continue
                    prefix = getattr(unit, "Prefix", None) or ""
                    name = getattr(unit, "Name", None) or unit.is_a()
                    units[str(unit_type)] = f"{prefix}{name}".strip()
        except Exception as e:
            logger.debug(f"Could not read project units: {e}")
        return units

    def build_report(self, ifc_path: Optional[Path] = None) -> Dict[str, Any]:
        """Assemble the machine-readable run report."""
        ambiguous = {
            name: sorted(owners)
            for name, owners in sorted(self._property_owners.items())
            if len(owners) > 1
        }
        return {
            "source": {
                "ifc_file": str(ifc_path) if ifc_path else str(self.settings.ifc_file_path),
                "schema": self._schema_name,
                "units": self._project_units(),
            },
            "stats": dict(self.stats),
            "relationship_counts": dict(sorted(self.relationship_counts.items())),
            "unmatched_endpoints": dict(sorted(self.unmatched_endpoints.items())),
            "relationship_census": self.relationship_census,
            "ambiguous_property_names": ambiguous,
        }

    def run(
        self,
        ifc_path: Optional[Path] = None,
        wipe: bool = True,
        report_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Execute the full ETL pipeline.

        Args:
            ifc_path: Optional path to IFC file (uses config if not provided)
            wipe: Whether to wipe the database before loading (default: True)
            report_path: Optional path for the JSON coverage/statistics report

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

            # Step 7: Census what the IFC held versus what was materialised
            self.census_relationships()

            # Log summary
            logger.info("=" * 60)
            logger.info("ETL Pipeline Complete")
            logger.info(f"  Nodes created: {self.stats['nodes_created']}")
            logger.info(f"  Relationships created: {self.stats['relationships_created']}")
            logger.info(f"  IFC attributes written: {self.stats['attributes_written']}")
            logger.info(f"  Properties flattened: {self.stats['properties_flattened']}")
            logger.info(f"    of which qualified: {self.stats['qualified_keys_written']}")
            logger.info(f"  Material layer nodes: {self.stats['material_layer_nodes']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            if self.stats["fallback_suffix_keys"]:
                logger.warning(
                    f"  Numeric-suffix fallback keys: "
                    f"{self.stats['fallback_suffix_keys']} (expected 0 -- two "
                    f"qualified names collided)"
                )
            logger.info("=" * 60)

            discarded = {
                name: entry for name, entry in self.relationship_census.items()
                if entry["disposition"] == "DISCARDED"
            }
            if discarded:
                logger.info("IfcRelationship subtypes NOT materialised:")
                for name, entry in discarded.items():
                    logger.info(f"  {name}: {entry['count']}")
            else:
                logger.info("Every IfcRelationship subtype in this file was retained")

            if report_path:
                report = self.build_report(ifc_path)
                Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=False))
                logger.info(f"Wrote run report to {report_path}")

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
        python -m src.etl.loader /path/to/model.ifc --report /app/results/etl.json
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Load an IFC file into Neo4j as a labelled property graph."
    )
    parser.add_argument(
        "ifc_path", nargs="?", default=None,
        help="Path to the IFC file (defaults to IFC_FILE_PATH from the environment)",
    )
    parser.add_argument(
        "--report", default=None, metavar="PATH",
        help="Write a JSON run report (statistics, relationship census, "
             "ambiguous property names) to PATH",
    )
    parser.add_argument(
        "--no-wipe", action="store_true",
        help="Do not wipe the database before loading",
    )
    args = parser.parse_args()

    loader = IFCToNeo4jLoader()
    stats = loader.run(
        ifc_path=Path(args.ifc_path) if args.ifc_path else None,
        wipe=not args.no_wipe,
        report_path=Path(args.report) if args.report else None,
    )

    # Exit with error code if there were errors
    sys.exit(1 if stats["errors"] > 0 else 0)


if __name__ == "__main__":
    main()
