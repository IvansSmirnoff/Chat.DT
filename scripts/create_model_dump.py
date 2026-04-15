#!/usr/bin/env python3
"""
Create Model Dump for Direct QA Experiments.

This script exports IFC model data to a JSON file that can be used
by the Direct QA experimental settings (DIRECT_QA_BASELINE and 
DIRECT_QA_GROUNDED). The LLM receives this dump in its prompt to
answer questions without database access.

Two modes are supported:
1. From Neo4j: Export nodes already loaded into the database
2. From IFC: Parse the IFC file directly (requires ifcopenshell)

Usage:
    # From Neo4j (requires running database)
    python scripts/create_model_dump.py --source neo4j --output data/model_dump.json
    
    # From IFC file directly
    python scripts/create_model_dump.py --source ifc --ifc-path data/model.ifc --output data/model_dump.json

Author: NL2Cypher Research Team
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def export_from_neo4j(
    uri: str,
    user: str,
    password: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Export all nodes from Neo4j database.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        limit: Optional limit on number of nodes per label
        
    Returns:
        List of node dictionaries with GlobalId, type, and properties
    """
    from neo4j import GraphDatabase
    
    logger.info(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    
    nodes = []
    
    with driver.session() as session:
        # Get all node labels
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        logger.info(f"Found {len(labels)} labels: {labels}")
        
        # Export nodes for each label
        for label in labels:
            # Skip non-IFC labels if any
            if not label.startswith("Ifc"):
                continue
                
            query = f"MATCH (n:{label}) RETURN n"
            if limit:
                query += f" LIMIT {limit}"
            
            result = session.run(query)
            label_count = 0
            
            for record in result:
                node = record["n"]
                node_dict = dict(node)
                node_dict["type"] = label
                
                # Ensure GlobalId is present and first
                if "GlobalId" in node_dict:
                    global_id = node_dict.pop("GlobalId")
                    node_dict = {"GlobalId": global_id, "type": label, **node_dict}
                
                nodes.append(node_dict)
                label_count += 1
            
            logger.info(f"  Exported {label_count} {label} nodes")
    
    driver.close()
    logger.info(f"Total nodes exported: {len(nodes)}")
    return nodes


def export_from_ifc(
    ifc_path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Export elements directly from IFC file.
    
    Args:
        ifc_path: Path to IFC file
        limit: Optional limit on number of elements per type
        
    Returns:
        List of element dictionaries with GlobalId, type, and properties
    """
    try:
        import ifcopenshell
    except ImportError:
        logger.error("ifcopenshell is required for IFC export. Install it with: pip install ifcopenshell")
        sys.exit(1)
    
    logger.info(f"Opening IFC file: {ifc_path}")
    ifc = ifcopenshell.open(str(ifc_path))
    
    nodes = []
    type_counts: Dict[str, int] = {}
    
    # Get all IfcProduct elements (building elements)
    products = ifc.by_type("IfcProduct")
    logger.info(f"Found {len(products)} IfcProduct elements")
    
    for product in products:
        ifc_type = product.is_a()
        
        # Apply limit per type if specified
        if limit:
            type_counts[ifc_type] = type_counts.get(ifc_type, 0) + 1
            if type_counts[ifc_type] > limit:
                continue
        
        # Build node dictionary
        node_dict = {
            "GlobalId": product.GlobalId,
            "type": ifc_type,
            "Name": product.Name if hasattr(product, "Name") else None,
            "Description": product.Description if hasattr(product, "Description") else None,
            "ObjectType": product.ObjectType if hasattr(product, "ObjectType") else None,
        }
        
        # Extract property sets
        psets = _get_property_sets(product)
        node_dict.update(psets)
        
        # Remove None values for cleaner output
        node_dict = {k: v for k, v in node_dict.items() if v is not None}
        
        nodes.append(node_dict)
    
    # Log type distribution
    type_summary = {}
    for node in nodes:
        t = node["type"]
        type_summary[t] = type_summary.get(t, 0) + 1
    
    logger.info("Exported elements by type:")
    for t, count in sorted(type_summary.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {count}")
    
    logger.info(f"Total elements exported: {len(nodes)}")
    return nodes


def _get_property_sets(product) -> Dict[str, Any]:
    """Extract all property sets and quantity sets from an IFC product."""
    properties = {}
    
    try:
        # Get property sets via IsDefinedBy relationship
        if hasattr(product, "IsDefinedBy"):
            for definition in product.IsDefinedBy:
                if hasattr(definition, "RelatingPropertyDefinition"):
                    prop_def = definition.RelatingPropertyDefinition
                    
                    # Handle IfcPropertySet
                    if prop_def.is_a("IfcPropertySet"):
                        for prop in prop_def.HasProperties:
                            name = _sanitize_name(prop.Name)
                            value = _get_property_value(prop)
                            if value is not None:
                                properties[name] = value
                    
                    # Handle IfcElementQuantity
                    elif prop_def.is_a("IfcElementQuantity"):
                        for quantity in prop_def.Quantities:
                            name = _sanitize_name(quantity.Name)
                            value = _get_quantity_value(quantity)
                            if value is not None:
                                properties[name] = value
    except Exception as e:
        logger.debug(f"Error extracting properties: {e}")
    
    return properties


def _sanitize_name(name: str) -> str:
    """Sanitize property name for Neo4j compatibility."""
    # Replace spaces and special chars with underscores
    sanitized = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    # Remove any remaining special characters
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return sanitized


def _get_property_value(prop) -> Any:
    """Extract value from an IFC property."""
    try:
        if hasattr(prop, "NominalValue") and prop.NominalValue:
            value = prop.NominalValue.wrappedValue
            # Convert to Python native types
            if isinstance(value, (int, float, str, bool)):
                return value
            return str(value)
    except Exception:
        pass
    return None


def _get_quantity_value(quantity) -> Any:
    """Extract value from an IFC quantity."""
    try:
        # Try different quantity types
        for attr in ["LengthValue", "AreaValue", "VolumeValue", "CountValue", "WeightValue", "TimeValue"]:
            if hasattr(quantity, attr):
                value = getattr(quantity, attr)
                if value is not None:
                    return float(value) if isinstance(value, (int, float)) else value
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create model dump for Direct QA experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export from Neo4j database
    python scripts/create_model_dump.py --source neo4j
    
    # Export from IFC file
    python scripts/create_model_dump.py --source ifc --ifc-path data/model.ifc
    
    # Limit elements per type (for testing)
    python scripts/create_model_dump.py --source neo4j --limit 10
        """
    )
    
    parser.add_argument(
        "--source",
        choices=["neo4j", "ifc"],
        default="neo4j",
        help="Data source: 'neo4j' (from database) or 'ifc' (from file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/model_dump.json"),
        help="Output JSON file path (default: data/model_dump.json)"
    )
    parser.add_argument(
        "--ifc-path",
        type=Path,
        help="Path to IFC file (required if source=ifc)"
    )
    parser.add_argument(
        "--neo4j-uri",
        help="Neo4j URI (default: from settings or NEO4J_URI env var)"
    )
    parser.add_argument(
        "--neo4j-user",
        help="Neo4j username (default: from settings)"
    )
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password (default: from settings)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit elements per type (for testing smaller dumps)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (larger file, but readable)"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if args.source == "ifc" and not args.ifc_path:
        settings = get_settings()
        args.ifc_path = Path(settings.ifc_file_path)
        if not args.ifc_path.exists():
            parser.error("--ifc-path is required when source=ifc")
    
    # Export data
    if args.source == "neo4j":
        settings = get_settings()
        nodes = export_from_neo4j(
            uri=args.neo4j_uri or settings.neo4j_uri,
            user=args.neo4j_user or settings.neo4j_user,
            password=args.neo4j_password or settings.neo4j_password_value,
            limit=args.limit,
        )
    else:
        nodes = export_from_ifc(
            ifc_path=args.ifc_path,
            limit=args.limit,
        )
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(args.output, "w") as f:
        if args.pretty:
            json.dump(nodes, f, indent=2, ensure_ascii=False)
        else:
            json.dump(nodes, f, ensure_ascii=False)
    
    file_size_kb = args.output.stat().st_size / 1024
    logger.info(f"Model dump saved to: {args.output} ({file_size_kb:.1f} KB)")
    logger.info(f"Use with: python -m src.eval.cli --model-dump {args.output} ...")


if __name__ == "__main__":
    main()
