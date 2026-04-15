"""
ETL module for transforming IFC data into Neo4j Property Graph.
"""

from .loader import IFCToNeo4jLoader

__all__ = ["IFCToNeo4jLoader"]
