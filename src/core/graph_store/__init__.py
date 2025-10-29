"""
Graph store implementations for MnemoGraph.

Provides abstract base and concrete implementations for graph storage.

Available backends:
- Neo4jGraphStore: Production-grade, distributed graph database
"""

from src.core.graph_store.base import GraphStore
from src.core.graph_store.neo4j_store import Neo4jGraphStore

__all__ = [
    "GraphStore",
    "Neo4jGraphStore",
]
