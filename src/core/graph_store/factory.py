"""Factory for creating graph stores."""

from .base import GraphStore
from .neo4j_store import Neo4jGraphStore
from .sqlite_store import SQLiteGraphStore


def create_graph_store(backend: str = "sqlite", **kwargs) -> GraphStore:
    """
    Factory function to create graph stores.

    Args:
        backend: Type of backend ("sqlite" or "neo4j")
        **kwargs: Backend-specific arguments

    Returns:
        GraphStore instance

    Raises:
        ValueError: If backend is not supported
    """
    if backend == "sqlite":
        return SQLiteGraphStore(db_path=kwargs.get("db_path", "mnemograph.db"))
    elif backend == "neo4j":
        return Neo4jGraphStore(
            uri=kwargs.get("uri", "bolt://localhost:7687"),
            user=kwargs.get("user", "neo4j"),
            password=kwargs.get("password", "mnemograph123"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
