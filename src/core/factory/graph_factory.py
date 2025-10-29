"""
Factory for creating graph store backends.
"""

from src.config import Config
from src.core.graph_store.base import GraphStore
from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.core.graph_store.sqlite_store import SQLiteGraphStore


class GraphStoreFactory:
    """Factory for creating graph store backends from configuration."""

    @staticmethod
    def create(config: Config) -> GraphStore:
        """
        Create graph store from configuration.

        Args:
            config: Main configuration object

        Returns:
            Graph store instance

        Raises:
            ValueError: If backend is not supported
        """
        if config.graph_backend == "neo4j":
            return Neo4jGraphStore(
                uri=config.neo4j.uri,
                username=config.neo4j.username,
                password=config.neo4j.password,
                database=config.neo4j.database,
            )
        elif config.graph_backend == "sqlite":
            return SQLiteGraphStore(db_path=config.sqlite.db_path)
        else:
            raise ValueError(f"Unsupported graph backend: {config.graph_backend}")
