"""
Factory for creating graph store backends.
"""

from src.config import Config
from src.core.graph_store.base import GraphStore
from src.core.graph_store.neo4j_store import Neo4jGraphStore


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
        else:
            raise ValueError(f"Unsupported graph backend: {config.graph_backend}")
