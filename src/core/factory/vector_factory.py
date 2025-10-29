"""
Factory for creating vector store backends.
"""

from urllib.parse import urlparse

from src.config import QdrantConfig
from src.core.vector_store.base import VectorStore
from src.core.vector_store.qdrant import QdrantStore


class VectorStoreFactory:
    """Factory for creating vector store backends from configuration."""

    @staticmethod
    def create(config: QdrantConfig, vector_size: int) -> VectorStore:
        """
        Create vector store from configuration.

        Args:
            config: Qdrant configuration
            vector_size: Embedding dimension size

        Returns:
            Vector store instance
        """
        # Parse URL to extract host and port
        parsed = urlparse(config.url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333

        return QdrantStore(
            host=host,
            port=port,
            collection_name=config.collection_name,
            vector_size=vector_size,
            use_grpc=config.use_grpc,
            use_quantization=config.use_quantization,
            quantization_type=config.quantization_type,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construct=config.hnsw_ef_construct,
            on_disk=config.on_disk,
        )
