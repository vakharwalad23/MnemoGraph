"""Pluggable embedding providers."""

from .base import EmbeddingProvider
from .factory import create_embedding_provider
from .ollama import OllamaEmbedding

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbedding",
    "create_embedding_provider",
]
