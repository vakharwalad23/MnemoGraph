"""Pluggable embedding providers."""

from .base import EmbeddingProvider
from .ollama import OllamaEmbedding
from .factory import create_embedding_provider

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbedding",
    "create_embedding_provider",
]