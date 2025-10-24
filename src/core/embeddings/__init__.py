"""
Embedder abstraction layer for text embeddings.

Supported providers:
- Ollama (native SDK)
- OpenAI (official SDK)
"""
from src.core.embeddings.base import Embedder
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.embeddings.openai import OpenAIEmbedder

__all__ = [
    "Embedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
]
