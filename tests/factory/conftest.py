"""
Shared test fixtures for all test modules.
"""

import pytest

from src.core.embeddings.ollama import OllamaEmbedder


@pytest.fixture
def ollama_embedder():
    """Create Ollama embedder for testing."""
    return OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text", timeout=120.0)
