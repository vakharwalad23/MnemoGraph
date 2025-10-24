"""
Ollama embedder using native ollama-python SDK.
"""

import asyncio

import ollama

from src.core.embeddings.base import Embedder


class OllamaEmbedder(Embedder):
    """
    Ollama embedder for generating text embeddings.

    Uses native ollama-python SDK for embedding generation.
    Supports models like nomic-embed-text, mxbai-embed-large, etc.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama embedder.

        Args:
            host: Ollama server URL
            model: Embedding model name (e.g., "nomic-embed-text", "mxbai-embed-large")
            timeout: Request timeout in seconds
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self._dimension = None  # Cache dimension

        # Create async client
        self.client = ollama.AsyncClient(host=host)

    async def embed(self, text: str, **kwargs) -> list[float]:
        """
        Generate embedding for text using Ollama.

        Args:
            text: Text to embed
            **kwargs: Additional options passed to Ollama

        Returns:
            Embedding vector as list of floats
        """
        response = await self.client.embeddings(model=self.model, prompt=text, **kwargs)

        return response["embedding"]

    async def batch_embed(
        self, texts: list[str], batch_size: int = 32, **kwargs
    ) -> list[list[float]]:
        """
        Batch embed multiple texts with concurrency.

        Ollama processes requests sequentially on server,
        but we use asyncio for concurrent requests.

        Args:
            texts: List of texts to embed
            batch_size: Number of concurrent requests
            **kwargs: Additional options

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Process batch concurrently
            tasks = [self.embed(text, **kwargs) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)

            embeddings.extend(batch_embeddings)

        return embeddings

    async def get_dimension(self) -> int:
        """
        Get embedding dimension.
        Caches result after first call.

        Returns:
            Embedding vector dimension
        """
        if self._dimension is None:
            test_embedding = await self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    async def close(self):
        """Close client (Ollama SDK handles cleanup internally)."""
        pass
