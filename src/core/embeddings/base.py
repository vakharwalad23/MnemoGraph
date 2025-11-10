"""
Abstract base class for embedding providers.
Handles text to vector embeddings for semantic search.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract base for embedding providers.

    Responsibilities:
    - Generate vector embeddings for text
    - Batch processing for efficiency
    - Consistent vector dimensions
    """

    @abstractmethod
    async def embed(self, text: str, **kwargs) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed
            **kwargs: Provider-specific parameters

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValidationError: If text is invalid
            EmbeddingError: If embedding generation fails
        """
        pass

    async def batch_embed(
        self, texts: list[str], batch_size: int = 32, **kwargs
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Default implementation processes sequentially.
        Override for provider-specific batch optimization.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            **kwargs: Provider-specific parameters

        Returns:
            List of embedding vectors (same order as input texts)
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(text, **kwargs)
            embeddings.append(embedding)
        return embeddings

    async def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Default implementation embeds a test string.
        Override for efficiency if dimension is known.

        Returns:
            Embedding vector dimension
        """
        test_embedding = await self.embed("test")
        return len(test_embedding)

    @abstractmethod
    async def close(self):
        """
        Close any open connections.

        Optional to override if provider needs cleanup.
        """
        pass
