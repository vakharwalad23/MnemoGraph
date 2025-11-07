"""
OpenAI embedder using official SDK.
"""

from openai import AsyncOpenAI

from src.core.embeddings.base import Embedder
from src.utils.exceptions import EmbeddingError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbedder(Embedder):
    """
    OpenAI embedder for generating text embeddings.

    Uses official OpenAI SDK with support for batch processing.
    Supports models like text-embedding-3-small, text-embedding-3-large, etc.
    """

    # Known dimensions for OpenAI embedding models
    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model name (e.g., "text-embedding-3-small")
            organization: Optional organization ID
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
        """
        self.model = model

        self.client = AsyncOpenAI(
            api_key=api_key, organization=organization, base_url=base_url, timeout=timeout
        )

    async def embed(self, text: str, **kwargs) -> list[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed
            **kwargs: Additional parameters (e.g., dimensions, user)

        Returns:
            Embedding vector as list of floats

        Raises:
            ValidationError: If text is invalid
            EmbeddingError: If OpenAI API call fails
        """
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")

        try:
            response = await self.client.embeddings.create(model=self.model, input=text, **kwargs)

            if not response.data or len(response.data) == 0:
                raise EmbeddingError("OpenAI returned empty embedding response")

            return response.data[0].embedding
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"OpenAI embedding error: {e}",
                extra={"model": self.model, "error": str(e), "error_type": type(e).__name__},
            )
            raise EmbeddingError(f"OpenAI embedding error: {e}") from e

    async def batch_embed(
        self, texts: list[str], batch_size: int = 2048, **kwargs
    ) -> list[list[float]]:
        """
        Batch embed using OpenAI's native batch API.

        OpenAI supports up to 2048 inputs per request.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per request (max 2048)
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors

        Raises:
            ValidationError: If texts list is invalid
            EmbeddingError: If batch embedding fails
        """
        if not texts:
            raise ValidationError("Texts list cannot be empty")

        try:
            embeddings = []

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = await self.client.embeddings.create(
                    model=self.model, input=batch, **kwargs
                )

                if not response.data:
                    raise EmbeddingError("OpenAI returned empty batch embedding response")

                # Extract embeddings (response.data is already ordered)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"OpenAI batch embedding error: {e}",
                extra={
                    "model": self.model,
                    "batch_size": batch_size,
                    "num_texts": len(texts),
                    "error": str(e),
                },
            )
            raise EmbeddingError(f"OpenAI batch embedding error: {e}") from e

    async def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Uses known dimensions for OpenAI models for efficiency.
        Falls back to test embedding if model not recognized.

        Returns:
            Embedding vector dimension
        """
        # Return known dimension if available
        if self.model in self._MODEL_DIMENSIONS:
            return self._MODEL_DIMENSIONS[self.model]

        # Otherwise, generate test embedding
        return await super().get_dimension()

    async def close(self):
        """Close OpenAI client."""
        await self.client.close()
