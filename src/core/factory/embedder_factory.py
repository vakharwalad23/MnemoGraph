"""
Factory for creating embedder providers.
"""

from src.config import EmbedderConfig
from src.core.embeddings.base import Embedder
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.embeddings.openai import OpenAIEmbedder


class EmbedderFactory:
    """Factory for creating embedder providers from configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> Embedder:
        """
        Create embedder from configuration.

        Args:
            config: Embedder configuration

        Returns:
            Embedder instance

        Raises:
            ValueError: If provider is not supported
        """
        if config.provider == "ollama":
            return OllamaEmbedder(
                host=config.base_url,
                model=config.model,
                timeout=config.timeout,
            )
        elif config.provider == "openai":
            if not config.api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIEmbedder(
                api_key=config.api_key,
                model=config.model,
                timeout=config.timeout,
            )
        else:
            raise ValueError(f"Unsupported embedder provider: {config.provider}")

    @staticmethod
    async def get_dimension(embedder: Embedder, config: EmbedderConfig | None = None) -> int:
        """
        Get embedding dimension with fallback logic.

        Priority:
        1. From config if provided
        2. From embedder.dimension property if available
        3. From actual embedding test

        Args:
            embedder: Embedder instance
            config: Optional embedder config with dimension hint

        Returns:
            Embedding dimension
        """
        # Try config first
        if config and hasattr(config, "dimension") and config.dimension:
            return config.dimension

        # Try embedder property
        if hasattr(embedder, "dimension") and embedder.dimension:
            return embedder.dimension

        # Fallback: test embedding
        test_embedding = await embedder.embed("test")
        return len(test_embedding)
