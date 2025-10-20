"""Factory for creating embedding providers."""

from typing import Optional
from .base import EmbeddingProvider
from .ollama import OllamaEmbedding


def create_embedding_provider(
    provider_type: str = "ollama",
    model: Optional[str] = None,
    host: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider_type: Type of provider ("ollama", "openai", "sentence-transformers")
        model: Model name (provider-specific)
        host: Host URL for API providers
        **kwargs: Additional provider-specific arguments
        
    Returns:
        EmbeddingProvider instance
        
    Raises:
        ValueError: If provider_type is not supported
    """
    if provider_type == "ollama":
        return OllamaEmbedding(
            model=model or "nomic-embed-text",
            host=host or "http://localhost:11434",
            **kwargs
        )
    elif provider_type == "openai":
        # Placeholder for future implementation
        raise NotImplementedError("OpenAI provider not yet implemented")
    elif provider_type == "sentence-transformers":
        # Placeholder for future implementation
        raise NotImplementedError("SentenceTransformers provider not yet implemented")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")