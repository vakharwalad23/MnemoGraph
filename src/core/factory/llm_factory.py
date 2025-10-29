"""
Factory for creating LLM providers.
"""

from src.config import LLMConfig
from src.core.llm.base import LLMProvider
from src.core.llm.ollama import OllamaLLM
from src.core.llm.openai import OpenAILLM


class LLMFactory:
    """Factory for creating LLM providers from configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMProvider:
        """
        Create LLM provider from configuration.

        Args:
            config: LLM configuration

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is not supported
        """
        if config.provider == "ollama":
            return OllamaLLM(
                host=config.base_url,
                model=config.model,
                timeout=config.timeout,
            )
        elif config.provider == "openai":
            if not config.api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAILLM(
                api_key=config.api_key,
                model=config.model,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
