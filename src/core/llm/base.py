"""
Abstract base class for LLM providers.
Handles text generation with optional structured outputs.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMProvider(ABC):
    """
    Abstract base for LLM text generation providers.

    Responsibilities:
    - Text completion/generation
    - Structured output (JSON/Pydantic models)
    - Chat-based interactions
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseModel | str:
        """
        Generate completion from prompt.

        Args:
            prompt: The input prompt
            response_format: Optional Pydantic model for structured output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Provider-specific parameters

        Returns:
            Pydantic model instance if response_format provided, else string

        Raises:
            ValueError: If structured output parsing fails
            Exception: Provider-specific errors
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Close any open connections.
        Optional to override if provider needs cleanup.
        """
        # Default implementation does nothing
        # Providers should override if cleanup is needed
