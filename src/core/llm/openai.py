"""
OpenAI LLM provider using official SDK.
"""

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.core.llm.base import LLMProvider


class OpenAILLM(LLMProvider):
    """
    OpenAI LLM provider for text generation.

    Uses official OpenAI SDK with native structured output support.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
            organization: Optional organization ID
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
        """
        self.model = model

        self.client = AsyncOpenAI(
            api_key=api_key, organization=organization, base_url=base_url, timeout=timeout
        )

    async def complete(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseModel | str:
        """
        Generate completion using OpenAI.

        Uses native structured outputs (Parse API) when response_format is provided.

        Args:
            prompt: Input prompt
            response_format: Optional Pydantic model for structured output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (e.g., stop, presence_penalty)

        Returns:
            Pydantic model if response_format provided, else string
        """
        messages = [{"role": "user", "content": prompt}]

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # Use structured outputs if response format provided
        if response_format:
            # OpenAI's native structured output support (Parse API)
            response = await self.client.beta.chat.completions.parse(
                **params, response_format=response_format
            )

            return response.choices[0].message.parsed

        # Regular completion
        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    async def close(self):
        """Close OpenAI client."""
        await self.client.close()
