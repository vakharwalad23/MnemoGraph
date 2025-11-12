"""
OpenAI LLM provider using official SDK.
"""

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.core.llm.base import LLMProvider
from src.utils.exceptions import LLMError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        Raises:
            LLMError: If OpenAI API call fails
            ValidationError: If structured output parsing fails
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty")

        messages = [{"role": "user", "content": prompt}]

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        try:
            if response_format:
                response = await self.client.beta.chat.completions.parse(
                    **params, response_format=response_format
                )

                parsed = response.choices[0].message.parsed
                if not parsed:
                    raise ValidationError("OpenAI returned empty parsed response")

                return parsed

            response = await self.client.chat.completions.create(**params)
            content = response.choices[0].message.content

            if not content:
                raise LLMError("OpenAI returned empty content")

            return content
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"OpenAI API error: {e}",
                extra={"model": self.model, "error": str(e), "error_type": type(e).__name__},
            )
            raise LLMError(f"OpenAI API error: {e}") from e

    async def close(self):
        """Close OpenAI client."""
        await self.client.close()
