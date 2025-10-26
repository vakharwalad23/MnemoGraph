"""
Ollama LLM provider using native ollama-python SDK.
"""

import json

import ollama
from pydantic import BaseModel

from src.core.llm.base import LLMProvider


class OllamaLLM(LLMProvider):
    """
    Ollama LLM provider for text generation.

    Uses native ollama-python SDK for chat completions
    with JSON mode for structured outputs.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama LLM provider.

        Args:
            host: Ollama server URL
            model: Model name for text generation (e.g., "llama3.1", "mistral")
            timeout: Request timeout in seconds
        """
        self.host = host
        self.model = model
        self.timeout = timeout

        # Create async client
        self.client = ollama.AsyncClient(host=host)

    async def complete(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseModel | str:
        """
        Generate completion using Ollama.

        Supports structured output via JSON mode and schema validation.

        Args:
            prompt: Input prompt
            response_format: Optional Pydantic model for structured JSON output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional options (passed to Ollama)

        Returns:
            Pydantic model if response_format provided, else string

        Raises:
            ValueError: If structured output parsing fails
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs.get("options", {}),
        }

        # Handle structured output
        format_type = None
        messages = [{"role": "user", "content": prompt}]

        if response_format:
            format_type = "json"

            # Create simple example from schema instead of full schema
            schema = response_format.model_json_schema()

            # Build example JSON structure from schema properties
            example = {}
            properties = schema.get("properties", {})

            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "string")

                # Create example values based on type
                if field_type == "string":
                    example[field_name] = f"<{field_name}>"
                elif field_type == "number" or field_type == "integer":
                    example[field_name] = (
                        0.5 if "confidence" in field_name or "relevance" in field_name else 1
                    )
                elif field_type == "boolean":
                    example[field_name] = True
                elif field_type == "array":
                    example[field_name] = []
                elif field_type == "object":
                    example[field_name] = {}
                else:
                    example[field_name] = None

            example_str = json.dumps(example, indent=2)

            enhanced_prompt = f"""{prompt}

You MUST respond with valid JSON matching this structure:
{example_str}

IMPORTANT:
- Replace placeholder values like "<field_name>" with actual content
- All fields marked as REQUIRED in descriptions must be included
- Return ONLY valid JSON, no markdown formatting or extra text
- Do not return the schema itself, return actual data"""

            messages = [{"role": "user", "content": enhanced_prompt}]

        # Make request
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            format=format_type,
            options=options,
            **{k: v for k, v in kwargs.items() if k != "options"},
        )

        content = response["message"]["content"]

        # Parse structured output
        if response_format:
            try:
                # Clean JSON if wrapped in markdown
                cleaned = self._extract_json(content)

                # Debug: Check if LLM returned schema instead of data
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "properties" in parsed and "type" in parsed:
                        raise ValueError(
                            "LLM returned the JSON schema instead of actual data. "
                            "This usually means the prompt needs to be clearer about expecting actual values, not the schema definition."
                        )
                except json.JSONDecodeError:
                    pass  # Will be caught below

                return response_format.model_validate_json(cleaned)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse structured output: {e}\n"
                    f"Raw response (first 500 chars): {content[:500]}\n"
                    f"Expected format: {response_format.__name__}"
                ) from e

        return content

    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from content that might have markdown formatting.

        Args:
            content: Raw content that may contain JSON

        Returns:
            Cleaned JSON string
        """
        content = content.strip()

        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return content

    async def close(self):
        """Close client (Ollama SDK handles cleanup internally)."""
        pass
