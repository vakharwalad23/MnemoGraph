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
        model: str = "llama3.1",
        timeout: float = 120.0
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
        **kwargs
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
            **kwargs.get("options", {})
        }

        # Handle structured output
        format_type = None
        messages = [{"role": "user", "content": prompt}]

        if response_format:
            format_type = "json"

            # Enhance prompt with schema for better results
            schema = response_format.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            enhanced_prompt = f"""{prompt}

Respond with valid JSON matching this exact schema:
{schema_str}

Requirements:
- Use exact field names from schema
- Match data types exactly
- Include all required fields
- Return ONLY valid JSON, no markdown or extra text"""

            messages = [{"role": "user", "content": enhanced_prompt}]

        # Make request
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            format=format_type,
            options=options,
            **{k: v for k, v in kwargs.items() if k != "options"}
        )

        content = response["message"]["content"]

        # Parse structured output
        if response_format:
            try:
                # Clean JSON if wrapped in markdown
                cleaned = self._extract_json(content)
                return response_format.model_validate_json(cleaned)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse structured output: {e}\n"
                    f"Raw response: {content}\n"
                    f"Expected schema: {response_format.model_json_schema()}"
                )

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

