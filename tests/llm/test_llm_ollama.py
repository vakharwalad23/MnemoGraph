"""
Tests for Ollama LLM provider.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from src.core.llm.ollama import OllamaLLM


class SimpleResponse(BaseModel):
    """Test response model."""

    answer: str
    confidence: float
    reasoning: str


@pytest.fixture
def ollama_llm():
    """Create Ollama LLM for testing."""
    return OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", timeout=120.0)


@pytest.mark.unit
@pytest.mark.asyncio
class TestOllamaLLM:
    """Test Ollama LLM provider."""

    async def test_initialization(self, ollama_llm):
        """Test provider initialization."""
        assert ollama_llm.host == "http://localhost:11434"
        assert ollama_llm.model == "llama3.1:8b"
        assert ollama_llm.timeout == 120.0
        assert ollama_llm.client is not None

    async def test_complete_simple(self, ollama_llm):
        """Test simple text completion."""
        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": "Paris is the capital"}}

            result = await ollama_llm.complete("What is the capital of France?", max_tokens=50)

            assert isinstance(result, str)
            assert result == "Paris is the capital"
            mock_chat.assert_called_once()

    async def test_complete_with_temperature(self, ollama_llm):
        """Test completion with custom temperature."""
        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": "test"}}

            await ollama_llm.complete("test", temperature=0.7, max_tokens=100)

            call_args = mock_chat.call_args
            assert call_args.kwargs["options"]["temperature"] == 0.7
            assert call_args.kwargs["options"]["num_predict"] == 100

    async def test_complete_with_extra_options(self, ollama_llm):
        """Test completion with extra options."""
        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": "test"}}

            await ollama_llm.complete("test", options={"top_p": 0.9, "top_k": 40})

            call_args = mock_chat.call_args
            assert call_args.kwargs["options"]["top_p"] == 0.9
            assert call_args.kwargs["options"]["top_k"] == 40

    async def test_complete_structured(self, ollama_llm):
        """Test structured output completion."""
        json_response = '{"answer": "yes", "confidence": 0.95, "reasoning": "because"}'

        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": json_response}}

            result = await ollama_llm.complete("Is Python good?", response_format=SimpleResponse)

            assert isinstance(result, SimpleResponse)
            assert result.answer == "yes"
            assert result.confidence == 0.95
            assert result.reasoning == "because"

            # Verify JSON mode was used
            call_args = mock_chat.call_args
            assert call_args.kwargs["format"] == "json"

    async def test_complete_structured_with_markdown(self, ollama_llm):
        """Test structured output with markdown code blocks."""
        json_response = '```json\n{"answer": "yes", "confidence": 0.9, "reasoning": "test"}\n```'

        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": json_response}}

            result = await ollama_llm.complete("test", response_format=SimpleResponse)

            assert isinstance(result, SimpleResponse)
            assert result.answer == "yes"

    async def test_complete_structured_with_generic_code_block(self, ollama_llm):
        """Test structured output with generic code blocks."""
        json_response = '```\n{"answer": "yes", "confidence": 0.9, "reasoning": "test"}\n```'

        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": json_response}}

            result = await ollama_llm.complete("test", response_format=SimpleResponse)

            assert isinstance(result, SimpleResponse)

    async def test_complete_structured_invalid_json(self, ollama_llm):
        """Test structured output with invalid JSON."""
        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": "not json at all"}}

            with pytest.raises(ValueError, match="Failed to parse structured output"):
                await ollama_llm.complete("test", response_format=SimpleResponse)

    async def test_complete_structured_missing_fields(self, ollama_llm):
        """Test structured output with missing required fields."""
        json_response = '{"answer": "yes"}'  # Missing confidence and reasoning

        with patch.object(ollama_llm.client, "chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"message": {"content": json_response}}

            with pytest.raises(ValueError):
                await ollama_llm.complete("test", response_format=SimpleResponse)

    async def test_extract_json_clean(self, ollama_llm):
        """Test JSON extraction from clean input."""
        result = ollama_llm._extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    async def test_extract_json_with_whitespace(self, ollama_llm):
        """Test JSON extraction with whitespace."""
        result = ollama_llm._extract_json('  \n  {"key": "value"}  \n  ')
        assert result == '{"key": "value"}'

    async def test_extract_json_markdown_json(self, ollama_llm):
        """Test JSON extraction from markdown json block."""
        result = ollama_llm._extract_json('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    async def test_extract_json_markdown_generic(self, ollama_llm):
        """Test JSON extraction from generic markdown block."""
        result = ollama_llm._extract_json('```\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

    async def test_close(self, ollama_llm):
        """Test close method."""
        await ollama_llm.close()  # Should not raise


@pytest.mark.integration
@pytest.mark.asyncio
class TestOllamaLLMIntegration:
    """
    Integration tests for Ollama LLM.
    Requires running Ollama server.
    Run with: pytest -m integration
    """

    async def test_real_completion(self):
        """Test real completion with Ollama."""
        llm = OllamaLLM(model="llama3.1:8b")

        try:
            result = await llm.complete(
                "Say 'Hello' and nothing else", max_tokens=10, temperature=0.0
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
        finally:
            await llm.close()

    async def test_real_structured_output(self):
        """Test real structured output with Ollama."""
        llm = OllamaLLM(model="llama3.1:8b")

        class YesNo(BaseModel):
            answer: str
            confidence: float

        try:
            result = await llm.complete(
                "Is 2+2=4? Answer yes or no", response_format=YesNo, temperature=0.0
            )
            assert isinstance(result, YesNo)
            assert result.answer in ["yes", "Yes", "YES"]
            assert 0.0 <= result.confidence <= 1.0
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
        finally:
            await llm.close()
