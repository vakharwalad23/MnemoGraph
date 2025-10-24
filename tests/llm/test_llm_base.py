"""
Tests for LLM base class.
"""
import pytest
from pydantic import BaseModel

from src.core.llm.base import LLMProvider


class MockLLM(LLMProvider):
    """Mock LLM provider for testing."""

    async def complete(self, prompt: str, response_format=None, **kwargs):
        if response_format:
            # Return mock structured response
            return response_format(answer="test", confidence=0.9)
        return "test response"

    async def close(self):
        """Mock close implementation."""
        pass


class TestResponse(BaseModel):
    """Test response model."""
    answer: str
    confidence: float


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMProviderBase:
    """Test base LLM provider functionality."""

    async def test_abstract_instantiation(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMProvider()

    async def test_complete_interface(self):
        """Test complete method interface."""
        provider = MockLLM()
        result = await provider.complete("test prompt")
        assert isinstance(result, str)
        assert result == "test response"

    async def test_complete_with_response_format(self):
        """Test complete with structured output."""
        provider = MockLLM()
        result = await provider.complete(
            "test prompt",
            response_format=TestResponse
        )
        assert isinstance(result, TestResponse)
        assert result.answer == "test"
        assert result.confidence == 0.9

    async def test_complete_with_parameters(self):
        """Test complete with custom parameters."""
        provider = MockLLM()
        result = await provider.complete(
            "test",
            max_tokens=100,
            temperature=0.7
        )
        assert result == "test response"

    async def test_close_default(self):
        """Test default close implementation."""
        provider = MockLLM()
        await provider.close()  # Should not raise

