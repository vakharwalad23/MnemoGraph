"""
Tests for OpenAI LLM provider.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from src.core.llm.openai import OpenAILLM


class SimpleResponse(BaseModel):
    """Test response model."""
    answer: str
    confidence: float


@pytest.fixture
def openai_llm():
    """Create OpenAI LLM for testing."""
    return OpenAILLM(
        api_key="test-key",
        model="gpt-4o",
        timeout=120.0
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestOpenAILLM:
    """Test OpenAI LLM provider."""

    async def test_initialization(self, openai_llm):
        """Test provider initialization."""
        assert openai_llm.model == "gpt-4o"
        assert openai_llm.client is not None

    async def test_initialization_with_organization(self):
        """Test initialization with organization."""
        llm = OpenAILLM(
            api_key="test-key",
            organization="org-123"
        )
        assert llm.client is not None

    async def test_initialization_with_base_url(self):
        """Test initialization with custom base URL."""
        llm = OpenAILLM(
            api_key="test-key",
            base_url="https://custom.openai.com"
        )
        assert llm.client is not None

    async def test_complete_simple(self, openai_llm):
        """Test simple completion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="test response"))
        ]

        with patch.object(openai_llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await openai_llm.complete("test prompt")

            assert isinstance(result, str)
            assert result == "test response"
            mock_create.assert_called_once()

    async def test_complete_with_parameters(self, openai_llm):
        """Test completion with custom parameters."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="test"))
        ]

        with patch.object(openai_llm.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await openai_llm.complete(
                "test",
                temperature=0.8,
                max_tokens=500,
                stop=["END"]
            )

            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.8
            assert call_args.kwargs["max_tokens"] == 500
            assert call_args.kwargs["stop"] == ["END"]

    async def test_complete_structured(self, openai_llm):
        """Test structured output with Parse API."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(
                parsed=SimpleResponse(answer="yes", confidence=0.9)
            ))
        ]

        with patch.object(openai_llm.client.beta.chat.completions, 'parse', new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = mock_response

            result = await openai_llm.complete(
                "test",
                response_format=SimpleResponse
            )

            assert isinstance(result, SimpleResponse)
            assert result.answer == "yes"
            assert result.confidence == 0.9
            mock_parse.assert_called_once()

    async def test_complete_structured_with_extra_params(self, openai_llm):
        """Test structured output with extra parameters."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(
                parsed=SimpleResponse(answer="yes", confidence=0.9)
            ))
        ]

        with patch.object(openai_llm.client.beta.chat.completions, 'parse', new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = mock_response

            await openai_llm.complete(
                "test",
                response_format=SimpleResponse,
                temperature=0.5,
                presence_penalty=0.6
            )

            call_args = mock_parse.call_args
            assert call_args.kwargs["temperature"] == 0.5
            assert call_args.kwargs["presence_penalty"] == 0.6
            assert call_args.kwargs["response_format"] == SimpleResponse

    async def test_close(self, openai_llm):
        """Test close method."""
        with patch.object(openai_llm.client, 'close', new_callable=AsyncMock) as mock_close:
            await openai_llm.close()
            mock_close.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
class TestOpenAILLMIntegration:
    """
    Integration tests for OpenAI LLM.
    Requires OPENAI_API_KEY environment variable.
    Run with: pytest -m integration
    """

    async def test_real_completion(self):
        """Test real completion with OpenAI."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini")

        try:
            result = await llm.complete(
                "Say 'Hi' and nothing else",
                max_tokens=10,
                temperature=0.0
            )
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            await llm.close()

    async def test_real_structured_output(self):
        """Test real structured output with OpenAI."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = OpenAILLM(api_key=api_key, model="gpt-4o-mini")

        class YesNo(BaseModel):
            answer: str
            confidence: float

        try:
            result = await llm.complete(
                "Is 2+2=4? Answer yes or no",
                response_format=YesNo,
                temperature=0.0
            )
            assert isinstance(result, YesNo)
            assert result.answer.lower() in ["yes", "true"]
            assert 0.0 <= result.confidence <= 1.0
        finally:
            await llm.close()

