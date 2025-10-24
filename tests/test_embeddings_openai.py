"""
Tests for OpenAI embedder.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.embeddings.openai import OpenAIEmbedder


@pytest.fixture
def openai_embedder():
    """Create OpenAI embedder for testing."""
    return OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
        timeout=120.0
    )


@pytest.mark.asyncio
class TestOpenAIEmbedder:
    """Test OpenAI embedder."""

    async def test_initialization(self, openai_embedder):
        """Test embedder initialization."""
        assert openai_embedder.model == "text-embedding-3-small"
        assert openai_embedder.client is not None

    async def test_initialization_with_organization(self):
        """Test initialization with organization."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            organization="org-123"
        )
        assert embedder.client is not None

    async def test_initialization_with_base_url(self):
        """Test initialization with custom base URL."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            base_url="https://custom.openai.com"
        )
        assert embedder.client is not None

    async def test_embed(self, openai_embedder):
        """Test embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        with patch.object(openai_embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await openai_embedder.embed("test text")

            assert isinstance(result, list)
            assert result == [0.1, 0.2, 0.3]
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input="test text"
            )

    async def test_embed_with_kwargs(self, openai_embedder):
        """Test embedding with extra kwargs."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]

        with patch.object(openai_embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await openai_embedder.embed("test", dimensions=512, user="user123")

            call_args = mock_create.call_args
            assert call_args.kwargs["dimensions"] == 512
            assert call_args.kwargs["user"] == "user123"

    async def test_batch_embed(self, openai_embedder):
        """Test batch embedding."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4])
        ]

        with patch.object(openai_embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            texts = ["text1", "text2"]
            results = await openai_embedder.batch_embed(texts)

            assert len(results) == 2
            assert results[0] == [0.1, 0.2]
            assert results[1] == [0.3, 0.4]
            mock_create.assert_called_once()

    async def test_batch_embed_large(self, openai_embedder):
        """Test batch embedding with multiple batches."""
        # First batch (2048 items)
        mock_response1 = MagicMock()
        mock_response1.data = [MagicMock(embedding=[float(i)]) for i in range(2048)]

        # Second batch (remaining 100 items)
        mock_response2 = MagicMock()
        mock_response2.data = [MagicMock(embedding=[float(i)]) for i in range(100)]

        with patch.object(openai_embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [mock_response1, mock_response2]

            texts = [f"text{i}" for i in range(2148)]
            results = await openai_embedder.batch_embed(texts)

            assert len(results) == 2148
            assert mock_create.call_count == 2

    async def test_batch_embed_custom_batch_size(self, openai_embedder):
        """Test batch embedding with custom batch size."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1]),
            MagicMock(embedding=[0.2])
        ]

        with patch.object(openai_embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            texts = ["text1", "text2", "text3", "text4"]
            results = await openai_embedder.batch_embed(texts, batch_size=2)

            assert len(results) == 4
            assert mock_create.call_count == 2

    async def test_batch_embed_empty_list(self, openai_embedder):
        """Test batch embedding with empty list."""
        results = await openai_embedder.batch_embed([])
        assert results == []

    async def test_get_dimension_known_model(self, openai_embedder):
        """Test get_dimension for known model."""
        dim = await openai_embedder.get_dimension()
        assert dim == 1536  # Known dimension for text-embedding-3-small

    async def test_get_dimension_known_model_large(self):
        """Test get_dimension for large model."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            model="text-embedding-3-large"
        )
        dim = await embedder.get_dimension()
        assert dim == 3072

    async def test_get_dimension_ada(self):
        """Test get_dimension for ada-002."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            model="text-embedding-ada-002"
        )
        dim = await embedder.get_dimension()
        assert dim == 1536

    async def test_get_dimension_unknown_model(self):
        """Test get_dimension for unknown model (fallback to test embed)."""
        embedder = OpenAIEmbedder(
            api_key="test-key",
            model="unknown-model"
        )

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])]

        with patch.object(embedder.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            dim = await embedder.get_dimension()
            assert dim == 4
            mock_create.assert_called_once()

    async def test_close(self, openai_embedder):
        """Test close method."""
        with patch.object(openai_embedder.client, 'close', new_callable=AsyncMock) as mock_close:
            await openai_embedder.close()
            mock_close.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
class TestOpenAIEmbedderIntegration:
    """
    Integration tests for OpenAI embedder.
    Requires OPENAI_API_KEY environment variable.
    Run with: pytest -m integration
    """

    async def test_real_embedding(self):
        """Test real embedding with OpenAI."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=api_key)

        try:
            result = await embedder.embed("test text")
            assert isinstance(result, list)
            assert len(result) == 1536  # text-embedding-3-small
            assert all(isinstance(x, float) for x in result)
        finally:
            await embedder.close()

    async def test_real_batch_embedding(self):
        """Test real batch embedding with OpenAI."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=api_key)

        try:
            texts = ["hello", "world", "test"]
            results = await embedder.batch_embed(texts)
            assert len(results) == 3
            assert all(isinstance(emb, list) for emb in results)
            assert all(len(emb) == 1536 for emb in results)
        finally:
            await embedder.close()

    async def test_real_dimension(self):
        """Test real dimension retrieval."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=api_key)

        try:
            dim = await embedder.get_dimension()
            assert dim == 1536
        finally:
            await embedder.close()

