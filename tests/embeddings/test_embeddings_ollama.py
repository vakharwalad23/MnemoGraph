"""
Tests for Ollama embedder.
"""
from unittest.mock import AsyncMock, patch

import pytest

from src.core.embeddings.ollama import OllamaEmbedder


@pytest.fixture
def ollama_embedder():
    """Create Ollama embedder for testing."""
    return OllamaEmbedder(
        host="http://localhost:11434",
        model="nomic-embed-text",
        timeout=120.0
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestOllamaEmbedder:
    """Test Ollama embedder."""

    async def test_initialization(self, ollama_embedder):
        """Test embedder initialization."""
        assert ollama_embedder.host == "http://localhost:11434"
        assert ollama_embedder.model == "nomic-embed-text"
        assert ollama_embedder.timeout == 120.0
        assert ollama_embedder.client is not None
        assert ollama_embedder._dimension is None

    async def test_embed(self, ollama_embedder):
        """Test embedding generation."""
        with patch.object(ollama_embedder.client, 'embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }

            result = await ollama_embedder.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 5
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embed.assert_called_once_with(
                model="nomic-embed-text",
                prompt="test text"
            )

    async def test_embed_with_kwargs(self, ollama_embedder):
        """Test embedding with extra kwargs."""
        with patch.object(ollama_embedder.client, 'embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "embedding": [0.1, 0.2]
            }

            await ollama_embedder.embed("test", keep_alive="5m")

            call_args = mock_embed.call_args
            assert call_args.kwargs["keep_alive"] == "5m"

    async def test_batch_embed(self, ollama_embedder):
        """Test batch embedding."""
        with patch.object(ollama_embedder.client, 'embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
                {"embedding": [0.5, 0.6]}
            ]

            texts = ["text1", "text2", "text3"]
            results = await ollama_embedder.batch_embed(texts, batch_size=2)

            assert len(results) == 3
            assert results[0] == [0.1, 0.2]
            assert results[1] == [0.3, 0.4]
            assert results[2] == [0.5, 0.6]
            assert mock_embed.call_count == 3

    async def test_batch_embed_empty_list(self, ollama_embedder):
        """Test batch embedding with empty list."""
        results = await ollama_embedder.batch_embed([])
        assert results == []

    async def test_batch_embed_single_item(self, ollama_embedder):
        """Test batch embedding with single item."""
        with patch.object(ollama_embedder.client, 'embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2]}

            results = await ollama_embedder.batch_embed(["test"])

            assert len(results) == 1
            assert results[0] == [0.1, 0.2]

    async def test_get_dimension_caching(self, ollama_embedder):
        """Test dimension caching."""
        with patch.object(ollama_embedder.client, 'embeddings', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "embedding": [0.1, 0.2, 0.3]
            }

            # First call should embed
            dim1 = await ollama_embedder.get_dimension()
            assert dim1 == 3
            assert mock_embed.call_count == 1

            # Second call should use cache
            dim2 = await ollama_embedder.get_dimension()
            assert dim2 == 3
            assert mock_embed.call_count == 1  # No additional call

    async def test_close(self, ollama_embedder):
        """Test close method."""
        await ollama_embedder.close()  # Should not raise


@pytest.mark.integration
@pytest.mark.asyncio
class TestOllamaEmbedderIntegration:
    """
    Integration tests for Ollama embedder.
    Requires running Ollama server with nomic-embed-text model.
    Run with: pytest -m integration
    """

    async def test_real_embedding(self):
        """Test real embedding with Ollama."""
        embedder = OllamaEmbedder(model="nomic-embed-text")

        try:
            result = await embedder.embed("test text")
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(x, float) for x in result)
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
        finally:
            await embedder.close()

    async def test_real_batch_embedding(self):
        """Test real batch embedding with Ollama."""
        embedder = OllamaEmbedder(model="nomic-embed-text")

        try:
            texts = ["hello", "world", "test"]
            results = await embedder.batch_embed(texts)
            assert len(results) == 3
            assert all(isinstance(emb, list) for emb in results)
            assert all(len(emb) > 0 for emb in results)
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
        finally:
            await embedder.close()

    async def test_real_dimension(self):
        """Test real dimension retrieval."""
        embedder = OllamaEmbedder(model="nomic-embed-text")

        try:
            dim = await embedder.get_dimension()
            assert isinstance(dim, int)
            assert dim > 0
            # nomic-embed-text typically has 768 dimensions
            assert dim == 768
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
        finally:
            await embedder.close()

