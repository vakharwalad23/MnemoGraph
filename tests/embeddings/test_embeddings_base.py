"""
Tests for embeddings base class.
"""

import pytest

from src.core.embeddings.base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder for testing."""

    async def embed(self, text: str, **kwargs):
        # Return fixed-size embedding based on text length
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    async def close(self):
        """Mock close implementation."""
        pass


@pytest.mark.unit
@pytest.mark.asyncio
class TestEmbedderBase:
    """Test base Embedder functionality."""

    async def test_abstract_instantiation(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Embedder()

    async def test_embed_interface(self):
        """Test embed method interface."""
        embedder = MockEmbedder()
        result = await embedder.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(x, float) for x in result)

    async def test_batch_embed_default(self):
        """Test default batch_embed implementation."""
        embedder = MockEmbedder()
        texts = ["text1", "text2", "text3"]
        results = await embedder.batch_embed(texts)

        assert len(results) == 3
        assert all(len(emb) == 5 for emb in results)
        assert all(isinstance(emb, list) for emb in results)

    async def test_batch_embed_with_batch_size(self):
        """Test batch_embed with custom batch size."""
        embedder = MockEmbedder()
        texts = ["text1", "text2", "text3", "text4", "text5"]
        results = await embedder.batch_embed(texts, batch_size=2)

        assert len(results) == 5

    async def test_get_dimension_default(self):
        """Test default get_dimension implementation."""
        embedder = MockEmbedder()
        dimension = await embedder.get_dimension()
        assert isinstance(dimension, int)
        assert dimension == 5

    async def test_close_default(self):
        """Test default close implementation."""
        embedder = MockEmbedder()
        await embedder.close()  # Should not raise

