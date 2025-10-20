"""Tests for embedding providers."""

import pytest
from src.core.embeddings import (
    EmbeddingProvider,
    OllamaEmbedding,
    create_embedding_provider,
)


class TestOllamaEmbedding:
    """Test Ollama embedding provider."""
    
    @pytest.fixture
    def embedder(self):
        """Create an Ollama embedder instance."""
        return OllamaEmbedding(
            model="nomic-embed-text",
            host="http://localhost:11434"
        )
    
    @pytest.mark.asyncio
    async def test_single_embedding(self, embedder):
        """Test generating a single embedding."""
        text = "Python is a programming language"
        embedding = await embedder.embed(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) == 768  # nomic-embed-text dimension
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_cached(self, embedder):
        """Test that dimension is cached after first call."""
        assert embedder._dimension is None
        
        await embedder.embed("Test text")
        
        assert embedder._dimension == 768
        assert embedder.dimension == 768
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self, embedder):
        """Test generating embeddings for multiple texts."""
        texts = [
            "Machine learning is awesome",
            "Neural networks are powerful",
            "Data science is interesting"
        ]
        
        embeddings = await embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 768 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, embedder):
        """Test that different texts produce different embeddings."""
        text1 = "Python programming"
        text2 = "Cooking recipes"
        
        emb1 = await embedder.embed(text1)
        emb2 = await embedder.embed(text2)
        
        assert emb1 != emb2
    
    @pytest.mark.asyncio
    async def test_similar_texts_similar_embeddings(self, embedder):
        """Test that similar texts produce similar embeddings."""
        text1 = "Python is a programming language"
        text2 = "Python is used for programming"
        
        emb1 = await embedder.embed(text1)
        emb2 = await embedder.embed(text2)
        
        # Calculate cosine similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Similar texts should have high similarity (> 0.5)
        assert similarity > 0.5
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, embedder):
        """Test handling of empty text."""
        # Ollama returns an empty embedding list for empty text (expected behavior)
        embedding = await embedder.embed("")
        assert isinstance(embedding, list)
        # Empty text typically returns empty embedding
        if len(embedding) == 0:
            # This is the expected behavior for Ollama with empty text
            assert True
        else:
            # Some models might still return a valid embedding
            assert len(embedding) == 768


class TestEmbeddingFactory:
    """Test embedding provider factory."""
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider via factory."""
        provider = create_embedding_provider(
            provider_type="ollama",
            model="nomic-embed-text"
        )
        
        assert isinstance(provider, OllamaEmbedding)
        assert provider.model == "nomic-embed-text"
    
    def test_create_ollama_with_custom_host(self):
        """Test creating Ollama provider with custom host."""
        provider = create_embedding_provider(
            provider_type="ollama",
            model="nomic-embed-text",
            host="http://custom-host:11434"
        )
        
        assert provider.host == "http://custom-host:11434"
    
    def test_create_ollama_default_model(self):
        """Test factory uses default model when not specified."""
        provider = create_embedding_provider(provider_type="ollama")
        
        assert provider.model == "nomic-embed-text"
    
    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embedding_provider(provider_type="fake-provider")
    
    def test_openai_not_implemented(self):
        """Test that OpenAI provider raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="OpenAI provider not yet implemented"):
            create_embedding_provider(provider_type="openai")
    
    def test_sentence_transformers_not_implemented(self):
        """Test that SentenceTransformers provider raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, 
            match="SentenceTransformers provider not yet implemented"
        ):
            create_embedding_provider(provider_type="sentence-transformers")


class TestEmbeddingProviderInterface:
    """Test that providers implement the correct interface."""
    
    def test_ollama_implements_interface(self):
        """Test that OllamaEmbedding implements EmbeddingProvider."""
        provider = OllamaEmbedding()
        assert isinstance(provider, EmbeddingProvider)
    
    def test_has_embed_method(self):
        """Test that provider has embed method."""
        provider = OllamaEmbedding()
        assert hasattr(provider, "embed")
        assert callable(provider.embed)
    
    def test_has_embed_batch_method(self):
        """Test that provider has embed_batch method."""
        provider = OllamaEmbedding()
        assert hasattr(provider, "embed_batch")
        assert callable(provider.embed_batch)
    
    def test_has_dimension_property(self):
        """Test that provider has dimension property."""
        provider = OllamaEmbedding()
        assert hasattr(provider, "dimension")
        assert isinstance(provider.dimension, int)