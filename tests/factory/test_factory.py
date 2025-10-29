"""
Tests for factory classes.

Tests the creation of components using factories.
"""

import pytest

from src.config import Config, EmbedderConfig, LLMConfig, QdrantConfig
from src.core.embeddings.base import Embedder
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.embeddings.openai import OpenAIEmbedder
from src.core.factory import EmbedderFactory, GraphStoreFactory, LLMFactory, VectorStoreFactory
from src.core.graph_store.base import GraphStore
from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.core.graph_store.sqlite_store import SQLiteGraphStore
from src.core.llm.base import LLMProvider
from src.core.llm.ollama import OllamaLLM
from src.core.llm.openai import OpenAILLM
from src.core.vector_store.base import VectorStore
from src.core.vector_store.qdrant import QdrantStore


class TestLLMFactory:
    """Test LLM factory."""

    def test_create_ollama_llm(self):
        """Test creating Ollama LLM provider."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1:8b",
            base_url="http://localhost:11434",
        )

        llm = LLMFactory.create(config)

        assert isinstance(llm, OllamaLLM)
        assert isinstance(llm, LLMProvider)
        assert llm.model == "llama3.1:8b"

    def test_create_openai_llm(self):
        """Test creating OpenAI LLM provider."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test-key",
        )

        llm = LLMFactory.create(config)

        assert isinstance(llm, OpenAILLM)
        assert isinstance(llm, LLMProvider)
        assert llm.model == "gpt-4o-mini"

    def test_create_openai_without_api_key_raises_error(self):
        """Test that OpenAI without API key raises error."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key=None,  # Missing API key
        )

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            LLMFactory.create(config)

    def test_create_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises error."""
        config = LLMConfig(
            provider="unsupported",
            model="some-model",
        )

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create(config)

    def test_create_from_full_config(self):
        """Test creating LLM from full Config object."""
        config = Config()
        config.llm.provider = "ollama"
        config.llm.model = "llama3.2:latest"

        llm = LLMFactory.create(config.llm)

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3.2:latest"


class TestEmbedderFactory:
    """Test embedder factory."""

    def test_create_ollama_embedder(self):
        """Test creating Ollama embedder."""
        config = EmbedderConfig(
            provider="ollama",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
        )

        embedder = EmbedderFactory.create(config)

        assert isinstance(embedder, OllamaEmbedder)
        assert isinstance(embedder, Embedder)
        assert embedder.model == "nomic-embed-text"

    def test_create_openai_embedder(self):
        """Test creating OpenAI embedder."""
        config = EmbedderConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test-key",
        )

        embedder = EmbedderFactory.create(config)

        assert isinstance(embedder, OpenAIEmbedder)
        assert isinstance(embedder, Embedder)

    def test_create_openai_embedder_without_api_key_raises_error(self):
        """Test that OpenAI embedder without API key raises error."""
        config = EmbedderConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key=None,
        )

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbedderFactory.create(config)

    def test_create_unsupported_embedder_raises_error(self):
        """Test that unsupported embedder raises error."""
        config = EmbedderConfig(
            provider="unsupported",
            model="some-model",
        )

        with pytest.raises(ValueError, match="Unsupported embedder provider"):
            EmbedderFactory.create(config)

    @pytest.mark.asyncio
    async def test_get_dimension_from_config(self):
        """Test getting dimension from config."""
        config = EmbedderConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,  # Explicit dimension
        )
        embedder = EmbedderFactory.create(config)

        dimension = await EmbedderFactory.get_dimension(embedder, config)

        assert dimension == 768

    @pytest.mark.asyncio
    async def test_get_dimension_from_embedder_property(self):
        """Test getting dimension from embedder property."""
        config = EmbedderConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=None,  # No explicit dimension
        )
        embedder = EmbedderFactory.create(config)

        # Mock dimension property
        embedder.dimension = 768

        dimension = await EmbedderFactory.get_dimension(embedder, config)

        assert dimension == 768

    @pytest.mark.asyncio
    async def test_get_dimension_auto_detect(self, ollama_embedder):
        """Test auto-detecting dimension from actual embedding.

        This tests the fallback mechanism where if dimension is not in config
        and not available as a property, it actually embeds a test string
        to determine the dimension.
        """
        config = EmbedderConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=None,
        )

        # Note: This test requires actual Ollama connection
        # It tests the complete fallback: config -> property -> auto-detect
        dimension = await EmbedderFactory.get_dimension(ollama_embedder, config)

        # nomic-embed-text has 768 dimensions
        assert dimension == 768
        assert isinstance(dimension, int)
        assert dimension > 0


class TestGraphStoreFactory:
    """Test graph store factory."""

    def test_create_sqlite_graph_store(self):
        """Test creating SQLite graph store."""
        config = Config()
        config.graph_backend = "sqlite"
        config.sqlite.db_path = ":memory:"

        graph_store = GraphStoreFactory.create(config)

        assert isinstance(graph_store, SQLiteGraphStore)
        assert isinstance(graph_store, GraphStore)
        assert graph_store.db_path == ":memory:"

    def test_create_neo4j_graph_store(self):
        """Test creating Neo4j graph store."""
        config = Config()
        config.graph_backend = "neo4j"
        config.neo4j.uri = "bolt://localhost:7687"
        config.neo4j.username = "neo4j"
        config.neo4j.password = "password"

        graph_store = GraphStoreFactory.create(config)

        assert isinstance(graph_store, Neo4jGraphStore)
        assert isinstance(graph_store, GraphStore)
        assert graph_store.uri == "bolt://localhost:7687"

    def test_create_unsupported_graph_backend_raises_error(self):
        """Test that unsupported graph backend raises error."""
        config = Config()
        config.graph_backend = "unsupported"

        with pytest.raises(ValueError, match="Unsupported graph backend"):
            GraphStoreFactory.create(config)

    def test_create_with_custom_neo4j_settings(self):
        """Test creating Neo4j with custom settings."""
        config = Config()
        config.graph_backend = "neo4j"
        config.neo4j.uri = "neo4j://custom:7687"
        config.neo4j.username = "admin"
        config.neo4j.password = "secret"
        config.neo4j.database = "custom_db"

        graph_store = GraphStoreFactory.create(config)

        assert isinstance(graph_store, Neo4jGraphStore)
        assert graph_store.uri == "neo4j://custom:7687"
        assert graph_store.username == "admin"
        assert graph_store.database == "custom_db"


class TestVectorStoreFactory:
    """Test vector store factory."""

    def test_create_qdrant_store(self):
        """Test creating Qdrant vector store."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_memories",
            use_grpc=False,
            hnsw_m=16,
        )
        vector_size = 768

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert isinstance(vector_store, QdrantStore)
        assert isinstance(vector_store, VectorStore)
        assert vector_store.collection_name == "test_memories"
        assert vector_store.vector_size == 768
        assert vector_store.host == "localhost"
        assert vector_store.port == 6333

    def test_create_with_quantization(self):
        """Test creating Qdrant with quantization settings."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="memories",
            use_quantization=True,
            quantization_type="int8",
            use_grpc=True,
        )
        vector_size = 1536

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert isinstance(vector_store, QdrantStore)
        assert vector_store.use_quantization is True
        assert vector_store.quantization_type == "int8"
        assert vector_store.use_grpc is True

    def test_create_with_custom_hnsw_settings(self):
        """Test creating Qdrant with custom HNSW settings."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="memories",
            hnsw_m=32,
            hnsw_ef_construct=200,
        )
        vector_size = 768

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert isinstance(vector_store, QdrantStore)
        assert vector_store.collection_name == "memories"
        assert vector_store.hnsw_m == 32
        assert vector_store.hnsw_ef_construct == 200

    def test_create_with_on_disk_storage(self):
        """Test creating Qdrant with on-disk storage."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="memories",
            on_disk=True,
        )
        vector_size = 768

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert isinstance(vector_store, QdrantStore)
        assert vector_store.collection_name == "memories"
        assert vector_store.on_disk is True

    def test_create_with_all_advanced_settings(self):
        """Test creating Qdrant with all advanced configuration options."""
        config = QdrantConfig(
            url="http://custom-qdrant:6334",
            collection_name="advanced_test",
            use_grpc=False,
            hnsw_m=64,
            hnsw_ef_construct=400,
            use_quantization=False,
            quantization_type="scalar",
            on_disk=True,
        )
        vector_size = 1536

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert isinstance(vector_store, QdrantStore)
        assert vector_store.host == "custom-qdrant"
        assert vector_store.port == 6334
        assert vector_store.collection_name == "advanced_test"
        assert vector_store.vector_size == 1536
        assert vector_store.use_grpc is False
        assert vector_store.hnsw_m == 64
        assert vector_store.hnsw_ef_construct == 400
        assert vector_store.use_quantization is False
        assert vector_store.quantization_type == "scalar"
        assert vector_store.on_disk is True

    def test_url_parsing_with_custom_port(self):
        """Test that URL parsing correctly extracts host and port."""
        config = QdrantConfig(
            url="http://192.168.1.100:6334",
            collection_name="test",
        )
        vector_size = 768

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert vector_store.host == "192.168.1.100"
        assert vector_store.port == 6334

    def test_url_parsing_without_port(self):
        """Test URL parsing with default port."""
        config = QdrantConfig(
            url="http://qdrant-server",
            collection_name="test",
        )
        vector_size = 768

        vector_store = VectorStoreFactory.create(config, vector_size)

        assert vector_store.host == "qdrant-server"
        assert vector_store.port == 6333  # Default port


class TestFactoryIntegration:
    """Test factory integration with full config."""

    def test_create_all_components_from_config(self):
        """Test creating all components from a single config."""
        config = Config()
        config.llm.provider = "ollama"
        config.embedder.provider = "ollama"
        config.graph_backend = "sqlite"

        # Create all components
        llm = LLMFactory.create(config.llm)
        embedder = EmbedderFactory.create(config.embedder)
        graph_store = GraphStoreFactory.create(config)
        vector_store = VectorStoreFactory.create(config.qdrant, 768)

        # Verify all created successfully
        assert isinstance(llm, LLMProvider)
        assert isinstance(embedder, Embedder)
        assert isinstance(graph_store, GraphStore)
        assert isinstance(vector_store, VectorStore)

    def test_create_openai_stack(self):
        """Test creating full OpenAI stack."""
        config = Config()
        config.llm.provider = "openai"
        config.llm.model = "gpt-4o-mini"
        config.llm.api_key = "sk-test"
        config.embedder.provider = "openai"
        config.embedder.model = "text-embedding-3-small"
        config.embedder.api_key = "sk-test"
        config.embedder.dimension = 1536

        llm = LLMFactory.create(config.llm)
        embedder = EmbedderFactory.create(config.embedder)

        assert isinstance(llm, OpenAILLM)
        assert isinstance(embedder, OpenAIEmbedder)

    def test_create_mixed_providers(self):
        """Test creating components with mixed providers."""
        config = Config()
        config.llm.provider = "openai"
        config.llm.api_key = "sk-test"
        config.embedder.provider = "ollama"
        config.graph_backend = "neo4j"

        llm = LLMFactory.create(config.llm)
        embedder = EmbedderFactory.create(config.embedder)
        graph_store = GraphStoreFactory.create(config)

        # Mixed providers should work
        assert isinstance(llm, OpenAILLM)
        assert isinstance(embedder, OllamaEmbedder)
        assert isinstance(graph_store, Neo4jGraphStore)
