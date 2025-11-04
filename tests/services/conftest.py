"""Fixtures for service tests.

Fixtures use function scope to avoid event loop issues.
Each test gets fresh instances but cleanup happens after all tests via pytest hooks.

Configuration is loaded from:
1. .env.test file (if exists)
2. Environment variables
3. Default test values
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config
from src.core.factory import EmbedderFactory, GraphStoreFactory, LLMFactory, VectorStoreFactory
from src.models.memory import Memory, MemoryStatus, NodeType

# Configuration


def get_test_config() -> Config:
    """
    Get test configuration from environment or defaults.

    Loads from .env.test if exists, otherwise uses environment variables or defaults.
    """
    env_test_path = Path(".env.test")
    if env_test_path.exists():
        return Config.from_env(env_file=env_test_path)
    return Config.from_env()


# Fixtures


@pytest.fixture
async def ollama_llm() -> AsyncGenerator:
    """
    Create LLM provider for testing using configuration.

    Uses provider from config (.env.test or environment variables).
    Tests will skip if provider is not available.
    """
    test_config = get_test_config()
    llm = LLMFactory.create(test_config.llm)

    try:
        # Test connection by making a simple request
        await llm.complete("test", max_tokens=10)
        yield llm
    except Exception as e:
        pytest.skip(f"LLM provider not available: {e}")
    finally:
        await llm.close()


@pytest.fixture
async def ollama_embedder() -> AsyncGenerator:
    """
    Create embedder for testing using configuration.

    Uses provider from config (.env.test or environment variables).
    Tests will skip if embedder is not available.
    """
    test_config = get_test_config()
    embedder = EmbedderFactory.create(test_config.embedder)

    try:
        # Test connection
        await embedder.embed("test")
        yield embedder
    except Exception as e:
        pytest.skip(f"Embedder not available: {e}")
    finally:
        await embedder.close()


@pytest.fixture
async def qdrant_vector_store() -> AsyncGenerator:
    """
    Create Qdrant vector store for testing.

    Requires Qdrant to be running.
    Tests will skip if Qdrant is not available.
    Uses a test collection that gets cleaned up after test.
    """
    import uuid

    test_config = get_test_config()

    # Use unique collection name for each test
    collection_name = f"test_{test_config.qdrant.collection_name}_{uuid.uuid4().hex[:8]}"
    test_qdrant_config = test_config.qdrant.model_copy()
    test_qdrant_config.collection_name = collection_name

    # Get embedding dimension
    embedder = EmbedderFactory.create(test_config.embedder)
    try:
        vector_size = await EmbedderFactory.get_dimension(embedder, test_config.embedder)
        await embedder.close()
    except Exception:
        await embedder.close()
        pytest.skip("Cannot determine embedding dimension")
        return

    store = VectorStoreFactory.create(test_qdrant_config, vector_size)

    try:
        await store.initialize()
        yield store
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")
    finally:
        # Clean up: delete test collection
        try:
            if store.client:
                await store.client.delete_collection(collection_name)
        except Exception:
            pass
        await store.close()


@pytest.fixture
async def neo4j_graph_store() -> AsyncGenerator:
    """
    Create Neo4j graph store for testing (requires running Neo4j).

    Uses configuration from .env.test or environment.
    Cleans up test data after each test.
    """
    test_config = get_test_config()
    store = GraphStoreFactory.create(test_config)

    # Skip if not Neo4j
    from src.core.graph_store.neo4j_store import Neo4jGraphStore

    if not isinstance(store, Neo4jGraphStore):
        pytest.skip("Test requires Neo4j backend")

    try:
        await store.initialize()
        yield store
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
    finally:
        # Clean up: Delete all test nodes and relationships
        try:
            if store.driver:
                async with store.driver.session(database=store.database) as session:
                    # Delete all Memory nodes and their relationships
                    await session.run("MATCH (m:Memory) DETACH DELETE m")
        except Exception:
            pass
        await store.close()


@pytest.fixture
def sample_memory() -> Memory:
    """Create a sample memory for testing."""
    test_config = get_test_config()
    # Use configured dimension or default
    dim = test_config.embedder.dimension if test_config.embedder.dimension else 768

    return Memory(
        id="mem_test_1",
        content="This is a test memory about Python async programming",
        type=NodeType.MEMORY,
        embedding=[0.1] * dim,
        status=MemoryStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_memories() -> list[Memory]:
    """Create multiple sample memories."""
    test_config = get_test_config()
    # Use configured dimension or default
    dim = test_config.embedder.dimension if test_config.embedder.dimension else 768

    return [
        Memory(
            id=f"mem_test_{i}",
            content=f"Test memory {i} about programming",
            type=NodeType.MEMORY,
            embedding=[0.1 * i] * dim,
            status=MemoryStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    config = get_test_config()
    # Override with test settings
    config.llm_relationships.context_window = 20  # Smaller for testing
    config.llm_relationships.min_confidence = 0.5
    config.llm_relationships.min_derived_confidence = 0.7
    config.llm_relationships.enable_auto_invalidation = False
    return config


# Fixture Aliases for Backward Compatibility


@pytest.fixture
async def mock_llm(ollama_llm):
    """Alias for ollama_llm to maintain test compatibility."""
    return ollama_llm


@pytest.fixture
async def mock_embedder(ollama_embedder):
    """Alias for ollama_embedder to maintain test compatibility."""
    return ollama_embedder


@pytest.fixture
async def mock_vector_store(qdrant_vector_store):
    """Alias for qdrant_vector_store to maintain test compatibility."""
    return qdrant_vector_store


@pytest.fixture
def mock_sync_manager(neo4j_graph_store, mock_vector_store):
    """Create a sync manager for testing."""
    from src.services.memory_sync import MemorySyncManager

    return MemorySyncManager(
        graph_store=neo4j_graph_store,
        vector_store=mock_vector_store,
        max_retries=3,
        retry_delay=0.5,
    )
