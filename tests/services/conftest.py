"""Fixtures for service tests.

Fixtures use function scope to avoid event loop issues.
Each test gets fresh instances but cleanup happens after all tests via pytest hooks.
"""

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest

from src.config import Config
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.core.graph_store.sqlite_store import SQLiteGraphStore
from src.core.llm.ollama import OllamaLLM
from src.core.vector_store.qdrant import QdrantStore
from src.models.memory import Memory, MemoryStatus, NodeType

# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
async def ollama_llm() -> AsyncGenerator[OllamaLLM, None]:
    """
    Create real Ollama LLM provider for testing.

    Requires Ollama to be running with llama3.1:8b model.
    Tests will skip if Ollama is not available.
    """
    llm = OllamaLLM(host="http://localhost:11434", model="llama3.1:8b", timeout=120.0)

    try:
        # Test connection by making a simple request
        await llm.complete("test", max_tokens=10)
        yield llm
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")
    finally:
        await llm.close()


@pytest.fixture
async def ollama_embedder() -> AsyncGenerator[OllamaEmbedder, None]:
    """
    Create real Ollama embedder for testing.

    Requires Ollama to be running with nomic-embed-text model.
    Tests will skip if Ollama is not available.
    """
    embedder = OllamaEmbedder(
        host="http://localhost:11434", model="nomic-embed-text", timeout=120.0
    )

    try:
        # Test connection
        await embedder.embed("test")
        yield embedder
    except Exception as e:
        pytest.skip(f"Ollama embedder not available: {e}")
    finally:
        await embedder.close()


@pytest.fixture
async def qdrant_vector_store() -> AsyncGenerator[QdrantStore, None]:
    """
    Create real Qdrant vector store for testing.

    Requires Qdrant to be running on localhost:6333.
    Tests will skip if Qdrant is not available.
    Uses a test collection that gets cleaned up after test.
    """
    import uuid

    # Use unique collection name for each test
    collection_name = f"test_memories_{uuid.uuid4().hex[:8]}"

    store = QdrantStore(
        host="localhost",
        port=6333,
        collection_name=collection_name,
        vector_size=768,  # nomic-embed-text dimension
        use_grpc=False,
        use_quantization=False,
    )

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
async def sqlite_graph_store() -> AsyncGenerator[SQLiteGraphStore, None]:
    """
    Create SQLite graph store for testing.

    Uses in-memory database for fast, isolated tests.
    """
    store = SQLiteGraphStore(db_path=":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def neo4j_graph_store() -> AsyncGenerator[Neo4jGraphStore, None]:
    """
    Create Neo4j graph store for testing (requires running Neo4j).

    Cleans up test data after each test.
    """
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
    )
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
    return Memory(
        id="mem_test_1",
        content="This is a test memory about Python async programming",
        type=NodeType.MEMORY,
        embedding=[0.1] * 768,  # nomic-embed-text dimension
        status=MemoryStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_memories() -> list[Memory]:
    """Create multiple sample memories."""
    return [
        Memory(
            id=f"mem_test_{i}",
            content=f"Test memory {i} about programming",
            type=NodeType.MEMORY,
            embedding=[0.1 * i] * 768,  # nomic-embed-text dimension
            status=MemoryStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    from src.config import Config

    config = Config()
    # Override with test settings
    config.llm_relationships.context_window = 20  # Smaller for testing
    config.llm_relationships.min_confidence = 0.5
    config.llm_relationships.min_derived_confidence = 0.7
    config.llm_relationships.enable_auto_invalidation = False
    return config


# ═══════════════════════════════════════════════════════════
# Fixture Aliases for Backward Compatibility
# ═══════════════════════════════════════════════════════════


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
