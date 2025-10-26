"""Fixtures for service tests."""

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import pytest

from src.config import Config
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.core.graph_store.sqlite_store import SQLiteGraphStore
from src.core.llm.ollama import OllamaLLM
from src.core.vector_store.base import SearchResult, VectorStore
from src.models.memory import Memory, MemoryStatus, NodeType


# ═══════════════════════════════════════════════════════════
# Mock Vector Store
# ═══════════════════════════════════════════════════════════


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize store."""
        self.initialized = True

    async def upsert_memory(self, memory: Memory) -> None:
        """Store memory."""
        self.memories[memory.id] = memory

    async def batch_upsert(self, memories: list[Memory]) -> None:
        """Batch upsert."""
        for memory in memories:
            await self.upsert_memory(memory)

    async def search_similar(
        self,
        vector: list[float],
        limit: int = 10,
        filters: dict = None,
        score_threshold: float = None,
    ) -> list[SearchResult]:
        """Search for similar memories."""
        results = []
        for memory in self.memories.values():
            # Simple cosine similarity
            if memory.embedding:
                similarity = sum(a * b for a, b in zip(vector, memory.embedding, strict=False)) / (
                    sum(a * a for a in vector) ** 0.5
                    * sum(b * b for b in memory.embedding) ** 0.5
                    or 1
                )
            else:
                similarity = 0.5

            if score_threshold is None or similarity >= score_threshold:
                # Apply filters
                if filters:
                    if "status" in filters and memory.status.value not in filters["status"]:
                        continue
                    if "created_after" in filters and memory.created_at.isoformat() < filters[
                        "created_after"
                    ]:
                        continue

                results.append(SearchResult(memory=memory, score=similarity))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def search_by_payload(self, filter: dict, limit: int = 10) -> list[SearchResult]:
        """Search by payload."""
        results = []
        for memory in self.memories.values():
            # Simple filter matching
            results.append(SearchResult(memory=memory, score=1.0))
            if len(results) >= limit:
                break
        return results

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get memory by ID."""
        return self.memories.get(memory_id)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete memory."""
        self.memories.pop(memory_id, None)

    async def count_memories(self, filters: dict = None) -> int:
        """Count memories."""
        return len(self.memories)

    async def close(self) -> None:
        """Close store."""
        pass


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
async def ollama_llm() -> AsyncGenerator[OllamaLLM, None]:
    """
    Create real Ollama LLM provider for testing.

    Requires Ollama to be running with llama3.1 model.
    Tests will skip if Ollama is not available.
    """
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1",
        timeout=120.0
    )

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
        host="http://localhost:11434",
        model="nomic-embed-text",
        timeout=120.0
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
async def mock_vector_store() -> AsyncGenerator[MockVectorStore, None]:
    """Create mock vector store."""
    store = MockVectorStore()
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def sqlite_graph_store() -> AsyncGenerator[SQLiteGraphStore, None]:
    """Create SQLite graph store for testing."""
    store = SQLiteGraphStore(db_path=":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def neo4j_graph_store() -> AsyncGenerator[Neo4jGraphStore, None]:
    """Create Neo4j graph store for testing (requires running Neo4j)."""
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
        # Clean up test data
        await store.close()


@pytest.fixture
def sample_memory() -> Memory:
    """Create a sample memory for testing."""
    return Memory(
        id="mem_test_1",
        content="This is a test memory about Python async programming",
        type=NodeType.MEMORY,
        embedding=[0.1] * 16,
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
            embedding=[0.1 * i] * 16,
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
