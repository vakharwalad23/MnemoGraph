"""
Tests for vector store base class.
"""

import pytest

from src.core.vector_store.base import SearchResult, VectorStore
from src.models.memory import Memory, MemoryStatus


class MockVectorStore(VectorStore):
    """Mock vector store for testing."""

    def __init__(self):
        """Initialize mock store."""
        self.memories: dict[str, Memory] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """Mock initialize implementation."""
        self.initialized = True

    async def upsert_memory(self, memory: Memory) -> None:
        """Mock upsert implementation."""
        if not memory:
            raise ValueError("Memory cannot be None")
        if not memory.id:
            raise ValueError("Memory ID cannot be empty")
        if not memory.embedding or len(memory.embedding) == 0:
            raise ValueError("Memory must have an embedding")
        self.memories[memory.id] = memory

    async def batch_upsert(self, memories: list[Memory], batch_size: int = 100) -> None:
        """Mock batch upsert implementation."""
        for memory in memories:
            await self.upsert_memory(memory)

    async def search_similar(
        self,
        vector: list[float],
        limit: int = 10,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Mock search similar implementation."""
        results = []
        for memory in self.memories.values():
            # Simple filter matching
            if filters:
                if "status" in filters:
                    if memory.status.value != filters["status"]:
                        continue
                if "type" in filters:
                    if memory.type.value != filters["type"]:
                        continue
                if "user_id" in filters:
                    if memory.user_id != filters["user_id"]:
                        continue

            # Mock similarity score
            score = 0.8
            if score_threshold and score < score_threshold:
                continue

            results.append(SearchResult(memory=memory, score=score, metadata={}))

        return results[:limit]

    async def search_by_payload(self, filter: dict, limit: int = 10) -> list[SearchResult]:
        """Mock search by payload implementation."""
        results = []
        for memory in self.memories.values():
            match = True
            for key, value in filter.items():
                if key == "status" and memory.status.value != value:
                    match = False
                    break
                if key == "type" and memory.type.value != value:
                    match = False
                    break
                if key == "user_id" and memory.user_id != value:
                    match = False
                    break

            if match:
                results.append(SearchResult(memory=memory, score=1.0, metadata={}))

        return results[:limit]

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Mock get memory implementation."""
        return self.memories.get(memory_id)

    async def delete_memory(self, memory_id: str) -> None:
        """Mock delete memory implementation."""
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} not found")
        del self.memories[memory_id]

    async def count_memories(self, filters: dict | None = None) -> int:
        """Mock count memories implementation."""
        if not filters:
            return len(self.memories)

        count = 0
        for memory in self.memories.values():
            match = True
            for key, value in filters.items():
                if key == "status" and memory.status.value != value:
                    match = False
                    break
                if key == "type" and memory.type.value != value:
                    match = False
                    break
                if key == "user_id" and memory.user_id != value:
                    match = False
                    break

            if match:
                count += 1

        return count

    async def close(self) -> None:
        """Mock close implementation."""
        self.memories.clear()


@pytest.mark.unit
@pytest.mark.asyncio
class TestVectorStoreBase:
    """Test base VectorStore functionality."""

    async def test_abstract_instantiation(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VectorStore()

    async def test_initialize_interface(self):
        """Test initialize method interface."""
        store = MockVectorStore()
        await store.initialize()
        assert store.initialized is True

    async def test_upsert_memory_interface(self):
        """Test upsert_memory method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await store.upsert_memory(memory)
        assert "test-1" in store.memories

    async def test_upsert_memory_validation(self):
        """Test upsert_memory validation."""
        store = MockVectorStore()

        # Test None memory
        with pytest.raises(ValueError, match="Memory cannot be None"):
            await store.upsert_memory(None)

        # Test empty ID
        from datetime import datetime

        memory = Memory(
            id="",
            content="Test",
            embedding=[0.1],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValueError, match="Memory ID cannot be empty"):
            await store.upsert_memory(memory)

        # Test missing embedding
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValueError, match="Memory must have an embedding"):
            await store.upsert_memory(memory)

    async def test_batch_upsert_interface(self):
        """Test batch_upsert method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[0.1, 0.2, 0.3],
                user_id="user-1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]

        await store.batch_upsert(memories)
        assert len(store.memories) == 5

    async def test_search_similar_interface(self):
        """Test search_similar method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory)

        results = await store.search_similar([0.1, 0.2, 0.3], limit=10)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].memory.id == "test-1"

    async def test_search_similar_with_filters(self):
        """Test search_similar with filters."""
        from datetime import datetime

        store = MockVectorStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            status=MemoryStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            status=MemoryStatus.HISTORICAL,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory1)
        await store.upsert_memory(memory2)

        results = await store.search_similar(
            [0.1, 0.2, 0.3], filters={"status": "active"}, limit=10
        )
        assert len(results) == 1
        assert results[0].memory.id == "test-1"

    async def test_search_similar_with_score_threshold(self):
        """Test search_similar with score threshold."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory)

        # Mock returns 0.8, threshold 0.9 should filter it out
        results = await store.search_similar([0.1, 0.2, 0.3], score_threshold=0.9, limit=10)
        assert len(results) == 0

    async def test_search_by_payload_interface(self):
        """Test search_by_payload method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            status=MemoryStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory)

        results = await store.search_by_payload({"status": "active"}, limit=10)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    async def test_get_memory_interface(self):
        """Test get_memory method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory)

        result = await store.get_memory("test-1")
        assert result is not None
        assert result.id == "test-1"

        result = await store.get_memory("nonexistent")
        assert result is None

    async def test_delete_memory_interface(self):
        """Test delete_memory method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.upsert_memory(memory)

        await store.delete_memory("test-1")
        assert "test-1" not in store.memories

        with pytest.raises(ValueError, match="not found"):
            await store.delete_memory("nonexistent")

    async def test_count_memories_interface(self):
        """Test count_memories method interface."""
        from datetime import datetime

        store = MockVectorStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[0.1, 0.2, 0.3],
                user_id="user-1",
                status=MemoryStatus.ACTIVE if i % 2 == 0 else MemoryStatus.HISTORICAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]

        for memory in memories:
            await store.upsert_memory(memory)

        total = await store.count_memories()
        assert total == 5

        active_count = await store.count_memories({"status": "active"})
        assert active_count == 3  # 0, 2, 4 are active

    async def test_close_interface(self):
        """Test close method interface."""
        store = MockVectorStore()
        await store.close()  # Should not raise
