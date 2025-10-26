"""
Tests for InvalidationManager service.

Tests memory invalidation, validation, and supersession checking.
"""

from datetime import datetime, timedelta

import pytest

from src.models.memory import Memory, MemoryStatus, NodeType
from src.services.invalidation_manager import InvalidationManager


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvalidationManagerUnit:
    """Unit tests for InvalidationManager."""

    async def test_initialization(self, mock_llm, sqlite_graph_store, mock_vector_store):
        """Test manager initialization."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        assert manager.llm == mock_llm
        assert manager.graph_store == sqlite_graph_store
        assert manager.vector_store == mock_vector_store

    async def test_should_validate_never_validated(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memory
    ):
        """Test validation needed when never validated."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Memory without last_validated
        assert manager._should_validate(sample_memory) is True

    async def test_should_validate_recent_memory(
        self, mock_llm, sqlite_graph_store, mock_vector_store
    ):
        """Test validation logic for recent memories."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Recent memory (< 30 days old)
        recent_mem = Memory(
            id="recent_mem",
            content="Recent memory",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
            created_at=datetime.now() - timedelta(days=10),
            updated_at=datetime.now(),
        )
        recent_mem.metadata["last_validated"] = (datetime.now() - timedelta(days=50)).isoformat()

        # Should not validate (validated 50 days ago, needs 90 days)
        assert manager._should_validate(recent_mem) is False

        # Update last validation to 100 days ago
        recent_mem.metadata["last_validated"] = (datetime.now() - timedelta(days=100)).isoformat()
        assert manager._should_validate(recent_mem) is True

    async def test_should_validate_old_memory(
        self, mock_llm, sqlite_graph_store, mock_vector_store
    ):
        """Test validation logic for old memories."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Old memory (> 180 days)
        old_mem = Memory(
            id="old_mem",
            content="Old memory",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
            created_at=datetime.now() - timedelta(days=200),
            updated_at=datetime.now(),
        )
        old_mem.metadata["last_validated"] = (datetime.now() - timedelta(days=20)).isoformat()

        # Should validate (validated 20 days ago, needs 14 days for old memories)
        assert manager._should_validate(old_mem) is True

    async def test_calculate_similarity(self, mock_llm, sqlite_graph_store, mock_vector_store):
        """Test cosine similarity calculation."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert manager._calculate_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        assert manager._calculate_similarity(vec3, vec4) == pytest.approx(0.0)

        # Opposite vectors
        vec5 = [1.0, 0.0, 0.0]
        vec6 = [-1.0, 0.0, 0.0]
        assert manager._calculate_similarity(vec5, vec6) == pytest.approx(-1.0)

    async def test_check_invalidation(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memory
    ):
        """Test invalidation checking."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Check invalidation
        result = await manager.check_invalidation(sample_memory)

        assert result is not None
        assert hasattr(result, "memory_id")
        assert hasattr(result, "status")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "confidence")
        assert result.memory_id == sample_memory.id


@pytest.mark.unit
@pytest.mark.sqlite
@pytest.mark.asyncio
class TestInvalidationManagerSQLite:
    """Integration tests with SQLite."""

    async def test_validate_on_access_active(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memory
    ):
        """Test validating active memory on access."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Add memory to store
        await sqlite_graph_store.add_node(sample_memory)

        # Save original count before validation
        original_count = sample_memory.access_count

        # Validate on access
        result = await manager.validate_on_access(sample_memory)

        assert result.id == sample_memory.id
        assert result.access_count == original_count + 1

    async def test_validate_on_access_already_invalidated(
        self, mock_llm, sqlite_graph_store, mock_vector_store
    ):
        """Test validating already invalidated memory."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Create invalidated memory
        invalid_mem = Memory(
            id="invalid_mem",
            content="Invalidated",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
            status=MemoryStatus.INVALIDATED,
        )

        await sqlite_graph_store.add_node(invalid_mem)

        # Should return without checking
        result = await manager.validate_on_access(invalid_mem)

        assert result.status == MemoryStatus.INVALIDATED
        assert result.access_count == 1  # Still incremented

    async def test_mark_invalidated(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memory
    ):
        """Test marking memory as invalidated."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        await sqlite_graph_store.add_node(sample_memory)

        # Create invalidation result
        from src.models.version import InvalidationResult

        result = InvalidationResult(
            memory_id=sample_memory.id,
            status="invalidated",
            reasoning="Test invalidation",
            confidence=0.9,
        )

        # Mark as invalidated
        updated = await manager._mark_invalidated(sample_memory, result)

        assert updated.status == MemoryStatus.INVALIDATED
        assert updated.invalidation_reason == "Test invalidation"
        assert updated.valid_until is not None

    async def test_check_supersession(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memories
    ):
        """Test supersession checking."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Add memories
        for mem in sample_memories:
            await sqlite_graph_store.add_node(mem)

        # New memory
        new_mem = Memory(
            id="new_mem",
            content="Updated information about programming",
            type=NodeType.MEMORY,
            embedding=[0.8] * 768,
        )

        # Check supersession
        superseded = await manager.check_supersession(new_mem, sample_memories)

        assert isinstance(superseded, list)
        # With real LLM, we get analysis results
        # (No call_count check since we're using real Ollama)

    async def test_get_memory_context(
        self, mock_llm, sqlite_graph_store, mock_vector_store, sample_memory
    ):
        """Test getting memory context."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        await sqlite_graph_store.add_node(sample_memory)

        # Get context
        context = await manager._get_memory_context(sample_memory)

        assert isinstance(context, str)
        assert len(context) > 0


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestInvalidationManagerNeo4j:
    """Integration tests with Neo4j."""

    async def test_validate_on_access_neo4j(
        self, mock_llm, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test validation with Neo4j store."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
        )

        await neo4j_graph_store.add_node(sample_memory)

        # Validate
        result = await manager.validate_on_access(sample_memory)

        assert result is not None
        assert result.id == sample_memory.id

    async def test_check_supersession_neo4j(
        self, mock_llm, neo4j_graph_store, mock_vector_store, sample_memories
    ):
        """Test supersession checking with Neo4j."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
        )

        # Add memories
        for mem in sample_memories:
            await neo4j_graph_store.add_node(mem)

        new_mem = Memory(
            id="new_mem",
            content="Updated information",
            type=NodeType.MEMORY,
            embedding=[0.9] * 768,
        )

        # Check supersession
        superseded = await manager.check_supersession(new_mem, sample_memories[:2])

        assert isinstance(superseded, list)


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.asyncio
class TestInvalidationManagerBackgroundWorker:
    """Tests for background worker functionality."""

    async def test_start_stop_worker(self, mock_llm, sqlite_graph_store, mock_vector_store):
        """Test starting and stopping background worker."""
        manager = InvalidationManager(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
        )

        # Start worker
        manager.start_background_worker(interval_hours=24)

        assert manager._worker_task is not None
        assert not manager._worker_task.done()

        # Stop worker
        manager.stop_background_worker()

        # Wait a bit for cancellation
        import asyncio

        await asyncio.sleep(0.1)

        assert manager._worker_task.cancelled() or manager._worker_task.done()
