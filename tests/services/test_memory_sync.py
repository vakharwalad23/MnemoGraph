"""
Tests for MemorySyncManager service.

Tests synchronization between graph and vector stores with retry logic.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.models.memory import MemoryStatus
from src.services.memory_sync import MemorySyncManager, SyncError


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMemorySyncManager:
    """Integration tests for MemorySyncManager with real stores."""

    @pytest.fixture
    def sync_manager(self, neo4j_graph_store, mock_vector_store):
        """Create sync manager with real stores."""
        return MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

    async def test_initialization(self, neo4j_graph_store, mock_vector_store):
        """Test sync manager initialization."""
        manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=5,
            retry_delay=1.0,
        )

        assert manager.graph_store == neo4j_graph_store
        assert manager.vector_store == mock_vector_store
        assert manager.max_retries == 5
        assert manager.retry_delay == 1.0

    async def test_sync_memory_full_success(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test successful full memory sync."""
        # Add to graph store first
        await neo4j_graph_store.add_node(sample_memory)

        # Sync to vector store
        await sync_manager.sync_memory_full(sample_memory)

        # Verify in vector store
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.content == sample_memory.content
        assert retrieved.status == sample_memory.status

    async def test_sync_memory_full_with_retry(
        self, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test successful sync after retry."""
        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Create sync manager with failing then succeeding vector store
        call_count = 0

        original_upsert = mock_vector_store.upsert_memory

        async def failing_upsert(memory):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return await original_upsert(memory)

        mock_vector_store.upsert_memory = failing_upsert

        sync_manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

        # Should succeed after retries
        await sync_manager.sync_memory_full(sample_memory)

        # Verify synced
        mock_vector_store.upsert_memory = original_upsert
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert call_count == 3

    async def test_sync_memory_full_max_retries_exceeded(
        self, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test sync failure after max retries."""
        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Create sync manager with always failing vector store
        async def always_fail(memory):
            raise Exception("Persistent error")

        mock_vector_store.upsert_memory = always_fail

        sync_manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

        with pytest.raises(SyncError) as exc_info:
            await sync_manager.sync_memory_full(sample_memory)

        assert "failed after 3 attempts" in str(exc_info.value)

    async def test_sync_memory_status(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test sync_memory_status delegates to sync_memory_full."""
        sample_memory.status = MemoryStatus.HISTORICAL

        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Sync status
        await sync_manager.sync_memory_status(sample_memory)

        # Verify synced
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert retrieved.status == MemoryStatus.HISTORICAL

    async def test_sync_memory_invalidation(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test sync_memory_invalidation delegates to sync_memory_full."""
        sample_memory.status = MemoryStatus.INVALIDATED
        sample_memory.invalidation_reason = "Test invalidation"
        sample_memory.valid_until = datetime.now()

        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Sync invalidation
        await sync_manager.sync_memory_invalidation(sample_memory)

        # Verify synced
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert retrieved.status == MemoryStatus.INVALIDATED
        assert retrieved.invalidation_reason == "Test invalidation"

    async def test_sync_memory_supersession(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test sync_memory_supersession delegates to sync_memory_full."""
        sample_memory.status = MemoryStatus.SUPERSEDED
        sample_memory.superseded_by = "mem_test_2"
        sample_memory.valid_until = datetime.now()

        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Sync supersession
        await sync_manager.sync_memory_supersession(sample_memory)

        # Verify synced
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is not None
        assert retrieved.status == MemoryStatus.SUPERSEDED
        assert retrieved.superseded_by == "mem_test_2"

    async def test_sync_memory_deletion_success(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test successful memory deletion."""
        # Add to both stores
        await neo4j_graph_store.add_node(sample_memory)
        await mock_vector_store.upsert_memory(sample_memory)

        # Delete from graph
        await neo4j_graph_store.delete_node(sample_memory.id)

        # Sync deletion
        await sync_manager.sync_memory_deletion(sample_memory.id)

        # Verify deleted from vector store
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is None

    async def test_sync_memory_deletion_with_retry(
        self, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test memory deletion with retry."""
        # Add to both stores
        await neo4j_graph_store.add_node(sample_memory)
        await mock_vector_store.upsert_memory(sample_memory)

        # Delete from graph
        await neo4j_graph_store.delete_node(sample_memory.id)

        # Create sync manager with failing then succeeding delete
        call_count = 0
        original_delete = mock_vector_store.delete_memory

        async def failing_delete(memory_id):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Delete failed")
            return await original_delete(memory_id)

        mock_vector_store.delete_memory = failing_delete

        sync_manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

        await sync_manager.sync_memory_deletion(sample_memory.id)

        # Verify deleted
        mock_vector_store.delete_memory = original_delete
        retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert retrieved is None
        assert call_count == 2

    async def test_sync_batch_memories_all_success(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memories
    ):
        """Test batch sync with all successes."""
        # Add all to graph store
        for memory in sample_memories:
            await neo4j_graph_store.add_node(memory)

        # Batch sync to vector store
        result = await sync_manager.sync_batch_memories(sample_memories)

        assert result["success"] == 5
        assert result["failed"] == 0
        assert len(result["errors"]) == 0

        # Verify all in vector store
        for memory in sample_memories:
            retrieved = await mock_vector_store.get_memory(memory.id)
            assert retrieved is not None

    async def test_sync_batch_memories_partial_failure(
        self, neo4j_graph_store, mock_vector_store, sample_memories
    ):
        """Test batch sync with some failures."""
        # Add all to graph store
        for memory in sample_memories:
            await neo4j_graph_store.add_node(memory)

        # Create sync manager with failing upsert for second memory
        call_count = 0
        original_upsert = mock_vector_store.upsert_memory

        async def selective_fail(memory):
            nonlocal call_count
            call_count += 1
            # Fail for second memory (mem_test_2)
            if memory.id == "mem_test_2":
                raise Exception("Sync failed")
            return await original_upsert(memory)

        mock_vector_store.upsert_memory = selective_fail

        sync_manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=2,  # Lower for faster test
            retry_delay=0.05,
        )

        result = await sync_manager.sync_batch_memories(sample_memories)

        assert result["success"] == 4
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["memory_id"] == "mem_test_2"

    async def test_exponential_backoff(self, neo4j_graph_store, mock_vector_store, sample_memory):
        """Test that retry delay increases exponentially."""
        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Create sync manager with always failing vector store
        async def always_fail(memory):
            raise Exception("Always fail")

        mock_vector_store.upsert_memory = always_fail

        sync_manager = MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(SyncError):
                await sync_manager.sync_memory_full(sample_memory)

            # Check exponential backoff: 0.1, 0.2 (no sleep after last attempt)
            assert mock_sleep.call_count == 2
            calls = mock_sleep.call_args_list
            assert calls[0][0][0] == 0.1  # First retry
            assert calls[1][0][0] == 0.2  # Second retry


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMemorySyncManagerValidation:
    """Tests for sync validation and repair functionality."""

    @pytest.fixture
    def sync_manager(self, neo4j_graph_store, mock_vector_store):
        """Create sync manager with real stores."""
        return MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

    async def test_validate_sync_consistency_in_sync(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test validation when stores are in sync."""
        # Add to both stores
        await neo4j_graph_store.add_node(sample_memory)
        await mock_vector_store.upsert_memory(sample_memory)

        # Validate
        result = await sync_manager.validate_sync_consistency(sample_memory.id)

        assert result["in_sync"] is True
        assert len(result["mismatches"]) == 0

    async def test_validate_sync_consistency_out_of_sync(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test validation when stores are out of sync."""
        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Add to vector store with different status
        modified_memory = sample_memory.model_copy()
        modified_memory.status = MemoryStatus.HISTORICAL
        await mock_vector_store.upsert_memory(modified_memory)

        # Validate
        result = await sync_manager.validate_sync_consistency(sample_memory.id)

        assert result["in_sync"] is False
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["field"] == "status"
        assert result["mismatches"][0]["graph"] == "active"
        assert result["mismatches"][0]["vector"] == "historical"

    async def test_validate_sync_consistency_missing_in_vector(
        self, sync_manager, neo4j_graph_store, sample_memory
    ):
        """Test validation when memory missing in vector store."""
        # Add only to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Validate
        result = await sync_manager.validate_sync_consistency(sample_memory.id)

        assert result["in_sync"] is False
        assert "not found in vector store" in result["error"]

    async def test_validate_sync_consistency_missing_in_graph(
        self, sync_manager, mock_vector_store, sample_memory
    ):
        """Test validation when memory missing in graph store."""
        # Add only to vector store
        await mock_vector_store.upsert_memory(sample_memory)

        # Validate
        result = await sync_manager.validate_sync_consistency(sample_memory.id)

        assert result["in_sync"] is False
        assert "not found in graph store" in result["error"]

    async def test_repair_sync_success(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test successful sync repair."""
        # Create a clean copy for graph store (ACTIVE status)
        graph_memory = sample_memory.model_copy(deep=True)
        graph_memory.status = MemoryStatus.ACTIVE
        await neo4j_graph_store.add_node(graph_memory)

        # Add to vector store with different status (HISTORICAL)
        vector_memory = sample_memory.model_copy(deep=True)
        vector_memory.status = MemoryStatus.HISTORICAL
        await mock_vector_store.upsert_memory(vector_memory)

        # Repair (should sync from graph to vector)
        success = await sync_manager.repair_sync(sample_memory.id)

        assert success is True

        # Verify repaired - should now match graph store (ACTIVE)
        repaired_memory = await mock_vector_store.get_memory(sample_memory.id)
        assert repaired_memory.status == MemoryStatus.ACTIVE  # From graph store

    async def test_repair_sync_missing_in_graph(
        self, sync_manager, mock_vector_store, sample_memory
    ):
        """Test repair when memory missing in graph store."""
        # Add only to vector store
        await mock_vector_store.upsert_memory(sample_memory)

        # Try to repair
        success = await sync_manager.repair_sync(sample_memory.id)

        assert success is False

    async def test_validate_and_repair_batch(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memories
    ):
        """Test batch validation and repair."""
        # Add all to graph store with ACTIVE status (creating clean copies)
        for memory in sample_memories:
            graph_memory = memory.model_copy(deep=True)
            graph_memory.status = MemoryStatus.ACTIVE
            await neo4j_graph_store.add_node(graph_memory)

        # Add first 3 to vector store correctly
        for memory in sample_memories[:3]:
            vector_memory = memory.model_copy(deep=True)
            vector_memory.status = MemoryStatus.ACTIVE
            await mock_vector_store.upsert_memory(vector_memory)

        # Add 4th with wrong status (HISTORICAL instead of ACTIVE)
        fourth_memory = sample_memories[3].model_copy(deep=True)
        fourth_memory.status = MemoryStatus.HISTORICAL
        await mock_vector_store.upsert_memory(fourth_memory)

        # Add 5th with wrong status too (so it can be repaired - needs embedding!)
        fifth_memory = sample_memories[4].model_copy(deep=True)
        fifth_memory.status = MemoryStatus.HISTORICAL
        await mock_vector_store.upsert_memory(fifth_memory)

        # Validate and repair all
        memory_ids = [m.id for m in sample_memories]
        result = await sync_manager.validate_and_repair_batch(memory_ids)

        assert result["validated"] == 5
        assert result["in_sync"] == 3  # First 3 are in sync
        assert result["out_of_sync"] == 2  # 4th and 5th have wrong status
        assert result["repaired"] == 2  # Both repaired
        assert result["failed"] == 0

        # Verify repairs - all should now have ACTIVE status (from graph store)
        for memory in sample_memories:
            vector_memory = await mock_vector_store.get_memory(memory.id)
            assert vector_memory is not None
            assert vector_memory.status == MemoryStatus.ACTIVE  # All match graph now


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMemorySyncManagerMemoryStates:
    """Tests for syncing memories in different states."""

    @pytest.fixture
    def sync_manager(self, neo4j_graph_store, mock_vector_store):
        """Create sync manager with real stores."""
        return MemorySyncManager(
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            max_retries=3,
            retry_delay=0.1,
        )

    async def test_sync_memory_with_supersession(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test syncing memory with supersession state."""
        # Create superseded memory
        sample_memory.status = MemoryStatus.SUPERSEDED
        sample_memory.superseded_by = "mem_test_2"
        sample_memory.valid_until = datetime.now()
        sample_memory.invalidation_reason = "Superseded by newer version"

        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Sync to vector store
        await sync_manager.sync_memory_supersession(sample_memory)

        # Verify in vector store
        vector_memory = await mock_vector_store.get_memory(sample_memory.id)
        assert vector_memory.status == MemoryStatus.SUPERSEDED
        assert vector_memory.superseded_by == "mem_test_2"
        assert vector_memory.valid_until is not None

    async def test_sync_memory_with_invalidation(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memory
    ):
        """Test syncing memory with invalidation state."""
        # Create invalidated memory
        sample_memory.status = MemoryStatus.INVALIDATED
        sample_memory.invalidation_reason = "No longer relevant"
        sample_memory.valid_until = datetime.now()

        # Add to graph store
        await neo4j_graph_store.add_node(sample_memory)

        # Sync to vector store
        await sync_manager.sync_memory_invalidation(sample_memory)

        # Verify in vector store
        vector_memory = await mock_vector_store.get_memory(sample_memory.id)
        assert vector_memory.status == MemoryStatus.INVALIDATED
        assert vector_memory.invalidation_reason == "No longer relevant"
        assert vector_memory.valid_until is not None

    async def test_sync_batch_with_different_statuses(
        self, sync_manager, neo4j_graph_store, mock_vector_store, sample_memories
    ):
        """Test batch sync with memories in different states."""
        # Set different statuses
        sample_memories[0].status = MemoryStatus.ACTIVE
        sample_memories[1].status = MemoryStatus.HISTORICAL
        sample_memories[2].status = MemoryStatus.SUPERSEDED
        sample_memories[2].superseded_by = "mem_test_99"
        sample_memories[3].status = MemoryStatus.INVALIDATED
        sample_memories[3].invalidation_reason = "Test invalidation"
        sample_memories[4].status = MemoryStatus.ACTIVE

        # Add all to graph store
        for memory in sample_memories:
            await neo4j_graph_store.add_node(memory)

        # Batch sync to vector store
        result = await sync_manager.sync_batch_memories(sample_memories)

        assert result["success"] == 5
        assert result["failed"] == 0

        # Verify all in vector store with correct status
        for memory in sample_memories:
            vector_memory = await mock_vector_store.get_memory(memory.id)
            assert vector_memory is not None
            assert vector_memory.status == memory.status
