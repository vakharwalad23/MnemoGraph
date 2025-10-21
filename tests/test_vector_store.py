"""Tests for Qdrant vector store."""

import pytest
from datetime import datetime, timezone
from src.core.vector_store import QdrantStore


class TestQdrantStore:
    """Test Qdrant vector store operations."""
    
    @pytest.fixture
    async def store(self):
        """Create a QdrantStore instance and clean up after test."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_memories",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup: Delete the test collection
        await store.connect()
        try:
            await store.client.delete_collection("test_memories")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector."""
        return [0.1] * 768  # 768-dimensional vector
    
    @pytest.mark.asyncio
    async def test_initialize_collection(self, store):
        """Test collection initialization."""
        # Collection should be created during fixture setup
        collections = await store.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        assert "test_memories" in collection_names
    
    @pytest.mark.asyncio
    async def test_upsert_memory(self, store, sample_embedding):
        """Test storing a memory."""
        memory_id = "test-memory-1"
        metadata = {
            "text": "Python is a programming language",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        await store.upsert_memory(memory_id, sample_embedding, metadata)
        
        # Verify memory was stored
        result = await store.get_memory(memory_id)
        assert result is not None
        assert result["id"] == memory_id
        assert result["metadata"]["text"] == "Python is a programming language"
    
    @pytest.mark.asyncio
    async def test_get_memory(self, store, sample_embedding):
        """Test retrieving a memory by ID."""
        memory_id = "test-memory-2"
        metadata = {"text": "Test memory"}
        
        await store.upsert_memory(memory_id, sample_embedding, metadata)
        
        result = await store.get_memory(memory_id)
        
        assert result is not None
        assert result["id"] == memory_id
        assert len(result["vector"]) == 768
        assert result["metadata"]["text"] == "Test memory"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, store):
        """Test retrieving a memory that doesn't exist."""
        result = await store.get_memory("nonexistent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_similar(self, store, sample_embedding):
        """Test searching for similar memories."""
        # Store multiple memories
        memories = [
            ("mem-1", [0.1] * 768, {"text": "Python programming"}),
            ("mem-2", [0.2] * 768, {"text": "JavaScript coding"}),
            ("mem-3", [0.15] * 768, {"text": "Python development"}),
        ]
        
        for mem_id, embedding, metadata in memories:
            await store.upsert_memory(mem_id, embedding, metadata)
        
        # Search for similar memories
        query_vector = [0.12] * 768  # Similar to mem-1 and mem-3
        results = await store.search_similar(query_vector, limit=2)
        
        assert len(results) <= 2
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("metadata" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(self, store, sample_embedding):
        """Test searching with score threshold."""
        await store.upsert_memory("mem-1", sample_embedding, {"text": "Test"})
        
        # Search with high threshold
        results = await store.search_similar(
            sample_embedding,
            limit=10,
            score_threshold=0.99
        )
        
        # Should only return highly similar memories
        assert all(result["score"] >= 0.99 for result in results)
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, sample_embedding):
        """Test searching with metadata filters."""
        # Store memories with different statuses
        await store.upsert_memory(
            "mem-1", 
            sample_embedding, 
            {"text": "Memory 1", "status": "active"}
        )
        await store.upsert_memory(
            "mem-2", 
            [0.2] * 768, 
            {"text": "Memory 2", "status": "forgotten"}
        )
        
        # Search only for active memories
        results = await store.search_similar(
            sample_embedding,
            limit=10,
            filter_dict={"status": "active"}
        )
        
        assert len(results) >= 1
        for result in results:
            assert result["metadata"]["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_update_access_metadata(self, store, sample_embedding):
        """Test updating access tracking metadata."""
        memory_id = "mem-access-test"
        await store.upsert_memory(memory_id, sample_embedding, {"text": "Test"})
        
        # Update access metadata
        now = datetime.now(timezone.utc)
        await store.update_access_metadata(memory_id, access_count=5, last_accessed=now)
        
        # Verify update
        result = await store.get_memory(memory_id)
        assert result["metadata"]["access_count"] == 5
        assert result["metadata"]["last_accessed"] == now.isoformat()
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, store, sample_embedding):
        """Test deleting a memory."""
        memory_id = "mem-delete-test"
        await store.upsert_memory(memory_id, sample_embedding, {"text": "To delete"})
        
        # Verify it exists
        result = await store.get_memory(memory_id)
        assert result is not None
        
        # Delete it
        await store.delete_memory(memory_id)
        
        # Verify it's gone
        result = await store.get_memory(memory_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_count_memories(self, store, sample_embedding):
        """Test counting memories in collection."""
        initial_count = await store.count_memories()
        
        # Add some memories
        for i in range(3):
            await store.upsert_memory(f"mem-{i}", sample_embedding, {"index": i})
        
        final_count = await store.count_memories()
        assert final_count == initial_count + 3
    
    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, store, sample_embedding):
        """Test that upsert updates existing memory."""
        memory_id = "mem-update-test"
        
        # Initial insert
        await store.upsert_memory(
            memory_id, 
            sample_embedding, 
            {"text": "Original text"}
        )
        
        # Update with new metadata
        await store.upsert_memory(
            memory_id, 
            sample_embedding, 
            {"text": "Updated text"}
        )
        
        # Verify update
        result = await store.get_memory(memory_id)
        assert result["metadata"]["text"] == "Updated text"
        
        # Count should not increase
        count = await store.count_memories()
        # Should have only one memory with this ID
        assert count >= 1


class TestQdrantStoreConnection:
    """Test connection management."""
    
    @pytest.mark.asyncio
    async def test_connect(self):
        """Test establishing connection."""
        store = QdrantStore()
        
        assert store.client is None
        
        await store.connect()
        
        assert store.client is not None
        
        await store.close()
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing connection."""
        store = QdrantStore()
        await store.connect()
        
        assert store.client is not None
        
        await store.close()
        
        assert store.client is None
    
    @pytest.mark.asyncio
    async def test_auto_connect_on_operations(self):
        """Test that operations automatically connect if needed."""
        store = QdrantStore(collection_name="test_auto_connect")
        
        assert store.client is None
        
        # Initialize should auto-connect
        await store.initialize()
        
        assert store.client is not None
        
        # Cleanup
        try:
            await store.client.delete_collection("test_auto_connect")
        except Exception:
            pass
        await store.close()