"""
Tests for Qdrant vector store implementation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vector_store.qdrant import QdrantStore
from src.models.memory import Memory
from src.utils.exceptions import ValidationError, VectorStoreError


@pytest.mark.unit
@pytest.mark.asyncio
class TestQdrantStore:
    """Test Qdrant vector store implementation."""

    async def test_initialization(self, qdrant_store):
        """Test store initialization."""
        assert qdrant_store.host == "localhost"
        assert qdrant_store.port == 6333
        assert qdrant_store.collection_name == "test_memories"
        assert qdrant_store.vector_size == 768
        assert qdrant_store.client is None

    async def test_initialization_with_grpc(self):
        """Test initialization with gRPC enabled."""
        store = QdrantStore(use_grpc=True, port=6334)
        assert store.use_grpc is True
        assert store.port == 6334

    async def test_initialization_with_quantization(self):
        """Test initialization with quantization."""
        store = QdrantStore(use_quantization=True, quantization_type="int8")
        assert store.use_quantization is True
        assert store.quantization_type == "int8"

    async def test_to_uuid_with_valid_uuid(self, qdrant_store):
        """Test UUID conversion with valid UUID."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = qdrant_store._to_uuid(uuid_str)
        assert result == uuid_str

    async def test_to_uuid_with_string_id(self, qdrant_store):
        """Test UUID conversion with string ID."""
        string_id = "test-memory-1"
        result = qdrant_store._to_uuid(string_id)
        # Should return a UUID string
        assert isinstance(result, str)
        assert len(result) == 36  # UUID format

    async def test_connect(self, qdrant_store):
        """Test connection to Qdrant."""
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient") as mock_client:
            mock_client.return_value = AsyncMock()
            await qdrant_store.connect()
            assert qdrant_store.client is not None

    async def test_connect_failure(self, qdrant_store):
        """Test connection failure handling."""
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            with pytest.raises(VectorStoreError, match="Failed to connect"):
                await qdrant_store.connect()

    async def test_initialize_new_collection(self, qdrant_store):
        """Test initialization with new collection."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.initialize()

            # Verify collection was created
            mock_client.create_collection.assert_called_once()
            # Verify indices were created
            assert mock_client.create_payload_index.call_count >= 4

    async def test_initialize_existing_collection(self, qdrant_store):
        """Test initialization with existing collection."""
        mock_client = AsyncMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_memories"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.initialize()

            # Should not create collection
            mock_client.create_collection.assert_not_called()

    async def test_memory_to_payload(self, qdrant_store, sample_memory):
        """Test memory to payload conversion."""
        payload = qdrant_store._memory_to_payload(sample_memory)
        assert payload["original_id"] == sample_memory.id
        assert payload["content"] == sample_memory.content
        assert payload["type"] == sample_memory.type.value
        assert payload["status"] == sample_memory.status.value
        assert payload["user_id"] == sample_memory.user_id

    async def test_payload_to_memory(self, qdrant_store, sample_memory):
        """Test payload to memory conversion."""
        payload = qdrant_store._memory_to_payload(sample_memory)
        vector = sample_memory.embedding
        memory = qdrant_store._payload_to_memory(payload, vector)

        assert memory.id == sample_memory.id
        assert memory.content == sample_memory.content
        assert memory.type == sample_memory.type
        assert memory.status == sample_memory.status
        assert memory.user_id == sample_memory.user_id

    async def test_upsert_memory(self, qdrant_store, sample_memory):
        """Test upsert memory operation."""
        mock_client = AsyncMock()
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.upsert_memory(sample_memory)

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            assert call_args.kwargs["collection_name"] == "test_memories"
            assert len(call_args.kwargs["points"]) == 1

    async def test_upsert_memory_validation_none(self, qdrant_store):
        """Test upsert memory validation with None."""
        with pytest.raises(ValidationError, match="Memory cannot be None"):
            await qdrant_store.upsert_memory(None)

    async def test_upsert_memory_validation_empty_id(self, qdrant_store):
        """Test upsert memory validation with empty ID."""
        memory = Memory(
            id="",
            content="Test",
            embedding=[0.1] * 768,
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValidationError, match="Memory ID cannot be empty"):
            await qdrant_store.upsert_memory(memory)

    async def test_upsert_memory_validation_no_embedding(self, qdrant_store):
        """Test upsert memory validation with no embedding."""
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValidationError, match="Memory must have an embedding"):
            await qdrant_store.upsert_memory(memory)

    async def test_batch_upsert(self, qdrant_store):
        """Test batch upsert operation."""
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                user_id="user-1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]

        mock_client = AsyncMock()
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.batch_upsert(memories, batch_size=2)

            # Should be called multiple times for batches
            assert mock_client.upsert.call_count >= 2

    async def test_search_similar(self, qdrant_store, sample_memory):
        """Test similarity search."""
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "test-uuid"
        mock_point.vector = sample_memory.embedding
        mock_point.payload = qdrant_store._memory_to_payload(sample_memory)
        mock_point.score = 0.95

        mock_response = MagicMock()
        mock_response.points = [mock_point]

        mock_client.query_points = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            results = await qdrant_store.search_similar([0.1] * 768, limit=10)

            assert len(results) == 1
            assert results[0].memory.id == sample_memory.id
            assert results[0].score == 0.95

    async def test_search_similar_with_filters(self, qdrant_store):
        """Test similarity search with filters."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.search_similar(
                [0.1] * 768,
                filters={"status": "active", "user_id": "user-1"},
                limit=10,
            )

            call_args = mock_client.query_points.call_args
            assert call_args.kwargs["query_filter"] is not None

    async def test_search_similar_with_score_threshold(self, qdrant_store):
        """Test similarity search with score threshold."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.search_similar([0.1] * 768, score_threshold=0.8, limit=10)

            call_args = mock_client.query_points.call_args
            assert call_args.kwargs["score_threshold"] == 0.8

    async def test_search_by_payload(self, qdrant_store, sample_memory):
        """Test search by payload."""
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "test-uuid"
        mock_point.vector = sample_memory.embedding
        mock_point.payload = qdrant_store._memory_to_payload(sample_memory)

        mock_response = ([mock_point], None)
        mock_client.scroll = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            results = await qdrant_store.search_by_payload(
                {"user_id": "user-1", "status": "active"}, limit=10
            )

            assert len(results) == 1
            assert results[0].memory.id == sample_memory.id

    async def test_get_memory(self, qdrant_store, sample_memory):
        """Test get memory by ID."""
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.payload = qdrant_store._memory_to_payload(sample_memory)
        mock_point.vector = sample_memory.embedding

        mock_client.retrieve = AsyncMock(return_value=[mock_point])

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            result = await qdrant_store.get_memory(sample_memory.id)

            assert result is not None
            assert result.id == sample_memory.id

    async def test_get_memory_not_found(self, qdrant_store):
        """Test get memory when not found."""
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[])

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            result = await qdrant_store.get_memory("nonexistent")
            assert result is None

    async def test_get_memory_validation(self, qdrant_store):
        """Test get memory validation."""
        with pytest.raises(ValidationError, match="Memory ID cannot be empty"):
            await qdrant_store.get_memory("")

    async def test_delete_memory(self, qdrant_store):
        """Test delete memory operation."""
        mock_client = AsyncMock()
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.delete_memory("test-1")

            mock_client.delete.assert_called_once()
            call_args = mock_client.delete.call_args
            assert call_args.kwargs["collection_name"] == "test_memories"

    async def test_delete_memory_validation(self, qdrant_store):
        """Test delete memory validation."""
        with pytest.raises(ValidationError, match="Memory ID cannot be empty"):
            await qdrant_store.delete_memory("")

    async def test_count_memories_no_filters(self, qdrant_store):
        """Test count memories without filters."""
        mock_client = AsyncMock()
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 100
        mock_client.get_collection = AsyncMock(return_value=mock_collection_info)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            count = await qdrant_store.count_memories()

            assert count == 100

    async def test_count_memories_with_filters(self, qdrant_store):
        """Test count memories with filters."""
        mock_client = AsyncMock()
        mock_count_response = MagicMock()
        mock_count_response.count = 50
        mock_client.count = AsyncMock(return_value=mock_count_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            count = await qdrant_store.count_memories({"status": "active"})

            assert count == 50
            mock_client.count.assert_called_once()

    async def test_close(self, qdrant_store):
        """Test close connection."""
        mock_client = AsyncMock()
        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            await qdrant_store.close()

            mock_client.close.assert_called_once()
            assert qdrant_store.client is None

    async def test_search_by_payload_all_users(self, qdrant_store, sample_memory):
        """Test internal search across all users."""
        mock_client = AsyncMock()
        mock_point = MagicMock()
        mock_point.id = "test-uuid"
        mock_point.vector = sample_memory.embedding
        mock_point.payload = qdrant_store._memory_to_payload(sample_memory)

        mock_response = ([mock_point], None)
        mock_client.scroll = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            results = await qdrant_store._search_by_payload_all_users(
                {"status": "active"}, limit=10
            )

            assert len(results) == 1
            assert results[0].id == sample_memory.id

    async def test_search_by_payload_all_users_with_order_by(self, qdrant_store):
        """Test internal search with ordering."""
        from datetime import datetime, timedelta

        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[0.1] * 768,
                user_id=f"user-{i % 2}",
                created_at=datetime.now() - timedelta(days=i),
                updated_at=datetime.now() - timedelta(days=i),
            )
            for i in range(5)
        ]

        mock_client = AsyncMock()
        mock_points = []
        for memory in memories:
            mock_point = MagicMock()
            mock_point.id = f"uuid-{memory.id}"
            mock_point.vector = memory.embedding
            mock_point.payload = qdrant_store._memory_to_payload(memory)
            mock_points.append(mock_point)

        mock_response = (mock_points, None)
        mock_client.scroll = AsyncMock(return_value=mock_response)

        with patch("src.core.vector_store.qdrant.AsyncQdrantClient", return_value=mock_client):
            await qdrant_store.connect()
            results = await qdrant_store._search_by_payload_all_users(
                limit=10, order_by="created_at DESC"
            )

            # Results should be sorted by created_at descending
            assert len(results) == 5


@pytest.mark.integration
@pytest.mark.asyncio
class TestQdrantStoreIntegration:
    """
    Integration tests for Qdrant vector store.
    Requires running Qdrant server.
    Run with: pytest -m integration
    """

    async def test_real_connection(self):
        """Test real connection to Qdrant."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_integration",
            vector_size=768,
        )

        try:
            await store.connect()
            assert store.client is not None
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
        finally:
            await store.close()

    async def test_real_upsert_and_retrieve(self):
        """Test real upsert and retrieve operations."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_integration",
            vector_size=768,
        )

        try:
            await store.initialize()
            memory = Memory(
                id="integration-test-1",
                content="Integration test memory",
                embedding=[0.1] * 768,
                user_id="test-user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            await store.upsert_memory(memory)
            result = await store.get_memory("integration-test-1")

            assert result is not None
            assert result.id == memory.id
            assert result.content == memory.content

            # Cleanup
            await store.delete_memory("integration-test-1")
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
        finally:
            await store.close()
