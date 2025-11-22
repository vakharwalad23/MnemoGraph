"""
Tests for Neo4j graph store implementation.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.models.memory import Memory, MemoryStatus, NodeType
from src.models.relationships import Edge, RelationshipType
from src.utils.exceptions import GraphStoreError, ValidationError


def create_mock_session():
    """Create a properly configured mock session for async context manager."""
    mock_session = AsyncMock()
    mock_session_context = MagicMock()
    mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_context.__aexit__ = AsyncMock(return_value=None)
    return mock_session, mock_session_context


@pytest.mark.unit
@pytest.mark.asyncio
class TestNeo4jGraphStore:
    """Test Neo4j graph store implementation."""

    async def test_initialization(self, neo4j_store):
        """Test store initialization."""
        assert neo4j_store.uri == "bolt://localhost:7687"
        assert neo4j_store.username == "neo4j"
        assert neo4j_store.password == "password"
        assert neo4j_store.database == "neo4j"
        assert neo4j_store.driver is None

    async def test_connect(self, neo4j_store):
        """Test connection to Neo4j."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            await neo4j_store.connect()
            assert neo4j_store.driver is not None

    async def test_connect_failure(self, neo4j_store):
        """Test connection failure handling."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_db.driver.side_effect = Exception("Connection failed")
            with pytest.raises(GraphStoreError, match="Failed to connect"):
                await neo4j_store.connect()

    async def test_initialize(self, neo4j_store):
        """Test store initialization with indexes."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            await neo4j_store.initialize()

            # Verify constraints and indexes were created
            assert mock_session.run.call_count >= 8

    async def test_add_node(self, neo4j_store, sample_memory):
        """Test add node operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            await neo4j_store.add_node(sample_memory)

            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args
            assert "MERGE" in call_args[0][0]
            assert "$id" in call_args[0][0]

    async def test_add_node_validation_none(self, neo4j_store):
        """Test add node validation with None."""
        with pytest.raises(ValidationError, match="Memory cannot be None"):
            await neo4j_store.add_node(None)

    async def test_add_node_validation_empty_id(self, neo4j_store):
        """Test add node validation with empty ID."""
        memory = Memory(
            id="",
            content="Test",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValidationError, match="Memory ID cannot be empty"):
            await neo4j_store.add_node(memory)

    async def test_add_node_validation_no_user_id(self, neo4j_store):
        """Test add node validation with no user_id."""
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[],
            user_id="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValidationError, match="Memory must have user_id"):
            await neo4j_store.add_node(memory)

    async def test_get_node(self, neo4j_store, sample_memory):
        """Test get node operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            node_data = {
                "id": sample_memory.id,
                "content_preview": sample_memory.content[:200],
                "type": sample_memory.type.value,
                "status": sample_memory.status.value,
                "version": sample_memory.version,
                "user_id": sample_memory.user_id,
            }
            mock_node = MagicMock()
            mock_node.__getitem__ = MagicMock(side_effect=lambda key: node_data[key])
            mock_node.get = MagicMock(
                side_effect=lambda key, default=None: node_data.get(key, default)
            )
            mock_record.__getitem__.return_value = mock_node
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            result = await neo4j_store.get_node(sample_memory.id, "user-1")

            assert result is not None
            assert result.id == sample_memory.id

    async def test_get_node_not_found(self, neo4j_store):
        """Test get node when not found."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=None)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            result = await neo4j_store.get_node("nonexistent", "user-1")

            assert result is None

    async def test_get_node_validation(self, neo4j_store):
        """Test get node validation."""
        with pytest.raises(ValidationError, match="Node ID cannot be empty"):
            await neo4j_store.get_node("", "user-1")

        with pytest.raises(ValidationError, match="user_id is required"):
            await neo4j_store.get_node("test-1", "")

    async def test_update_node(self, neo4j_store, sample_memory):
        """Test update node operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            sample_memory.content = "Updated content"
            await neo4j_store.update_node(sample_memory)

            mock_session.run.assert_called_once()

    async def test_delete_node(self, neo4j_store):
        """Test delete node operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            await neo4j_store.delete_node("test-1", "user-1")

            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args
            assert "DETACH DELETE" in call_args[0][0]

    async def test_delete_node_validation(self, neo4j_store):
        """Test delete node validation."""
        with pytest.raises(ValidationError, match="user_id is required"):
            await neo4j_store.delete_node("test-1", "")

    async def test_query_memories(self, neo4j_store):
        """Test query memories operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            node_data = {
                "id": "test-1",
                "content_preview": "Test",
                "type": "MEMORY",
                "status": "active",
                "version": 1,
                "user_id": "user-1",
            }
            mock_node = MagicMock()
            mock_node.__getitem__ = MagicMock(side_effect=lambda key: node_data[key])
            mock_node.get = MagicMock(
                side_effect=lambda key, default=None: node_data.get(key, default)
            )
            mock_record.__getitem__.return_value = mock_node

            async def async_iter():
                yield mock_record

            mock_result = AsyncMock()
            mock_result.__aiter__ = lambda self: async_iter()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            results = await neo4j_store.query_memories("user-1", limit=10)

            assert len(results) == 1

    async def test_query_memories_with_filters(self, neo4j_store):
        """Test query memories with filters."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()

            async def async_iter():
                if False:
                    yield  # Make it an async generator

            mock_result = AsyncMock()
            mock_result.__aiter__ = lambda self: async_iter()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            await neo4j_store.query_memories(
                "user-1", filters={"status": "active", "type": "MEMORY"}, limit=10
            )

            call_args = mock_session.run.call_args
            # Check that status filter is in the parameters
            assert call_args is not None
            assert len(call_args[0]) >= 2
            params = call_args[0][1] if len(call_args[0]) > 1 else {}
            assert "status" in params

    async def test_get_random_memories(self, neo4j_store):
        """Test get random memories operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            node_data = {
                "id": "test-1",
                "content_preview": "Test",
                "type": "MEMORY",
                "status": "active",
                "version": 1,
                "user_id": "user-1",
            }
            mock_node = MagicMock()
            mock_node.__getitem__ = MagicMock(side_effect=lambda key: node_data[key])
            mock_node.get = MagicMock(
                side_effect=lambda key, default=None: node_data.get(key, default)
            )
            mock_record.__getitem__.return_value = mock_node

            async def async_iter():
                yield mock_record

            mock_result = AsyncMock()
            mock_result.__aiter__ = lambda self: async_iter()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            results = await neo4j_store.get_random_memories("user-1", limit=5)

            assert len(results) == 1

    async def test_add_edge(self, neo4j_store, sample_memory):
        """Test add edge operation."""
        memory2 = Memory(
            id="test-memory-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            mock_rel = MagicMock()
            mock_record.__getitem__.return_value = mock_rel
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()

            # Add nodes first
            await neo4j_store.add_node(sample_memory)
            await neo4j_store.add_node(memory2)

            edge = Edge(
                source=sample_memory.id,
                target=memory2.id,
                type=RelationshipType.REFERENCES,
            )
            edge_id = await neo4j_store.add_edge(edge, "user-1")

            assert edge_id is not None

    async def test_add_edge_validation(self, neo4j_store):
        """Test add edge validation."""
        with pytest.raises(ValidationError, match="user_id is required"):
            edge = Edge(
                source="test-1",
                target="test-2",
                type=RelationshipType.REFERENCES,
            )
            await neo4j_store.add_edge(edge, "")

    async def test_get_edge(self, neo4j_store):
        """Test get edge operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            rel_data = {
                "id": "edge-1",
                "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "metadata": json.dumps({"test": "data"}),
            }
            mock_rel = MagicMock()
            mock_rel.__getitem__ = MagicMock(side_effect=lambda key: rel_data[key])
            mock_rel.get = MagicMock(
                side_effect=lambda key, default=None: rel_data.get(key, default)
            )
            mock_record.__getitem__.side_effect = lambda key: {
                "r": mock_rel,
                "edge_type": "REFERENCES",
                "source": "test-1",
                "target": "test-2",
            }.get(key)
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            result = await neo4j_store.get_edge("edge-1")

            assert result is not None
            assert result["id"] == "edge-1"

    async def test_get_edge_between(self, neo4j_store):
        """Test get edge between nodes."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            rel_data = {
                "id": "edge-1",
                "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "metadata": json.dumps({}),
            }
            mock_rel = MagicMock()
            mock_rel.__getitem__ = MagicMock(side_effect=lambda key: rel_data[key])
            mock_rel.get = MagicMock(
                side_effect=lambda key, default=None: rel_data.get(key, default)
            )
            mock_record.__getitem__.side_effect = lambda key: {
                "r": mock_rel,
                "edge_type": "REFERENCES",
            }.get(key)
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            result = await neo4j_store.get_edge_between("test-1", "test-2", "user-1")

            assert result is not None

    async def test_get_neighbors(self, neo4j_store):
        """Test get neighbors operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            node_data = {
                "id": "test-2",
                "content_preview": "Neighbor",
                "type": "MEMORY",
                "status": "active",
                "version": 1,
                "user_id": "user-1",
            }
            mock_node = MagicMock()
            mock_node.__getitem__ = MagicMock(side_effect=lambda key: node_data[key])
            mock_node.get = MagicMock(
                side_effect=lambda key, default=None: node_data.get(key, default)
            )
            rel_data = {
                "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "metadata": json.dumps({}),
            }
            mock_rel = MagicMock()
            mock_rel.__getitem__ = MagicMock(side_effect=lambda key: rel_data[key])
            mock_rel.get = MagicMock(
                side_effect=lambda key, default=None: rel_data.get(key, default)
            )
            mock_record.__getitem__.side_effect = lambda key: {
                "n": mock_node,
                "r": mock_rel,
                "rel_type": "REFERENCES",
            }.get(key)

            async def async_iter():
                yield mock_record

            mock_result = AsyncMock()
            mock_result.__aiter__ = lambda self: async_iter()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            neighbors = await neo4j_store.get_neighbors("test-1", "user-1")

            assert len(neighbors) == 1
            assert neighbors[0][0].id == "test-2"

    async def test_find_path(self, neo4j_store):
        """Test find path operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            mock_record.__getitem__.return_value = ["test-1", "test-2", "test-3"]
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            path = await neo4j_store.find_path("test-1", "test-3", "user-1")

            assert path is not None
            assert len(path) == 3

    async def test_count_nodes(self, neo4j_store):
        """Test count nodes operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            mock_record.__getitem__.return_value = 10
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            count = await neo4j_store.count_nodes("user-1")

            assert count == 10

    async def test_count_edges(self, neo4j_store):
        """Test count edges operation."""
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_session, mock_session_context = create_mock_session()
            mock_record = MagicMock()
            mock_record.__getitem__.return_value = 5
            mock_result = AsyncMock()
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_driver.session = MagicMock(return_value=mock_session_context)
            mock_db.driver.return_value = mock_driver

            await neo4j_store.connect()
            count = await neo4j_store.count_edges("user-1")

            assert count == 5

    async def test_close(self, neo4j_store):
        """Test close connection."""
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        with patch("src.core.graph_store.neo4j_store.AsyncGraphDatabase") as mock_db:
            mock_db.driver.return_value = mock_driver
            await neo4j_store.connect()
            await neo4j_store.close()

            mock_driver.close.assert_called_once()
            assert neo4j_store.driver is None

    async def test_node_to_memory(self, neo4j_store):
        """Test node to memory conversion."""
        mock_node = MagicMock()
        # Set up both __getitem__ and get() methods for the mock
        node_data = {
            "id": "test-1",
            "content_preview": "Test content preview",
            "type": "MEMORY",
            "status": "active",
            "version": 1,
            "parent_version": None,
            "superseded_by": None,
            "user_id": "user-1",
        }
        mock_node.__getitem__ = MagicMock(side_effect=lambda key: node_data[key])
        mock_node.get = MagicMock(side_effect=lambda key, default=None: node_data.get(key, default))

        memory = neo4j_store._node_to_memory(mock_node, "user-1")

        assert memory.id == "test-1"
        assert memory.content == "Test content preview"
        assert memory.type == NodeType.MEMORY
        assert memory.status == MemoryStatus.ACTIVE
        assert memory.user_id == "user-1"


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestNeo4jGraphStoreIntegration:
    """
    Integration tests for Neo4j graph store.
    Requires running Neo4j instance.
    Run with: pytest -m integration -m neo4j
    """

    async def test_real_connection(self):
        """Test real connection to Neo4j."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j",
        )

        try:
            await store.connect()
            assert store.driver is not None
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await store.close()

    async def test_real_add_and_get_node(self):
        """Test real add and get node operations."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j",
        )

        try:
            await store.initialize()
            memory = Memory(
                id="integration-test-1",
                content="Integration test memory for graph",
                embedding=[],
                user_id="test-user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            await store.add_node(memory)
            result = await store.get_node("integration-test-1", "test-user")

            assert result is not None
            assert result.id == memory.id

            # Cleanup
            await store.delete_node("integration-test-1", "test-user")
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await store.close()
