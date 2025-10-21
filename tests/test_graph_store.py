"""Tests for graph store implementations."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from src.core.graph_store import (
    GraphStore,
    SQLiteGraphStore,
    Neo4jGraphStore,
    create_graph_store,
)
from src.models import NodeType, RelationshipType, MemoryStatus


class TestSQLiteGraphStore:
    """Test SQLite graph store implementation."""
    
    @pytest.fixture
    async def store(self, tmp_path):
        """Create a SQLite graph store instance."""
        db_path = tmp_path / "test_graph.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()
    
    @pytest.mark.asyncio
    async def test_initialize(self, store):
        """Test database initialization."""
        # Should create tables without error
        await store.initialize()
        assert store.conn is not None
    
    @pytest.mark.asyncio
    async def test_add_node(self, store):
        """Test adding a node."""
        await store.add_node(
            node_id="node-1",
            node_type=NodeType.MEMORY,
            data={"text": "Test memory"},
            status=MemoryStatus.NEW
        )
        
        # Retrieve and verify
        node = await store.get_node("node-1")
        assert node is not None
        assert node.id == "node-1"
        assert node.type == NodeType.MEMORY
        assert node.data["text"] == "Test memory"
        assert node.status == MemoryStatus.NEW
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_node(self, store):
        """Test retrieving a node that doesn't exist."""
        node = await store.get_node("nonexistent")
        assert node is None
    
    @pytest.mark.asyncio
    async def test_add_edge(self, store):
        """Test adding an edge between nodes."""
        # Create two nodes
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        
        # Add edge
        edge_id = await store.add_edge(
            source_id="node-1",
            target_id="node-2",
            edge_type=RelationshipType.SIMILAR_TO,
            weight=0.85,
            metadata={"similarity": "high"}
        )
        
        assert edge_id is not None
    
    @pytest.mark.asyncio
    async def test_get_neighbors_outgoing(self, store):
        """Test getting outgoing neighbors."""
        # Create nodes
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        # Add edges
        await store.add_edge("node-1", "node-2", RelationshipType.SIMILAR_TO, 0.9)
        await store.add_edge("node-1", "node-3", RelationshipType.UPDATES, 0.8)
        
        # Get outgoing neighbors
        neighbors = await store.get_neighbors("node-1", direction="outgoing")
        
        assert len(neighbors) == 2
        neighbor_ids = [n["node"].id for n in neighbors]
        assert "node-2" in neighbor_ids
        assert "node-3" in neighbor_ids
    
    @pytest.mark.asyncio
    async def test_get_neighbors_incoming(self, store):
        """Test getting incoming neighbors."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        
        await store.add_edge("node-2", "node-1", RelationshipType.SIMILAR_TO, 0.9)
        
        # Get incoming neighbors
        neighbors = await store.get_neighbors("node-1", direction="incoming")
        
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "node-2"
    
    @pytest.mark.asyncio
    async def test_get_neighbors_filtered_by_type(self, store):
        """Test getting neighbors filtered by edge type."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        await store.add_edge("node-1", "node-2", RelationshipType.SIMILAR_TO, 0.9)
        await store.add_edge("node-1", "node-3", RelationshipType.UPDATES, 0.8)
        
        # Filter by SIMILAR_TO only
        neighbors = await store.get_neighbors(
            "node-1",
            edge_types=[RelationshipType.SIMILAR_TO],
            direction="outgoing"
        )
        
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "node-2"
        assert neighbors[0]["edge_type"] == RelationshipType.SIMILAR_TO
    
    @pytest.mark.asyncio
    async def test_update_node_status(self, store):
        """Test updating node status."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Test"})
        
        # Update status
        await store.update_node_status("node-1", MemoryStatus.ACTIVE)
        
        # Verify
        node = await store.get_node("node-1")
        assert node.status == MemoryStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_update_node_access(self, store):
        """Test updating node access tracking."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Test"})
        
        now = datetime.now(timezone.utc)
        await store.update_node_access("node-1", access_count=5, last_accessed=now)
        
        node = await store.get_node("node-1")
        assert node.access_count == 5
        assert node.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_update_edge_weight(self, store):
        """Test updating edge weight."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        
        edge_id = await store.add_edge(
            "node-1", "node-2", RelationshipType.SIMILAR_TO, 0.5
        )
        
        # Update weight
        await store.update_edge_weight(edge_id, 0.9)
        
        # Verify through neighbors
        neighbors = await store.get_neighbors("node-1", direction="outgoing")
        assert neighbors[0]["edge_weight"] == 0.9
    
    @pytest.mark.asyncio
    async def test_mark_forgotten(self, store):
        """Test marking a node as forgotten."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Test"})
        
        await store.mark_forgotten("node-1")
        
        node = await store.get_node("node-1")
        assert node.status == MemoryStatus.FORGOTTEN
    
    @pytest.mark.asyncio
    async def test_find_path(self, store):
        """Test finding path between nodes."""
        # Create a chain: node-1 -> node-2 -> node-3
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        await store.add_edge("node-1", "node-2", RelationshipType.FOLLOWS)
        await store.add_edge("node-2", "node-3", RelationshipType.FOLLOWS)
        
        # Find path
        path = await store.find_path("node-1", "node-3", max_depth=3)
        
        assert path is not None
        assert path == ["node-1", "node-2", "node-3"]
    
    @pytest.mark.asyncio
    async def test_find_path_no_connection(self, store):
        """Test finding path when no connection exists."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        
        # No edge between them
        path = await store.find_path("node-1", "node-2", max_depth=3)
        
        assert path is None
    
    @pytest.mark.asyncio
    async def test_delete_node(self, store):
        """Test deleting a node."""
        await store.add_node("node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_edge("node-1", "node-2", RelationshipType.SIMILAR_TO)
        
        # Delete node-1
        await store.delete_node("node-1")
        
        # Verify deletion
        node = await store.get_node("node-1")
        assert node is None
        
        # Verify node-2 still exists
        node2 = await store.get_node("node-2")
        assert node2 is not None


class TestNeo4jGraphStore:
    """Test Neo4j graph store implementation."""
    
    @pytest.fixture
    async def store(self):
        """Create a Neo4j graph store instance."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mnemograph123"
        )
        await store.initialize()
        yield store
        
        # Cleanup: Delete all test nodes
        driver = await store._get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n:Node) DETACH DELETE n")
        
        await store.close()
    
    @pytest.mark.asyncio
    async def test_initialize(self, store):
        """Test Neo4j initialization."""
        await store.initialize()
        assert store.driver is not None
    
    @pytest.mark.asyncio
    async def test_add_node(self, store):
        """Test adding a node."""
        await store.add_node(
            node_id="neo-node-1",
            node_type=NodeType.MEMORY,
            data={"text": "Test memory"},
            status=MemoryStatus.NEW
        )
        
        # Retrieve and verify
        node = await store.get_node("neo-node-1")
        assert node is not None
        assert node.id == "neo-node-1"
        assert node.type == NodeType.MEMORY
        assert node.data["text"] == "Test memory"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_node(self, store):
        """Test retrieving a node that doesn't exist."""
        node = await store.get_node("nonexistent")
        assert node is None
    
    @pytest.mark.asyncio
    async def test_add_edge(self, store):
        """Test adding an edge."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("neo-node-2", NodeType.MEMORY, {"text": "Node 2"})
        
        edge_id = await store.add_edge(
            source_id="neo-node-1",
            target_id="neo-node-2",
            edge_type=RelationshipType.SIMILAR_TO,
            weight=0.85
        )
        
        assert edge_id is not None
    
    @pytest.mark.asyncio
    async def test_get_neighbors_outgoing(self, store):
        """Test getting outgoing neighbors."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("neo-node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("neo-node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        await store.add_edge("neo-node-1", "neo-node-2", RelationshipType.SIMILAR_TO, 0.9)
        await store.add_edge("neo-node-1", "neo-node-3", RelationshipType.UPDATES, 0.8)
        
        neighbors = await store.get_neighbors("neo-node-1", direction="outgoing")
        
        assert len(neighbors) == 2
        neighbor_ids = [n["node"].id for n in neighbors]
        assert "neo-node-2" in neighbor_ids
        assert "neo-node-3" in neighbor_ids
    
    @pytest.mark.asyncio
    async def test_get_neighbors_filtered_by_type(self, store):
        """Test getting neighbors filtered by edge type."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("neo-node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("neo-node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        await store.add_edge("neo-node-1", "neo-node-2", RelationshipType.SIMILAR_TO, 0.9)
        await store.add_edge("neo-node-1", "neo-node-3", RelationshipType.UPDATES, 0.8)
        
        neighbors = await store.get_neighbors(
            "neo-node-1",
            edge_types=[RelationshipType.SIMILAR_TO],
            direction="outgoing"
        )
        
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "neo-node-2"
    
    @pytest.mark.asyncio
    async def test_update_node_status(self, store):
        """Test updating node status."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Test"})
        
        await store.update_node_status("neo-node-1", MemoryStatus.ACTIVE)
        
        node = await store.get_node("neo-node-1")
        assert node.status == MemoryStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_update_node_access(self, store):
        """Test updating node access tracking."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Test"})
        
        now = datetime.now(timezone.utc)
        await store.update_node_access("neo-node-1", access_count=5, last_accessed=now)
        
        node = await store.get_node("neo-node-1")
        assert node.access_count == 5
        assert node.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_mark_forgotten(self, store):
        """Test marking a node as forgotten."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Test"})
        
        await store.mark_forgotten("neo-node-1")
        
        node = await store.get_node("neo-node-1")
        assert node.status == MemoryStatus.FORGOTTEN
    
    @pytest.mark.asyncio
    async def test_find_path(self, store):
        """Test finding path between nodes."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("neo-node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_node("neo-node-3", NodeType.MEMORY, {"text": "Node 3"})
        
        await store.add_edge("neo-node-1", "neo-node-2", RelationshipType.FOLLOWS)
        await store.add_edge("neo-node-2", "neo-node-3", RelationshipType.FOLLOWS)
        
        path = await store.find_path("neo-node-1", "neo-node-3", max_depth=3)
        
        assert path is not None
        assert len(path) == 3
        assert path[0] == "neo-node-1"
        assert path[-1] == "neo-node-3"
    
    @pytest.mark.asyncio
    async def test_delete_node(self, store):
        """Test deleting a node."""
        await store.add_node("neo-node-1", NodeType.MEMORY, {"text": "Node 1"})
        await store.add_node("neo-node-2", NodeType.MEMORY, {"text": "Node 2"})
        await store.add_edge("neo-node-1", "neo-node-2", RelationshipType.SIMILAR_TO)
        
        await store.delete_node("neo-node-1")
        
        node = await store.get_node("neo-node-1")
        assert node is None


class TestGraphStoreFactory:
    """Test graph store factory."""
    
    def test_create_sqlite_store(self):
        """Test creating SQLite store via factory."""
        store = create_graph_store(backend="sqlite", db_path="test.db")
        
        assert isinstance(store, SQLiteGraphStore)
        assert store.db_path == "test.db"
    
    def test_create_neo4j_store(self):
        """Test creating Neo4j store via factory."""
        store = create_graph_store(
            backend="neo4j",
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test123"
        )
        
        assert isinstance(store, Neo4jGraphStore)
        assert store.uri == "bolt://localhost:7687"
        assert store.user == "neo4j"
    
    def test_create_sqlite_default(self):
        """Test factory uses default parameters."""
        store = create_graph_store(backend="sqlite")
        
        assert isinstance(store, SQLiteGraphStore)
        assert store.db_path == "mnemograph.db"
    
    def test_unknown_backend_raises_error(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_graph_store(backend="fake-backend")


class TestGraphStoreInterface:
    """Test that implementations follow the interface."""
    
    def test_sqlite_implements_interface(self):
        """Test that SQLiteGraphStore implements GraphStore."""
        store = SQLiteGraphStore()
        assert isinstance(store, GraphStore)
    
    def test_neo4j_implements_interface(self):
        """Test that Neo4jGraphStore implements GraphStore."""
        store = Neo4jGraphStore()
        assert isinstance(store, GraphStore)
    
    def test_has_required_methods(self):
        """Test that stores have all required methods."""
        required_methods = [
            "initialize",
            "add_node",
            "add_edge",
            "get_node",
            "get_neighbors",
            "update_node_status",
            "update_node_access",
            "update_edge_weight",
            "mark_forgotten",
            "find_path",
            "delete_node",
            "close",
        ]
        
        for store_class in [SQLiteGraphStore, Neo4jGraphStore]:
            store = store_class()
            for method in required_methods:
                assert hasattr(store, method)
                assert callable(getattr(store, method))