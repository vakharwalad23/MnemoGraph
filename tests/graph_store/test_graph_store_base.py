"""
Tests for graph store base class.
"""

import pytest

from src.core.graph_store.base import GraphStore
from src.models.memory import Memory, MemoryStatus
from src.models.relationships import Edge, RelationshipType


class MockGraphStore(GraphStore):
    """Mock graph store for testing."""

    def __init__(self):
        """Initialize mock store."""
        self.nodes: dict[str, Memory] = {}
        self.edges: dict[str, dict] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """Mock initialize implementation."""
        self.initialized = True

    async def add_node(self, memory: Memory) -> None:
        """Mock add node implementation."""
        if not memory:
            raise ValueError("Memory cannot be None")
        if not memory.id:
            raise ValueError("Memory ID cannot be empty")
        if not memory.user_id:
            raise ValueError("Memory must have user_id")
        self.nodes[memory.id] = memory

    async def get_node(self, node_id: str, user_id: str) -> Memory | None:
        """Mock get node implementation."""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        if node.user_id != user_id:
            return None
        return node

    async def update_node(self, memory: Memory) -> None:
        """Mock update node implementation."""
        await self.add_node(memory)

    async def delete_node(self, node_id: str, user_id: str) -> None:
        """Mock delete node implementation."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.user_id != user_id:
            raise ValueError("User ID mismatch")
        del self.nodes[node_id]
        # Remove edges
        edges_to_remove = [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge["source"] == node_id or edge["target"] == node_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]

    async def query_memories(
        self,
        user_id: str,
        filters: dict | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Mock query memories implementation."""
        results = []
        for node in self.nodes.values():
            if node.user_id != user_id:
                continue

            if filters:
                if "status" in filters and node.status.value != filters["status"]:
                    continue
                if "type" in filters and node.type.value != filters["type"]:
                    continue

            results.append(node)

        if order_by:
            # Simple sorting
            reverse = "DESC" in order_by.upper()
            field = order_by.replace("DESC", "").replace("ASC", "").strip()
            if hasattr(Memory, field):
                results.sort(key=lambda m: getattr(m, field, ""), reverse=reverse)

        return results[:limit]

    async def get_random_memories(
        self, user_id: str, filters: dict | None = None, limit: int = 10
    ) -> list[Memory]:
        """Mock get random memories implementation."""
        all_memories = await self.query_memories(user_id, filters, limit=1000)
        import random

        return random.sample(all_memories, min(limit, len(all_memories)))

    async def add_edge(self, edge: dict | Edge, user_id: str) -> str:
        """Mock add edge implementation."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required")

        if isinstance(edge, Edge):
            source_id = edge.source
            target_id = edge.target
            edge_type = edge.type.value if isinstance(edge.type, RelationshipType) else edge.type
        else:
            source_id = edge["source"]
            target_id = edge["target"]
            edge_type = edge["type"]

        # Verify both nodes exist and belong to user
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found")

        if self.nodes[source_id].user_id != user_id or self.nodes[target_id].user_id != user_id:
            raise ValueError("User ID mismatch")

        edge_id = f"edge-{len(self.edges)}"
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "user_id": user_id,
        }
        return edge_id

    async def get_edge(self, edge_id: str) -> dict | None:
        """Mock get edge implementation."""
        return self.edges.get(edge_id)

    async def get_edge_between(
        self, source_id: str, target_id: str, user_id: str, relationship_type: str | None = None
    ) -> dict | None:
        """Mock get edge between implementation."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required")
        for edge in self.edges.values():
            if edge["source"] == source_id and edge["target"] == target_id:
                if edge["user_id"] != user_id:
                    continue
                if relationship_type and edge["type"] != relationship_type:
                    continue
                return edge
        return None

    async def update_edge(self, edge: dict) -> None:
        """Mock update edge implementation."""
        edge_id = edge["id"]
        if edge_id in self.edges:
            self.edges[edge_id].update(edge)

    async def delete_edge(self, edge_id: str) -> None:
        """Mock delete edge implementation."""
        if edge_id in self.edges:
            del self.edges[edge_id]

    async def get_neighbors(
        self,
        node_id: str,
        user_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = 100,
    ) -> list[tuple[Memory, Edge]]:
        """Mock get neighbors implementation."""
        neighbors = []
        for edge in self.edges.values():
            if direction == "outgoing" and edge["source"] == node_id:
                if relationship_types and edge["type"] not in relationship_types:
                    continue
                target = self.nodes.get(edge["target"])
                if target and target.user_id == user_id:
                    edge_obj = Edge(
                        source=edge["source"],
                        target=edge["target"],
                        type=RelationshipType(edge["type"]),
                    )
                    neighbors.append((target, edge_obj))
            elif direction == "incoming" and edge["target"] == node_id:
                if relationship_types and edge["type"] not in relationship_types:
                    continue
                source = self.nodes.get(edge["source"])
                if source and source.user_id == user_id:
                    edge_obj = Edge(
                        source=edge["source"],
                        target=edge["target"],
                        type=RelationshipType(edge["type"]),
                    )
                    neighbors.append((source, edge_obj))

        return neighbors[:limit]

    async def find_path(
        self, start_id: str, end_id: str, user_id: str, max_depth: int = 5
    ) -> list[str] | None:
        """Mock find path implementation."""
        # Simple BFS path finding
        from collections import deque

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current, path = queue.popleft()

            if current == end_id:
                return path

            if len(path) >= max_depth:
                continue

            for edge in self.edges.values():
                if edge["source"] == current and edge["target"] not in visited:
                    target = self.nodes.get(edge["target"])
                    if target and target.user_id == user_id:
                        visited.add(edge["target"])
                        queue.append((edge["target"], path + [edge["target"]]))

        return None

    async def count_nodes(self, user_id: str, filters: dict | None = None) -> int:
        """Mock count nodes implementation."""
        count = 0
        for node in self.nodes.values():
            if node.user_id != user_id:
                continue

            if filters:
                if "status" in filters and node.status.value != filters["status"]:
                    continue

            count += 1

        return count

    async def count_edges(self, user_id: str, relationship_type: str | None = None) -> int:
        """Mock count edges implementation."""
        count = 0
        for edge in self.edges.values():
            if edge["user_id"] != user_id:
                continue
            if relationship_type and edge["type"] != relationship_type:
                continue
            count += 1
        return count

    async def close(self) -> None:
        """Mock close implementation."""
        self.nodes.clear()
        self.edges.clear()


@pytest.mark.unit
@pytest.mark.asyncio
class TestGraphStoreBase:
    """Test base GraphStore functionality."""

    async def test_abstract_instantiation(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GraphStore()

    async def test_initialize_interface(self):
        """Test initialize method interface."""
        store = MockGraphStore()
        await store.initialize()
        assert store.initialized is True

    async def test_add_node_interface(self):
        """Test add_node method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await store.add_node(memory)
        assert "test-1" in store.nodes

    async def test_add_node_validation(self):
        """Test add_node validation."""
        store = MockGraphStore()

        # Test None memory
        with pytest.raises(ValueError, match="Memory cannot be None"):
            await store.add_node(None)

        # Test empty ID
        from datetime import datetime

        memory = Memory(
            id="",
            content="Test",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValueError, match="Memory ID cannot be empty"):
            await store.add_node(memory)

        # Test missing user_id
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[],
            user_id="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with pytest.raises(ValueError, match="Memory must have user_id"):
            await store.add_node(memory)

    async def test_get_node_interface(self):
        """Test get_node method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory = Memory(
            id="test-1",
            content="Test content",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory)

        result = await store.get_node("test-1", "user-1")
        assert result is not None
        assert result.id == "test-1"

        result = await store.get_node("nonexistent", "user-1")
        assert result is None

        # Test user isolation
        result = await store.get_node("test-1", "user-2")
        assert result is None

    async def test_update_node_interface(self):
        """Test update_node method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory = Memory(
            id="test-1",
            content="Original",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory)

        memory.content = "Updated"
        await store.update_node(memory)

        updated = await store.get_node("test-1", "user-1")
        assert updated.content == "Updated"

    async def test_delete_node_interface(self):
        """Test delete_node method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory = Memory(
            id="test-1",
            content="Test",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory)

        await store.delete_node("test-1", "user-1")
        assert "test-1" not in store.nodes

    async def test_query_memories_interface(self):
        """Test query_memories method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[],
                user_id="user-1",
                status=MemoryStatus.ACTIVE if i % 2 == 0 else MemoryStatus.HISTORICAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]

        for memory in memories:
            await store.add_node(memory)

        results = await store.query_memories("user-1")
        assert len(results) == 5

        results = await store.query_memories("user-1", filters={"status": "active"})
        assert len(results) == 3

    async def test_get_random_memories_interface(self):
        """Test get_random_memories method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[],
                user_id="user-1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(10)
        ]

        for memory in memories:
            await store.add_node(memory)

        results = await store.get_random_memories("user-1", limit=5)
        assert len(results) == 5

    async def test_add_edge_interface(self):
        """Test add_edge method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge = Edge(
            source="test-1",
            target="test-2",
            type=RelationshipType.REFERENCES,
        )
        edge_id = await store.add_edge(edge, "user-1")
        assert edge_id is not None

    async def test_add_edge_user_isolation(self):
        """Test add_edge enforces user isolation."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge = Edge(
            source="test-1",
            target="test-2",
            type=RelationshipType.REFERENCES,
        )

        with pytest.raises(ValueError, match="User ID mismatch"):
            await store.add_edge(edge, "user-1")

    async def test_get_edge_interface(self):
        """Test get_edge method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge = Edge(
            source="test-1",
            target="test-2",
            type=RelationshipType.REFERENCES,
        )
        edge_id = await store.add_edge(edge, "user-1")

        result = await store.get_edge(edge_id)
        assert result is not None
        assert result["source"] == "test-1"

    async def test_get_edge_between_interface(self):
        """Test get_edge_between method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge = Edge(
            source="test-1",
            target="test-2",
            type=RelationshipType.REFERENCES,
        )
        await store.add_edge(edge, "user-1")

        result = await store.get_edge_between("test-1", "test-2", "user-1")
        assert result is not None

    async def test_get_edge_between_validation(self):
        """Test get_edge_between validation."""
        store = MockGraphStore()
        with pytest.raises(ValueError, match="user_id is required"):
            await store.get_edge_between("test-1", "test-2", "")

    async def test_get_neighbors_interface(self):
        """Test get_neighbors method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge = Edge(
            source="test-1",
            target="test-2",
            type=RelationshipType.REFERENCES,
        )
        await store.add_edge(edge, "user-1")

        neighbors = await store.get_neighbors("test-1", "user-1")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == "test-2"

    async def test_find_path_interface(self):
        """Test find_path method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Test {i}",
                embedding=[],
                user_id="user-1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(3)
        ]

        for memory in memories:
            await store.add_node(memory)

        # Create path: test-0 -> test-1 -> test-2
        edge1 = Edge(source="test-0", target="test-1", type=RelationshipType.REFERENCES)
        edge2 = Edge(source="test-1", target="test-2", type=RelationshipType.REFERENCES)
        await store.add_edge(edge1, "user-1")
        await store.add_edge(edge2, "user-1")

        path = await store.find_path("test-0", "test-2", "user-1")
        assert path is not None
        assert len(path) == 3
        assert path[0] == "test-0"
        assert path[-1] == "test-2"

    async def test_count_nodes_interface(self):
        """Test count_nodes method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                embedding=[],
                user_id="user-1",
                status=MemoryStatus.ACTIVE if i % 2 == 0 else MemoryStatus.HISTORICAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]

        for memory in memories:
            await store.add_node(memory)

        total = await store.count_nodes("user-1")
        assert total == 5

        active_count = await store.count_nodes("user-1", filters={"status": "active"})
        assert active_count == 3

    async def test_count_edges_interface(self):
        """Test count_edges method interface."""
        from datetime import datetime

        store = MockGraphStore()
        memory1 = Memory(
            id="test-1",
            content="Test 1",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        memory2 = Memory(
            id="test-2",
            content="Test 2",
            embedding=[],
            user_id="user-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await store.add_node(memory1)
        await store.add_node(memory2)

        edge1 = Edge(source="test-1", target="test-2", type=RelationshipType.REFERENCES)
        edge2 = Edge(source="test-2", target="test-1", type=RelationshipType.SIMILAR_TO)
        await store.add_edge(edge1, "user-1")
        await store.add_edge(edge2, "user-1")

        total = await store.count_edges("user-1")
        assert total == 2

        ref_count = await store.count_edges("user-1", relationship_type="REFERENCES")
        assert ref_count == 1

    async def test_close_interface(self):
        """Test close method interface."""
        store = MockGraphStore()
        await store.close()  # Should not raise
