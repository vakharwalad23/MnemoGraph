"""
Base interface for graph storage - redesigned for Phase 2 & 3.

Clean interface that works with new Memory model and relationship system.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.models.memory import Memory
from src.models.relationships import Edge


class GraphStore(ABC):
    """Abstract base class for graph storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the graph store (create tables/schema)."""
        pass

    # ═══════════════════════════════════════════════════════════
    # NODE OPERATIONS (Memory nodes)
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    async def add_node(self, memory: Memory) -> None:
        """
        Add a memory node to the graph.

        Args:
            memory: Memory object to store
        """
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Memory | None:
        """
        Retrieve a memory node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Memory or None if not found
        """
        pass

    @abstractmethod
    async def update_node(self, memory: Memory) -> None:
        """
        Update an existing memory node.

        Args:
            memory: Updated memory object
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node and its edges.

        Args:
            node_id: Node identifier
        """
        pass

    @abstractmethod
    async def query_memories(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Query memories with filters.

        Args:
            filters: Filter conditions (e.g., {"status": "active", "created_after": "2024-01-01"})
            order_by: Sort order (e.g., "created_at DESC")
            limit: Maximum results

        Returns:
            List of matching memories
        """
        pass

    @abstractmethod
    async def get_random_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[Memory]:
        """
        Get random memories for sampling.

        Args:
            filters: Optional filters
            limit: Number of memories

        Returns:
            List of random memories
        """
        pass

    # ═══════════════════════════════════════════════════════════
    # EDGE OPERATIONS (Relationships)
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    async def add_edge(self, edge: dict[str, Any] | Edge) -> str:
        """
        Add an edge between two nodes.

        Args:
            edge: Edge dict or Edge object with source, target, type, metadata

        Returns:
            Edge ID
        """
        pass

    @abstractmethod
    async def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        """
        Get edge by ID.

        Args:
            edge_id: Edge identifier

        Returns:
            Edge dict or None
        """
        pass

    @abstractmethod
    async def get_edge_between(
        self, source_id: str, target_id: str, relationship_type: str | None = None
    ) -> dict[str, Any] | None:
        """
        Find edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Optional filter by type

        Returns:
            Edge dict or None
        """
        pass

    @abstractmethod
    async def update_edge(self, edge: dict[str, Any]) -> None:
        """
        Update edge metadata.

        Args:
            edge: Edge dict with id and updated fields
        """
        pass

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> None:
        """
        Delete an edge.

        Args:
            edge_id: Edge identifier
        """
        pass

    # ═══════════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = 100,
    ) -> list[tuple[Memory, Edge]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node identifier
            relationship_types: Filter by specific types
            direction: "outgoing", "incoming", or "both"
            depth: Traversal depth
            limit: Maximum results

        Returns:
            List of tuples with (Memory, Edge)
        """
        pass

    @abstractmethod
    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[str] | None:
        """
        Find shortest path between two nodes.

        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path depth

        Returns:
            List of node IDs forming the path, or None
        """
        pass

    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    async def count_nodes(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count nodes matching filters.

        Args:
            filters: Optional filter conditions

        Returns:
            Count of nodes
        """
        pass

    @abstractmethod
    async def count_edges(self, relationship_type: str | None = None) -> int:
        """
        Count edges.

        Args:
            relationship_type: Optional filter by type

        Returns:
            Count of edges
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the graph store."""
        pass
