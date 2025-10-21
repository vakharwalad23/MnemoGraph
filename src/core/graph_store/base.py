"""Base interface for graph storage."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.models import Node, Edge, NodeType, RelationshipType, MemoryStatus


class GraphStore(ABC):
    """Abstract base class for graph storage implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the graph store (create tables/schema)."""
        pass
    
    @abstractmethod
    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        data: Dict[str, Any],
        status: MemoryStatus = MemoryStatus.NEW
    ) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of the node
            data: Node data as dictionary
            status: Memory status
        """
        pass
    
    @abstractmethod
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: RelationshipType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight
            metadata: Additional edge metadata
            
        Returns:
            Edge ID
        """
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node or None if not found
        """
        pass
    
    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[RelationshipType]] = None,
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Node identifier
            edge_types: Filter by specific edge types
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of neighbor nodes with edge information
        """
        pass
    
    @abstractmethod
    async def update_node_status(
        self,
        node_id: str,
        status: MemoryStatus
    ) -> None:
        """
        Update node status.
        
        Args:
            node_id: Node identifier
            status: New status
        """
        pass
    
    @abstractmethod
    async def update_node_access(
        self,
        node_id: str,
        access_count: int,
        last_accessed: datetime
    ) -> None:
        """
        Update node access tracking.
        
        Args:
            node_id: Node identifier
            access_count: Updated access count
            last_accessed: Last access timestamp
        """
        pass
    
    @abstractmethod
    async def update_edge_weight(
        self,
        edge_id: str,
        weight: float
    ) -> None:
        """
        Update edge weight.
        
        Args:
            edge_id: Edge identifier
            weight: New weight
        """
        pass
    
    @abstractmethod
    async def mark_forgotten(self, node_id: str) -> None:
        """
        Mark a node as forgotten.
        
        Args:
            node_id: Node identifier
        """
        pass
    
    @abstractmethod
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """
        Find path between two nodes.
        
        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path depth
            
        Returns:
            List of node IDs forming the path, or None
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
    async def close(self) -> None:
        """Close the connection to the graph store."""
        pass