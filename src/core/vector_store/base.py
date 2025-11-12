"""
Base interface for vector storage

Clean interface that works with new Memory model and embedding system.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from src.models.memory import Memory


class SearchResult(BaseModel):
    """Vector search result."""

    memory: Memory
    score: float
    metadata: dict[str, Any] = {}


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the vector store (create collections/indices).

        Raises:
            VectorStoreError: If initialization fails
        """
        pass

    @abstractmethod
    async def upsert_memory(self, memory: Memory) -> None:
        """
        Store or update a memory with its embedding.

        Args:
            memory: Memory object with embedding

        Raises:
            ValidationError: If memory is invalid
            VectorStoreError: If upsert operation fails
        """
        pass

    @abstractmethod
    async def batch_upsert(self, memories: list[Memory]) -> None:
        """
        Batch upsert multiple memories for efficiency.

        Args:
            memories: List of Memory objects
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar memories by vector.

        Args:
            vector: Query embedding vector
            limit: Maximum results
            filters: Optional payload filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results with memories and scores
        """
        pass

    @abstractmethod
    async def search_by_payload(
        self,
        filter: dict[str, Any],
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search by payload/metadata filters.

        Args:
            filter: Payload filter conditions
            limit: Maximum results

        Returns:
            List of matching memories
        """
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory or None if not found

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from the store.

        Args:
            memory_id: Memory identifier

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If deletion operation fails
        """
        pass

    @abstractmethod
    async def count_memories(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count memories matching filters.

        Args:
            filters: Optional filter conditions

        Returns:
            Number of memories
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the vector store."""
        pass
