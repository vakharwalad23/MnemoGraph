"""
Unified Memory Store Facade - Single entry point for all memory operations.

This facade provides a clean abstraction over graph and vector stores,
ensuring consistent access patterns and proper synchronization.

Key responsibilities:
- Unified CRUD operations for memories
- Atomic access tracking
- Relationship management
- Search operations across both stores
- Automatic synchronization between stores
"""

import asyncio
from datetime import datetime
from typing import Any

from src.core.graph_store.base import GraphStore
from src.core.vector_store.base import VectorStore
from src.models.memory import Memory
from src.models.relationships import Edge
from src.utils.exceptions import (
    GraphStoreError,
    MemoryError,
    ValidationError,
    VectorStoreError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """
    Unified facade for all memory storage operations.

    This class ensures consistent access patterns and proper abstraction.
    Services should use this instead of calling stores directly.

    Architecture:
    - Vector store: Source of truth for all memory data and metadata
    - Graph store: Relationship management and graph queries
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
    ):
        """
        Initialize MemoryStore facade.

        Args:
            vector_store: Vector database for semantic search
            graph_store: Graph database for relationships
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        logger.info("MemoryStore facade initialized")

    async def get_memory(
        self,
        memory_id: str,
        track_access: bool = True,
        include_relationships: bool = False,
    ) -> Memory | None:
        """
        Get memory from vector store (source of truth).

        Args:
            memory_id: Unique identifier for the memory
            track_access: Whether to increment access count and update last_accessed
            include_relationships: Whether to include relationship data from graph

        Returns:
            Memory object if found, None otherwise

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If vector store operation fails
        """
        if not memory_id or not memory_id.strip():
            raise ValidationError("memory_id cannot be empty")

        try:
            # Get from vector store (source of truth)
            memory = await self.vector_store.get_memory(memory_id)

            if not memory:
                logger.debug(
                    f"Memory not found: {memory_id}",
                    extra={"memory_id": memory_id, "operation": "get_memory"},
                )
                return None

            # Track access if requested
            if track_access:
                await self._track_access(memory_id)
                # Reload to get updated access counts
                memory = await self.vector_store.get_memory(memory_id)

            # Optionally enrich with relationships from graph
            if include_relationships and memory:
                # Attach relationship data (can be used by services)
                memory.metadata["_relationships"] = await self._get_relationship_summary(memory_id)

            return memory

        except VectorStoreError:
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Failed to get memory: {e}") from e

    async def create_memory(self, memory: Memory) -> Memory:
        """
        Create memory in vector store + minimal node in graph.

        Args:
            memory: Memory to create (must have embedding)

        Returns:
            Created memory

        Raises:
            ValidationError: If memory is invalid
            VectorStoreError: If vector store operation fails
            GraphStoreError: If graph store operation fails
        """
        if not memory.id or not memory.content:
            raise ValidationError("Memory must have id and content")
        if not memory.embedding:
            raise ValidationError("Memory must have embedding for vector storage")

        logger.info(
            f"Creating memory: {memory.id}",
            extra={"memory_id": memory.id, "operation": "create_memory"},
        )

        try:
            # Create in vector store (source of truth)
            await self.vector_store.upsert_memory(memory)
            logger.debug(
                f"Memory stored in vector store: {memory.id}",
                extra={"memory_id": memory.id},
            )

            # Create minimal node in graph store (for relationships)
            await self.graph_store.add_node(memory)
            logger.debug(
                f"Memory node created in graph: {memory.id}",
                extra={"memory_id": memory.id},
            )

            logger.info(
                f"Memory created successfully: {memory.id}",
                extra={"memory_id": memory.id},
            )
            return memory

        except (VectorStoreError, GraphStoreError):
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating memory {memory.id}: {e}",
                extra={"memory_id": memory.id, "error": str(e)},
            )
            raise MemoryError(f"Failed to create memory: {e}") from e

    async def update_memory(self, memory: Memory) -> Memory:
        """
        Update memory in vector store + sync minimal node.

        Args:
            memory: Updated memory

        Returns:
            Updated memory

        Raises:
            ValidationError: If memory is invalid
            VectorStoreError: If vector store operation fails
        """
        if not memory.id:
            raise ValidationError("Memory must have id")

        logger.info(
            f"Updating memory: {memory.id}",
            extra={"memory_id": memory.id, "operation": "update_memory"},
        )

        try:
            # Update timestamp
            memory.updated_at = datetime.now()

            # If embedding is missing, try to retrieve from vector store
            if not memory.embedding or len(memory.embedding) == 0:
                logger.debug(
                    f"Embedding missing for {memory.id}, attempting to retrieve from vector store",
                    extra={"memory_id": memory.id},
                )
                existing_memory = await self.vector_store.get_memory(memory.id)
                if (
                    existing_memory
                    and existing_memory.embedding
                    and len(existing_memory.embedding) > 0
                ):
                    memory.embedding = existing_memory.embedding
                    logger.debug(
                        f"Retrieved embedding for {memory.id} from vector store",
                        extra={"memory_id": memory.id},
                    )
                else:
                    raise ValidationError(
                        f"Cannot update {memory.id}: no embedding found. "
                        "Memory needs to be re-embedded with embedder.embed() before updating."
                    )

            # Update in vector store with simple retry (2 attempts)
            max_retries = 2
            retry_delay = 0.5

            for attempt in range(max_retries):
                try:
                    await self.vector_store.upsert_memory(memory)
                    logger.debug(
                        f"Memory updated in vector store: {memory.id}",
                        extra={"memory_id": memory.id},
                    )
                    break
                except VectorStoreError as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Update failed for {memory.id} (attempt {attempt + 1}/{max_retries}): {e}. Retrying...",
                            extra={"memory_id": memory.id, "attempt": attempt + 1},
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        raise

            # Sync minimal node to graph store
            await self.graph_store.update_node(memory)
            logger.debug(
                f"Memory node synced to graph: {memory.id}",
                extra={"memory_id": memory.id},
            )

            logger.info(
                f"Memory updated successfully: {memory.id}",
                extra={"memory_id": memory.id},
            )
            return memory

        except (ValidationError, VectorStoreError):
            # Let validation and store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error updating memory {memory.id}: {e}",
                extra={"memory_id": memory.id, "error": str(e)},
            )
            raise MemoryError(f"Failed to update memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete memory from both stores.

        Args:
            memory_id: Memory ID to delete

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If vector store deletion fails
            GraphStoreError: If graph store deletion fails
        """
        if not memory_id or not memory_id.strip():
            raise ValidationError("memory_id cannot be empty")

        logger.info(
            f"Deleting memory: {memory_id}",
            extra={"memory_id": memory_id, "operation": "delete_memory"},
        )

        try:
            # Delete from vector store (source of truth)
            await self.vector_store.delete_memory(memory_id)
            logger.debug(
                f"Memory deleted from vector store: {memory_id}",
                extra={"memory_id": memory_id},
            )

            # Delete node and edges from graph store
            await self.graph_store.delete_node(memory_id)
            logger.debug(
                f"Memory node deleted from graph: {memory_id}",
                extra={"memory_id": memory_id},
            )

            logger.info(
                f"Memory deleted successfully: {memory_id}",
                extra={"memory_id": memory_id},
            )

        except (VectorStoreError, GraphStoreError):
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error deleting memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Failed to delete memory: {e}") from e

    async def get_relationships(
        self,
        memory_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[Edge]:
        """
        Get relationships from graph store.

        Args:
            memory_id: Memory ID
            relationship_types: Filter by specific types
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of edges

        Raises:
            ValidationError: If memory_id is invalid
            GraphStoreError: If graph store operation fails
        """
        if not memory_id or not memory_id.strip():
            raise ValidationError("memory_id cannot be empty")

        try:
            # Get neighbors from graph (returns list of tuples)
            neighbors = await self.graph_store.get_neighbors(
                node_id=memory_id,
                relationship_types=relationship_types,
                direction=direction,
                depth=1,
                limit=1000,
            )

            # Extract just the edges
            edges = [edge for _, edge in neighbors]
            return edges

        except GraphStoreError:
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting relationships for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Failed to get relationships: {e}") from e

    async def add_relationship(self, edge: Edge) -> str:
        """
        Add relationship to graph store.

        Args:
            edge: Edge to add

        Returns:
            Edge ID

        Raises:
            ValidationError: If edge is invalid
            GraphStoreError: If graph store operation fails
        """
        if not edge.source or not edge.target:
            raise ValidationError("Edge must have source and target")

        try:
            edge_id = await self.graph_store.add_edge(edge)
            logger.debug(
                f"Relationship added: {edge.source} -> {edge.target}",
                extra={
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.value,
                },
            )
            return edge_id

        except GraphStoreError:
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error adding relationship: {e}",
                extra={"error": str(e)},
            )
            raise MemoryError(f"Failed to add relationship: {e}") from e

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        track_access: bool = False,
        score_threshold: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Search vector store for similar memories.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            filters: Optional filters (status, type, etc.)
            track_access: Whether to track access for results
            score_threshold: Minimum similarity score

        Returns:
            List of (Memory, score) tuples

        Raises:
            ValidationError: If query_embedding is invalid
            VectorStoreError: If search operation fails
        """
        if not query_embedding:
            raise ValidationError("query_embedding cannot be empty")

        try:
            # Search vector store
            results = await self.vector_store.search_similar(
                vector=query_embedding,
                limit=limit,
                filters=filters,
                score_threshold=score_threshold,
            )

            # Track access for results if requested
            if track_access:
                for result in results:
                    await self._track_access(result.memory.id)

            return [(result.memory, result.score) for result in results]

        except VectorStoreError:
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in search_similar: {e}",
                extra={"error": str(e)},
            )
            raise MemoryError(f"Failed to search similar memories: {e}") from e

    async def get_neighbors(
        self,
        memory_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = 100,
        track_access: bool = True,
    ) -> list[tuple[Memory, Edge]]:
        """
        Get neighbors from graph, then fetch full memories from vector.

        Args:
            memory_id: Starting memory ID
            relationship_types: Filter by specific types
            direction: "outgoing", "incoming", or "both"
            depth: Traversal depth
            limit: Maximum results
            track_access: Whether to track access for source memory

        Returns:
            List of (Memory, Edge) tuples with complete memory data

        Raises:
            ValidationError: If memory_id is invalid
            GraphStoreError: If graph store operation fails
            VectorStoreError: If vector store operation fails
        """
        if not memory_id or not memory_id.strip():
            raise ValidationError("memory_id cannot be empty")

        try:
            # Track access for source memory if requested
            if track_access:
                await self._track_access(memory_id)

            # Get edges from graph store
            neighbors = await self.graph_store.get_neighbors(
                node_id=memory_id,
                relationship_types=relationship_types,
                direction=direction,
                depth=depth,
                limit=limit,
            )

            # Fetch full memories from vector store (source of truth)
            enriched_neighbors = []
            for memory, edge in neighbors:
                # Get full memory from vector store
                full_memory = await self.vector_store.get_memory(memory.id)
                if full_memory:
                    enriched_neighbors.append((full_memory, edge))
                else:
                    # Fallback to graph memory if not in vector store
                    enriched_neighbors.append((memory, edge))

            return enriched_neighbors

        except (GraphStoreError, VectorStoreError):
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting neighbors for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Failed to get neighbors: {e}") from e

    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[str] | None:
        """
        Find path between two memories in the graph.

        Args:
            start_id: Start memory ID
            end_id: End memory ID
            max_depth: Maximum path length

        Returns:
            List of memory IDs forming path, or None if no path found

        Raises:
            ValidationError: If start_id or end_id is invalid
            GraphStoreError: If graph store operation fails
        """
        if not start_id or not end_id:
            raise ValidationError("start_id and end_id are required")

        try:
            return await self.graph_store.find_path(start_id, end_id, max_depth)
        except GraphStoreError:
            # Let store errors propagate
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error finding path: {e}",
                extra={"start_id": start_id, "end_id": end_id, "error": str(e)},
            )
            raise MemoryError(f"Failed to find path: {e}") from e

    async def _track_access(self, memory_id: str) -> None:
        """
        Atomically track memory access.

        Updates access_count and last_accessed in vector store.
        This is an internal method called by get_memory and get_neighbors.

        Args:
            memory_id: Memory ID to track
        """
        try:
            # Get current memory from vector store
            memory = await self.vector_store.get_memory(memory_id)
            if not memory:
                logger.warning(
                    f"Cannot track access for non-existent memory: {memory_id}",
                    extra={"memory_id": memory_id},
                )
                return

            # Increment access count and update timestamps
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            memory.updated_at = datetime.now()

            # Update in vector store (metadata-only update is efficient)
            await self.vector_store.upsert_memory(memory)

            logger.debug(
                f"Access tracked for {memory_id} (count: {memory.access_count})",
                extra={"memory_id": memory_id, "access_count": memory.access_count},
            )

        except VectorStoreError as e:
            # Log but don't raise - access tracking failures shouldn't break reads
            logger.warning(
                f"Failed to track access for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
        except Exception as e:
            # Log unexpected errors but don't raise
            logger.warning(
                f"Unexpected error tracking access for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )

    async def _get_relationship_summary(self, memory_id: str) -> dict[str, Any]:
        """
        Get summary of relationships for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            Dict with relationship counts by type and direction
        """
        try:
            outgoing = await self.graph_store.get_neighbors(
                node_id=memory_id, direction="outgoing", depth=1, limit=1000
            )
            incoming = await self.graph_store.get_neighbors(
                node_id=memory_id, direction="incoming", depth=1, limit=1000
            )

            return {
                "outgoing_count": len(outgoing),
                "incoming_count": len(incoming),
                "total_count": len(outgoing) + len(incoming),
            }
        except Exception as e:
            logger.warning(
                f"Failed to get relationship summary for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            return {"outgoing_count": 0, "incoming_count": 0, "total_count": 0}

    # UTILITY METHODS

    async def count_memories(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count memories in vector store.

        Args:
            filters: Optional filter conditions

        Returns:
            Number of memories

        Raises:
            VectorStoreError: If count operation fails
        """
        try:
            return await self.vector_store.count_memories(filters)
        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error counting memories: {e}",
                extra={"error": str(e)},
            )
            raise MemoryError(f"Failed to count memories: {e}") from e
