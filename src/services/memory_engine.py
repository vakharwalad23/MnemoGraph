"""
Unified Memory Engine - Integrates all components.

Brings together:
- LLM & Embedder providers
- Memory Evolution & Invalidation
- Scalable Relationship Extraction
- Graph Store & Vector Store
"""

import asyncio
from datetime import datetime
from typing import Any

from src.config import Config
from src.core.embeddings.base import Embedder
from src.core.graph_store.base import GraphStore
from src.core.llm.base import LLMProvider
from src.core.memory_store.memory_store import MemoryStore
from src.core.vector_store.base import VectorStore
from src.models.memory import Memory, NodeType
from src.models.relationships import Edge, RelationshipBundle
from src.models.version import InvalidationResult, MemoryEvolution, VersionChain
from src.services.invalidation_manager import InvalidationManager
from src.services.llm_relationship_engine import LLMRelationshipEngine
from src.services.memory_evolution import MemoryEvolutionService
from src.utils.exceptions import (
    EmbeddingError,
    GraphStoreError,
    MemoryError,
    NotFoundError,
    ValidationError,
    VectorStoreError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryEngine:
    """
    Unified Memory Engine integrating all components.

    Features:
    - Add memories with automatic relationship extraction
    - Query memories with intelligent filtering
    - Memory versioning and evolution
    - Intelligent invalidation
    - Vector + Graph search
    - Background workers
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: Embedder,
        graph_store: GraphStore,
        vector_store: VectorStore,
        config: Config,
    ):
        """
        Initialize Memory Engine.

        Args:
            llm: LLM provider for text generation
            embedder: Embedder for generating embeddings
            graph_store: Graph database (Neo4j)
            vector_store: Vector database (Qdrant)
            config: Configuration object
        """
        self.llm = llm
        self.embedder = embedder
        self.config = config

        self.memory_store = MemoryStore(
            vector_store=vector_store,
            graph_store=graph_store,
        )

        # Direct store references kept for infrastructure operations only (initialize, close, statistics)
        self.graph_store = graph_store
        self.vector_store = vector_store

        self.evolution = MemoryEvolutionService(
            llm=llm,
            memory_store=self.memory_store,
            embedder=embedder,
        )

        self.invalidation = InvalidationManager(
            llm=llm,
            memory_store=self.memory_store,
        )

        self.relationship_engine = LLMRelationshipEngine(
            llm_provider=llm,
            embedder=embedder,
            memory_store=self.memory_store,
            config=config,
        )

    async def initialize(self) -> None:
        """Initialize all stores."""
        logger.info("Initializing Memory Engine")

        await self.graph_store.initialize()
        logger.info("Graph store initialized")

        await self.vector_store.initialize()
        logger.info("Vector store initialized")

        if self.config.llm_relationships.enable_auto_invalidation:
            self.invalidation.start_background_worker(interval_hours=24)
            logger.info("Background invalidation worker started")

        logger.info("Memory Engine ready")

    async def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_type: NodeType = NodeType.MEMORY,
    ) -> tuple[Memory, RelationshipBundle]:
        """
        Add a new memory with automatic relationship extraction.

        Args:
            content: Memory content
            metadata: Optional metadata
            memory_type: Type of memory node

        Returns:
            Tuple of (Memory, RelationshipBundle)
        Raises:
            ValidationError: If content is invalid
            EmbeddingError: If embedding generation fails
            MemoryError: If memory creation fails
        """
        if not content or not content.strip():
            raise ValidationError("Memory content cannot be empty")

        logger.info(
            f"Adding memory: {content[:50]}",
            extra={"operation": "add_memory", "memory_type": memory_type.value},
        )

        try:
            # Generate embedding
            logger.debug("Generating embedding")
            embedding = await self.embedder.embed(content)
        except Exception as e:
            logger.error(
                f"Failed to generate embedding: {e}",
                extra={"operation": "add_memory", "error": str(e)},
            )
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

        try:
            # Create memory object
            memory = Memory(
                id=self._generate_id(),
                content=content,
                type=memory_type,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # Process with relationship engine (handles graph + vector store)
            logger.debug("Extracting relationships")
            extraction = await self.relationship_engine.process_new_memory(memory)

            logger.info(
                f"Memory added: {memory.id}",
                extra={
                    "memory_id": memory.id,
                    "relationships": len(extraction.relationships),
                    "derived_insights": len(extraction.derived_insights),
                },
            )

            return memory, extraction
        except (GraphStoreError, VectorStoreError) as e:
            logger.error(
                f"Failed to add memory: {e}",
                extra={"operation": "add_memory", "error": str(e), "error_type": type(e).__name__},
            )
            raise MemoryError(f"Failed to add memory: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error adding memory: {e}",
                extra={"operation": "add_memory", "error": str(e)},
            )
            raise MemoryError(f"Unexpected error adding memory: {e}") from e

    async def get_memory(
        self,
        memory_id: str,
        validate: bool = True,
        track_access: bool = True,
        include_relationships: bool = False,
        relationship_limit: int = 100,
    ) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier
            validate: Whether to check validity on access
            track_access: Whether to track this access (increments access_count)
            include_relationships: Whether to include relationship data in metadata
            relationship_limit: Maximum number of relationships to include

        Returns:
            Memory or None if not found

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If vector store operation fails
            MemoryError: If unexpected error occurs
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        try:
            memory = await self.memory_store.get_memory(
                memory_id=memory_id,
                track_access=track_access,
                include_relationships=include_relationships,
                relationship_limit=relationship_limit,
            )

            if not memory:
                logger.debug(
                    f"Memory not found: {memory_id}",
                    extra={"memory_id": memory_id, "operation": "get_memory"},
                )
                return None

            if validate and self.config.llm_relationships.enable_auto_invalidation:
                memory = await self.invalidation.validate_on_access(memory)

            return memory

        except (ValidationError, VectorStoreError):
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "operation": "get_memory", "error": str(e)},
            )
            raise MemoryError(f"Failed to get memory: {e}") from e

    async def update_memory(
        self, memory_id: str, new_content: str
    ) -> tuple[Memory, MemoryEvolution]:
        """
        Update memory with versioning.

        Args:
            memory_id: Memory to update
            new_content: New content

        Returns:
            Tuple of (updated_memory, evolution_result)
        Raises:
            ValidationError: If inputs are invalid
            NotFoundError: If memory not found
            MemoryError: If update fails
        """
        if not memory_id:
            raise ValidationError("memory_id is required")
        if not new_content or not new_content.strip():
            raise ValidationError("new_content cannot be empty")

        logger.info(
            f"Updating memory: {memory_id}",
            extra={"memory_id": memory_id, "operation": "update_memory"},
        )

        try:
            current = await self.memory_store.get_memory(
                memory_id=memory_id, track_access=False, include_relationships=False
            )
            if not current:
                raise NotFoundError(f"Memory not found: {memory_id}")

            evolution = await self.evolution.evolve_memory(current, new_content)

            if evolution.new_version:
                new_memory = await self.memory_store.get_memory(
                    memory_id=evolution.new_version, track_access=False, include_relationships=False
                )
                logger.info(
                    "Memory updated with new version",
                    extra={"memory_id": memory_id, "new_version": evolution.new_version},
                )
                return new_memory, evolution

            updated_memory = await self.memory_store.get_memory(
                memory_id=memory_id, track_access=False, include_relationships=False
            )
            logger.info(
                f"Memory {evolution.action}",
                extra={"memory_id": memory_id, "action": evolution.action},
            )
            return updated_memory, evolution
        except (NotFoundError, ValidationError):
            raise
        except (GraphStoreError, VectorStoreError) as e:
            logger.error(
                f"Failed to update memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e), "error_type": type(e).__name__},
            )
            raise MemoryError(f"Failed to update memory: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error updating memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Unexpected error updating memory: {e}") from e

    async def update_memory_metadata(self, memory_id: str, new_metadata: dict[str, Any]) -> Memory:
        """
        Update memory metadata.

        Args:
            memory_id: Memory to update
            new_metadata: New metadata dictionary

        Returns:
            Updated memory

        Raises:
            ValidationError: If inputs are invalid
            NotFoundError: If memory not found
            MemoryError: If update fails
        """
        if not memory_id:
            raise ValidationError("memory_id is required")
        if not isinstance(new_metadata, dict):
            raise ValidationError("new_metadata must be a dictionary")

        logger.info(
            f"Updating metadata for memory: {memory_id}",
            extra={"memory_id": memory_id, "operation": "update_metadata"},
        )

        try:
            memory = await self.memory_store.get_memory(
                memory_id=memory_id, track_access=False, include_relationships=False
            )
            if not memory:
                raise NotFoundError(f"Memory not found: {memory_id}")

            if new_metadata:
                memory.metadata.update(new_metadata)
                await self.memory_store.update_memory(memory)

            logger.info(f"Metadata updated for memory: {memory_id}", extra={"memory_id": memory_id})

            return memory

        except (NotFoundError, ValidationError):
            raise
        except VectorStoreError as e:
            logger.error(
                f"Failed to update metadata for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e), "error_type": type(e).__name__},
            )
            raise MemoryError(f"Failed to update metadata: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error updating metadata for {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Unexpected error updating metadata: {e}") from e

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from both stores.

        Args:
            memory_id: Memory to delete

        Raises:
            ValidationError: If memory_id is invalid
            MemoryError: If deletion fails
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        logger.info(
            f"Deleting memory: {memory_id}",
            extra={"memory_id": memory_id, "operation": "delete_memory"},
        )

        try:
            await self.memory_store.delete_memory(memory_id)

            logger.info(f"Memory deleted: {memory_id}", extra={"memory_id": memory_id})

        except (ValidationError, VectorStoreError, GraphStoreError) as e:
            logger.error(
                f"Failed to delete memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e), "error_type": type(e).__name__},
            )
            raise MemoryError(f"Failed to delete memory: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error deleting memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise MemoryError(f"Unexpected error deleting memory: {e}") from e

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        track_access: bool = False,
    ) -> list[tuple[Memory, float]]:
        """
        Search for similar memories by semantic meaning.

        Args:
            query: Query text
            limit: Maximum results
            filters: Optional filters (status, type, etc.)
            score_threshold: Minimum similarity score
            track_access: Whether to track access for results (default: False)

        Returns:
            List of (Memory, score) tuples

        Raises:
            EmbeddingError: If embedding generation fails
            VectorStoreError: If search fails
        """
        try:
            query_embedding = await self.embedder.embed(query)

            results = await self.memory_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                filters=filters,
                score_threshold=score_threshold,
                track_access=track_access,
            )

            return results

        except Exception as e:
            logger.error(
                f"Search similar failed: {e}",
                extra={"query": query[:50], "error": str(e)},
            )
            raise

    async def query_memories(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Query memories from graph store with filters.

        Args:
            filters: Filter conditions
            order_by: Sort order
            limit: Maximum results

        Returns:
            List of memories
        """
        return await self.graph_store.query_memories(
            filters=filters,
            order_by=order_by,
            limit=limit,
        )

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
        Get related memories via graph relationships.

        Args:
            memory_id: Starting memory
            relationship_types: Filter by types
            direction: "outgoing", "incoming", or "both"
            depth: Traversal depth
            limit: Maximum results
            track_access: Whether to track access for source memory

        Returns:
            List of tuples with (Memory, Edge) - memories are fetched from vector store

        Raises:
            ValidationError: If memory_id is invalid
            GraphStoreError: If graph traversal fails
            VectorStoreError: If memory fetch fails
        """
        return await self.memory_store.get_neighbors(
            memory_id=memory_id,
            relationship_types=relationship_types,
            direction=direction,
            depth=depth,
            limit=limit,
            track_access=track_access,
        )

    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[str] | None:
        """
        Find path between two memories.

        Args:
            start_id: Start memory ID
            end_id: End memory ID
            max_depth: Maximum path length

        Returns:
            List of memory IDs forming path, or None

        Raises:
            ValidationError: If start_id or end_id is invalid
            GraphStoreError: If path finding fails
        """
        return await self.memory_store.find_path(start_id, end_id, max_depth)

    async def get_version_history(self, memory_id: str) -> VersionChain:
        """
        Get complete version history for a memory.

        Args:
            memory_id: Memory ID

        Returns:
            VersionChain object
        """
        return await self.evolution.get_version_history(memory_id)

    async def rollback_to_version(self, version_id: str) -> Memory:
        """
        Rollback to a previous version.

        Args:
            version_id: Version to rollback to

        Returns:
            New memory with rolled back content
        """
        return await self.evolution.rollback_to_version(version_id)

    async def time_travel_query(self, query: str, as_of: datetime, limit: int = 10) -> list[Memory]:
        """
        Query memories as they existed at a point in time.

        Args:
            query: Query text
            as_of: Point in time
            limit: Maximum results

        Returns:
            List of memories valid at that time
        """
        query_embedding = await self.embedder.embed(query)

        return await self.evolution.time_travel_query(query_embedding, as_of, limit)

    async def check_memory_validity(self, memory_id: str) -> InvalidationResult:
        """
        Check if a memory is still valid.

        Args:
            memory_id: Memory to check

        Returns:
            InvalidationResult

        Raises:
            ValidationError: If memory_id is invalid
            NotFoundError: If memory not found
        """
        if not memory_id:
            raise ValidationError("memory_id is required")

        memory = await self.memory_store.get_memory(
            memory_id=memory_id, track_access=False, include_relationships=False
        )
        if not memory:
            raise NotFoundError(f"Memory not found: {memory_id}")

        return await self.invalidation.check_invalidation(memory)

    async def invalidate_memory(self, memory_id: str, reason: str) -> None:
        """
        Manually invalidate a memory.

        Args:
            memory_id: Memory to invalidate
            reason: Invalidation reason

        Raises:
            ValidationError: If inputs are invalid
            NotFoundError: If memory not found
        """
        if not memory_id:
            raise ValidationError("memory_id is required")
        if not reason or not reason.strip():
            raise ValidationError("reason cannot be empty")

        memory = await self.memory_store.get_memory(
            memory_id=memory_id, track_access=False, include_relationships=False
        )
        if not memory:
            raise NotFoundError(f"Memory not found: {memory_id}")

        from src.models.version import InvalidationResult, InvalidationStatus

        result = InvalidationResult(
            memory_id=memory_id,
            status=InvalidationStatus.INVALIDATED,
            reasoning=reason,
            confidence=1.0,
        )

        await self.invalidation._mark_invalidated(memory, result)

    # STATISTICS & MONITORING

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get memory engine statistics.

        Returns:
            Statistics dictionary
        """
        active_count = await self.graph_store.count_nodes({"status": "active"})
        historical_count = await self.graph_store.count_nodes({"status": "historical"})
        superseded_count = await self.graph_store.count_nodes({"status": "superseded"})

        total_edges = await self.graph_store.count_edges()

        vector_count = await self.vector_store.count_memories()

        return {
            "memories": {
                "active": active_count,
                "historical": historical_count,
                "superseded": superseded_count,
                "total_graph": active_count + historical_count + superseded_count,
                "total_vector": vector_count,
            },
            "relationships": {
                "total": total_edges,
            },
        }

    # LIFECYCLE MANAGEMENT

    async def close(self) -> None:
        """Close all connections and stop workers."""
        logger.info("Shutting down Memory Engine")

        # Stop background workers and wait for them to finish
        self.invalidation.stop_background_worker()

        # Wait for background task to complete cancellation
        if hasattr(self.invalidation, "_worker_task") and self.invalidation._worker_task:
            try:
                await self.invalidation._worker_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception as e:
                logger.warning(f"Error during worker shutdown: {e}")

        # Close stores
        await self.graph_store.close()
        await self.vector_store.close()

        # Close providers
        await self.llm.close()
        await self.embedder.close()

        logger.info("Memory Engine shutdown complete")

    # HELPER METHODS

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        from uuid import uuid4

        return f"mem_{uuid4().hex[:12]}"
