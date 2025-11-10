"""
Memory Sync Manager - Ensures consistency between graph and vector stores.

Handles:
- Atomic updates to both stores
- Selective field synchronization
- Retry logic for failures
- Validation and repair utilities
"""

import asyncio
from typing import Any

from src.core.graph_store.base import GraphStore
from src.core.vector_store.base import VectorStore
from src.models.memory import Memory
from src.utils.exceptions import GraphStoreError, SyncError, VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemorySyncManager:
    """
    Manages synchronization between graph and vector stores.

    - Vector store = Source of truth (ALL memory data and metadata)
    - Graph store = Minimal nodes (id, content_preview, type, status, version info)

    Vector store stores (complete memory):
    - content (full text)
    - embedding (vector representation)
    - metadata (complete metadata dict)
    - status, type, confidence
    - all timestamps (created_at, updated_at, valid_from, valid_until)
    - version info (version, parent_version, superseded_by)
    - access tracking (access_count, last_accessed)
    - invalidation info (invalidation_reason)

    Graph store stores (minimal node):
    - id (for relationships)
    - content_preview (first 200 chars for display)
    - type, status (for filtering)
    - version, parent_version, superseded_by (for version chain traversal)

    Sync direction: Vector store (source) -> Graph store (minimal sync)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize sync manager.

        Args:
            graph_store: Graph database
            vector_store: Vector database
            max_retries: Maximum retry attempts for sync operations
            retry_delay: Delay between retries in seconds
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def _retry_operation(self, operation, operation_name: str):
        """
        Retry an async operation with exponential backoff.

        Args:
            operation: Async callable to retry
            operation_name: Name for logging

        Returns:
            Result of operation

        Raises:
            SyncError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await operation()
            except (GraphStoreError, VectorStoreError) as e:
                # Store-specific errors are retryable
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s...",
                        extra={
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "error_type": type(e).__name__,
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries} attempts",
                        extra={
                            "operation": operation_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
            except Exception as e:
                # Unexpected errors should not be retried
                logger.error(
                    f"{operation_name} failed with unexpected error: {e}",
                    extra={
                        "operation": operation_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise SyncError(
                    f"{operation_name} failed: {e}",
                    context={"operation": operation_name, "error_type": type(e).__name__},
                ) from e

        raise SyncError(
            f"{operation_name} failed after {self.max_retries} attempts: {last_error}",
            context={"operation": operation_name, "max_retries": self.max_retries},
        ) from last_error

    async def sync_memory_status(self, memory: Memory) -> None:
        """
        Sync memory status to vector store.

        Args:
            memory: Memory object with updated status

        Raises:
            SyncError: If sync fails after retries
        """
        await self.sync_memory_full(memory)

    async def sync_memory_invalidation(self, memory: Memory) -> None:
        """
        Sync memory invalidation to vector store.

        Updates: status, invalidation_reason, valid_until, updated_at

        Args:
            memory: Memory object with invalidation state

        Raises:
            SyncError: If sync fails after retries
        """
        await self.sync_memory_full(memory)

    async def sync_memory_supersession(self, memory: Memory) -> None:
        """
        Sync memory supersession to vector store.

        Updates old memory: status, superseded_by, valid_until, updated_at

        Args:
            memory: Memory object with supersession state

        Raises:
            SyncError: If sync fails after retries
        """
        await self.sync_memory_full(memory)

    async def sync_memory_full(self, memory: Memory) -> None:
        """
        Full sync of memory to vector store.

        Use for: new memories, content updates, embedding changes, status updates,
        invalidation, supersession, or any other memory state changes.

        If embedding is missing, automatically retrieves it from vector store.

        Args:
            memory: Complete memory object to sync

        Raises:
            SyncError: If sync fails after retries or if embedding cannot be found
            VectorStoreError: If vector store operations fail
        """
        # If embedding is missing or empty, try to get it from vector store
        if not memory.embedding or len(memory.embedding) == 0:
            logger.debug(
                f"Embedding missing for {memory.id}, attempting to retrieve from vector store",
                extra={"memory_id": memory.id, "operation": "sync_memory_full"},
            )
            try:
                vector_memory = await self.vector_store.get_memory(memory.id)
                if vector_memory and vector_memory.embedding and len(vector_memory.embedding) > 0:
                    memory.embedding = vector_memory.embedding
                    logger.debug(
                        f"Retrieved embedding for {memory.id} from vector store",
                        extra={"memory_id": memory.id},
                    )
                else:
                    raise SyncError(
                        f"Cannot sync {memory.id}: no embedding found. "
                        "Memory needs to be re-embedded with embedder.embed() before syncing.",
                        context={"memory_id": memory.id, "operation": "sync_memory_full"},
                    )
            except VectorStoreError as e:
                raise SyncError(
                    f"Failed to retrieve embedding for {memory.id}: {e}",
                    context={"memory_id": memory.id, "operation": "retrieve_embedding"},
                ) from e

        async def _sync():
            try:
                await self.vector_store.upsert_memory(memory)
                logger.debug(
                    f"Synced memory {memory.id}",
                    extra={"memory_id": memory.id, "status": memory.status.value},
                )
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to upsert memory {memory.id}: {e}", context={"memory_id": memory.id}
                ) from e

        await self._retry_operation(_sync, f"sync_memory_full({memory.id})")

    async def sync_memory_deletion(self, memory_id: str) -> None:
        """
        Sync memory deletion to vector store.

        Args:
            memory_id: Memory ID to delete

        Raises:
            SyncError: If sync fails after retries
            VectorStoreError: If vector store deletion fails
        """

        async def _sync():
            try:
                await self.vector_store.delete_memory(memory_id)
                logger.debug(
                    f"Synced deletion for {memory_id}",
                    extra={"memory_id": memory_id, "operation": "delete"},
                )
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to delete memory {memory_id}: {e}", context={"memory_id": memory_id}
                ) from e

        await self._retry_operation(_sync, f"sync_memory_deletion({memory_id})")

    async def sync_batch_memories(self, memories: list[Memory]) -> dict[str, Any]:
        """
        Batch sync multiple memories to vector store.

        More efficient than individual syncs for bulk operations.

        Args:
            memories: List of memories to sync

        Returns:
            Dict with sync results: {"success": count, "failed": count, "errors": [...]}
        """
        results = {"success": 0, "failed": 0, "errors": []}

        for memory in memories:
            try:
                await self.sync_memory_full(memory)
                results["success"] += 1
            except SyncError as e:
                results["failed"] += 1
                results["errors"].append({"memory_id": memory.id, "error": str(e)})
                logger.error(
                    f"Batch sync failed for {memory.id}",
                    extra={"memory_id": memory.id, "error": str(e), "operation": "batch_sync"},
                )

        logger.info(
            f"Batch sync completed: {results['success']} success, {results['failed']} failed",
            extra={"success_count": results["success"], "failed_count": results["failed"]},
        )
        return results

    async def validate_sync_consistency(self, memory_id: str) -> dict[str, Any]:
        """
        Validate that graph and vector stores are in sync for critical fields.

        Args:
            memory_id: Memory ID to validate

        Returns:
            Dict with validation results: {"in_sync": bool, "mismatches": [...]}
        """
        try:
            graph_memory = await self.graph_store.get_node(memory_id)
            vector_memory = await self.vector_store.get_memory(memory_id)

            if not graph_memory:
                return {
                    "in_sync": False,
                    "error": "Memory not found in graph store",
                }

            if not vector_memory:
                return {
                    "in_sync": False,
                    "error": "Memory not found in vector store",
                }

            # Check critical fields
            mismatches = []

            if graph_memory.status != vector_memory.status:
                mismatches.append(
                    {
                        "field": "status",
                        "graph": graph_memory.status.value,
                        "vector": vector_memory.status.value,
                    }
                )

            if graph_memory.type != vector_memory.type:
                mismatches.append(
                    {
                        "field": "type",
                        "graph": graph_memory.type.value,
                        "vector": vector_memory.type.value,
                    }
                )

            if graph_memory.valid_until != vector_memory.valid_until:
                mismatches.append(
                    {
                        "field": "valid_until",
                        "graph": (
                            graph_memory.valid_until.isoformat()
                            if graph_memory.valid_until
                            else None
                        ),
                        "vector": (
                            vector_memory.valid_until.isoformat()
                            if vector_memory.valid_until
                            else None
                        ),
                    }
                )

            if graph_memory.superseded_by != vector_memory.superseded_by:
                mismatches.append(
                    {
                        "field": "superseded_by",
                        "graph": graph_memory.superseded_by,
                        "vector": vector_memory.superseded_by,
                    }
                )

            if graph_memory.version != vector_memory.version:
                mismatches.append(
                    {
                        "field": "version",
                        "graph": graph_memory.version,
                        "vector": vector_memory.version,
                    }
                )

            return {
                "in_sync": len(mismatches) == 0,
                "mismatches": mismatches,
            }

        except Exception as e:
            logger.error(f"Validation failed for {memory_id}: {e}")
            return {
                "in_sync": False,
                "error": f"Validation error: {str(e)}",
            }

    async def repair_sync(self, memory_id: str) -> bool:
        """
        Repair sync inconsistencies by syncing from graph to vector.

        Graph store is the source of truth for metadata, vector store provides embedding.

        Args:
            memory_id: Memory ID to repair

        Returns:
            True if repair successful, False otherwise
        """
        try:
            graph_memory = await self.graph_store.get_node(memory_id)
            if not graph_memory:
                logger.warning(f"Cannot repair {memory_id}: not found in graph store")
                return False

            # Get embedding from vector store (if exists)
            vector_memory = await self.vector_store.get_memory(memory_id)
            if vector_memory and vector_memory.embedding:
                # Use existing embedding from vector store
                graph_memory.embedding = vector_memory.embedding
            else:
                # No embedding in vector store - cannot sync without embedding
                logger.error(
                    f"Cannot repair {memory_id}: no embedding found in vector store. "
                    "Memory needs to be re-embedded."
                )
                return False

            await self.sync_memory_full(graph_memory)
            logger.info(f"Repaired sync for {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Repair failed for {memory_id}: {e}")
            return False

    async def validate_and_repair_batch(self, memory_ids: list[str]) -> dict[str, Any]:
        """
        Validate and repair multiple memories.

        Args:
            memory_ids: List of memory IDs to check

        Returns:
            Dict with results: {"validated": count, "repaired": count, "failed": count}
        """
        results = {
            "validated": 0,
            "in_sync": 0,
            "out_of_sync": 0,
            "repaired": 0,
            "failed": 0,
        }

        for memory_id in memory_ids:
            try:
                validation = await self.validate_sync_consistency(memory_id)
                results["validated"] += 1

                if validation["in_sync"]:
                    results["in_sync"] += 1
                else:
                    results["out_of_sync"] += 1
                    # Attempt repair
                    if await self.repair_sync(memory_id):
                        results["repaired"] += 1
                    else:
                        results["failed"] += 1
            except Exception as e:
                logger.error(f"Validation/repair failed for {memory_id}: {e}")
                results["failed"] += 1

        logger.info(
            f"Batch validation/repair: {results['in_sync']} in sync, "
            f"{results['repaired']} repaired, {results['failed']} failed"
        )
        return results
