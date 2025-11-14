"""
Qdrant vector store implementation

Clean implementation that works with new Memory model and optimized for performance.
"""

from datetime import datetime
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid5

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from src.core.vector_store.base import SearchResult, VectorStore
from src.models.memory import Memory, MemoryStatus, NodeType
from src.utils.exceptions import ValidationError, VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantStore(VectorStore):
    """
    Optimized Qdrant vector store for memory embeddings.

    Features:
    - gRPC connection for 2-3x speed
    - HNSW indexing for fast search
    - Int8 quantization for memory efficiency
    - Batch operations
    - Payload indexing for fast filtering
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "memories",
        vector_size: int = 768,
        use_grpc: bool = False,  # Set to True for production
        use_quantization: bool = False,  # Set to True for large datasets
        quantization_type: str = "int8",  # int8 or scalar
        hnsw_m: int = 16,  # HNSW M parameter (16-32 recommended)
        # HNSW ef_construct (higher = better quality)
        hnsw_ef_construct: int = 100,
        on_disk: bool = False,  # Store vectors on disk instead of RAM
    ):
        """
        Initialize Qdrant store.

        Args:
            host: Qdrant host
            port: Qdrant port (6333 for HTTP, 6334 for gRPC)
            collection_name: Collection name
            vector_size: Embedding dimension
            use_grpc: Use gRPC connection (faster)
            use_quantization: Use int8 quantization (memory efficient)
            quantization_type: Type of quantization (int8, scalar)
            hnsw_m: HNSW M parameter (connections per node)
            hnsw_ef_construct: HNSW ef_construct parameter
            on_disk: Store vectors on disk (reduces RAM usage)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.use_grpc = use_grpc
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.on_disk = on_disk
        self.client: AsyncQdrantClient | None = None

    def _to_uuid(self, id_str: str) -> str:
        """
        Convert string ID to UUID format consistently.

        Args:
            id_str: String identifier

        Returns:
            UUID string
        """
        try:
            UUID(id_str)
            return id_str
        except ValueError:
            return str(uuid5(NAMESPACE_DNS, id_str))

    async def connect(self) -> None:
        """
        Establish connection to Qdrant.

        Raises:
            VectorStoreError: If connection fails
        """
        if self.client is None:
            try:
                self.client = AsyncQdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=self.use_grpc,
                    timeout=30,
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to Qdrant: {e}",
                    extra={"host": self.host, "port": self.port, "error": str(e)},
                )
                raise VectorStoreError(f"Failed to connect to Qdrant: {e}") from e

    async def initialize(self) -> None:
        """
        Initialize the collection with optimized settings.

        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            await self.connect()

            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create with optimized settings
                vectors_config = VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    # HNSW indexing for fast search
                    hnsw_config=HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct,
                        full_scan_threshold=10000,
                    ),
                    on_disk=self.on_disk,
                )

                # Add quantization if enabled
                if self.use_quantization:
                    vectors_config.quantization_config = ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    )

                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,
                        memmap_threshold=50000,
                    ),
                )

                # Create payload indices for fast filtering
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="status",
                    field_schema="keyword",
                )

                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="type",
                    field_schema="keyword",
                )

                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="created_at",
                    field_schema="datetime",
                )

                # CRITICAL: User index for fast filtering
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema="keyword",
                )
        except Exception as e:
            logger.error(
                f"Failed to initialize Qdrant collection: {e}",
                extra={"collection": self.collection_name, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to initialize Qdrant collection: {e}") from e

    def _memory_to_payload(self, memory: Memory) -> dict[str, Any]:
        """
        Convert Memory to Qdrant payload.

        Args:
            memory: Memory object

        Returns:
            Payload dictionary
        """
        return {
            "original_id": memory.id,
            "content": memory.content,
            "type": memory.type.value,
            "status": memory.status.value,
            "user_id": memory.user_id,
            "version": memory.version,
            "parent_version": memory.parent_version,
            "valid_from": memory.valid_from.isoformat(),
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            "superseded_by": memory.superseded_by,
            "invalidation_reason": memory.invalidation_reason,
            "confidence": memory.confidence,
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "metadata": memory.metadata,
        }

    def _payload_to_memory(self, payload: dict[str, Any], vector: list[float]) -> Memory:
        """
        Convert Qdrant payload to Memory.

        Args:
            payload: Qdrant payload
            vector: Embedding vector

        Returns:
            Memory object
        """
        return Memory(
            id=payload["original_id"],
            content=payload["content"],
            type=NodeType(payload["type"]),
            embedding=vector,
            user_id=payload["user_id"],
            version=payload.get("version", 1),
            parent_version=payload.get("parent_version"),
            valid_from=datetime.fromisoformat(payload["valid_from"]),
            valid_until=(
                datetime.fromisoformat(payload["valid_until"])
                if payload.get("valid_until")
                else None
            ),
            status=MemoryStatus(payload["status"]),
            superseded_by=payload.get("superseded_by"),
            invalidation_reason=payload.get("invalidation_reason"),
            metadata=payload.get("metadata", {}),
            confidence=payload.get("confidence", 1.0),
            access_count=payload.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(payload["last_accessed"])
                if payload.get("last_accessed")
                else None
            ),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )

    async def upsert_memory(self, memory: Memory) -> None:
        """
        Store or update a memory with its embedding.

        Args:
            memory: Memory object with embedding

        Raises:
            ValidationError: If memory is invalid
            VectorStoreError: If upsert operation fails
        """
        if not memory:
            raise ValidationError("Memory cannot be None")
        if not memory.id:
            raise ValidationError("Memory ID cannot be empty")
        if not memory.embedding or len(memory.embedding) == 0:
            raise ValidationError("Memory must have an embedding")

        try:
            await self.connect()

            uuid_id = self._to_uuid(memory.id)
            payload = self._memory_to_payload(memory)

            point = PointStruct(
                id=uuid_id,
                vector=memory.embedding,
                payload=payload,
            )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,  # Wait for write to complete for consistency
            )
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to upsert memory {memory.id}: {e}",
                extra={"memory_id": memory.id, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to upsert memory: {e}") from e

    async def batch_upsert(self, memories: list[Memory], batch_size: int = 100) -> None:
        """
        Batch upsert multiple memories for efficiency.

        Args:
            memories: List of Memory objects
            batch_size: Batch size for upload
        """
        await self.connect()

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]

            points = [
                PointStruct(
                    id=self._to_uuid(mem.id),
                    vector=mem.embedding,
                    payload=self._memory_to_payload(mem),
                )
                for mem in batch
            ]

            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=False,
            )

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
            List of search results
        """
        await self.connect()

        # Build filter
        query_filter = None
        if filters:
            conditions = []

            # Status filter
            if "status" in filters:
                status_values = (
                    filters["status"]
                    if isinstance(filters["status"], list)
                    else [filters["status"]]
                )
                conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchAny(any=status_values),
                    )
                )

            # Type filter
            if "type" in filters:
                type_values = (
                    filters["type"] if isinstance(filters["type"], list) else [filters["type"]]
                )
                conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchAny(any=type_values),
                    )
                )

            # User filter - CRITICAL for multi-user isolation
            if "user_id" in filters:
                conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=filters["user_id"]),
                    )
                )

            # Date range filter
            if "created_after" in filters:
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(gte=filters["created_after"]),
                    )
                )

            if conditions:
                query_filter = Filter(must=conditions)

        # Search
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to SearchResults
        results = []
        for point in response.points:
            memory = self._payload_to_memory(point.payload, point.vector)
            results.append(
                SearchResult(
                    memory=memory,
                    score=point.score,
                    metadata={"qdrant_id": str(point.id)},
                )
            )

        return results

    async def search_by_payload(
        self, filter: dict[str, Any], limit: int = 10
    ) -> list[SearchResult]:
        """
        Search by payload/metadata filters.

        Args:
            filter: Payload filter conditions (must include user_id for security)
            limit: Maximum results

        Returns:
            List of matching memories
        """
        await self.connect()

        # Build filter conditions
        conditions = []
        for key, value in filter.items():
            if "$contains" in str(value):
                # Handle contains operations
                continue
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        # Scroll through results (no vector search)
        response = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to SearchResults
        results = []
        for point in response[0]:  # response is tuple (points, next_offset)
            memory = self._payload_to_memory(point.payload, point.vector)
            results.append(
                SearchResult(
                    memory=memory,
                    score=1.0,  # No score for payload search
                    metadata={"qdrant_id": str(point.id)},
                )
            )

        return results

    # INTERNAL METHODS FOR BACKGROUND OPERATIONS
    # These methods bypass user_id filtering and should ONLY be used by internal services
    # (e.g., background workers, admin operations). They maintain security by ensuring
    # all returned memories still have valid user_id values.

    async def _search_by_payload_all_users(
        self, filter: dict[str, Any] | None = None, limit: int = 100, order_by: str | None = None
    ) -> list[Memory]:
        """
        INTERNAL: Search memories across all users by payload filters.

        WARNING: This method bypasses user_id filtering and should ONLY be used
        by internal services (background workers, admin operations).

        Args:
            filter: Optional payload filter conditions (user_id will be ignored)
            limit: Maximum results
            order_by: Sort order (e.g., "created_at") - supports ASC/DESC

        Returns:
            List of memories from all users
        """
        await self.connect()

        # Build filter conditions (excluding user_id if present)
        conditions = []
        if filter:
            for key, value in filter.items():
                # Skip user_id filter for internal queries
                if key == "user_id":
                    continue
                if "$contains" in str(value):
                    # Handle contains operations
                    continue
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        query_filter = Filter(must=conditions) if conditions else None

        # Scroll through results (no vector search)
        # Note: Qdrant scroll doesn't support order_by directly, we'll sort in Python if needed
        response = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to Memory objects
        memories = []
        for point in response[0]:  # response is tuple (points, next_offset)
            memory = self._payload_to_memory(point.payload, point.vector)
            # Only include memories with valid user_id
            if memory.user_id:
                memories.append(memory)

        # Sort in Python if order_by specified
        if order_by:
            reverse = False
            if " DESC" in order_by:
                reverse = True
                order_by = order_by.replace(" DESC", "").strip()
            elif " ASC" in order_by:
                order_by = order_by.replace(" ASC", "").strip()

            if hasattr(Memory, order_by) or order_by in ["created_at", "updated_at"]:
                try:
                    memories.sort(
                        key=lambda m: getattr(m, order_by, datetime.min) or datetime.min,
                        reverse=reverse,
                    )
                except Exception:
                    logger.warning(f"Failed to sort by {order_by}, returning unsorted results")

        return memories[:limit]

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
        if not memory_id or not memory_id.strip():
            raise ValidationError("Memory ID cannot be empty")

        try:
            await self.connect()

            uuid_id = self._to_uuid(memory_id)

            results = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[uuid_id],
                with_vectors=True,
            )

            if not results:
                return None

            point = results[0]
            return self._payload_to_memory(point.payload, point.vector)
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to retrieve memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to retrieve memory: {e}") from e

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from the store.

        Args:
            memory_id: Memory identifier

        Raises:
            ValidationError: If memory_id is invalid
            VectorStoreError: If deletion operation fails
        """
        if not memory_id or not memory_id.strip():
            raise ValidationError("Memory ID cannot be empty")

        try:
            await self.connect()

            uuid_id = self._to_uuid(memory_id)

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[uuid_id],
                wait=True,  # Wait for delete to complete for consistency
            )
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to delete memory {memory_id}: {e}",
                extra={"memory_id": memory_id, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to delete memory: {e}") from e

    async def count_memories(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count memories matching filters.

        Args:
            filters: Optional filter conditions

        Returns:
            Number of memories
        """
        await self.connect()

        if not filters:
            collection_info = await self.client.get_collection(self.collection_name)
            return collection_info.points_count

        # Count with filters
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            query_filter = Filter(must=conditions)

        response = await self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
        )

        return response.count

    async def close(self) -> None:
        """Close the connection to Qdrant."""
        if self.client is not None:
            await self.client.close()
            self.client = None
