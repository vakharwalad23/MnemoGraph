"""
Unified Memory Engine - Integrates all components.

Brings together:
- LLM & Embedder providers
- Memory Evolution & Invalidation
- Scalable Relationship Extraction
- Graph Store & Vector Store
"""

from datetime import datetime
from typing import Any

from src.config import Config
from src.core.embeddings.base import Embedder
from src.core.graph_store.base import GraphStore
from src.core.llm.base import LLMProvider
from src.core.vector_store.base import VectorStore
from src.models.memory import Memory, NodeType
from src.models.relationships import Edge, RelationshipBundle
from src.models.version import InvalidationResult, MemoryEvolution, VersionChain
from src.services.invalidation_manager import InvalidationManager
from src.services.llm_relationship_engine import LLMRelationshipEngine
from src.services.memory_evolution import MemoryEvolutionService
from src.services.memory_sync import MemorySyncManager


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
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.config = config

        # Initialize sync manager first
        self.sync_manager = MemorySyncManager(
            graph_store=graph_store,
            vector_store=vector_store,
            max_retries=3,
            retry_delay=0.5,
        )

        # Initialize Phase 2 & 3 services
        self.evolution = MemoryEvolutionService(
            llm=llm,
            graph_store=graph_store,
            embedder=embedder,
            vector_store=vector_store,
            sync_manager=self.sync_manager,
        )

        self.invalidation = InvalidationManager(
            llm=llm,
            graph_store=graph_store,
            vector_store=vector_store,
            sync_manager=self.sync_manager,
        )

        self.relationship_engine = LLMRelationshipEngine(
            llm_provider=llm,
            embedder=embedder,
            vector_store=vector_store,
            graph_store=graph_store,
            config=config,
            sync_manager=self.sync_manager,
        )

    async def initialize(self) -> None:
        """Initialize all stores."""
        print("🔧 Initializing Memory Engine...")

        await self.graph_store.initialize()
        print("   ✓ Graph store initialized")

        await self.vector_store.initialize()
        print("   ✓ Vector store initialized")

        # Start background invalidation worker if enabled
        if self.config.llm_relationships.enable_auto_invalidation:
            self.invalidation.start_background_worker(interval_hours=24)
            print("   ✓ Background invalidation worker started")

        print("✅ Memory Engine ready!")

    # ═══════════════════════════════════════════════════════════
    # CORE MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════

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
        """
        print(f"\n📝 Adding memory: {content[:50]}...")

        # Generate embedding
        print("   🔢 Generating embedding...")
        embedding = await self.embedder.embed(content)

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
        print("   🔗 Extracting relationships...")
        extraction = await self.relationship_engine.process_new_memory(memory)

        print(f"✅ Memory added: {memory.id}")
        print(f"   Relationships: {len(extraction.relationships)}")
        print(f"   Derived insights: {len(extraction.derived_insights)}")

        return memory, extraction

    async def get_memory(self, memory_id: str, validate: bool = True) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier
            validate: Whether to check validity on access

        Returns:
            Memory or None if not found
        """
        # Try graph store first (has full data)
        memory = await self.graph_store.get_node(memory_id)

        if not memory:
            return None

        # Validate on access if enabled
        if validate and self.config.llm_relationships.enable_auto_invalidation:
            memory = await self.invalidation.validate_on_access(memory)

        return memory

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
        """
        print(f"\n🔄 Updating memory: {memory_id}")

        # Get current memory
        current = await self.graph_store.get_node(memory_id)
        if not current:
            raise ValueError(f"Memory not found: {memory_id}")

        # Evolve memory (Phase 2)
        evolution = await self.evolution.evolve_memory(current, new_content)

        # If new version created, process relationships
        if evolution.new_version:
            new_memory = await self.graph_store.get_node(evolution.new_version)

            # Generate new embedding
            new_memory.embedding = await self.embedder.embed(new_content)

            # Update both stores with sync manager
            await self.graph_store.update_node(new_memory)
            await self.sync_manager.sync_memory_full(new_memory)

            print(f"✅ Memory updated with new version: {evolution.new_version}")

            return new_memory, evolution

        # For augment, sync the updated memory
        updated_memory = await self.graph_store.get_node(memory_id)
        if updated_memory:
            await self.sync_manager.sync_memory_full(updated_memory)

        print(f"✅ Memory augmented: {memory_id}")
        return current, evolution

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from both stores.

        Args:
            memory_id: Memory to delete
        """
        print(f"🗑️  Deleting memory: {memory_id}")

        # Delete from graph store first (source of truth)
        await self.graph_store.delete_node(memory_id)

        # Sync deletion to vector store
        await self.sync_manager.sync_memory_deletion(memory_id)

        print(f"✅ Memory deleted: {memory_id}")

    # ═══════════════════════════════════════════════════════════
    # SEARCH & QUERY OPERATIONS
    # ═══════════════════════════════════════════════════════════

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Search for similar memories by semantic meaning.

        Args:
            query: Query text
            limit: Maximum results
            filters: Optional filters (status, type, etc.)
            score_threshold: Minimum similarity score

        Returns:
            List of (Memory, score) tuples
        """
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Search vector store
        results = await self.vector_store.search_similar(
            vector=query_embedding,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )

        return [(result.memory, result.score) for result in results]

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
    ) -> list[tuple[Memory, Edge]]:
        """
        Get related memories via graph relationships.

        Args:
            memory_id: Starting memory
            relationship_types: Filter by types
            direction: "outgoing", "incoming", or "both"
            depth: Traversal depth
            limit: Maximum results

        Returns:
            List of tuples with (Memory, Edge)
        """
        return await self.graph_store.get_neighbors(
            node_id=memory_id,
            relationship_types=relationship_types,
            direction=direction,
            depth=depth,
            limit=limit,
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
        """
        return await self.graph_store.find_path(start_id, end_id, max_depth)

    # ═══════════════════════════════════════════════════════════
    # VERSIONING & HISTORY OPERATIONS
    # ═══════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════
    # INVALIDATION OPERATIONS
    # ═══════════════════════════════════════════════════════════

    async def check_memory_validity(self, memory_id: str) -> InvalidationResult:
        """
        Check if a memory is still valid.

        Args:
            memory_id: Memory to check

        Returns:
            InvalidationResult
        """
        memory = await self.graph_store.get_node(memory_id)
        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")

        return await self.invalidation.check_invalidation(memory)

    async def invalidate_memory(self, memory_id: str, reason: str) -> None:
        """
        Manually invalidate a memory.

        Args:
            memory_id: Memory to invalidate
            reason: Invalidation reason
        """
        memory = await self.graph_store.get_node(memory_id)
        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")

        from src.models.version import InvalidationResult

        result = InvalidationResult(
            memory_id=memory_id,
            status="invalidated",
            reasoning=reason,
            confidence=1.0,
        )

        await self.invalidation._mark_invalidated(memory, result)

    # ═══════════════════════════════════════════════════════════
    # STATISTICS & MONITORING
    # ═══════════════════════════════════════════════════════════

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get memory engine statistics.

        Returns:
            Statistics dictionary
        """
        # Count nodes by status
        active_count = await self.graph_store.count_nodes({"status": "active"})
        historical_count = await self.graph_store.count_nodes({"status": "historical"})
        superseded_count = await self.graph_store.count_nodes({"status": "superseded"})

        # Count edges
        total_edges = await self.graph_store.count_edges()

        # Count vector store memories
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

    # ═══════════════════════════════════════════════════════════
    # LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    async def close(self) -> None:
        """Close all connections and stop workers."""
        print("🛑 Shutting down Memory Engine...")

        # Stop background workers
        self.invalidation.stop_background_worker()

        # Close stores
        await self.graph_store.close()
        await self.vector_store.close()

        # Close providers
        await self.llm.close()
        await self.embedder.close()

        print("✅ Memory Engine shutdown complete")

    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        from uuid import uuid4

        return f"mem_{uuid4().hex[:12]}"
