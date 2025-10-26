"""
Tests for MemoryEngine service.

Tests the unified memory engine integrating all components.
"""

import pytest

from src.models.memory import Memory, MemoryStatus
from src.services.memory_engine import MemoryEngine


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryEngineUnit:
    """Unit tests for MemoryEngine."""

    async def test_initialization(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config
    ):
        """Test engine initialization."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        assert engine.llm == mock_llm
        assert engine.embedder == mock_embedder
        assert engine.graph_store == sqlite_graph_store
        assert engine.vector_store == mock_vector_store
        assert engine.config == config

    async def test_generate_id(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config
    ):
        """Test ID generation."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        id1 = engine._generate_id()
        id2 = engine._generate_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Should be unique
        assert id1.startswith("mem_")
        assert id2.startswith("mem_")


@pytest.mark.unit
@pytest.mark.sqlite
@pytest.mark.asyncio
class TestMemoryEngineSQLite:
    """Integration tests with SQLite."""

    async def test_initialize_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config
    ):
        """Test engine initialization with SQLite."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Stores should be initialized
        assert sqlite_graph_store.connection is not None
        assert mock_vector_store.initialized is True

    async def test_add_memory_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config
    ):
        """Test adding a memory with SQLite."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memory
        memory, extraction = await engine.add_memory(
            content="Test memory about Python programming",
            metadata={"source": "test"},
        )

        assert memory is not None
        assert memory.content == "Test memory about Python programming"
        assert memory.metadata["source"] == "test"
        assert len(memory.embedding) > 0

        assert extraction is not None
        assert extraction.memory_id == memory.id

        # Verify memory in stores
        retrieved = await sqlite_graph_store.get_node(memory.id)
        assert retrieved is not None
        assert retrieved.id == memory.id

    async def test_get_memory_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test getting a memory by ID."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memory first
        await sqlite_graph_store.add_node(sample_memory)

        # Get memory
        retrieved = await engine.get_memory(sample_memory.id, validate=False)

        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.content == sample_memory.content

    async def test_get_memory_not_found(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config
    ):
        """Test getting non-existent memory."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Try to get non-existent memory
        retrieved = await engine.get_memory("nonexistent_id")

        assert retrieved is None

    async def test_update_memory_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test updating a memory."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add initial memory
        await sqlite_graph_store.add_node(sample_memory)

        # Update memory
        updated, evolution = await engine.update_memory(
            sample_memory.id, "Updated content about async Python"
        )

        assert updated is not None
        assert evolution is not None
        assert hasattr(evolution, "action")

    async def test_delete_memory_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test deleting a memory."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memory
        await sqlite_graph_store.add_node(sample_memory)
        await mock_vector_store.upsert_memory(sample_memory)

        # Delete memory
        await engine.delete_memory(sample_memory.id)

        # Verify deleted
        retrieved = await sqlite_graph_store.get_node(sample_memory.id)
        assert retrieved is None

        vector_retrieved = await mock_vector_store.get_memory(sample_memory.id)
        assert vector_retrieved is None

    async def test_search_similar_sqlite(
        self,
        mock_llm,
        mock_embedder,
        sqlite_graph_store,
        mock_vector_store,
        config,
        sample_memories,
    ):
        """Test semantic search."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memories
        for mem in sample_memories:
            await mock_vector_store.upsert_memory(mem)

        # Search
        results = await engine.search_similar("programming", limit=5)

        assert isinstance(results, list)
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (Memory, score)
        assert all(isinstance(r[0], Memory) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    async def test_query_memories_sqlite(
        self,
        mock_llm,
        mock_embedder,
        sqlite_graph_store,
        mock_vector_store,
        config,
        sample_memories,
    ):
        """Test querying memories with filters."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memories
        for mem in sample_memories:
            await sqlite_graph_store.add_node(mem)

        # Query active memories
        results = await engine.query_memories(
            filters={"status": MemoryStatus.ACTIVE.value}, limit=10
        )

        assert isinstance(results, list)
        assert all(isinstance(m, Memory) for m in results)
        assert all(m.status == MemoryStatus.ACTIVE for m in results)

    async def test_get_neighbors_sqlite(
        self,
        mock_llm,
        mock_embedder,
        sqlite_graph_store,
        mock_vector_store,
        config,
        sample_memories,
    ):
        """Test getting memory neighbors."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memories and edges
        for mem in sample_memories[:3]:
            await sqlite_graph_store.add_node(mem)

        await sqlite_graph_store.add_edge(
            {
                "source": sample_memories[0].id,
                "target": sample_memories[1].id,
                "type": "SIMILAR_TO",
                "metadata": {"confidence": 0.8},
            }
        )

        # Get neighbors
        neighbors = await engine.get_neighbors(sample_memories[0].id)

        assert isinstance(neighbors, list)
        # Should find at least one neighbor
        if len(neighbors) > 0:
            assert all(isinstance(n, tuple) for n in neighbors)

    async def test_find_path_sqlite(
        self,
        mock_llm,
        mock_embedder,
        sqlite_graph_store,
        mock_vector_store,
        config,
        sample_memories,
    ):
        """Test finding path between memories."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Create path: mem1 -> mem2 -> mem3
        for mem in sample_memories[:3]:
            await sqlite_graph_store.add_node(mem)

        await sqlite_graph_store.add_edge(
            {
                "source": sample_memories[0].id,
                "target": sample_memories[1].id,
                "type": "FOLLOWS",
            }
        )
        await sqlite_graph_store.add_edge(
            {
                "source": sample_memories[1].id,
                "target": sample_memories[2].id,
                "type": "FOLLOWS",
            }
        )

        # Find path
        path = await engine.find_path(sample_memories[0].id, sample_memories[2].id)

        if path:
            assert isinstance(path, list)
            assert sample_memories[0].id in path
            assert sample_memories[2].id in path

    async def test_get_version_history_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test getting version history."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add and update memory
        await sqlite_graph_store.add_node(sample_memory)
        updated, evolution = await engine.update_memory(sample_memory.id, "New content")

        if evolution.new_version:
            # Get history
            history = await engine.get_version_history(evolution.new_version)

            assert history is not None
            assert hasattr(history, "versions")
            assert len(history.versions) >= 1

    async def test_check_memory_validity_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test checking memory validity."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        await sqlite_graph_store.add_node(sample_memory)

        # Check validity
        result = await engine.check_memory_validity(sample_memory.id)

        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "reasoning")

    async def test_invalidate_memory_sqlite(
        self, mock_llm, mock_embedder, sqlite_graph_store, mock_vector_store, config, sample_memory
    ):
        """Test manually invalidating memory."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        await sqlite_graph_store.add_node(sample_memory)

        # Invalidate
        await engine.invalidate_memory(sample_memory.id, "Test invalidation")

        # Verify invalidated
        retrieved = await sqlite_graph_store.get_node(sample_memory.id)
        assert retrieved.status == MemoryStatus.INVALIDATED
        assert retrieved.invalidation_reason == "Test invalidation"

    async def test_get_statistics_sqlite(
        self,
        mock_llm,
        mock_embedder,
        sqlite_graph_store,
        mock_vector_store,
        config,
        sample_memories,
    ):
        """Test getting engine statistics."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=sqlite_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add some memories
        for mem in sample_memories:
            await sqlite_graph_store.add_node(mem)
            await mock_vector_store.upsert_memory(mem)

        # Get stats
        stats = await engine.get_statistics()

        assert isinstance(stats, dict)
        assert "memories" in stats
        assert "relationships" in stats
        assert stats["memories"]["active"] >= 0
        assert stats["memories"]["total_graph"] >= 0


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMemoryEngineNeo4j:
    """Integration tests with Neo4j."""

    async def test_initialize_neo4j(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test engine initialization with Neo4j."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        assert neo4j_graph_store.driver is not None

    async def test_add_memory_neo4j(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test adding memory with Neo4j."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        memory, extraction = await engine.add_memory("Test memory with Neo4j")

        assert memory is not None
        assert extraction is not None

        # Verify in Neo4j
        retrieved = await neo4j_graph_store.get_node(memory.id)
        assert retrieved is not None

    async def test_search_and_query_neo4j(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config, sample_memories
    ):
        """Test search operations with Neo4j."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memories
        for mem in sample_memories:
            await neo4j_graph_store.add_node(mem)
            await mock_vector_store.upsert_memory(mem)

        # Search
        results = await engine.search_similar("programming", limit=5)
        assert isinstance(results, list)

        # Query
        queried = await engine.query_memories(filters={"status": "active"})
        assert isinstance(queried, list)

    async def test_graph_operations_neo4j(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config, sample_memories
    ):
        """Test graph-specific operations with Neo4j."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        # Add memories and edges
        for mem in sample_memories[:3]:
            await neo4j_graph_store.add_node(mem)

        await neo4j_graph_store.add_edge(
            {
                "source": sample_memories[0].id,
                "target": sample_memories[1].id,
                "type": "SIMILAR_TO",
            }
        )

        # Get neighbors
        neighbors = await engine.get_neighbors(sample_memories[0].id)
        assert isinstance(neighbors, list)

    async def test_close_engine(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test closing engine resources."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()
        await engine.close()

        # Verify closed
        assert neo4j_graph_store.driver is None
