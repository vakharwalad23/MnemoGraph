"""
Tests for MemoryEngine service.

Tests the unified memory engine integrating all components.
"""

import pytest

from src.services.memory_engine import MemoryEngine


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryEngineUnit:
    """Unit tests for MemoryEngine."""

    async def test_initialization(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test engine initialization."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        assert engine.llm == mock_llm
        assert engine.embedder == mock_embedder
        assert engine.graph_store == neo4j_graph_store
        assert engine.vector_store == mock_vector_store
        assert engine.config == config

    async def test_generate_id(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test ID generation."""
        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
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
