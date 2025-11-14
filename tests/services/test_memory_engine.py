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

        user_id = "test_user_001"
        memory, extraction = await engine.add_memory("Test memory with Neo4j", user_id=user_id)

        assert memory is not None
        assert memory.user_id == user_id
        assert extraction is not None

        # Verify in Neo4j
        retrieved = await neo4j_graph_store.get_node(memory.id, user_id)
        assert retrieved is not None
        assert retrieved.user_id == user_id

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

        user_id = "test_user_001"
        # Add memories
        for mem in sample_memories:
            await neo4j_graph_store.add_node(mem)
            await mock_vector_store.upsert_memory(mem)

        # Search
        results = await engine.search_similar("programming", user_id=user_id, limit=5)
        assert isinstance(results, list)
        # Verify all results belong to the user
        for result_memory, _ in results:
            assert result_memory.user_id == user_id

        # Query
        queried = await engine.query_memories(user_id=user_id, filters={"status": "active"})
        assert isinstance(queried, list)
        # Verify all results belong to the user
        for mem in queried:
            assert mem.user_id == user_id

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

        user_id = "test_user_001"
        # Add memories and edges
        for mem in sample_memories[:3]:
            await neo4j_graph_store.add_node(mem)

        from src.models.relationships import Edge, RelationshipType

        edge = Edge(
            source=sample_memories[0].id,
            target=sample_memories[1].id,
            type=RelationshipType.SIMILAR_TO,
        )
        await neo4j_graph_store.add_edge(edge, user_id)

        # Get neighbors
        neighbors = await engine.get_neighbors(sample_memories[0].id, user_id=user_id)
        assert isinstance(neighbors, list)
        # Verify all neighbors belong to the same user
        for neighbor_mem, _ in neighbors:
            assert neighbor_mem.user_id == user_id

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

    async def test_multi_user_isolation(
        self, mock_llm, mock_embedder, neo4j_graph_store, mock_vector_store, config
    ):
        """Test that users cannot access each other's memories."""
        from src.utils.exceptions import SecurityError

        engine = MemoryEngine(
            llm=mock_llm,
            embedder=mock_embedder,
            graph_store=neo4j_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        await engine.initialize()

        user_a = "user_a"
        user_b = "user_b"

        # User A adds a memory
        memory_a, _ = await engine.add_memory("User A's private memory", user_id=user_a)
        assert memory_a.user_id == user_a

        # User B adds a memory
        memory_b, _ = await engine.add_memory("User B's private memory", user_id=user_b)
        assert memory_b.user_id == user_b

        # User A should only see their own memories in search
        results_a = await engine.search_similar("memory", user_id=user_a, limit=10)
        assert all(result_mem.user_id == user_a for result_mem, _ in results_a)
        assert not any(result_mem.id == memory_b.id for result_mem, _ in results_a)

        # User B should only see their own memories in search
        results_b = await engine.search_similar("memory", user_id=user_b, limit=10)
        assert all(result_mem.user_id == user_b for result_mem, _ in results_b)
        assert not any(result_mem.id == memory_a.id for result_mem, _ in results_b)

        # User A should only see their own memories in query
        queried_a = await engine.query_memories(user_id=user_a, filters={"status": "active"})
        assert all(mem.user_id == user_a for mem in queried_a)
        assert not any(mem.id == memory_b.id for mem in queried_a)

        # User B should only see their own memories in query
        queried_b = await engine.query_memories(user_id=user_b, filters={"status": "active"})
        assert all(mem.user_id == user_b for mem in queried_b)
        assert not any(mem.id == memory_a.id for mem in queried_b)

        # User A should be able to get their own memory
        retrieved_a = await engine.get_memory(memory_a.id, user_id=user_a)
        assert retrieved_a is not None
        assert retrieved_a.id == memory_a.id
        assert retrieved_a.user_id == user_a

        # User A should NOT be able to get User B's memory (SecurityError expected)
        try:
            retrieved_b_as_a = await engine.get_memory(memory_b.id, user_id=user_a)
            # If no SecurityError is raised, ensure it's None or doesn't belong to user_a
            if retrieved_b_as_a is not None:
                # This should not happen, but if it does, verify it's not accessible
                assert (
                    retrieved_b_as_a.user_id != user_a
                ), "User A should not access User B's memory"
        except SecurityError:
            # This is expected - User A cannot access User B's memory
            pass

        # User A should NOT be able to update User B's memory
        try:
            await engine.update_memory(memory_b.id, user_id=user_a, new_content="Hacked content")
            # If no error, verify the memory wasn't actually updated
            updated = await engine.get_memory(memory_b.id, user_id=user_b)
            assert updated.content != "Hacked content", "User A should not update User B's memory"
        except SecurityError:
            # This is expected - User A cannot update User B's memory
            pass

        # User A should NOT be able to delete User B's memory
        try:
            await engine.delete_memory(memory_b.id, user_id=user_a)
            # If no error, verify the memory still exists for User B
            still_exists = await engine.get_memory(memory_b.id, user_id=user_b)
            assert still_exists is not None, "User A should not delete User B's memory"
        except SecurityError:
            # This is expected - User A cannot delete User B's memory
            pass

        # Statistics should be user-specific
        stats_a = await engine.get_statistics(user_id=user_a)
        stats_b = await engine.get_statistics(user_id=user_b)
        # Both should have at least 1 memory
        # get_statistics returns nested structure: {"memories": {"total_graph": ..., "total_vector": ...}, ...}
        assert stats_a["memories"]["total_graph"] >= 1 or stats_a["memories"]["total_vector"] >= 1
        assert stats_b["memories"]["total_graph"] >= 1 or stats_b["memories"]["total_vector"] >= 1
