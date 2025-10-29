"""
Tests for MemoryEvolutionService.

Tests memory versioning, evolution analysis, and time travel queries.
"""

import pytest

from src.services.memory_evolution import MemoryEvolutionService


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryEvolutionServiceUnit:
    """Unit tests for MemoryEvolutionService."""

    async def test_initialization(self, mock_llm, neo4j_graph_store, mock_embedder):
        """Test service initialization."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        assert service.llm == mock_llm
        assert service.graph_store == neo4j_graph_store
        assert service.embedder == mock_embedder

    async def test_analyze_evolution(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memory
    ):
        """Test evolution analysis."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        # Analyze evolution
        analysis = await service._analyze_evolution(sample_memory, "Updated information")

        assert analysis is not None
        assert hasattr(analysis, "action")
        assert hasattr(analysis, "reasoning")
        assert hasattr(analysis, "change_description")
        assert hasattr(analysis, "confidence")
        assert analysis.action in ["update", "augment", "replace", "preserve"]


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMemoryEvolutionServiceNeo4j:
    """Integration tests with Neo4j."""

    async def test_evolve_memory_neo4j(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memory
    ):
        """Test memory evolution with Neo4j."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        await neo4j_graph_store.add_node(sample_memory)

        # Evolve memory
        result = await service.evolve_memory(sample_memory, "Updated content")

        assert result is not None
        assert result.current_version == sample_memory.id

    async def test_get_version_history_neo4j(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memory
    ):
        """Test version history with Neo4j."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        await neo4j_graph_store.add_node(sample_memory)

        # Create version
        result = await service.evolve_memory(sample_memory, "New version")

        if result.new_version:
            # Get history
            history = await service.get_version_history(result.new_version)

            assert history is not None
            assert len(history.versions) >= 1

    async def test_rollback_to_version_neo4j(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memory
    ):
        """Test rollback with Neo4j."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        await neo4j_graph_store.add_node(sample_memory)

        result = await service.evolve_memory(sample_memory, "Version 2")

        if result.new_version:
            # Rollback
            rolled_back = await service.rollback_to_version(sample_memory.id)

            assert rolled_back is not None
            assert rolled_back.content == sample_memory.content

    async def test_time_travel_query_neo4j(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memories
    ):
        """Test time travel query with Neo4j (temporal-only mode)."""
        from datetime import datetime, timedelta

        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        # Add memories with different time ranges
        now = datetime.now()
        for i, memory in enumerate(sample_memories[:3]):
            # Set temporal validity
            memory.valid_from = now - timedelta(days=10 - i)
            memory.valid_until = None if i == 2 else now - timedelta(days=5 - i)
            await neo4j_graph_store.add_node(memory)

        # Query at a specific time (7 days ago) - temporal only
        as_of = now - timedelta(days=7)
        results = await service.time_travel_query(
            query_embedding=None,  # No semantic search
            as_of=as_of,
            limit=10,
            use_semantic_search=False,
        )

        assert isinstance(results, list)
        # Should only return memories that were valid at that time
        for memory in results:
            assert memory.valid_from <= as_of
            assert memory.valid_until is None or memory.valid_until > as_of

    async def test_time_travel_query_with_semantic_search(
        self, mock_llm, neo4j_graph_store, mock_embedder, mock_vector_store, sample_memories
    ):
        """Test time travel query with semantic search enabled."""
        from datetime import datetime, timedelta

        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )

        # Add memories with different time ranges
        now = datetime.now()
        for i, memory in enumerate(sample_memories[:3]):
            # Set temporal validity
            memory.valid_from = now - timedelta(days=10 - i)
            memory.valid_until = None if i == 2 else now - timedelta(days=5 - i)
            await neo4j_graph_store.add_node(memory)
            await mock_vector_store.upsert_memory(memory)

        # Query at a specific time with semantic search
        as_of = now - timedelta(days=7)
        query_embedding = [0.1] * 768
        results = await service.time_travel_query(
            query_embedding=query_embedding,
            as_of=as_of,
            limit=10,
            use_semantic_search=True,
        )

        assert isinstance(results, list)
        # Should return semantically similar memories valid at that time
        for memory in results:
            assert memory.valid_from <= as_of
            assert memory.valid_until is None or memory.valid_until > as_of

    async def test_time_travel_query_fallback_without_vector_store(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memories
    ):
        """Test time travel query falls back to temporal-only when vector_store is None."""
        from datetime import datetime, timedelta

        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
            vector_store=None,  # No vector store
        )

        # Add memories
        now = datetime.now()
        for i, memory in enumerate(sample_memories[:3]):
            memory.valid_from = now - timedelta(days=10 - i)
            memory.valid_until = None if i == 2 else now - timedelta(days=5 - i)
            await neo4j_graph_store.add_node(memory)

        # Try semantic search but should fallback to temporal-only
        as_of = now - timedelta(days=7)
        results = await service.time_travel_query(
            query_embedding=[0.1] * 768,
            as_of=as_of,
            limit=10,
            use_semantic_search=True,  # Requested but will fallback
        )

        assert isinstance(results, list)
        # Should still work with temporal filtering
        for memory in results:
            assert memory.valid_from <= as_of
            assert memory.valid_until is None or memory.valid_until > as_of

    async def test_get_current_version_neo4j(
        self, mock_llm, neo4j_graph_store, mock_embedder, sample_memory
    ):
        """Test getting current version with Neo4j."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=neo4j_graph_store,
            embedder=mock_embedder,
        )

        await neo4j_graph_store.add_node(sample_memory)

        # Create multiple versions
        result1 = await service.evolve_memory(sample_memory, "Version 2")

        if result1.new_version:
            # Get current should return the latest version
            current = await service.get_current_version(sample_memory.id)
            assert current is not None
            assert current.id == result1.new_version
