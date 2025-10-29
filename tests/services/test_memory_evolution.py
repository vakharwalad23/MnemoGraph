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
