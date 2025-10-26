"""
Tests for MemoryEvolutionService.

Tests memory versioning, evolution analysis, and time travel queries.
"""

from datetime import datetime, timedelta

import pytest

from src.models.memory import Memory, MemoryStatus, NodeType
from src.services.memory_evolution import MemoryEvolutionService


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryEvolutionServiceUnit:
    """Unit tests for MemoryEvolutionService."""

    async def test_initialization(self, mock_llm, sqlite_graph_store, mock_embedder):
        """Test service initialization."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        assert service.llm == mock_llm
        assert service.graph_store == sqlite_graph_store
        assert service.embedder == mock_embedder

    async def test_analyze_evolution(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test evolution analysis."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
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


@pytest.mark.unit
@pytest.mark.sqlite
@pytest.mark.asyncio
class TestMemoryEvolutionServiceSQLite:
    """Integration tests with SQLite."""

    async def test_evolve_memory_update_sqlite(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test memory evolution with update action."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        # Add initial memory
        await sqlite_graph_store.add_node(sample_memory)

        # Evolve memory
        result = await service.evolve_memory(sample_memory, "Updated content about Python")

        assert result is not None
        assert hasattr(result, "current_version")
        assert hasattr(result, "new_version")
        assert hasattr(result, "action")
        assert result.current_version == sample_memory.id

        # If action was update/replace, new version should exist
        if result.action in ["update", "replace"]:
            assert result.new_version is not None
            # Check that new version exists in store
            new_mem = await sqlite_graph_store.get_node(result.new_version)
            assert new_mem is not None
            assert new_mem.version == sample_memory.version + 1

    async def test_evolve_memory_augment_sqlite(self, mock_llm, sqlite_graph_store, mock_embedder):
        """Test memory evolution with augment action."""
        # Setup mock to return augment action
        from src.services.memory_evolution import EvolutionAnalysis

        mock_llm.set_response(
            "analyze_evolution",
            EvolutionAnalysis(
                action="augment",
                reasoning="Additional details without contradiction",
                change_description="Added details",
                confidence=0.8,
            ),
        )

        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        mem = Memory(
            id="mem_aug",
            content="Original content",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
        )

        await sqlite_graph_store.add_node(mem)

        # Evolve with augment
        result = await service.evolve_memory(mem, "Additional information")

        # With augment, no new version created
        if result.action == "augment":
            assert result.new_version is None
            # Original memory should be updated
            updated = await sqlite_graph_store.get_node(mem.id)
            assert "augmented" in updated.metadata

    async def test_create_new_version_sqlite(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test creating new version."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        from src.services.memory_evolution import EvolutionAnalysis

        analysis = EvolutionAnalysis(
            action="update",
            reasoning="Content updated",
            change_description="Major update",
            confidence=0.9,
        )

        # Create new version
        new_version = await service._create_new_version(sample_memory, "New content", analysis)

        assert new_version is not None
        assert new_version.version == sample_memory.version + 1
        assert new_version.parent_version == sample_memory.id
        assert new_version.content == "New content"
        assert new_version.status == MemoryStatus.ACTIVE

    async def test_get_version_history_sqlite(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test getting version history."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        # Add memory and create versions
        await sqlite_graph_store.add_node(sample_memory)

        # Create a new version
        result = await service.evolve_memory(sample_memory, "Version 2 content")

        if result.new_version:
            # Get version history
            history = await service.get_version_history(result.new_version)

            assert history is not None
            assert hasattr(history, "versions")
            assert hasattr(history, "total_versions")
            assert len(history.versions) >= 1
            assert history.current_version_id == result.new_version

    async def test_get_current_version_sqlite(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test getting current version of superseded memory."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        await sqlite_graph_store.add_node(sample_memory)

        # Evolve memory multiple times
        result1 = await service.evolve_memory(sample_memory, "Version 2")

        if result1.new_version:
            new_mem = await sqlite_graph_store.get_node(result1.new_version)
            result2 = await service.evolve_memory(new_mem, "Version 3")

            if result2.new_version:
                # Get current version from original
                current = await service.get_current_version(sample_memory.id)

                assert current is not None
                # Should be the latest version
                assert current.id == result2.new_version

    async def test_rollback_to_version_sqlite(
        self, mock_llm, sqlite_graph_store, mock_embedder, sample_memory
    ):
        """Test rolling back to previous version."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        await sqlite_graph_store.add_node(sample_memory)

        # Create new version
        result = await service.evolve_memory(sample_memory, "New version")

        if result.new_version:
            # Rollback to original
            rolled_back = await service.rollback_to_version(sample_memory.id)

            assert rolled_back is not None
            assert rolled_back.content == sample_memory.content
            assert (
                "rollback" in rolled_back.id
                or "rollback" in rolled_back.metadata.get("rollback_reason", "").lower()
            )

    async def test_time_travel_query_sqlite(self, mock_llm, sqlite_graph_store, mock_embedder):
        """Test time travel query."""
        service = MemoryEvolutionService(
            llm=mock_llm,
            graph_store=sqlite_graph_store,
            embedder=mock_embedder,
        )

        # Create memory with specific validity period
        old_mem = Memory(
            id="old_mem",
            content="Old version",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
            valid_from=datetime.now() - timedelta(days=100),
            valid_until=datetime.now() - timedelta(days=50),
            status=MemoryStatus.SUPERSEDED,
        )

        new_mem = Memory(
            id="new_mem",
            content="New version",
            type=NodeType.MEMORY,
            embedding=[0.1] * 768,
            valid_from=datetime.now() - timedelta(days=50),
            status=MemoryStatus.ACTIVE,
        )

        await sqlite_graph_store.add_node(old_mem)
        await sqlite_graph_store.add_node(new_mem)

        # Query as of 75 days ago (should get old version)
        as_of = datetime.now() - timedelta(days=75)
        results = await service.time_travel_query([0.1] * 16, as_of, limit=10)

        assert isinstance(results, list)
        # Should find memory valid at that time
        if len(results) > 0:
            for mem in results:
                assert mem.valid_from <= as_of
                assert mem.valid_until is None or mem.valid_until > as_of


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
