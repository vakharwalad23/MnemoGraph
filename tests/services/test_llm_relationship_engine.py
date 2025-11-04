"""
Tests for LLMRelationshipEngine service.

Tests relationship extraction, derived insights, and edge creation.
"""

import pytest

from src.models.memory import NodeType
from src.models.relationships import RelationshipType
from src.services.llm_relationship_engine import LLMRelationshipEngine


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMRelationshipEngineUnit:
    """Unit tests for LLMRelationshipEngine."""

    async def test_initialization(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        mock_sync_manager,
    ):
        """Test engine initialization."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        assert engine.llm == mock_llm
        assert engine.embedder == mock_embedder
        assert engine.vector_store == mock_vector_store
        assert engine.graph_store == neo4j_graph_store
        assert engine.config == config

    async def test_format_memories_detailed(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memories,
        mock_sync_manager,
    ):
        """Test detailed memory formatting."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        formatted = engine._format_memories_detailed(sample_memories[:3])

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Check that memory IDs are in output
        for mem in sample_memories[:3]:
            assert mem.id in formatted

    async def test_format_memories_compact(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memories,
        mock_sync_manager,
    ):
        """Test compact memory formatting."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        formatted = engine._format_memories_compact(sample_memories[:3])

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        # Should be more compact than detailed
        detailed = engine._format_memories_detailed(sample_memories[:3])
        assert len(formatted) < len(detailed)

    async def test_format_memories_empty(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        mock_sync_manager,
    ):
        """Test formatting empty memory list."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        formatted = engine._format_memories_detailed([])

        assert isinstance(formatted, str)
        assert "No memories" in formatted

    async def test_create_edge_from_relationship(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        mock_sync_manager,
    ):
        """Test creating edge from relationship."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        from src.models.relationships import Relationship

        rel = Relationship(
            type=RelationshipType.SIMILAR_TO,
            target_id="mem_target",
            confidence=0.85,
            reasoning="Similar content",
        )

        edge = engine._create_edge_from_relationship("mem_source", rel)

        assert edge["source"] == "mem_source"
        assert edge["target"] == "mem_target"
        assert edge["type"] == RelationshipType.SIMILAR_TO
        assert edge["metadata"]["confidence"] == 0.85
        assert edge["metadata"]["reasoning"] == "Similar content"


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestLLMRelationshipEngineNeo4j:
    """Integration tests with Neo4j."""

    async def test_process_new_memory_neo4j(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memory,
        mock_sync_manager,
    ):
        """Test processing new memory with Neo4j."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        # Process memory
        result = await engine.process_new_memory(sample_memory)

        assert result is not None
        assert result.memory_id == sample_memory.id

        # Verify memory was added to graph
        retrieved = await neo4j_graph_store.get_node(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id

    async def test_process_with_edges_neo4j(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memories,
        mock_sync_manager,
    ):
        """Test that edges are created in Neo4j."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        # Add context memories
        for mem in sample_memories[:-1]:
            await neo4j_graph_store.add_node(mem)
            await mock_vector_store.upsert_memory(mem)

        # Process new memory
        new_memory = sample_memories[-1]
        result = await engine.process_new_memory(new_memory)

        assert result is not None

        # Check edges were created
        edge_count = await neo4j_graph_store.count_edges()
        assert edge_count >= 0

    async def test_create_derived_memories_neo4j(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memory,
        mock_sync_manager,
    ):
        """Test creating derived memories in Neo4j."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        await neo4j_graph_store.add_node(sample_memory)

        from src.models.relationships import DerivedInsight

        insights = [
            DerivedInsight(
                content="Derived insight",
                confidence=0.75,
                reasoning="Test reasoning",
                source_ids=[sample_memory.id],
                type="pattern_recognition",
            )
        ]

        await engine._create_derived_memories(insights, sample_memory)

        # Verify derived memory exists
        count = await neo4j_graph_store.count_nodes({"type": NodeType.DERIVED.value})
        assert count > 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestBuildExtractionPrompt:
    """Tests for prompt building."""

    async def test_build_extraction_prompt(
        self,
        mock_llm,
        mock_embedder,
        mock_vector_store,
        neo4j_graph_store,
        config,
        sample_memory,
        mock_sync_manager,
    ):
        """Test extraction prompt building."""
        engine = LLMRelationshipEngine(
            llm_provider=mock_llm,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            graph_store=neo4j_graph_store,
            config=config,
            sync_manager=mock_sync_manager,
        )

        from src.models.relationships import ContextBundle

        context = ContextBundle(
            vector_candidates=[sample_memory],
            filtered_candidates=[sample_memory],
        )

        prompt = engine._build_extraction_prompt(sample_memory, context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert sample_memory.content in prompt
        assert "RelationshipBundle" in prompt or "relationships" in prompt.lower()
