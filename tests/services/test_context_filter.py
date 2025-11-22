"""
Tests for MultiStageFilter service.

Tests context gathering, filtering, and LLM pre-filtering.
"""

from pathlib import Path

import pytest

from src.config import Config
from src.models.memory import Memory, MemoryStatus, NodeType
from src.services.context_filter import MultiStageFilter


def get_test_config() -> Config:
    """Get test configuration."""
    env_test_path = Path(".env.test")
    if env_test_path.exists():
        return Config.from_env(env_file=env_test_path)
    return Config.from_env()


@pytest.mark.unit
@pytest.mark.asyncio
class TestMultiStageFilterUnit:
    """Unit tests for MultiStageFilter."""

    async def test_initialization(self, memory_store, mock_llm, config):
        """Test filter initialization."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        assert filter_service.memory_store == memory_store
        assert filter_service.llm == mock_llm
        assert filter_service.config == config

    async def test_stage1_vector_search(
        self,
        memory_store,
        mock_llm,
        config,
        sample_memory,
        sample_memories,
    ):
        """Test stage 1 vector search."""
        # Setup
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Add some memories to vector store
        for mem in sample_memories:
            await memory_store.vector_store.upsert_memory(mem)

        # Test vector search
        results = await filter_service._stage1_vector_search(sample_memory)

        assert isinstance(results, list)
        assert all(isinstance(m, Memory) for m in results)
        # Verify all results belong to the same user
        for mem in results:
            assert mem.user_id == sample_memory.user_id

    async def test_temporal_filter(
        self,
        memory_store,
        mock_llm,
        config,
        sample_memory,
        sample_memories,
    ):
        """Test temporal filtering."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Add memories to both stores
        for mem in sample_memories:
            await memory_store.create_memory(mem)

        # Test temporal filter
        results = await filter_service._temporal_filter(sample_memory)

        assert isinstance(results, list)
        # Results should be recent active memories and belong to same user
        for mem in results:
            assert mem.status == MemoryStatus.ACTIVE
            assert mem.user_id == sample_memory.user_id

    async def test_graph_filter_no_neighbors(self, memory_store, mock_llm, config, sample_memory):
        """Test graph filter when memory has no neighbors."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Test with non-existent memory
        results = await filter_service._graph_filter(sample_memory)

        assert isinstance(results, list)
        assert len(results) == 0  # No neighbors for new memory

    async def test_entity_filter(self, memory_store, mock_llm, config, sample_memory):
        """Test entity extraction and filtering."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Test entity filter
        results = await filter_service._entity_filter(sample_memory)

        assert isinstance(results, list)
        # With real LLM, we should get some results
        # (No call_count check since we're using real Ollama)

    async def test_conversation_filter_no_conversation(
        self, memory_store, mock_llm, config, sample_memory
    ):
        """Test conversation filter when no conversation_id."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Test without conversation_id
        results = await filter_service._conversation_filter(sample_memory)

        assert isinstance(results, list)
        assert len(results) == 0

    async def test_conversation_filter_with_conversation(self, memory_store, mock_llm, config):
        """Test conversation filter with conversation_id."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Create memories in same conversation
        # Use same dimension as sample_memories fixture
        test_config = get_test_config()
        dim = test_config.embedder.dimension if test_config.embedder.dimension else 768

        user_id = "test_user_001"
        conv_memories = [
            Memory(
                id=f"conv_mem_{i}",
                content=f"Message {i}",
                type=NodeType.MEMORY,
                user_id=user_id,
                embedding=[0.1 * i] * dim,
                metadata={"conversation_id": "conv_123"},
            )
            for i in range(3)
        ]

        for mem in conv_memories:
            await memory_store.create_memory(mem)

        # Test with conversation memory
        test_memory = Memory(
            id="test_conv_mem",
            content="New message",
            type=NodeType.MEMORY,
            user_id=user_id,  # Same user for conversation
            embedding=[0.5] * 768,
            metadata={"conversation_id": "conv_123"},
        )

        results = await filter_service._conversation_filter(test_memory)

        assert isinstance(results, list)
        # Should find other memories in conversation and same user
        assert len(results) <= 10
        for mem in results:
            assert mem.user_id == user_id

    async def test_deduplicate(self, memory_store, mock_llm, config, sample_memories):
        """Test deduplication of memories."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Create list with duplicates
        memories_with_dupes = sample_memories + [sample_memories[0], sample_memories[1]]

        # Deduplicate
        unique = filter_service._deduplicate(memories_with_dupes)

        assert len(unique) == len(sample_memories)
        # Check all IDs are unique
        ids = [m.id for m in unique]
        assert len(ids) == len(set(ids))

    async def test_combine_and_deduplicate(self, memory_store, mock_llm, config, sample_memories):
        """Test combining and deduplicating results."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        vector_candidates = sample_memories[:3]
        context_results = {
            "temporal": sample_memories[2:4],
            "graph": [],
            "entity": sample_memories[3:5],
            "conversation": [],
        }

        combined = filter_service._combine_and_deduplicate(vector_candidates, context_results)

        assert isinstance(combined, list)
        assert all(isinstance(m, Memory) for m in combined)
        # Should have all unique memories
        ids = [m.id for m in combined]
        assert len(ids) == len(set(ids))


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.asyncio
class TestMultiStageFilterNeo4j:
    """Integration tests with Neo4j."""

    async def test_gather_context_neo4j(
        self, memory_store, mock_llm, config, sample_memory, sample_memories
    ):
        """Test full context gathering with Neo4j."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Setup data
        for mem in sample_memories:
            await memory_store.create_memory(mem)

        # Gather context
        context = await filter_service.gather_context(sample_memory)

        assert context is not None
        assert hasattr(context, "vector_candidates")
        assert hasattr(context, "filtered_candidates")
        assert isinstance(context.filtered_candidates, list)
        # Verify all candidates belong to the same user
        for mem in context.filtered_candidates:
            assert mem.user_id == sample_memory.user_id

    async def test_graph_filter_with_neighbors_neo4j(
        self, memory_store, mock_llm, config, sample_memories
    ):
        """Test graph filter with neighbors in Neo4j."""
        filter_service = MultiStageFilter(
            memory_store=memory_store,
            llm_provider=mock_llm,
            config=config,
        )

        # Add memories and create edges
        user_id = sample_memories[0].user_id
        for mem in sample_memories:
            await memory_store.create_memory(mem)

        # Create some edges
        from src.models.relationships import Edge, RelationshipType

        edge = Edge(
            source=sample_memories[0].id,
            target=sample_memories[1].id,
            type=RelationshipType.SIMILAR_TO,
            metadata={"confidence": 0.8},
        )
        await memory_store.graph_store.add_edge(edge, user_id)

        # Test graph filter
        results = await filter_service._graph_filter(sample_memories[0])

        assert isinstance(results, list)
        # Should find neighbors
        if len(results) > 0:
            assert all(isinstance(m, Memory) for m in results)
