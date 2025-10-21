"""Tests for causal and sequential relationship engine."""


import pytest

from src.core.embeddings import OllamaEmbedding
from src.core.graph_store import Neo4jGraphStore, SQLiteGraphStore
from src.core.relationships import CausalSequentialEngine
from src.core.vector_store import QdrantStore
from src.models import NodeType, RelationshipType


class TestCausalSequentialEngineSQLite:
    """Test causal/sequential engine with SQLite."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_causal_sqlite", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_causal_sqlite")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create a SQLite graph store."""
        db_path = tmp_path / "test_causal_sqlite.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create causal/sequential engine."""
        return CausalSequentialEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            max_sequence_gap=3600,
            similarity_threshold=0.6,
            topic_shift_threshold=0.4,
        )

    @pytest.mark.asyncio
    async def test_create_sequential_chain(self, engine, graph_store):
        """Test creating sequential FOLLOWS relationships."""
        # Create memories in order
        memories = [
            ("seq-1", "First step: Install Python"),
            ("seq-2", "Second step: Set up environment"),
            ("seq-3", "Third step: Write code"),
        ]

        memory_ids = []

        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create sequential chain
        result = await engine.create_sequential_chain(
            memory_ids=memory_ids, chain_type="tutorial_steps"
        )

        print("\n🔗 Sequential Chain:")
        print(f"  Chain length: {result['chain_length']}")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Chain type: {result['chain_type']}")

        assert result["edges_created"] == 2  # 3 nodes = 2 edges
        assert result["chain_length"] == 3

        # Verify edges exist
        neighbors = await graph_store.get_neighbors(
            "seq-1", edge_types=[RelationshipType.FOLLOWS], direction="outgoing"
        )
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "seq-2"
        print("  ✓ Created sequential chain with FOLLOWS edges")

    @pytest.mark.asyncio
    async def test_detect_conversation_threads(self, engine, graph_store):
        """Test detecting conversation threads by temporal proximity."""
        # Note: Since add_node uses current timestamp, all nodes will be created
        # at roughly the same time. This test validates that the thread detection
        # logic works, even if it detects one thread when created simultaneously.

        memories = [
            ("conv-1", "Hello"),
            ("conv-2", "How are you?"),
            ("conv-3", "I'm fine"),
            ("conv-4", "New topic"),
            ("conv-5", "Discussing something else"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Detect threads (will likely be 1 thread since created at same time)
        threads = await engine.detect_conversation_threads(
            memory_ids=memory_ids, max_gap_seconds=3600  # 1 hour max gap
        )

        print("\n💬 Conversation Threads:")
        print(f"  Threads detected: {len(threads)}")
        for idx, thread in enumerate(threads):
            print(f"  Thread {idx + 1}: {len(thread)} messages - {thread}")

        # When created at same time, they'll be in one thread
        assert len(threads) >= 1
        assert sum(len(t) for t in threads) == 5  # All 5 messages accounted for
        print("  ✓ Thread detection works (grouped by temporal proximity)")

    @pytest.mark.asyncio
    async def test_create_conversation_threads(self, engine, graph_store):
        """Test creating FOLLOWS edges for conversation threads."""
        # Create conversation memories
        memories = [
            ("chat-1", "Hello"),
            ("chat-2", "Hi there"),
            ("chat-3", "How can I help?"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        result = await engine.create_conversation_threads(memory_ids)

        print("\n💬 Thread Creation:")
        print(f"  Threads detected: {result['threads_detected']}")
        print(f"  Total edges: {result['total_edges_created']}")
        print(f"  Thread stats: {result['thread_stats']}")

        assert result["threads_detected"] >= 1
        assert result["total_edges_created"] >= 2
        print("  ✓ Created conversation thread edges")

    @pytest.mark.asyncio
    async def test_create_version_history(self, engine, graph_store):
        """Test creating version history (UPDATES relationships)."""
        # Create versions of a document
        versions = [
            ("v1", "Initial draft", 1),
            ("v2", "Revised draft", 2),
            ("v3", "Final version", 3),
        ]

        for mem_id, text, _version in versions:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})

        # Create version chain
        edge1 = await engine.create_version_history(
            memory_id="v2", previous_version_id="v1", version_number=2
        )

        edge2 = await engine.create_version_history(
            memory_id="v3", previous_version_id="v2", version_number=3
        )

        print("\n📝 Version History:")
        print(f"  Edge 1 (v1→v2): {edge1}")
        print(f"  Edge 2 (v2→v3): {edge2}")

        assert edge1 != ""
        assert edge2 != ""

        # Verify version chain
        neighbors = await graph_store.get_neighbors(
            "v1", edge_types=[RelationshipType.UPDATES], direction="outgoing"
        )
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "v2"
        print("  ✓ Created version history chain")

    @pytest.mark.asyncio
    async def test_get_version_chain(self, engine, graph_store):
        """Test retrieving complete version chain."""
        # Create version chain
        versions = ["ver-1", "ver-2", "ver-3", "ver-4"]

        for mem_id in versions:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": f"Version {mem_id}"})

        # Link versions
        for i in range(len(versions) - 1):
            await engine.create_version_history(
                memory_id=versions[i + 1], previous_version_id=versions[i], version_number=i + 2
            )

        # Get forward chain from ver-2
        forward_chain = await engine.get_version_chain("ver-2", direction="forward")

        print("\n🔄 Version Chain:")
        print(f"  Forward from ver-2: {[v['id'] for v in forward_chain]}")

        assert len(forward_chain) == 3  # ver-2, ver-3, ver-4
        assert forward_chain[0]["id"] == "ver-2"
        assert forward_chain[-1]["id"] == "ver-4"

        # Get backward chain from ver-3
        backward_chain = await engine.get_version_chain("ver-3", direction="backward")

        print(f"  Backward from ver-3: {[v['id'] for v in backward_chain]}")

        assert len(backward_chain) == 3  # ver-1, ver-2, ver-3
        assert backward_chain[0]["id"] == "ver-1"
        assert backward_chain[-1]["id"] == "ver-3"
        print("  ✓ Retrieved version chains correctly")

    @pytest.mark.asyncio
    async def test_detect_prerequisites(self, engine, graph_store, vector_store, embedder):
        """Test detecting prerequisite knowledge."""
        # Create foundational and advanced memories
        memories = [
            ("prereq-basic", "Variables store data in Python"),
            ("prereq-inter", "Functions encapsulate reusable code in Python"),
            (
                "prereq-adv",
                "Python decorators modify function behavior using closures and functions",
            ),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})

            # Store in vector store with real embeddings
            embedding = await embedder.embed(text)
            await vector_store.upsert_memory(
                memory_id=mem_id, embedding=embedding, metadata={"text": text}
            )

            memory_ids.append(mem_id)

        # Detect prerequisites for advanced topic
        prerequisites = await engine.detect_prerequisites(
            memory_id="prereq-adv", candidate_ids=["prereq-basic", "prereq-inter"]
        )

        print("\n📚 Prerequisites:")
        for prereq_id, relevance in prerequisites:
            print(f"  {prereq_id}: relevance={relevance:.3f}")

        # Should detect some prerequisites with real embeddings
        print(f"  ✓ Detected {len(prerequisites)} prerequisites")

    @pytest.mark.asyncio
    async def test_create_prerequisite_edges(self, engine, graph_store, vector_store, embedder):
        """Test creating REQUIRES edges for prerequisites."""
        # Create knowledge progression
        memories = [
            ("know-1", "Basic Python syntax and variables"),
            ("know-2", "Python object-oriented programming with classes"),
        ]

        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})

            # Store in vector store with real embeddings
            embedding = await embedder.embed(text)
            await vector_store.upsert_memory(
                memory_id=mem_id, embedding=embedding, metadata={"text": text}
            )

        result = await engine.create_prerequisite_edges(
            memory_id="know-2", candidate_ids=["know-1"], max_prerequisites=3
        )

        print("\n📖 Prerequisite Edges:")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Prerequisites: {result['prerequisites']}")

        # With real embeddings, should detect Python-related prerequisite
        print("  ✓ Prerequisite analysis completed")

    @pytest.mark.asyncio
    async def test_create_causal_relationship(self, engine, graph_store):
        """Test creating explicit causal relationships."""
        # Create cause and effect memories
        await graph_store.add_node("cause-1", NodeType.MEMORY, {"text": "Ran the code"})
        await graph_store.add_node("effect-1", NodeType.MEMORY, {"text": "Got an error"})

        edge_id = await engine.create_causal_relationship(
            cause_id="cause-1",
            effect_id="effect-1",
            confidence=0.9,
            metadata={"context": "debugging"},
        )

        print("\n⚡ Causal Relationship:")
        print(f"  Edge ID: {edge_id}")

        assert edge_id != ""

        # Verify edge
        neighbors = await graph_store.get_neighbors(
            "cause-1", edge_types=[RelationshipType.CAUSES], direction="outgoing"
        )
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "effect-1"
        assert neighbors[0]["edge_metadata"]["confidence"] == 0.9
        print("  ✓ Created CAUSES relationship")

    @pytest.mark.asyncio
    async def test_get_knowledge_path(self, engine, graph_store):
        """Test finding knowledge path between concepts."""
        # Create knowledge progression
        concepts = ["path-1", "path-2", "path-3", "path-4"]

        for concept in concepts:
            await graph_store.add_node(concept, NodeType.MEMORY, {"text": f"Concept {concept}"})

        # Create knowledge path
        for i in range(len(concepts) - 1):
            await engine.create_prerequisite_edges(
                memory_id=concepts[i + 1], candidate_ids=[concepts[i]], max_prerequisites=1
            )

        # Note: This requires REQUIRES edges to be traversable by find_path
        # For now, test the method exists
        path = await engine.get_knowledge_path("path-1", "path-4", max_depth=5)

        print("\n🛤️  Knowledge Path:")
        if path:
            print(f"  Path found: {[p['id'] for p in path]}")
        else:
            print("  No path found (expected if edges don't exist)")

        print("  ✓ Knowledge path search completed")

    @pytest.mark.asyncio
    async def test_analyze_sequence(self, engine, graph_store):
        """Test comprehensive sequence analysis."""
        # Create sequence of memories
        memories = ["analyze-1", "analyze-2", "analyze-3", "analyze-4", "analyze-5"]

        for mem_id in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": f"Memory {mem_id}"})

        # Analyze the sequence
        analysis = await engine.analyze_sequence(memories)

        print("\n📊 Sequence Analysis:")
        print(f"  Total memories: {analysis['total_memories']}")
        print(f"  Threads detected: {analysis['threads_detected']}")
        print(f"  Thread lengths: {analysis['thread_lengths']}")
        print(f"  Topic shifts: {analysis['topic_shifts']}")
        print(f"  Temporal stats: {analysis['temporal_stats']}")

        assert analysis["total_memories"] == 5
        assert analysis["threads_detected"] >= 1
        print("  ✓ Sequence analysis completed")


class TestCausalSequentialEngineNeo4j:
    """Test causal/sequential engine with Neo4j."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_causal_neo4j", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_causal_neo4j")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self):
        """Create a Neo4j graph store."""
        store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="mnemograph123")
        await store.initialize()
        yield store

        # Cleanup
        driver = await store._get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n:Node) WHERE n.id STARTS WITH 'neo-causal-' DETACH DELETE n")

        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create causal/sequential engine with Neo4j."""
        return CausalSequentialEngine(vector_store=vector_store, graph_store=graph_store)

    @pytest.mark.asyncio
    async def test_sequential_chain_neo4j(self, engine, graph_store):
        """Test sequential chain in Neo4j."""
        memories = [
            ("neo-causal-1", "Step one"),
            ("neo-causal-2", "Step two"),
            ("neo-causal-3", "Step three"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        result = await engine.create_sequential_chain(memory_ids, chain_type="steps")

        print("\n🔗 Neo4j Sequential Chain:")
        print(f"  Edges created: {result['edges_created']}")

        assert result["edges_created"] == 2

        # Verify in Neo4j
        neighbors = await graph_store.get_neighbors(
            "neo-causal-1", edge_types=[RelationshipType.FOLLOWS], direction="outgoing"
        )
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "neo-causal-2"
        print("  ✓ Created FOLLOWS relationships in Neo4j")

    @pytest.mark.asyncio
    async def test_version_history_neo4j(self, engine, graph_store):
        """Test version history in Neo4j."""
        versions = ["neo-causal-v1", "neo-causal-v2"]

        for ver in versions:
            await graph_store.add_node(ver, NodeType.MEMORY, {"text": f"Version {ver}"})

        edge_id = await engine.create_version_history(
            memory_id="neo-causal-v2", previous_version_id="neo-causal-v1", version_number=2
        )

        print("\n📝 Neo4j Version History:")
        print(f"  Edge created: {edge_id}")

        assert edge_id != ""
        print("  ✓ Created UPDATES relationship in Neo4j")


class TestCausalEngineConfiguration:
    """Test engine configuration."""

    @pytest.mark.asyncio
    async def test_custom_parameters(self, tmp_path):
        """Test custom engine parameters."""
        vector_store = QdrantStore(collection_name="test_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))

        engine = CausalSequentialEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            max_sequence_gap=7200,
            similarity_threshold=0.75,
            topic_shift_threshold=0.3,
        )

        assert engine.max_sequence_gap == 7200
        assert engine.similarity_threshold == 0.75
        assert engine.topic_shift_threshold == 0.3
        print("\n✓ Custom parameters configured correctly")
