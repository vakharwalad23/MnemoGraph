"""Tests for relationship orchestrator."""

from datetime import UTC, datetime

import pytest

from src.config import Config
from src.core.embeddings import OllamaEmbedding
from src.core.graph_store import Neo4jGraphStore, SQLiteGraphStore
from src.core.vector_store import QdrantStore
from src.models import NodeType
from src.services import RelationshipOrchestrator


@pytest.mark.sqlite
class TestRelationshipOrchestrator:
    """Test relationship orchestrator with SQLite."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_orchestrator", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_orchestrator")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create SQLite graph store."""
        db_path = tmp_path / "test_orchestrator.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def config(self):
        """Create test configuration."""
        return Config(
            relationships={
                "auto_infer_on_add": True,
                "semantic": {"similarity_threshold": 0.5},  # Lower threshold for testing
                "temporal": {"update_similarity_threshold": 0.85},
            }
        )

    @pytest.fixture
    async def orchestrator(self, vector_store, graph_store, embedder, config):
        """Create relationship orchestrator."""
        return RelationshipOrchestrator(
            vector_store=vector_store, graph_store=graph_store, embedder=embedder, config=config
        )

    @pytest.mark.asyncio
    async def test_process_new_memory(self, orchestrator, graph_store, vector_store, embedder):
        """Test processing a new memory with automatic relationship inference."""
        # Add some context memories first
        context_ids = []
        for i, text in enumerate(
            [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Python has great libraries for data science",
            ]
        ):
            mem_id = f"context-{i}"
            embedding = await embedder.embed(text)

            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            await vector_store.upsert_memory(mem_id, embedding, {"text": text})
            context_ids.append(mem_id)

        # Add new memory
        new_text = "Python 3.11 introduces performance improvements"
        new_embedding = await embedder.embed(new_text)
        new_id = "new-memory"

        await graph_store.add_node(new_id, NodeType.MEMORY, {"text": new_text})
        await vector_store.upsert_memory(new_id, new_embedding, {"text": new_text})

        # Process with orchestrator
        stats = await orchestrator.process_new_memory(
            memory_id=new_id,
            text=new_text,
            embedding=new_embedding,
            created_at=datetime.now(UTC),
            context_memory_ids=context_ids,
        )

        print("\n🎯 Orchestrator Stats:")
        print(f"  Memory ID: {stats['memory_id']}")
        print(f"  Total relationships: {stats['relationships_created']}")
        print("  Engines run:")
        for engine, result in stats["engines_run"].items():
            if result["success"]:
                print(f"    ✓ {engine}: {result}")
            else:
                print(f"    ✗ {engine}: {result.get('error')}")

        # Verify all expected engines ran successfully
        assert "semantic" in stats["engines_run"]
        assert "temporal" in stats["engines_run"]
        assert "hierarchical" in stats["engines_run"]
        assert "cooccurrence" in stats["engines_run"]
        assert "causal" in stats["engines_run"]

        # All engines should complete successfully
        for engine, result in stats["engines_run"].items():
            assert result["success"], f"Engine {engine} failed: {result.get('error')}"

        # With related Python content and lower threshold, we should find relationships
        # At minimum, semantic similarity should find the Python-related memories
        assert (
            stats["relationships_created"] > 0
        ), "Expected relationships to be created between related memories"

        print(f"  ✓ Orchestrator processed memory with {len(stats['engines_run'])} engines")
        print(f"  ✓ Created {stats['relationships_created']} relationships")

    @pytest.mark.asyncio
    async def test_batch_process_memories(self, orchestrator, graph_store, vector_store, embedder):
        """Test batch processing multiple memories."""
        # Create multiple memories
        memory_ids = []
        texts = [
            "Python programming basics",
            "Advanced Python techniques",
            "Python for data analysis",
            "Machine learning with Python",
            "Web development with Python",
        ]

        for i, text in enumerate(texts):
            mem_id = f"batch-{i}"
            embedding = await embedder.embed(text)

            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            await vector_store.upsert_memory(mem_id, embedding, {"text": text})
            memory_ids.append(mem_id)

        # Batch process
        stats = await orchestrator.batch_process_memories(memory_ids=memory_ids, batch_size=3)

        print("\n📦 Batch Processing Stats:")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Batches processed: {stats['batches_processed']}")

        assert stats["total_memories"] == 5
        assert stats["total_relationships"] > 0
        print("  ✓ Batch processing completed successfully")

    @pytest.mark.asyncio
    async def test_create_conversation_thread(self, orchestrator, graph_store):
        """Test creating conversation thread."""
        # Create conversation memories
        messages = [
            ("msg-1", "Hello"),
            ("msg-2", "Hi, how are you?"),
            ("msg-3", "I'm doing well, thanks!"),
            ("msg-4", "What are you working on?"),
        ]

        message_ids = []
        for mem_id, text in messages:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            message_ids.append(mem_id)

        # Create thread
        result = await orchestrator.create_conversation_thread(
            memory_ids=message_ids, thread_metadata={"conversation_id": "conv-123"}
        )

        print("\n💬 Conversation Thread:")
        print(f"  Messages: {len(message_ids)}")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Chain type: {result['chain_type']}")

        assert result["edges_created"] == 3  # N-1 edges
        print("  ✓ Conversation thread created successfully")

    @pytest.mark.asyncio
    async def test_create_document_hierarchy(self, orchestrator, graph_store):
        """Test creating document hierarchy."""
        # Create document and chunks
        doc_id = "doc-1"
        await graph_store.add_node(
            doc_id, NodeType.DOCUMENT, {"title": "Python Tutorial", "type": "document"}
        )

        chunk_ids = []
        for i in range(5):
            chunk_id = f"chunk-{i}"
            await graph_store.add_node(
                chunk_id, NodeType.CHUNK, {"text": f"Chapter {i} content", "parent_id": doc_id}
            )
            chunk_ids.append(chunk_id)

        # Create hierarchy
        result = await orchestrator.create_document_hierarchy(
            document_id=doc_id, chunk_ids=chunk_ids
        )

        print("\n📄 Document Hierarchy:")
        print(f"  Document: {doc_id}")
        print(f"  Chunks: {len(chunk_ids)}")
        print(f"  Edges created: {result['edges_created']}")

        assert result["edges_created"] == 5
        print("  ✓ Document hierarchy created successfully")


@pytest.mark.neo4j
class TestRelationshipOrchestratorNeo4j:
    """Test relationship orchestrator with Neo4j graph store."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_orchestrator_neo4j", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_orchestrator_neo4j")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self):
        """Create Neo4j graph store."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mnemograph123",  # Update with actual password
        )
        await store.initialize()
        yield store

        # Cleanup - delete all test nodes (including TOPIC nodes)
        await store.driver.execute_query(
            "MATCH (n) WHERE n.id STARTS WITH 'context-' OR "
            "n.id STARTS WITH 'new-memory' OR "
            "n.id STARTS WITH 'batch-' OR "
            "n.id STARTS WITH 'msg-' OR "
            "n.id STARTS WITH 'doc-' OR "
            "n.id STARTS WITH 'chunk-' OR "
            "n.id STARTS WITH 'topic-' "
            "DETACH DELETE n"
        )
        await store.close()

    @pytest.fixture
    async def config(self):
        """Create test configuration."""
        return Config(
            relationships={
                "auto_infer_on_add": True,
                "semantic": {"similarity_threshold": 0.5},
                "temporal": {"update_similarity_threshold": 0.85},
            }
        )

    @pytest.fixture
    async def orchestrator(self, vector_store, graph_store, embedder, config):
        """Create relationship orchestrator."""
        return RelationshipOrchestrator(
            vector_store=vector_store, graph_store=graph_store, embedder=embedder, config=config
        )

    @pytest.mark.asyncio
    async def test_process_new_memory_neo4j(
        self, orchestrator, graph_store, vector_store, embedder
    ):
        """Test processing a new memory with Neo4j graph store."""
        # Add some context memories first
        context_ids = []
        for i, text in enumerate(
            [
                "Python is a programming language",
                "JavaScript is used for web development",
                "Python has great libraries for data science",
            ]
        ):
            mem_id = f"context-{i}"
            embedding = await embedder.embed(text)

            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            await vector_store.upsert_memory(mem_id, embedding, {"text": text})
            context_ids.append(mem_id)

        # Add new memory
        new_text = "Python 3.11 introduces performance improvements"
        new_embedding = await embedder.embed(new_text)
        new_id = "new-memory"

        await graph_store.add_node(new_id, NodeType.MEMORY, {"text": new_text})
        await vector_store.upsert_memory(new_id, new_embedding, {"text": new_text})

        # Process with orchestrator
        stats = await orchestrator.process_new_memory(
            memory_id=new_id,
            text=new_text,
            embedding=new_embedding,
            created_at=datetime.now(UTC),
            context_memory_ids=context_ids,
        )

        print("\n🎯 Orchestrator Stats (Neo4j):")
        print(f"  Memory ID: {stats['memory_id']}")
        print(f"  Total relationships: {stats['relationships_created']}")
        print("  Engines run:")
        for engine, result in stats["engines_run"].items():
            if result["success"]:
                print(f"    ✓ {engine}: {result}")
            else:
                print(f"    ✗ {engine}: {result.get('error')}")

        # Verify all expected engines ran successfully
        assert "semantic" in stats["engines_run"]
        assert "temporal" in stats["engines_run"]
        assert "hierarchical" in stats["engines_run"]
        assert "cooccurrence" in stats["engines_run"]
        assert "causal" in stats["engines_run"]

        # All engines should complete successfully
        for engine, result in stats["engines_run"].items():
            assert result["success"], f"Engine {engine} failed: {result.get('error')}"

        # With related Python content and lower threshold, we should find relationships
        assert (
            stats["relationships_created"] > 0
        ), "Expected relationships to be created between related memories"

        print(f"  ✓ Orchestrator processed memory with {len(stats['engines_run'])} engines")
        print(f"  ✓ Created {stats['relationships_created']} relationships")

    @pytest.mark.asyncio
    async def test_batch_process_memories_neo4j(
        self, orchestrator, graph_store, vector_store, embedder
    ):
        """Test batch processing with Neo4j."""
        # Create multiple memories
        memory_ids = []
        texts = [
            "Python programming basics",
            "Advanced Python techniques",
            "Python for data analysis",
            "Machine learning with Python",
            "Web development with Python",
        ]

        for i, text in enumerate(texts):
            mem_id = f"batch-{i}"
            embedding = await embedder.embed(text)

            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            await vector_store.upsert_memory(mem_id, embedding, {"text": text})
            memory_ids.append(mem_id)

        # Batch process
        stats = await orchestrator.batch_process_memories(memory_ids=memory_ids, batch_size=3)

        print("\n📦 Batch Processing Stats (Neo4j):")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Batches processed: {stats['batches_processed']}")

        assert stats["total_memories"] == 5
        assert stats["total_relationships"] > 0
        print("  ✓ Batch processing completed successfully")

    @pytest.mark.asyncio
    async def test_create_conversation_thread_neo4j(self, orchestrator, graph_store):
        """Test creating conversation thread with Neo4j."""
        # Create conversation memories
        messages = [
            ("msg-1", "Hello"),
            ("msg-2", "Hi, how are you?"),
            ("msg-3", "I'm doing well, thanks!"),
            ("msg-4", "What are you working on?"),
        ]

        message_ids = []
        for mem_id, text in messages:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            message_ids.append(mem_id)

        # Create thread
        result = await orchestrator.create_conversation_thread(
            memory_ids=message_ids, thread_metadata={"conversation_id": "conv-123"}
        )

        print("\n💬 Conversation Thread (Neo4j):")
        print(f"  Messages: {len(message_ids)}")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Chain type: {result['chain_type']}")

        assert result["edges_created"] == 3  # N-1 edges
        print("  ✓ Conversation thread created successfully")

    @pytest.mark.asyncio
    async def test_create_document_hierarchy_neo4j(self, orchestrator, graph_store):
        """Test creating document hierarchy with Neo4j."""
        # Create document and chunks
        doc_id = "doc-1"
        await graph_store.add_node(
            doc_id, NodeType.DOCUMENT, {"title": "Python Tutorial", "type": "document"}
        )

        chunk_ids = []
        for i in range(5):
            chunk_id = f"chunk-{i}"
            await graph_store.add_node(
                chunk_id, NodeType.CHUNK, {"text": f"Chapter {i} content", "parent_id": doc_id}
            )
            chunk_ids.append(chunk_id)

        # Create hierarchy
        result = await orchestrator.create_document_hierarchy(
            document_id=doc_id, chunk_ids=chunk_ids
        )

        print("\n📄 Document Hierarchy (Neo4j):")
        print(f"  Document: {doc_id}")
        print(f"  Chunks: {len(chunk_ids)}")
        print(f"  Edges created: {result['edges_created']}")

        assert result["edges_created"] == 5
        print("  ✓ Document hierarchy created successfully")


class TestOrchestratorConfiguration:
    """Test orchestrator with different configurations."""

    @pytest.mark.asyncio
    async def test_custom_config(self, tmp_path):
        """Test orchestrator with custom configuration."""
        # Custom config with high thresholds
        config = Config(
            relationships={
                "semantic": {"similarity_threshold": 0.9},
                "temporal": {"update_similarity_threshold": 0.95},
                "cooccurrence": {"min_cooccurrence_count": 5},
            }
        )

        vector_store = QdrantStore(collection_name="test_custom_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))
        embedder = OllamaEmbedding()

        orchestrator = RelationshipOrchestrator(
            vector_store=vector_store, graph_store=graph_store, embedder=embedder, config=config
        )

        # Check engine configs
        assert orchestrator.semantic_engine.similarity_threshold == 0.9
        assert orchestrator.temporal_engine.update_similarity_threshold == 0.95
        assert orchestrator.cooccurrence_engine.min_cooccurrence_count == 5
        assert orchestrator.hierarchical_engine.min_cluster_size == 2  # default
        assert orchestrator.causal_engine.similarity_threshold == 0.6  # default

        print("\n✓ Custom configuration applied correctly")
