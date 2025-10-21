"""Tests for hierarchical relationship engine."""

import pytest

from src.core.embeddings import OllamaEmbedding
from src.core.graph_store import Neo4jGraphStore, SQLiteGraphStore
from src.core.relationships import HierarchicalRelationshipEngine
from src.core.vector_store import QdrantStore
from src.models import NodeType, RelationshipType


class TestHierarchicalRelationshipEngineSQLite:
    """Test hierarchical relationship engine with SQLite and real embeddings."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_hierarchical_sqlite", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_hierarchical_sqlite")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create a SQLite graph store."""
        db_path = tmp_path / "test_hierarchical_sqlite.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create hierarchical relationship engine."""
        return HierarchicalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_cluster_size=2,
            num_topics=3,
            abstraction_threshold=0.7,
        )

    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "python_general": "Python programming",
            "python_specific": "Python is a high-level interpreted programming language with dynamic typing and automatic memory management",
            "python_functions": "Python functions are defined using the def keyword",
            "javascript": "JavaScript is used for web development",
            "cooking": "Baking chocolate chip cookies",
        }

        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)

        return embeddings

    @pytest.mark.asyncio
    async def test_create_parent_child_relationship(self, engine, graph_store):
        """Test creating parent-child relationship."""
        # Create parent and child nodes
        await graph_store.add_node("doc-1", NodeType.DOCUMENT, {"text": "Parent document"})
        await graph_store.add_node("chunk-1", NodeType.CHUNK, {"text": "Child chunk"})

        # Create relationship
        edge_id = await engine.create_parent_child_relationship(
            parent_id="doc-1", child_id="chunk-1", metadata={"chunk_index": 0}
        )

        assert edge_id is not None

        print("\n🌳 Parent-Child Relationship:")
        print(f"  Document → Chunk: {edge_id}")

        # Verify relationship
        neighbors = await graph_store.get_neighbors(
            "doc-1", edge_types=[RelationshipType.PARENT_OF], direction="outgoing"
        )

        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "chunk-1"
        print("  ✓ Verified: doc-1 is parent of chunk-1")

    @pytest.mark.asyncio
    async def test_chunk_document(self, engine, graph_store):
        """Test chunking a document into multiple pieces."""
        # Create document and chunks
        await graph_store.add_node("doc-main", NodeType.DOCUMENT, {"text": "Main document"})

        chunk_ids = []
        for i in range(5):
            chunk_id = f"chunk-{i}"
            await graph_store.add_node(chunk_id, NodeType.CHUNK, {"text": f"Chunk {i}"})
            chunk_ids.append(chunk_id)

        # Create parent-child relationships
        edges_created = await engine.chunk_document("doc-main", chunk_ids)

        assert edges_created == 5

        print("\n📄 Document Chunking:")
        print(f"  Document split into {len(chunk_ids)} chunks")
        print(f"  Created {edges_created} parent-child relationships")

        # Verify all chunks are children
        neighbors = await graph_store.get_neighbors(
            "doc-main", edge_types=[RelationshipType.PARENT_OF], direction="outgoing"
        )

        assert len(neighbors) == 5
        print(f"  ✓ All {len(neighbors)} chunks linked to document")

    @pytest.mark.asyncio
    async def test_create_topic_clusters_real_embeddings(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test topic clustering with real embeddings."""
        # Create memories with embeddings
        memories = [
            ("mem-py1", real_embeddings["python_general"], "Python general"),
            ("mem-py2", real_embeddings["python_specific"], "Python specific"),
            ("mem-py3", real_embeddings["python_functions"], "Python functions"),
            ("mem-js", real_embeddings["javascript"], "JavaScript"),
            ("mem-cook", real_embeddings["cooking"], "Cooking"),
        ]

        memory_ids = []
        for mem_id, emb, text in memories:
            await vector_store.upsert_memory(mem_id, emb, {"text": text})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create topic clusters
        result = await engine.create_topic_clusters(memory_ids=memory_ids, create_topic_nodes=True)

        print("\n🏷️  Topic Clustering:")
        print(f"  Found {result['num_clusters']} topic clusters")

        for topic_info in result["topic_nodes"]:
            print(f"  Topic {topic_info['cluster_id']}: {topic_info['size']} memories")
            print(f"    Members: {topic_info['members']}")

        # Should create at least one cluster
        assert result["num_clusters"] >= 1
        assert len(result["topic_nodes"]) >= 1
        print(f"  ✓ Created {len(result['topic_nodes'])} topic nodes")

    @pytest.mark.asyncio
    async def test_detect_abstraction_levels(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test detecting abstraction relationships (general ← specific)."""
        # Create memories with different abstraction levels
        memories = [
            ("mem-general", real_embeddings["python_general"], "Python programming"),
            (
                "mem-specific",
                real_embeddings["python_specific"],
                "Python is a high-level interpreted programming language with dynamic typing and automatic memory management",
            ),
        ]

        for mem_id, emb, text in memories:
            await vector_store.upsert_memory(mem_id, emb, {"text": text})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})

        # Detect abstraction levels
        abstractions = await engine.detect_abstraction_levels(
            memory_ids=["mem-general", "mem-specific"], create_edges=True
        )

        print("\n🔼 Abstraction Detection:")
        print(f"  Found {len(abstractions)} abstraction relationships")

        for specific, general, confidence in abstractions:
            print(f"  {specific} → {general} (confidence: {confidence:.4f})")
            print("  ✓ Specific details abstract to general concept")

        if abstractions:
            # Verify the relationship direction (specific → general)
            specific_id, general_id, _ = abstractions[0]
            assert specific_id == "mem-specific"
            assert general_id == "mem-general"

    @pytest.mark.asyncio
    async def test_get_hierarchy_path(self, engine, graph_store):
        """Test getting hierarchical path from leaf to root."""
        # Create a 3-level hierarchy: book → chapter → section
        await graph_store.add_node("book-1", NodeType.DOCUMENT, {"text": "Book"})
        await graph_store.add_node("chapter-1", NodeType.DOCUMENT, {"text": "Chapter 1"})
        await graph_store.add_node("section-1", NodeType.CHUNK, {"text": "Section 1.1"})

        # Create hierarchy
        await engine.create_parent_child_relationship("book-1", "chapter-1")
        await engine.create_parent_child_relationship("chapter-1", "section-1")

        # Get path from section to root
        path = await engine.get_hierarchy_path("section-1")

        print("\n📚 Hierarchy Path:")
        print(f"  Path: {' ← '.join(path)}")

        assert len(path) == 3
        assert path[0] == "section-1"
        assert path[-1] == "book-1"
        print(f"  ✓ Traversed from leaf to root: {len(path)} levels")

    @pytest.mark.asyncio
    async def test_get_descendants(self, engine, graph_store):
        """Test getting all descendants of a node."""
        # Create tree: root → [child1, child2] → [grandchild1, grandchild2]
        await graph_store.add_node("root", NodeType.DOCUMENT, {"text": "Root"})
        await graph_store.add_node("child1", NodeType.CHUNK, {"text": "Child 1"})
        await graph_store.add_node("child2", NodeType.CHUNK, {"text": "Child 2"})
        await graph_store.add_node("grandchild1", NodeType.CHUNK, {"text": "Grandchild 1"})
        await graph_store.add_node("grandchild2", NodeType.CHUNK, {"text": "Grandchild 2"})

        # Create hierarchy
        await engine.create_parent_child_relationship("root", "child1")
        await engine.create_parent_child_relationship("root", "child2")
        await engine.create_parent_child_relationship("child1", "grandchild1")
        await engine.create_parent_child_relationship("child2", "grandchild2")

        # Get all descendants
        descendants = await engine.get_descendants("root")

        print("\n🌲 Tree Traversal:")
        print(f"  Root has {len(descendants)} descendants")
        print(f"  Descendants: {descendants}")

        assert len(descendants) == 4  # 2 children + 2 grandchildren
        assert "child1" in descendants
        assert "child2" in descendants
        assert "grandchild1" in descendants
        assert "grandchild2" in descendants
        print(f"  ✓ Found all descendants across {len(descendants)} nodes")


class TestHierarchicalRelationshipEngineNeo4j:
    """Test hierarchical relationships with Neo4j."""

    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(model="nomic-embed-text", host="http://localhost:11434")

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_hierarchical_neo4j", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_hierarchical_neo4j")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self):
        """Create a Neo4j graph store."""
        store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="mnemograph123")
        await store.initialize()
        yield store

        # Cleanup - delete both regular nodes and topic nodes
        driver = await store._get_driver()
        async with driver.session() as session:
            # Delete all nodes that start with 'neo-hier-' AND topic nodes
            await session.run(
                """
                MATCH (n:Node)
                WHERE n.id STARTS WITH 'neo-hier-' OR n.id STARTS WITH 'topic-'
                DETACH DELETE n
                """
            )

        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create hierarchical relationship engine with Neo4j."""
        return HierarchicalRelationshipEngine(
            vector_store=vector_store, graph_store=graph_store, min_cluster_size=2, num_topics=2
        )

    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "topic1_mem1": "Machine learning algorithms",
            "topic1_mem2": "Neural networks and deep learning",
            "topic2_mem1": "Cooking pasta recipes",
            "topic2_mem2": "Italian cuisine techniques",
        }

        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)

        return embeddings

    @pytest.mark.asyncio
    async def test_chunk_document_neo4j(self, engine, graph_store):
        """Test document chunking with Neo4j."""
        # Create document and chunks
        await graph_store.add_node("neo-hier-doc", NodeType.DOCUMENT, {"text": "Document"})

        chunk_ids = []
        for i in range(3):
            chunk_id = f"neo-hier-chunk-{i}"
            await graph_store.add_node(chunk_id, NodeType.CHUNK, {"text": f"Chunk {i}"})
            chunk_ids.append(chunk_id)

        # Create relationships
        edges_created = await engine.chunk_document("neo-hier-doc", chunk_ids)

        print("\n📄 Neo4j Document Chunking:")
        print(f"  Created {edges_created} parent-child relationships")

        assert edges_created == 3

        # Verify in Neo4j
        neighbors = await graph_store.get_neighbors(
            "neo-hier-doc", edge_types=[RelationshipType.PARENT_OF], direction="outgoing"
        )

        assert len(neighbors) == 3
        print("  ✓ All chunks linked in Neo4j graph")

    @pytest.mark.asyncio
    async def test_topic_clusters_neo4j(self, engine, vector_store, graph_store, real_embeddings):
        """Test topic clustering with Neo4j backend."""
        # Create memories
        memory_ids = []
        for key, emb in real_embeddings.items():
            mem_id = f"neo-hier-{key}"
            text = key.replace("_", " ")
            await vector_store.upsert_memory(mem_id, emb, {"text": text})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create clusters
        result = await engine.create_topic_clusters(memory_ids=memory_ids, create_topic_nodes=True)

        print("\n🏷️  Neo4j Topic Clustering:")
        print(f"  Found {result['num_clusters']} clusters")
        for topic in result["topic_nodes"]:
            print(f"  Topic {topic['cluster_id']}: {topic['size']} members")

        assert result["num_clusters"] >= 1
        print("  ✓ Created topic nodes in Neo4j")


class TestHierarchicalEngineConfiguration:
    """Test engine configuration."""

    @pytest.mark.asyncio
    async def test_custom_clustering_parameters(self, tmp_path):
        """Test custom clustering parameters."""
        vector_store = QdrantStore(collection_name="test_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))

        engine = HierarchicalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_cluster_size=5,
            num_topics=10,
            abstraction_threshold=0.8,
        )

        assert engine.min_cluster_size == 5
        assert engine.num_topics == 10
        assert engine.abstraction_threshold == 0.8
        print("\n✓ Custom hierarchical parameters configured correctly")
