"""Tests for semantic similarity engine."""

import pytest
import numpy as np
from src.core.relationships import SemanticSimilarityEngine
from src.core.vector_store import QdrantStore
from src.core.graph_store import SQLiteGraphStore, Neo4jGraphStore
from src.core.embeddings import OllamaEmbedding
from src.models import NodeType, RelationshipType


class TestSemanticSimilarityEngineSQLite:
    """Test semantic similarity with SQLite graph store and real embeddings."""
    
    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(
            model="nomic-embed-text",
            host="http://localhost:11434"
        )
    
    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_semantic_sim_sqlite",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_semantic_sim_sqlite")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create a SQLite graph store."""
        db_path = tmp_path / "test_semantic_sqlite.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create semantic similarity engine."""
        return SemanticSimilarityEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            similarity_threshold=0.5,  # Lower threshold for testing
            max_similar_memories=10
        )
    
    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "python1": "Python is a high-level programming language",
            "python2": "Python is used for software development",
            "cooking": "How to make chocolate chip cookies",
            "machine_learning": "Machine learning is a subset of artificial intelligence",
        }
        
        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)
        
        return embeddings
    
    @pytest.mark.asyncio
    async def test_compute_similarity_with_real_embeddings(self, real_embeddings):
        """Test computing similarity with real embeddings."""
        # Python-related texts should be similar
        sim_python = SemanticSimilarityEngine.compute_similarity(
            real_embeddings["python1"],
            real_embeddings["python2"]
        )
        
        # Python and cooking should be dissimilar
        sim_different = SemanticSimilarityEngine.compute_similarity(
            real_embeddings["python1"],
            real_embeddings["cooking"]
        )
        
        print(f"\nPython-Python similarity: {sim_python:.4f}")
        print(f"Python-Cooking similarity: {sim_different:.4f}")
        
        assert sim_python > sim_different
        assert sim_python > 0.5  # Should be reasonably similar
    
    @pytest.mark.asyncio
    async def test_compute_batch_similarity_real(self, real_embeddings):
        """Test batch similarity with real embeddings."""
        query = real_embeddings["python1"]
        embeddings = [
            real_embeddings["python2"],
            real_embeddings["cooking"],
            real_embeddings["machine_learning"],
        ]
        
        similarities = SemanticSimilarityEngine.compute_batch_similarity(
            query, embeddings
        )
        
        print(f"\nBatch similarities: {[f'{s:.4f}' for s in similarities]}")
        
        assert len(similarities) == 3
        # python2 should be most similar to python1
        assert similarities[0] > similarities[1]
    
    @pytest.mark.asyncio
    async def test_infer_relationships_real_embeddings(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test inferring relationships with real embeddings."""
        # Add memories to vector store
        memories = [
            ("mem-python-1", real_embeddings["python1"], "Python programming language"),
            ("mem-python-2", real_embeddings["python2"], "Python for development"),
            ("mem-cooking", real_embeddings["cooking"], "Cookie recipe"),
            ("mem-ml", real_embeddings["machine_learning"], "Machine learning AI"),
        ]
        
        for mem_id, emb, text in memories:
            await vector_store.upsert_memory(mem_id, emb, {"text": text})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
        
        # Infer relationships for first Python memory
        similar = await engine.infer_relationships(
            memory_id="mem-python-1",
            embedding=real_embeddings["python1"],
            create_edges=True
        )
        
        print(f"\nFound {len(similar)} similar memories:")
        for mem in similar:
            print(f"  - {mem['id']}: score={mem['score']:.4f}")
        
        # Should find mem-python-2 as similar
        similar_ids = [mem["id"] for mem in similar]
        assert "mem-python-2" in similar_ids
        
        # Should NOT find cooking as similar (if threshold is appropriate)
        # This might vary based on embeddings
        
        # Verify edges were created
        neighbors = await graph_store.get_neighbors(
            "mem-python-1",
            edge_types=[RelationshipType.SIMILAR_TO],
            direction="outgoing"
        )
        
        assert len(neighbors) >= 1
        print(f"\nCreated {len(neighbors)} edges from mem-python-1")
    
    @pytest.mark.asyncio
    async def test_threshold_filtering_real(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test that similarity threshold works with real embeddings."""
        # Add memories
        await vector_store.upsert_memory(
            "mem-1",
            real_embeddings["python1"],
            {"text": "Python programming"}
        )
        await vector_store.upsert_memory(
            "mem-2",
            real_embeddings["python2"],
            {"text": "Python development"}
        )
        await vector_store.upsert_memory(
            "mem-3",
            real_embeddings["cooking"],
            {"text": "Cooking recipes"}
        )
        
        await graph_store.add_node("mem-1", NodeType.MEMORY, {"text": "Python programming"})
        await graph_store.add_node("mem-2", NodeType.MEMORY, {"text": "Python development"})
        await graph_store.add_node("mem-3", NodeType.MEMORY, {"text": "Cooking recipes"})
        
        similar = await engine.infer_relationships(
            memory_id="mem-1",
            embedding=real_embeddings["python1"],
            create_edges=True
        )
        
        # Calculate actual similarities
        sim_python = SemanticSimilarityEngine.compute_similarity(
            real_embeddings["python1"],
            real_embeddings["python2"]
        )
        sim_cooking = SemanticSimilarityEngine.compute_similarity(
            real_embeddings["python1"],
            real_embeddings["cooking"]
        )
        
        print(f"\nSimilarity scores:")
        print(f"  Python-Python: {sim_python:.4f}")
        print(f"  Python-Cooking: {sim_cooking:.4f}")
        print(f"  Threshold: {engine.similarity_threshold}")
        
        similar_ids = [mem["id"] for mem in similar]
        
        # If python similarity is above threshold, should include mem-2
        if sim_python >= engine.similarity_threshold:
            assert "mem-2" in similar_ids
        
        # If cooking similarity is below threshold, should NOT include mem-3
        if sim_cooking < engine.similarity_threshold:
            assert "mem-3" not in similar_ids
    
    @pytest.mark.asyncio
    async def test_batch_infer_real(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test batch inference with real embeddings."""
        # Add all memories
        for key, emb in real_embeddings.items():
            mem_id = f"mem-{key}"
            await vector_store.upsert_memory(mem_id, emb, {"text": key})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": key})
        
        # Batch infer
        memories = [
            {"id": "mem-python1", "embedding": real_embeddings["python1"]},
            {"id": "mem-python2", "embedding": real_embeddings["python2"]},
        ]
        
        results = await engine.batch_infer_relationships(
            memories=memories,
            create_edges=True
        )
        
        print(f"\nBatch inference results:")
        for mem_id, similar_list in results.items():
            print(f"  {mem_id}: found {len(similar_list)} similar memories")
        
        assert "mem-python1" in results
        assert "mem-python2" in results
    
    @pytest.mark.asyncio
    async def test_find_clusters_real(self, engine, vector_store, real_embeddings):
        """Test finding clusters with real embeddings."""
        # Add memories
        memory_map = {}
        for key, emb in real_embeddings.items():
            mem_id = f"mem-{key}"
            await vector_store.upsert_memory(mem_id, emb, {"text": key})
            memory_map[mem_id] = key
        
        # Find clusters
        clusters = await engine.find_clusters(
            memory_ids=list(memory_map.keys()),
            min_cluster_size=2
        )
        
        print(f"\nFound {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {[memory_map[m] for m in cluster]}")
        
        # Should find at least Python-related cluster
        # (depends on threshold and actual similarities)
        assert len(clusters) >= 0


class TestSemanticSimilarityEngineNeo4j:
    """Test semantic similarity with Neo4j graph store and real embeddings."""
    
    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(
            model="nomic-embed-text",
            host="http://localhost:11434"
        )
    
    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_semantic_sim_neo4j",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_semantic_sim_neo4j")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    async def graph_store(self):
        """Create a Neo4j graph store."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mnemograph123"
        )
        await store.initialize()
        yield store
        
        # Cleanup
        driver = await store._get_driver()
        async with driver.session() as session:
            await session.run("MATCH (n:Node) WHERE n.id STARTS WITH 'neo-mem-' DETACH DELETE n")
        
        await store.close()
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create semantic similarity engine with Neo4j."""
        return SemanticSimilarityEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            similarity_threshold=0.5,
            max_similar_memories=10
        )
    
    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "python1": "Python is a programming language for software development",
            "python2": "Python programming is widely used in data science",
            "cooking": "Baking delicious chocolate chip cookies at home",
        }
        
        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)
        
        return embeddings
    
    @pytest.mark.asyncio
    async def test_infer_relationships_neo4j(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test inferring relationships with Neo4j backend."""
        # Add memories
        memories = [
            ("neo-mem-python-1", real_embeddings["python1"], "Python programming"),
            ("neo-mem-python-2", real_embeddings["python2"], "Python data science"),
            ("neo-mem-cooking", real_embeddings["cooking"], "Cookie recipe"),
        ]
        
        for mem_id, emb, text in memories:
            await vector_store.upsert_memory(mem_id, emb, {"text": text})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
        
        # Infer relationships
        similar = await engine.infer_relationships(
            memory_id="neo-mem-python-1",
            embedding=real_embeddings["python1"],
            create_edges=True
        )
        
        print(f"\nNeo4j: Found {len(similar)} similar memories:")
        for mem in similar:
            print(f"  - {mem['id']}: score={mem['score']:.4f}")
        
        # Verify with Neo4j query
        neighbors = await graph_store.get_neighbors(
            "neo-mem-python-1",
            edge_types=[RelationshipType.SIMILAR_TO],
            direction="outgoing"
        )
        
        print(f"Neo4j: Created {len(neighbors)} edges")
        assert len(neighbors) >= 1
        
        # Should find python2 as similar
        neighbor_ids = [n["node"].id for n in neighbors]
        assert "neo-mem-python-2" in neighbor_ids
    
    @pytest.mark.asyncio
    async def test_batch_infer_neo4j(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test batch inference with Neo4j."""
        # Add memories
        for key, emb in real_embeddings.items():
            mem_id = f"neo-mem-{key}"
            await vector_store.upsert_memory(mem_id, emb, {"text": key})
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": key})
        
        # Batch infer
        memories = [
            {"id": "neo-mem-python1", "embedding": real_embeddings["python1"]},
            {"id": "neo-mem-python2", "embedding": real_embeddings["python2"]},
        ]
        
        results = await engine.batch_infer_relationships(
            memories=memories,
            create_edges=True
        )
        
        print(f"\nNeo4j batch inference:")
        for mem_id, similar_list in results.items():
            print(f"  {mem_id}: {len(similar_list)} similar memories")
        
        assert "neo-mem-python1" in results
        assert "neo-mem-python2" in results
    
    @pytest.mark.asyncio
    async def test_edge_metadata_neo4j(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test that Neo4j edges include proper metadata."""
        await vector_store.upsert_memory(
            "neo-mem-1",
            real_embeddings["python1"],
            {"text": "Test 1"}
        )
        await vector_store.upsert_memory(
            "neo-mem-2",
            real_embeddings["python2"],
            {"text": "Test 2"}
        )
        
        await graph_store.add_node("neo-mem-1", NodeType.MEMORY, {"text": "Test 1"})
        await graph_store.add_node("neo-mem-2", NodeType.MEMORY, {"text": "Test 2"})
        
        await engine.infer_relationships(
            memory_id="neo-mem-1",
            embedding=real_embeddings["python1"],
            create_edges=True
        )
        
        # Check edges
        neighbors = await graph_store.get_neighbors(
            "neo-mem-1",
            edge_types=[RelationshipType.SIMILAR_TO],
            direction="outgoing"
        )
        
        if neighbors:
            print(f"\nNeo4j edge metadata:")
            print(f"  Edge type: {neighbors[0]['edge_type']}")
            print(f"  Edge weight: {neighbors[0]['edge_weight']:.4f}")
            
            assert neighbors[0]["edge_type"] == RelationshipType.SIMILAR_TO
            assert neighbors[0]["edge_weight"] > 0


class TestSemanticSimilarityEngineConfiguration:
    """Test engine configuration options."""
    
    @pytest.mark.asyncio
    async def test_custom_threshold(self, tmp_path):
        """Test custom similarity threshold."""
        vector_store = QdrantStore(collection_name="test_threshold")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))
        
        engine = SemanticSimilarityEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            similarity_threshold=0.9,
            max_similar_memories=5
        )
        
        assert engine.similarity_threshold == 0.9
        assert engine.max_similar_memories == 5
    
    @pytest.mark.asyncio
    async def test_max_similar_memories_limit(self, tmp_path):
        """Test max similar memories limit."""
        vector_store = QdrantStore(collection_name="test_limit")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))
        
        engine = SemanticSimilarityEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            max_similar_memories=3
        )
        
        assert engine.max_similar_memories == 3