"""Tests for the Memory Engine."""

import pytest
from datetime import datetime, timezone
from src.services import MemoryEngine
from src.core.vector_store import QdrantStore
from src.core.graph_store import SQLiteGraphStore, Neo4jGraphStore
from src.core.embeddings import OllamaEmbedding
from src.config import Config


@pytest.mark.sqlite
class TestMemoryEngineSQLite:
    """Test Memory Engine with SQLite graph store."""
    
    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(
            model="nomic-embed-text",
            host="http://localhost:11434"
        )
    
    @pytest.fixture
    async def vector_store(self):
        """Create Qdrant vector store."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_memory_engine",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_memory_engine")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create SQLite graph store."""
        db_path = tmp_path / "test_memory_engine.db"
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
                "semantic": {"similarity_threshold": 0.5},
            }
        )
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store, embedder, config):
        """Create memory engine."""
        engine = MemoryEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            embedder=embedder,
            config=config
        )
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_add_memory_basic(self, engine):
        """Test adding a basic memory."""
        result = await engine.add_memory(
            text="Python is a programming language",
            metadata={"source": "test"}
        )
        
        print("\n📝 Add Memory Result:")
        print(f"  Memory ID: {result['memory_id']}")
        print(f"  Created at: {result['created_at']}")
        print(f"  Relationships: {result['relationships_created']}")
        
        assert "memory_id" in result
        assert result["text"] == "Python is a programming language"
        assert result["auto_infer_enabled"] is True
        print("  ✓ Memory added successfully")
    
    @pytest.mark.asyncio
    async def test_add_memory_with_relationships(self, engine):
        """Test adding memories that should create relationships."""
        # Add first memory
        result1 = await engine.add_memory(
            text="Python is a programming language"
        )
        
        # Add related memory
        result2 = await engine.add_memory(
            text="Python 3.11 introduces performance improvements"
        )
        
        print("\n🔗 Relationship Creation:")
        print(f"  Memory 1 ID: {result1['memory_id']}")
        print(f"  Memory 2 ID: {result2['memory_id']}")
        print(f"  Memory 2 relationships: {result2['relationships_created']}")
        
        # Second memory should have found relationships with first
        assert result2["relationships_created"] > 0
        print("  ✓ Relationships automatically created")
    
    @pytest.mark.asyncio
    async def test_query_memories(self, engine):
        """Test querying memories."""
        # Add some memories
        await engine.add_memory("Python is a programming language")
        await engine.add_memory("JavaScript is used for web development")
        await engine.add_memory("Python has great libraries for data science")
        
        # Query for Python-related memories
        results = await engine.query_memories(
            query="Python programming",
            limit=5
        )
        
        print("\n🔍 Query Results:")
        print(f"  Query: 'Python programming'")
        print(f"  Results found: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.4f}")
            print(f"     Text: {result['metadata'].get('text', 'N/A')[:50]}...")
        
        assert len(results) > 0
        # Most results should be about Python
        python_results = sum(
            1 for r in results 
            if 'python' in r['metadata'].get('text', '').lower()
        )
        assert python_results >= 2
        print("  ✓ Query returned relevant results")
    
    @pytest.mark.asyncio
    async def test_query_with_relationships(self, engine):
        """Test querying with relationship information."""
        # Add memories
        mem1 = await engine.add_memory("Python programming basics")
        mem2 = await engine.add_memory("Advanced Python techniques")
        
        # Query with relationships
        results = await engine.query_memories(
            query="Python",
            limit=5,
            include_relationships=True
        )
        
        print("\n🔗 Query with Relationships:")
        for result in results:
            print(f"  Memory: {result['id']}")
            if "relationships" in result:
                print(f"    Relationship count: {result['relationships']['count']}")
                print(f"    Relationship types: {result['relationships']['types']}")
        
        # At least one result should have relationships
        has_relationships = any(
            r.get("relationships", {}).get("count", 0) > 0 
            for r in results
        )
        assert has_relationships
        print("  ✓ Relationship information included")
    
    @pytest.mark.asyncio
    async def test_get_memory(self, engine):
        """Test retrieving a specific memory."""
        # Add memory
        result = await engine.add_memory(
            text="Test memory content",
            metadata={"category": "test"}
        )
        memory_id = result["memory_id"]
        
        # Retrieve it
        memory = await engine.get_memory(memory_id)
        
        print("\n📄 Retrieved Memory:")
        print(f"  ID: {memory_id}")
        print(f"  Text: {memory['metadata']['text']}")
        print(f"  Category: {memory['metadata'].get('category')}")
        
        assert memory is not None
        assert memory["id"] == memory_id
        assert memory["metadata"]["text"] == "Test memory content"
        assert memory["metadata"]["category"] == "test"
        print("  ✓ Memory retrieved successfully")
    
    @pytest.mark.asyncio
    async def test_get_memory_with_relationships(self, engine):
        """Test retrieving memory with relationship details."""
        # Add related memories
        mem1 = await engine.add_memory("First memory")
        mem2 = await engine.add_memory("Related second memory")
        
        # Get with relationships
        memory = await engine.get_memory(
            mem2["memory_id"],
            include_relationships=True
        )
        
        print("\n🔗 Memory with Relationships:")
        print(f"  Memory ID: {memory['id']}")
        if "relationships" in memory:
            print(f"  Relationship count: {memory['relationships']['count']}")
            print(f"  Neighbors: {len(memory['relationships']['neighbors'])}")
            for neighbor in memory['relationships']['neighbors'][:3]:
                print(f"    - {neighbor['id']} ({neighbor['type']})")
        
        assert "relationships" in memory
        print("  ✓ Relationships included")
    
    @pytest.mark.asyncio
    async def test_update_memory(self, engine):
        """Test updating a memory."""
        # Add memory
        result = await engine.add_memory("Original text")
        memory_id = result["memory_id"]
        
        # Update it
        update_result = await engine.update_memory(
            memory_id=memory_id,
            text="Updated text",
            metadata={"updated": True}
        )
        
        print("\n✏️ Update Result:")
        print(f"  Memory ID: {memory_id}")
        print(f"  Updated: {update_result['updated']}")
        print(f"  Text changed: {update_result['text_changed']}")
        
        # Verify update
        memory = await engine.get_memory(memory_id)
        assert memory["metadata"]["text"] == "Updated text"
        assert memory["metadata"]["updated"] is True
        print("  ✓ Memory updated successfully")
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, engine):
        """Test deleting a memory."""
        # Add memory
        result = await engine.add_memory("Memory to delete")
        memory_id = result["memory_id"]
        
        # Delete it
        delete_result = await engine.delete_memory(memory_id)
        
        print("\n🗑️ Delete Result:")
        print(f"  Memory ID: {memory_id}")
        print(f"  Deleted: {delete_result['deleted']}")
        print(f"  Relationships removed: {delete_result['relationships_removed']}")
        
        # Verify deletion
        memory = await engine.get_memory(memory_id)
        assert memory is None
        print("  ✓ Memory deleted successfully")
    
    @pytest.mark.asyncio
    async def test_add_conversation(self, engine):
        """Test adding a conversation."""
        messages = [
            {"text": "Hello, how are you?", "role": "user"},
            {"text": "I'm doing well, thank you!", "role": "assistant"},
            {"text": "What's the weather like?", "role": "user"},
            {"text": "It's sunny and warm today.", "role": "assistant"},
        ]
        
        result = await engine.add_conversation(
            messages=messages,
            metadata={"session": "test-session"}
        )
        
        print("\n💬 Conversation Result:")
        print(f"  Conversation ID: {result['conversation_id']}")
        print(f"  Message count: {result['message_count']}")
        print(f"  Edges created: {result['edges_created']}")
        
        assert result["message_count"] == 4
        assert result["edges_created"] == 3  # N-1 edges
        assert len(result["message_ids"]) == 4
        print("  ✓ Conversation added successfully")
    
    @pytest.mark.asyncio
    async def test_add_document(self, engine):
        """Test adding a document with chunking."""
        long_text = """
        Python is a high-level programming language. It was created by Guido van Rossum
        and first released in 1991. Python's design philosophy emphasizes code readability
        with its notable use of significant indentation.
        
        Python is dynamically typed and garbage-collected. It supports multiple programming
        paradigms, including structured, object-oriented and functional programming.
        
        Python is often described as a "batteries included" language due to its comprehensive
        standard library. It has a large and active community that contributes to a vast
        collection of third-party packages.
        """
        
        result = await engine.add_document(
            text=long_text,
            chunk_size=200,
            chunk_overlap=50,
            metadata={"title": "Python Introduction"}
        )
        
        print("\n📚 Document Result:")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Chunk count: {result['chunk_count']}")
        print(f"  Relationships: {result['relationships_created']}")
        
        assert result["chunk_count"] > 1
        assert result["relationships_created"] > 0
        assert len(result["chunk_ids"]) == result["chunk_count"]
        print("  ✓ Document added and chunked successfully")
    
    @pytest.mark.asyncio
    async def test_memory_without_auto_infer(self, engine):
        """Test adding memory without automatic relationship inference."""
        result = await engine.add_memory(
            text="Standalone memory",
            auto_infer_relationships=False
        )
        
        print("\n⚙️ Manual Mode Result:")
        print(f"  Memory ID: {result['memory_id']}")
        print(f"  Auto infer: {result['auto_infer_enabled']}")
        print(f"  Relationships: {result['relationships_created']}")
        
        assert result["auto_infer_enabled"] is False
        assert result["relationships_created"] == 0
        assert "engines_run" not in result
        print("  ✓ Memory added without automatic relationships")


@pytest.mark.neo4j
class TestMemoryEngineNeo4j:
    """Test Memory Engine with Neo4j graph store."""
    
    @pytest.fixture
    async def embedder(self):
        """Create Ollama embedder."""
        return OllamaEmbedding(
            model="nomic-embed-text",
            host="http://localhost:11434"
        )
    
    @pytest.fixture
    async def vector_store(self):
        """Create Qdrant vector store."""
        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_memory_engine_neo4j",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_memory_engine_neo4j")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    async def graph_store(self):
        """Create Neo4j graph store."""
        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="mnemograph123"
        )
        await store.initialize()
        yield store
        
        # Cleanup - delete all test nodes (everything since this is a test collection)
        await store.driver.execute_query(
            "MATCH (n) DETACH DELETE n"
        )
        await store.close()
    
    @pytest.fixture
    async def config(self):
        """Create test configuration."""
        return Config(
            relationships={
                "auto_infer_on_add": True,
                "semantic": {"similarity_threshold": 0.5},
            }
        )
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store, embedder, config):
        """Create memory engine."""
        engine = MemoryEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            embedder=embedder,
            config=config
        )
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_add_memory_neo4j(self, engine):
        """Test adding memory with Neo4j."""
        result = await engine.add_memory(
            text="Neo4j test memory",
            metadata={"source": "test"}
        )
        
        print("\n📝 Add Memory (Neo4j):")
        print(f"  Memory ID: {result['memory_id']}")
        print(f"  Relationships: {result['relationships_created']}")
        
        assert "memory_id" in result
        print("  ✓ Memory added to Neo4j successfully")
    
    @pytest.mark.asyncio
    async def test_query_neo4j(self, engine):
        """Test querying with Neo4j."""
        await engine.add_memory("Python programming")
        await engine.add_memory("Python data science")
        
        results = await engine.query_memories("Python", limit=5)
        
        print("\n🔍 Query (Neo4j):")
        print(f"  Results: {len(results)}")
        
        assert len(results) > 0
        print("  ✓ Query successful with Neo4j")
    
    @pytest.mark.asyncio
    async def test_conversation_neo4j(self, engine):
        """Test conversation with Neo4j."""
        messages = [
            {"text": "Hello"},
            {"text": "Hi there"},
            {"text": "How are you?"},
        ]
        
        result = await engine.add_conversation(messages)
        
        print("\n💬 Conversation (Neo4j):")
        print(f"  Messages: {result['message_count']}")
        print(f"  Edges: {result['edges_created']}")
        
        assert result["edges_created"] == 2
        print("  ✓ Conversation created in Neo4j")

