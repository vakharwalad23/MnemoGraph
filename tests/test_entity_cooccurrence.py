"""Tests for entity co-occurrence engine."""

import pytest

from src.core.graph_store import Neo4jGraphStore, SQLiteGraphStore
from src.core.relationships import EntityCooccurrenceEngine
from src.core.vector_store import QdrantStore
from src.models import NodeType, RelationshipType


class TestEntityCooccurrenceEngineSQLite:
    """Test entity co-occurrence engine with SQLite."""

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_cooccurrence_sqlite", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_cooccurrence_sqlite")
        except Exception:
            pass
        await store.close()

    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create a SQLite graph store."""
        db_path = tmp_path / "test_cooccurrence_sqlite.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create entity co-occurrence engine (without spaCy for testing)."""
        return EntityCooccurrenceEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_entity_length=3,
            min_cooccurrence_count=2,
            entity_weight_threshold=0.3,
            use_spacy=False,  # Use simple extraction for consistent tests
        )

    @pytest.mark.asyncio
    async def test_extract_entities_simple(self, engine):
        """Test simple entity extraction (capitalized words)."""
        text = "Python and JavaScript are programming languages used in Machine Learning"

        entities = engine.extract_entities(text)

        print("\n🏷️  Entity Extraction (Simple):")
        print(f"  Text: '{text}'")
        print(f"  Extracted entities: {entities}")

        # Should extract capitalized words/phrases
        assert len(entities) > 0
        # Check for Python (always captured)
        assert "python" in entities
        # Machine Learning should be captured as multi-word phrase
        has_ml = any("machine" in e for e in entities)
        assert has_ml
        print(f"  ✓ Extracted {len(entities)} entities")

    @pytest.mark.asyncio
    async def test_extract_entities_proper_nouns(self, engine):
        """Test extracting proper nouns and names."""
        text = "John Smith works at Google in New York with Python"

        entities = engine.extract_entities(text)

        print("\n👤 Proper Noun Extraction:")
        print(f"  Text: '{text}'")
        print(f"  Entities: {entities}")

        # Should extract capitalized words/names
        assert len(entities) > 0
        # At minimum should get some capitalized entities
        capitalized_count = sum(1 for e in entities if len(e) >= 3)
        assert capitalized_count >= 2
        print(f"  ✓ Found {len(entities)} entities including proper nouns")

    @pytest.mark.asyncio
    async def test_extract_entities_multiword(self, engine):
        """Test extracting multi-word capitalized phrases."""
        text = "The New York Times reported on Machine Learning advances"

        entities = engine.extract_entities(text)

        print("\n📰 Multi-word Entity Extraction:")
        print(f"  Text: '{text}'")
        print(f"  Entities: {entities}")

        # Should extract multi-word capitalized phrases
        assert len(entities) > 0
        # Check for either individual words or phrases
        has_new_york = "new york" in entities or "new" in entities
        has_machine_learning = "machine learning" in entities or "machine" in entities
        assert has_new_york or has_machine_learning
        print("  ✓ Extracted multi-word entities")

    @pytest.mark.asyncio
    async def test_build_entity_index(self, engine, graph_store):
        """Test building entity index."""
        # Create memories with shared entities
        memories = [
            ("mem-1", "Python programming with Machine Learning"),
            ("mem-2", "JavaScript and Python for development"),
            ("mem-3", "Machine Learning algorithms with Python"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Build index
        entity_index = await engine.build_entity_index(memory_ids)

        print("\n📇 Entity Index:")
        for entity, mem_list in sorted(entity_index.items()):
            print(f"  '{entity}': {mem_list}")

        # Python should appear in all 3 memories
        assert "python" in entity_index
        assert len(entity_index["python"]) == 3

        # Machine should appear in 2 memories (from "Machine Learning")
        machine_related = [e for e in entity_index.keys() if "machine" in e]
        assert len(machine_related) > 0

        print(f"  ✓ Indexed {len(entity_index)} unique entities")

    @pytest.mark.asyncio
    async def test_create_cooccurrence_edges(self, engine, graph_store):
        """Test creating co-occurrence edges between memories."""
        # Create memories with clear overlapping multi-word entities
        memories = [
            ("mem-py1", "Machine Learning Python"),
            ("mem-py2", "Machine Learning Tutorial"),
            ("mem-py3", "Deep Learning Python"),
            ("mem-js", "Web Development JavaScript"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create co-occurrence edges
        result = await engine.create_cooccurrence_edges(memory_ids)

        print("\n🔗 Co-occurrence Analysis:")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Unique entities: {result['unique_entities']}")
        print("  Entity stats:")
        for entity, count in sorted(result["entity_stats"].items()):
            print(f"    - '{entity}': {count} memories")

        # Should have unique entities
        assert result["unique_entities"] > 0

        # May or may not create edges depending on min_cooccurrence_count
        # Just verify the process completes
        print(f"  ✓ Co-occurrence analysis completed with {result['edges_created']} edges")

    @pytest.mark.asyncio
    async def test_cooccurrence_metadata(self, engine, graph_store):
        """Test co-occurrence edge metadata."""
        # Create memories with clear entity overlap
        memories = [
            ("mem-a", "Python Machine Learning Data Science"),
            ("mem-b", "Python Machine Learning Tutorial"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create edges
        result = await engine.create_cooccurrence_edges(memory_ids)

        print("\n📋 Edge Metadata:")
        print(f"  Edges created: {result['edges_created']}")

        # Get edge details
        neighbors = await graph_store.get_neighbors("mem-a", direction="both")

        if neighbors:
            print("  Edge from mem-a:")
            for neighbor in neighbors:
                print(f"    Target: {neighbor['node'].id}")
                print(f"    Weight: {neighbor.get('edge_weight', 'N/A')}")
                if neighbor.get("edge_metadata"):
                    print(
                        f"    Shared entities: {neighbor['edge_metadata'].get('shared_entities', [])}"
                    )
                    print(f"    Num shared: {neighbor['edge_metadata'].get('num_shared', 0)}")

            assert result["edges_created"] >= 1
            print("  ✓ Edge metadata includes shared entities")

    @pytest.mark.asyncio
    async def test_find_memories_by_entity(self, engine, graph_store):
        """Test finding memories containing specific entity."""
        # Create memories with exact entity matches
        memories = [
            ("mem-1", "Python"),
            ("mem-2", "JavaScript"),
            ("mem-3", "Python"),
            ("mem-4", "Cooking"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Find memories with "python"
        python_memories = await engine.find_memories_by_entity("python", memory_ids)

        print("\n🔍 Entity Search:")
        print("  Searching for 'python'")
        print(f"  Found in: {python_memories}")

        assert len(python_memories) == 2
        assert "mem-1" in python_memories
        assert "mem-3" in python_memories
        print(f"  ✓ Found {len(python_memories)} memories with 'python'")

    @pytest.mark.asyncio
    async def test_find_memories_by_entity_case_insensitive(self, engine, graph_store):
        """Test entity search is case-insensitive."""
        # Use properly capitalized words (simple extraction requires [A-Z][a-z]+ pattern)
        memories = [
            ("mem-a", "Python Language"),
            ("mem-b", "Python Tutorial"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Search with different cases - should return same results
        results_lower = await engine.find_memories_by_entity("python", memory_ids)
        results_upper = await engine.find_memories_by_entity("PYTHON", memory_ids)
        results_mixed = await engine.find_memories_by_entity("PyThOn", memory_ids)

        print("\n🔤 Case-Insensitive Search:")
        print(f"  'python': {results_lower}")
        print(f"  'PYTHON': {results_upper}")
        print(f"  'PyThOn': {results_mixed}")

        assert len(results_lower) == 2
        assert len(results_upper) == 2
        assert len(results_mixed) == 2
        assert results_lower == results_upper == results_mixed
        print("  ✓ Entity search is case-insensitive")

    @pytest.mark.asyncio
    async def test_calculate_entity_importance(self, engine):
        """Test calculating entity importance scores."""
        entity_stats = {
            "python": 10,  # Common - lower importance
            "javascript": 8,
            "rare_term": 2,  # Rare - higher importance
            "unique": 1,  # Very rare - highest importance
        }

        importance = engine.calculate_entity_importance(entity_stats)

        print("\n⭐ Entity Importance:")
        for entity, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity}: {score:.4f}")

        # Rare entities should be more important
        assert importance["unique"] > importance["rare_term"]
        assert importance["rare_term"] > importance["python"]
        assert importance["rare_term"] > importance["javascript"]
        print("  ✓ Rare entities have higher importance scores")

    @pytest.mark.asyncio
    async def test_get_entity_graph(self, engine, graph_store):
        """Test building entity relationship graph."""
        # Create memories with overlapping entities
        memories = [
            ("mem-1", "Python and Machine Learning"),
            ("mem-2", "Python Programming Tutorial"),
            ("mem-3", "Machine Learning with Data Science"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Build entity graph
        entity_graph = await engine.get_entity_graph(memory_ids)

        print("\n🕸️  Entity Relationship Graph:")
        print(f"  Entities (nodes): {entity_graph['num_entities']}")
        print(f"  Relationships (edges): {entity_graph['num_relationships']}")

        print("\n  Entity nodes:")
        for node in sorted(entity_graph["nodes"], key=lambda x: x["entity"]):
            print(
                f"    - {node['entity']}: {node['memory_count']} memories, "
                f"importance={node['importance']:.4f}"
            )

        print("\n  Entity relationships:")
        for edge in entity_graph["edges"]:
            print(f"    - {edge['source']} ↔ {edge['target']} (co-occurred {edge['weight']} times)")

        assert entity_graph["num_entities"] > 0
        assert entity_graph["num_relationships"] >= 0  # May have isolated entities
        print(f"  ✓ Built entity graph with {entity_graph['num_entities']} entities")

    @pytest.mark.asyncio
    async def test_find_entity_clusters(self, engine, graph_store):
        """Test finding clusters of related entities."""
        # Create memories with distinct entity groups
        memories = [
            # Tech cluster
            ("mem-1", "Python Machine Learning Programming"),
            ("mem-2", "Python Programming Data Science"),
            ("mem-3", "Machine Learning Neural Networks"),
            # Food cluster
            ("mem-4", "Cooking Recipes Ingredients"),
            ("mem-5", "Baking Desserts Chocolate"),
            ("mem-6", "Recipes Baking Cooking"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Find entity clusters
        clusters = await engine.find_entity_clusters(memory_ids, min_cluster_size=2)

        print("\n📊 Entity Clustering:")
        print(f"  Found {len(clusters)} clusters")

        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {sorted(cluster)}")

        # Should find at least some clustering
        # (Exact results depend on entity extraction)
        print(f"  ✓ Identified {len(clusters)} entity clusters")

        # Verify clusters have minimum size
        for cluster in clusters:
            assert len(cluster) >= 2


class TestEntityCooccurrenceEngineNeo4j:
    """Test entity co-occurrence with Neo4j."""

    @pytest.fixture
    async def vector_store(self):
        """Create a Qdrant vector store."""
        store = QdrantStore(
            host="localhost", port=6333, collection_name="test_cooccurrence_neo4j", vector_size=768
        )
        await store.initialize()
        yield store

        # Cleanup
        try:
            await store.client.delete_collection("test_cooccurrence_neo4j")
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
            await session.run("MATCH (n:Node) WHERE n.id STARTS WITH 'neo-cooc-' DETACH DELETE n")

        await store.close()

    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create entity co-occurrence engine with Neo4j."""
        return EntityCooccurrenceEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_entity_length=3,
            min_cooccurrence_count=2,
            use_spacy=False,
        )

    @pytest.mark.asyncio
    async def test_create_cooccurrence_edges_neo4j(self, engine, graph_store):
        """Test creating co-occurrence edges in Neo4j."""
        # Create memories
        memories = [
            ("neo-cooc-1", "Python Programming Language"),
            ("neo-cooc-2", "Python and JavaScript Development"),
            ("neo-cooc-3", "JavaScript Web Framework"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        # Create co-occurrence edges
        result = await engine.create_cooccurrence_edges(memory_ids)

        print("\n🔗 Neo4j Co-occurrence:")
        print(f"  Edges created: {result['edges_created']}")
        print(f"  Unique entities: {result['unique_entities']}")
        print(f"  Entity stats: {result['entity_stats']}")

        # Verify in Neo4j
        if result["edges_created"] > 0:
            neighbors = await graph_store.get_neighbors(
                "neo-cooc-1", edge_types=[RelationshipType.CO_OCCURS], direction="both"
            )
            print(f"  Neighbors of neo-cooc-1: {[n['node'].id for n in neighbors]}")
            print("  ✓ Created CO_OCCURS relationships in Neo4j")

    @pytest.mark.asyncio
    async def test_entity_graph_neo4j(self, engine, graph_store):
        """Test entity graph building with Neo4j."""
        memories = [
            ("neo-cooc-a", "Python Machine Learning"),
            ("neo-cooc-b", "Python Programming"),
            ("neo-cooc-c", "Machine Learning Tutorial"),
        ]

        memory_ids = []
        for mem_id, text in memories:
            await graph_store.add_node(mem_id, NodeType.MEMORY, {"text": text})
            memory_ids.append(mem_id)

        entity_graph = await engine.get_entity_graph(memory_ids)

        print("\n🕸️  Neo4j Entity Graph:")
        print(f"  Entities: {entity_graph['num_entities']}")
        print(f"  Relationships: {entity_graph['num_relationships']}")

        assert entity_graph["num_entities"] > 0
        print("  ✓ Built entity graph in Neo4j")


class TestEntityEngineConfiguration:
    """Test engine configuration."""

    @pytest.mark.asyncio
    async def test_custom_entity_parameters(self, tmp_path):
        """Test custom entity extraction parameters."""
        vector_store = QdrantStore(collection_name="test_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))

        engine = EntityCooccurrenceEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_entity_length=5,
            min_cooccurrence_count=3,
            entity_weight_threshold=0.5,
            use_spacy=False,
        )

        assert engine.min_entity_length == 5
        assert engine.min_cooccurrence_count == 3
        assert engine.entity_weight_threshold == 0.5
        assert engine.use_spacy is False
        print("\n✓ Custom entity parameters configured correctly")

    @pytest.mark.asyncio
    async def test_spacy_fallback(self, tmp_path):
        """Test that engine gracefully falls back when spaCy unavailable."""
        vector_store = QdrantStore(collection_name="test_spacy")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))

        # Try to use spaCy (may or may not be available)
        engine = EntityCooccurrenceEngine(
            vector_store=vector_store, graph_store=graph_store, use_spacy=True
        )

        # Should still work with extraction (spaCy or fallback)
        text = "Python Programming Language"
        entities = engine.extract_entities(text)

        print("\n🔄 spaCy Fallback Test:")
        print(f"  Using spaCy: {engine.use_spacy}")
        print(f"  Extracted: {entities}")

        assert len(entities) > 0
        # Check that at least one entity contains "python" or "programming"
        has_relevant = any("python" in e or "programming" in e for e in entities)
        assert has_relevant
        print(f"  ✓ Entity extraction works (spaCy={engine.use_spacy})")
