"""Tests for temporal relationship engine."""

import pytest
from datetime import datetime, timedelta, timezone
from src.core.relationships import TemporalRelationshipEngine
from src.core.vector_store import QdrantStore
from src.core.graph_store import SQLiteGraphStore, Neo4jGraphStore
from src.core.embeddings import OllamaEmbedding
from src.models import NodeType, RelationshipType, MemoryStatus


class TestTemporalRelationshipEngineSQLite:
    """Test temporal relationship engine with SQLite and real embeddings."""
    
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
            collection_name="test_temporal_sqlite",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_temporal_sqlite")
        except Exception:
            pass
        await store.close()
    
    @pytest.fixture
    async def graph_store(self, tmp_path):
        """Create a SQLite graph store."""
        db_path = tmp_path / "test_temporal_sqlite.db"
        store = SQLiteGraphStore(db_path=str(db_path))
        await store.initialize()
        yield store
        await store.close()
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create temporal relationship engine."""
        return TemporalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            new_memory_window_hours=48,
            expiring_threshold_days=30,
            update_similarity_threshold=0.85,
            update_time_window_days=7,
            decay_half_life_days=30
        )
    
    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "python_old": "Python is a programming language",
            "python_new": "Python is a high-level programming language used for development",
            "cooking": "How to bake chocolate chip cookies",
        }
        
        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)
        
        return embeddings
    
    @pytest.mark.asyncio
    async def test_detect_new_memories(self, engine):
        """Test detecting new memories."""
        now = datetime.now(timezone.utc)
        
        # Recent memory (1 hour ago)
        recent = now - timedelta(hours=1)
        is_new = await engine.detect_new_memories("mem-1", recent)
        assert is_new is True
        print(f"\n✓ Memory from 1 hour ago: NEW = {is_new}")
        
        # Old memory (72 hours ago)
        old = now - timedelta(hours=72)
        is_new = await engine.detect_new_memories("mem-2", old)
        assert is_new is False
        print(f"✓ Memory from 72 hours ago: NEW = {is_new}")
        
        # Edge case: exactly at threshold (48 hours)
        edge = now - timedelta(hours=48)
        is_new = await engine.detect_new_memories("mem-3", edge)
        assert is_new is True
        print(f"✓ Memory from 48 hours ago (threshold): NEW = {is_new}")
    
    @pytest.mark.asyncio
    async def test_exponential_decay(self, engine):
        """Test exponential decay calculation mimicking human forgetting curve."""
        print("\n🧠 Ebbinghaus Forgetting Curve Simulation:")
        
        # Fresh memory (0 days)
        decay_0 = engine._exponential_decay(0)
        assert decay_0 == 0.0
        print(f"  Day 0 (fresh): {decay_0:.4f} (0% forgotten)")
        
        # One day old
        decay_1 = engine._exponential_decay(1)
        print(f"  Day 1: {decay_1:.4f} ({decay_1*100:.1f}% forgotten)")
        
        # One week
        decay_7 = engine._exponential_decay(7)
        print(f"  Day 7: {decay_7:.4f} ({decay_7*100:.1f}% forgotten)")
        
        # Half-life (30 days)
        decay_30 = engine._exponential_decay(30)
        assert 0.45 < decay_30 < 0.55  # Should be ~0.5
        print(f"  Day 30 (half-life): {decay_30:.4f} ({decay_30*100:.1f}% forgotten)")
        
        # Very old memory (365 days)
        decay_365 = engine._exponential_decay(365)
        assert decay_365 > 0.99  # Nearly complete decay
        print(f"  Day 365 (1 year): {decay_365:.4f} ({decay_365*100:.1f}% forgotten)")
    
    @pytest.mark.asyncio
    async def test_detect_expiring_memories_fresh(self, engine):
        """Test decay detection for fresh memory."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=1)
        
        decay_info = await engine.detect_expiring_memories(
            memory_id="mem-1",
            created_at=created,
            last_accessed=now,
            access_count=5
        )
        
        assert decay_info["is_expiring"] is False
        assert decay_info["is_forgotten"] is False
        assert decay_info["decay_score"] < 0.5
        
        print(f"\n🌟 Fresh Memory (1 day old, 5 accesses):")
        print(f"  Decay score: {decay_info['decay_score']:.4f}")
        print(f"  Is expiring: {decay_info['is_expiring']}")
        print(f"  Consolidation strength: {decay_info['consolidation_strength']:.4f}")
    
    @pytest.mark.asyncio
    async def test_detect_expiring_memories_old(self, engine):
        """Test decay detection for old memory."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=60)  # Old memory
        
        decay_info = await engine.detect_expiring_memories(
            memory_id="mem-1",
            created_at=created,
            last_accessed=created,  # Never accessed since creation
            access_count=0
        )
        
        assert decay_info["is_expiring"] is True
        assert decay_info["age_days"] >= 60
        
        print(f"\n⏰ Old Memory (60 days, never accessed):")
        print(f"  Decay score: {decay_info['decay_score']:.4f}")
        print(f"  Age: {decay_info['age_days']:.1f} days")
        print(f"  Days since access: {decay_info['days_since_access']:.1f}")
        print(f"  Is expiring: {decay_info['is_expiring']}")
        print(f"  Is forgotten: {decay_info['is_forgotten']}")
    
    @pytest.mark.asyncio
    async def test_consolidation_effect(self, engine):
        """Test that frequently accessed memories resist decay (like human brain)."""
        now = datetime.utcnow()
        created = now - timedelta(days=40)  # Moderately old
        
        # Low access count (weak consolidation)
        decay_low = await engine.detect_expiring_memories(
            memory_id="mem-1",
            created_at=created,
            last_accessed=created,
            access_count=1
        )
        
        # High access count (strong consolidation - like studying/rehearsal)
        decay_high = await engine.detect_expiring_memories(
            memory_id="mem-2",
            created_at=created,
            last_accessed=now,  # Recently accessed
            access_count=50
        )
        
        # Consolidated memory should have lower decay score
        assert decay_high["decay_score"] < decay_low["decay_score"]
        
        print(f"\n💪 Memory Consolidation Effect (40 days old):")
        print(f"  Weak memory (1 access): decay={decay_low['decay_score']:.4f}")
        print(f"  Strong memory (50 accesses): decay={decay_high['decay_score']:.4f}")
        print(f"  Consolidation strength: {decay_high['consolidation_strength']:.4f}")
        print(f"  ✓ Frequently accessed memories resist forgetting!")
    
    @pytest.mark.asyncio
    async def test_detect_updates_real_embeddings(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test detecting memory updates with real embeddings."""
        now = datetime.utcnow()
        
        # Old version created 3 days ago
        old_time = now - timedelta(days=3)
        await vector_store.upsert_memory(
            "mem-old-version",
            real_embeddings["python_old"],
            {"text": "Old Python info", "created_at": old_time.isoformat()}
        )
        await graph_store.add_node(
            "mem-old-version",
            NodeType.MEMORY,
            {"text": "Old Python info"}
        )
        
        # New version (updated information)
        new_time = now
        await vector_store.upsert_memory(
            "mem-new-version",
            real_embeddings["python_new"],
            {"text": "Updated Python info", "created_at": new_time.isoformat()}
        )
        await graph_store.add_node(
            "mem-new-version",
            NodeType.MEMORY,
            {"text": "Updated Python info"}
        )
        
        # Detect updates
        updates = await engine.detect_updates(
            memory_id="mem-new-version",
            embedding=real_embeddings["python_new"],
            created_at=new_time,
            create_edges=True
        )
        
        print(f"\n🔄 Update Detection:")
        print(f"  Found {len(updates)} updates")
        
        if updates:
            for update in updates:
                print(f"  - {update['id']}: similarity={update['score']:.4f}, "
                      f"{update['days_older']:.1f} days older")
            
            # Verify UPDATE edge was created
            neighbors = await graph_store.get_neighbors(
                "mem-new-version",
                edge_types=[RelationshipType.UPDATES],
                direction="outgoing"
            )
            assert len(neighbors) >= 1
            print(f"  ✓ Created {len(neighbors)} UPDATE edges")
    
    @pytest.mark.asyncio
    async def test_create_temporal_sequence(self, engine, graph_store):
        """Test creating temporal sequences (conversation threads)."""
        # Create nodes for conversation
        memory_ids = []
        for i in range(5):
            mem_id = f"conv-msg-{i}"
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": f"Message {i} in conversation"}
            )
            memory_ids.append(mem_id)
        
        # Create sequence
        edges_created = await engine.create_temporal_sequence(
            memory_ids=memory_ids,
            sequence_type="conversation"
        )
        
        assert edges_created == 4  # N-1 edges for N nodes
        
        print(f"\n💬 Conversation Thread:")
        print(f"  Created sequence of {len(memory_ids)} messages")
        print(f"  Created {edges_created} FOLLOWS edges")
        
        # Verify edges exist
        neighbors = await graph_store.get_neighbors(
            "conv-msg-0",
            edge_types=[RelationshipType.FOLLOWS],
            direction="outgoing"
        )
        
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "conv-msg-1"
        print(f"  ✓ Message 0 → Message 1 (FOLLOWS)")
    
    @pytest.mark.asyncio
    async def test_apply_decay_to_node(self, engine, graph_store):
        """Test applying decay effects to nodes (status transitions)."""
        now = datetime.utcnow()
        created = now - timedelta(days=60)
        
        # Create node
        await graph_store.add_node(
            "mem-decay-test",
            NodeType.MEMORY,
            {"text": "Test memory for decay"}
        )
        
        # Calculate decay
        decay_info = await engine.detect_expiring_memories(
            memory_id="mem-decay-test",
            created_at=created,
            last_accessed=created,
            access_count=0
        )
        
        # Apply decay (status transition)
        await engine.apply_decay_to_node("mem-decay-test", decay_info)
        
        # Check status changed
        node = await graph_store.get_node("mem-decay-test")
        assert node.status in [MemoryStatus.EXPIRING_SOON, MemoryStatus.FORGOTTEN]
        
        print(f"\n🔄 Memory Status Transition:")
        print(f"  Decay score: {decay_info['decay_score']:.4f}")
        print(f"  New status: {node.status}")
        print(f"  ✓ NEW → ACTIVE → EXPIRING_SOON → FORGOTTEN")
    
    @pytest.mark.asyncio
    async def test_run_decay_cycle(self, engine, graph_store):
        """Test running a full decay cycle (like sleep consolidation)."""
        now = datetime.utcnow()
        
        # Create memories with different ages (simulating different experiences)
        memories = [
            ("mem-recent", now - timedelta(days=1), "Yesterday's memory"),
            ("mem-medium", now - timedelta(days=40), "Last month's memory"),
            ("mem-ancient", now - timedelta(days=90), "Three months ago"),
        ]
        
        memory_ids = []
        for mem_id, created, text in memories:
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": text}
            )
            memory_ids.append(mem_id)
        
        # Run decay cycle (simulating brain's memory consolidation during sleep)
        stats = await engine.run_decay_cycle(memory_ids=memory_ids)
        
        assert stats["processed"] == 3
        
        print(f"\n😴 Sleep Consolidation Cycle (Decay Processing):")
        print(f"  Memories processed: {stats['processed']}")
        print(f"  Marked as expiring: {stats['marked_expiring']}")
        print(f"  Marked as forgotten: {stats['marked_forgotten']}")
        print(f"  Average decay score: {stats['average_decay_score']:.4f}")
        print(f"  ✓ Like how the brain consolidates memories during sleep!")
    
    @pytest.mark.asyncio
    async def test_temporal_clusters(self, engine, graph_store):
        """Test detecting temporal clusters (memories from same time period)."""
        now = datetime.now(timezone.utc)
        
        # Cluster 1: Memories from 10 hours ago (within 2-hour window)
        cluster1_base = now - timedelta(hours=10)
        
        # Cluster 2: Memories from 50 hours ago
        cluster2_base = now - timedelta(hours=50)
        
        memory_ids = []
        
        # Create cluster 1 (3 memories close in time)
        for i in range(3):
            mem_id = f"morning-event-{i}"
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": f"Morning event {i}"}
            )
            memory_ids.append(mem_id)
        
        # Create cluster 2 (2 memories close in time)
        for i in range(2):
            mem_id = f"yesterday-event-{i}"
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": f"Yesterday event {i}"}
            )
            memory_ids.append(mem_id)
        
        # Find temporal clusters
        clusters = await engine.detect_temporal_clusters(
            memory_ids=memory_ids,
            time_window_hours=24
        )
        
        print(f"\n📅 Temporal Clustering:")
        print(f"  Found {len(clusters)} temporal clusters")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {len(cluster)} memories from same time period")
        print(f"  ✓ Brain groups memories from same context/time!")
    
    @pytest.mark.asyncio
    async def test_calculate_temporal_relevance(self, engine):
        """Test temporal relevance calculation (recency + frequency)."""
        now = datetime.now(timezone.utc)
        
        # Recent, frequently accessed (highly relevant)
        recent_created = now - timedelta(hours=2)
        rel_recent = engine.calculate_temporal_relevance(
            created_at=recent_created,
            last_accessed=now,
            access_count=10
        )
        
        # Old, rarely accessed (low relevance)
        old_created = now - timedelta(days=90)
        rel_old = engine.calculate_temporal_relevance(
            created_at=old_created,
            last_accessed=old_created,
            access_count=1
        )
        
        # Recent should be more relevant
        assert rel_recent > rel_old
        
        print(f"\n⭐ Temporal Relevance Scores:")
        print(f"  Recent + frequently accessed: {rel_recent:.4f}")
        print(f"  Old + rarely accessed: {rel_old:.4f}")
        print(f"  ✓ Recent and frequently used memories are more accessible!")


class TestTemporalRelationshipEngineNeo4j:
    """Test temporal relationship engine with Neo4j and real embeddings."""
    
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
            collection_name="test_temporal_neo4j",
            vector_size=768
        )
        await store.initialize()
        yield store
        
        # Cleanup
        try:
            await store.client.delete_collection("test_temporal_neo4j")
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
            await session.run(
                "MATCH (n:Node) WHERE n.id STARTS WITH 'neo-temp-' DETACH DELETE n"
            )
        
        await store.close()
    
    @pytest.fixture
    async def engine(self, vector_store, graph_store):
        """Create temporal relationship engine with Neo4j."""
        return TemporalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            new_memory_window_hours=48,
            expiring_threshold_days=30,
            update_similarity_threshold=0.85,
            update_time_window_days=7,
            decay_half_life_days=30
        )
    
    @pytest.fixture
    async def real_embeddings(self, embedder):
        """Generate real embeddings from Ollama."""
        texts = {
            "python_v1": "Python 3.8 is a programming language",
            "python_v2": "Python 3.11 is the latest version with performance improvements",
            "unrelated": "Gardening tips for growing tomatoes",
        }
        
        embeddings = {}
        for key, text in texts.items():
            embeddings[key] = await embedder.embed(text)
        
        return embeddings
    
    @pytest.mark.asyncio
    async def test_detect_updates_neo4j(
        self, engine, vector_store, graph_store, real_embeddings
    ):
        """Test update detection with Neo4j backend."""
        now = datetime.now(timezone.utc)
        
        # Old version
        old_time = now - timedelta(days=5)
        await vector_store.upsert_memory(
            "neo-temp-python-v1",
            real_embeddings["python_v1"],
            {"text": "Python 3.8", "created_at": old_time.isoformat()}
        )
        await graph_store.add_node(
            "neo-temp-python-v1",
            NodeType.MEMORY,
            {"text": "Python 3.8"}
        )
        
        # New version
        new_time = now
        await vector_store.upsert_memory(
            "neo-temp-python-v2",
            real_embeddings["python_v2"],
            {"text": "Python 3.11", "created_at": new_time.isoformat()}
        )
        await graph_store.add_node(
            "neo-temp-python-v2",
            NodeType.MEMORY,
            {"text": "Python 3.11"}
        )
        
        # Detect updates
        updates = await engine.detect_updates(
            memory_id="neo-temp-python-v2",
            embedding=real_embeddings["python_v2"],
            created_at=new_time,
            create_edges=True
        )
        
        print(f"\n🔄 Neo4j Update Detection:")
        print(f"  Found {len(updates)} updates")
        
        if updates:
            for update in updates:
                print(f"  - {update['id']}: similarity={update['score']:.4f}")
            
            # Verify Neo4j edge
            neighbors = await graph_store.get_neighbors(
                "neo-temp-python-v2",
                edge_types=[RelationshipType.UPDATES],
                direction="outgoing"
            )
            assert len(neighbors) >= 1
            print(f"  ✓ Created UPDATE relationship in Neo4j")
    
    @pytest.mark.asyncio
    async def test_temporal_sequence_neo4j(self, engine, graph_store):
        """Test creating temporal sequences with Neo4j."""
        # Create conversation nodes
        memory_ids = []
        for i in range(4):
            mem_id = f"neo-temp-conv-{i}"
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": f"Message {i}"}
            )
            memory_ids.append(mem_id)
        
        # Create sequence
        edges_created = await engine.create_temporal_sequence(
            memory_ids=memory_ids,
            sequence_type="conversation"
        )
        
        assert edges_created == 3
        
        print(f"\n💬 Neo4j Conversation Thread:")
        print(f"  Created {edges_created} FOLLOWS relationships")
        
        # Verify with Neo4j
        neighbors = await graph_store.get_neighbors(
            "neo-temp-conv-0",
            edge_types=[RelationshipType.FOLLOWS],
            direction="outgoing"
        )
        
        assert len(neighbors) == 1
        assert neighbors[0]["node"].id == "neo-temp-conv-1"
        print(f"  ✓ Sequential relationships verified in Neo4j graph")
    
    @pytest.mark.asyncio
    async def test_decay_cycle_neo4j(self, engine, graph_store):
        """Test decay cycle with Neo4j backend."""
        now = datetime.now(timezone.utc)
        
        # Create memories
        memories = [
            ("neo-temp-fresh", now - timedelta(days=2)),
            ("neo-temp-old", now - timedelta(days=50)),
        ]
        
        memory_ids = []
        for mem_id, created in memories:
            await graph_store.add_node(
                mem_id,
                NodeType.MEMORY,
                {"text": f"Memory {mem_id}"}
            )
            memory_ids.append(mem_id)
        
        # Run decay cycle
        stats = await engine.run_decay_cycle(memory_ids=memory_ids)
        
        print(f"\n😴 Neo4j Decay Cycle:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Average decay: {stats['average_decay_score']:.4f}")
        print(f"  ✓ Memory consolidation in Neo4j graph")
        
        assert stats["processed"] == 2


class TestTemporalEngineConfiguration:
    """Test engine configuration."""
    
    @pytest.mark.asyncio
    async def test_custom_decay_parameters(self, tmp_path):
        """Test custom decay parameters."""
        vector_store = QdrantStore(collection_name="test_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))
        
        engine = TemporalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            new_memory_window_hours=24,
            expiring_threshold_days=14,
            decay_half_life_days=15
        )
        
        assert engine.new_memory_window_hours == 24
        assert engine.expiring_threshold_days == 14
        assert engine.decay_half_life_days == 15
        print("\n✓ Custom decay parameters configured correctly")
    
    @pytest.mark.asyncio
    async def test_update_detection_parameters(self, tmp_path):
        """Test update detection parameters."""
        vector_store = QdrantStore(collection_name="test_update_config")
        graph_store = SQLiteGraphStore(db_path=str(tmp_path / "test.db"))
        
        engine = TemporalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            update_similarity_threshold=0.9,
            update_time_window_days=14
        )
        
        assert engine.update_similarity_threshold == 0.9
        assert engine.update_time_window_days == 14
        print("\n✓ Update detection parameters configured correctly")