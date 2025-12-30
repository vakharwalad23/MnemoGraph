"""
Tests for all model classes in MnemoGraph.

Test Organization:
1. Source Content Models: Note, Document, Chunk
2. Memory Model: Memory with source linkage
3. Relationship Models: Edge, Relationship, RelationshipBundle
4. Version Models: VersionChain, MemoryEvolution, InvalidationResult
5. Enums: NodeType, SourceType, MemoryStatus, ContentStatus, RelationshipType
6. Utility Functions: compute_content_hash
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from src.models import (
    Chunk,
    ContentStatus,
    ContextBundle,
    DerivedInsight,
    Document,
    Edge,
    InvalidationResult,
    InvalidationStatus,
    Memory,
    MemoryEvolution,
    MemoryStatus,
    NodeType,
    Note,
    Relationship,
    RelationshipBundle,
    RelationshipType,
    SourceType,
    VersionChain,
    VersionChange,
    compute_content_hash,
)


class TestNote:
    """Tests for Note source content model."""

    def test_note_creation_minimal(self):
        """Test note creation with minimal required fields."""
        note = Note(
            id="note_001",
            user_id="user_001",
            content="This is a test note.",
            content_hash="sha256:abc123",
        )

        assert note.id == "note_001"
        assert note.user_id == "user_001"
        assert note.content == "This is a test note."
        assert note.content_hash == "sha256:abc123"
        assert note.type == NodeType.NOTE
        assert note.status == ContentStatus.ACTIVE
        assert note.title is None
        assert note.tags == []
        assert note.metadata == {}
        assert note.memory_ids == []
        assert note.embedding == []

    def test_note_creation_full(self):
        """Test note creation with all fields."""
        now = datetime.now()
        note = Note(
            id="note_002",
            user_id="user_001",
            content="Full note content",
            content_hash="sha256:def456",
            embedding=[0.1, 0.2, 0.3],
            title="My Note Title",
            tags=["python", "testing"],
            metadata={"source": "manual", "priority": "high"},
            memory_ids=["mem_001", "mem_002"],
            status=ContentStatus.ARCHIVED,
            created_at=now,
            updated_at=now,
        )

        assert note.title == "My Note Title"
        assert note.tags == ["python", "testing"]
        assert note.metadata == {"source": "manual", "priority": "high"}
        assert note.memory_ids == ["mem_001", "mem_002"]
        assert note.status == ContentStatus.ARCHIVED
        assert note.embedding == [0.1, 0.2, 0.3]

    def test_note_content_preview(self):
        """Test content_preview property."""
        short_note = Note(
            id="note_003",
            user_id="user_001",
            content="Short content",
            content_hash="sha256:short",
        )
        assert short_note.content_preview == "Short content"

        long_content = "A" * 300
        long_note = Note(
            id="note_004",
            user_id="user_001",
            content=long_content,
            content_hash="sha256:long",
        )
        assert len(long_note.content_preview) == 200
        assert long_note.content_preview == "A" * 200

    def test_note_is_active(self):
        """Test is_active method."""
        active_note = Note(
            id="note_005",
            user_id="user_001",
            content="Active note",
            content_hash="sha256:active",
            status=ContentStatus.ACTIVE,
        )
        assert active_note.is_active() is True

        archived_note = Note(
            id="note_006",
            user_id="user_001",
            content="Archived note",
            content_hash="sha256:archived",
            status=ContentStatus.ARCHIVED,
        )
        assert archived_note.is_active() is False

    def test_note_validation_required_fields(self):
        """Test note validation for required fields."""
        with pytest.raises(ValidationError):
            Note(user_id="user_001", content="Missing id", content_hash="sha256:x")

        with pytest.raises(ValidationError):
            Note(id="note_001", content="Missing user_id", content_hash="sha256:x")

        with pytest.raises(ValidationError):
            Note(id="note_001", user_id="user_001", content_hash="sha256:x")

        with pytest.raises(ValidationError):
            Note(id="note_001", user_id="user_001", content="Missing hash")


class TestDocument:
    """Tests for Document source content model."""

    def test_document_creation_minimal(self):
        """Test document creation with minimal required fields."""
        doc = Document(
            id="doc_001",
            user_id="user_001",
            summary="This is a document summary.",
            content_hash="sha256:doc123",
        )

        assert doc.id == "doc_001"
        assert doc.user_id == "user_001"
        assert doc.summary == "This is a document summary."
        assert doc.content_hash == "sha256:doc123"
        assert doc.type == NodeType.DOCUMENT
        assert doc.status == ContentStatus.ACTIVE
        assert doc.chunk_ids == []
        assert doc.chunk_count == 0
        assert doc.memory_ids == []
        assert doc.title is None
        assert doc.source_url is None
        assert doc.total_tokens == 0
        assert doc.chunking_strategy == "fixed"
        assert doc.summary_model == ""

    def test_document_creation_full(self):
        """Test document creation with all fields."""
        now = datetime.now()
        doc = Document(
            id="doc_002",
            user_id="user_001",
            summary="Complete document summary",
            content_hash="sha256:docfull",
            embedding=[0.1, 0.2, 0.3, 0.4],
            chunk_ids=["doc_002_chunk_0", "doc_002_chunk_1", "doc_002_chunk_2"],
            chunk_count=3,
            title="My Research Paper",
            source_url="https://example.com/paper.pdf",
            total_tokens=5000,
            tags=["research", "AI"],
            metadata={"author": "John Doe"},
            memory_ids=["mem_010", "mem_011"],
            chunking_strategy="semantic",
            summary_model="llama3.1:8b",
            status=ContentStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

        assert doc.chunk_count == 3
        assert len(doc.chunk_ids) == 3
        assert doc.title == "My Research Paper"
        assert doc.source_url == "https://example.com/paper.pdf"
        assert doc.total_tokens == 5000
        assert doc.chunking_strategy == "semantic"
        assert doc.summary_model == "llama3.1:8b"

    def test_document_content_preview(self):
        """Test content_preview property returns summary."""
        doc = Document(
            id="doc_003",
            user_id="user_001",
            summary="Short summary",
            content_hash="sha256:prev",
        )
        assert doc.content_preview == "Short summary"

        long_summary = "B" * 300
        doc_long = Document(
            id="doc_004",
            user_id="user_001",
            summary=long_summary,
            content_hash="sha256:prevlong",
        )
        assert len(doc_long.content_preview) == 200

    def test_document_is_active(self):
        """Test is_active method."""
        active_doc = Document(
            id="doc_005",
            user_id="user_001",
            summary="Active document",
            content_hash="sha256:active",
        )
        assert active_doc.is_active() is True

    def test_document_has_chunks(self):
        """Test has_chunks method."""
        empty_doc = Document(
            id="doc_006",
            user_id="user_001",
            summary="No chunks",
            content_hash="sha256:empty",
        )
        assert empty_doc.has_chunks() is False

        doc_with_chunks = Document(
            id="doc_007",
            user_id="user_001",
            summary="Has chunks",
            content_hash="sha256:chunks",
            chunk_ids=["doc_007_chunk_0"],
            chunk_count=1,
        )
        assert doc_with_chunks.has_chunks() is True

    def test_document_validation(self):
        """Test document validation for required fields."""
        with pytest.raises(ValidationError):
            Document(user_id="user_001", summary="Missing id", content_hash="sha256:x")

        with pytest.raises(ValidationError):
            Document(id="doc_001", summary="Missing user_id", content_hash="sha256:x")


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation_minimal(self):
        """Test chunk creation with minimal required fields."""
        chunk = Chunk(
            id="doc_001_chunk_0",
            user_id="user_001",
            document_id="doc_001",
            content="This is chunk content.",
            content_hash="sha256:chunk0",
            chunk_index=0,
            chunk_total=3,
            start_token=0,
            end_token=1000,
        )

        assert chunk.id == "doc_001_chunk_0"
        assert chunk.user_id == "user_001"
        assert chunk.document_id == "doc_001"
        assert chunk.content == "This is chunk content."
        assert chunk.type == NodeType.CHUNK
        assert chunk.chunk_index == 0
        assert chunk.chunk_total == 3
        assert chunk.start_token == 0
        assert chunk.end_token == 1000
        assert chunk.overlap_tokens == 0
        assert chunk.summary == ""
        assert chunk.embedding == []

    def test_chunk_creation_full(self):
        """Test chunk creation with all fields including overlap."""
        chunk = Chunk(
            id="doc_001_chunk_1",
            user_id="user_001",
            document_id="doc_001",
            content="Chunk with overlap content.",
            summary="Summary of this chunk.",
            content_hash="sha256:chunk1",
            embedding=[0.5, 0.6, 0.7],
            chunk_index=1,
            chunk_total=3,
            start_token=900,
            end_token=1900,
            overlap_tokens=100,
        )

        assert chunk.chunk_index == 1
        assert chunk.overlap_tokens == 100
        assert chunk.summary == "Summary of this chunk."
        assert chunk.embedding == [0.5, 0.6, 0.7]

    def test_chunk_content_preview_with_summary(self):
        """Test content_preview prefers summary when available."""
        chunk = Chunk(
            id="doc_001_chunk_2",
            user_id="user_001",
            document_id="doc_001",
            content="Full chunk content here.",
            summary="Chunk summary",
            content_hash="sha256:chunk2",
            chunk_index=2,
            chunk_total=3,
            start_token=1800,
            end_token=2500,
        )
        assert chunk.content_preview == "Chunk summary"

    def test_chunk_content_preview_without_summary(self):
        """Test content_preview falls back to content when no summary."""
        chunk = Chunk(
            id="doc_001_chunk_3",
            user_id="user_001",
            document_id="doc_001",
            content="Only content, no summary.",
            content_hash="sha256:chunk3",
            chunk_index=0,
            chunk_total=1,
            start_token=0,
            end_token=500,
        )
        assert chunk.content_preview == "Only content, no summary."

    def test_chunk_is_first_and_is_last(self):
        """Test is_first and is_last methods."""
        first_chunk = Chunk(
            id="doc_001_chunk_0",
            user_id="user_001",
            document_id="doc_001",
            content="First chunk",
            content_hash="sha256:first",
            chunk_index=0,
            chunk_total=3,
            start_token=0,
            end_token=1000,
        )
        assert first_chunk.is_first() is True
        assert first_chunk.is_last() is False

        last_chunk = Chunk(
            id="doc_001_chunk_2",
            user_id="user_001",
            document_id="doc_001",
            content="Last chunk",
            content_hash="sha256:last",
            chunk_index=2,
            chunk_total=3,
            start_token=1800,
            end_token=2500,
        )
        assert last_chunk.is_first() is False
        assert last_chunk.is_last() is True

        single_chunk = Chunk(
            id="doc_002_chunk_0",
            user_id="user_001",
            document_id="doc_002",
            content="Only chunk",
            content_hash="sha256:only",
            chunk_index=0,
            chunk_total=1,
            start_token=0,
            end_token=500,
        )
        assert single_chunk.is_first() is True
        assert single_chunk.is_last() is True

    def test_chunk_validation(self):
        """Test chunk validation for required fields."""
        with pytest.raises(ValidationError):
            Chunk(
                id="chunk_001",
                user_id="user_001",
                content="Missing document_id",
                content_hash="sha256:x",
                chunk_index=0,
                chunk_total=1,
                start_token=0,
                end_token=100,
            )

        with pytest.raises(ValidationError):
            Chunk(
                id="chunk_001",
                user_id="user_001",
                document_id="doc_001",
                content="Invalid chunk_index",
                content_hash="sha256:x",
                chunk_index=-1,  # Must be >= 0
                chunk_total=1,
                start_token=0,
                end_token=100,
            )


class TestMemory:
    """Tests for Memory model with source linkage."""

    def test_memory_creation_minimal(self):
        """Test memory creation with minimal required fields."""
        memory = Memory(
            id="mem_001",
            content="Test memory content",
            user_id="user_001",
        )

        assert memory.id == "mem_001"
        assert memory.content == "Test memory content"
        assert memory.user_id == "user_001"
        assert memory.type == NodeType.MEMORY
        assert memory.status == MemoryStatus.ACTIVE
        assert memory.version == 1
        assert memory.confidence == 1.0
        assert memory.access_count == 0
        assert memory.content_hash == ""
        assert memory.source_id is None
        assert memory.source_type is None
        assert memory.source_chunk_id is None

    def test_memory_with_source_linkage_note(self):
        """Test memory linked to a Note source."""
        memory = Memory(
            id="mem_002",
            content="Memory extracted from note",
            user_id="user_001",
            content_hash="sha256:mem002",
            source_id="note_001",
            source_type=SourceType.NOTE,
        )

        assert memory.source_id == "note_001"
        assert memory.source_type == SourceType.NOTE
        assert memory.source_chunk_id is None

    def test_memory_with_source_linkage_document(self):
        """Test memory linked to a Document source."""
        memory = Memory(
            id="mem_003",
            content="Memory extracted from document",
            user_id="user_001",
            content_hash="sha256:mem003",
            source_id="doc_001",
            source_type=SourceType.DOCUMENT,
            source_chunk_id="doc_001_chunk_2",
        )

        assert memory.source_id == "doc_001"
        assert memory.source_type == SourceType.DOCUMENT
        assert memory.source_chunk_id == "doc_001_chunk_2"

    def test_memory_with_metadata(self):
        """Test memory creation with metadata."""
        metadata = {"source": "test", "importance": "high"}
        memory = Memory(
            id="mem_004",
            content="Test memory with metadata",
            user_id="user_001",
            metadata=metadata,
        )

        assert memory.metadata == metadata
        assert memory.metadata["source"] == "test"

    def test_memory_versioning(self):
        """Test memory versioning fields."""
        now = datetime.now()
        future = now + timedelta(days=30)

        memory = Memory(
            id="mem_005",
            content="Versioned memory",
            user_id="user_001",
            version=2,
            parent_version="mem_004",
            valid_from=now,
            valid_until=future,
        )

        assert memory.version == 2
        assert memory.parent_version == "mem_004"
        assert memory.valid_until is not None

    def test_memory_status_transitions(self):
        """Test memory status transitions."""
        memory = Memory(
            id="mem_006",
            content="Superseded memory",
            user_id="user_001",
            status=MemoryStatus.SUPERSEDED,
            superseded_by="mem_007",
            invalidation_reason="Updated with new information",
        )

        assert memory.status == MemoryStatus.SUPERSEDED
        assert memory.superseded_by == "mem_007"
        assert memory.invalidation_reason == "Updated with new information"

    def test_memory_validation_required_fields(self):
        """Test memory validation for required fields."""
        with pytest.raises(ValidationError):
            Memory(content="Missing ID", user_id="user_001")

        with pytest.raises(ValidationError):
            Memory(id="mem_001", user_id="user_001")  # Missing content

        with pytest.raises(ValidationError):
            Memory(id="mem_001", content="Missing user_id")

    def test_memory_validation_confidence_range(self):
        """Test memory confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Memory(
                id="mem_007",
                content="Invalid confidence",
                user_id="user_001",
                confidence=1.5,
            )

        with pytest.raises(ValidationError):
            Memory(
                id="mem_008",
                content="Invalid confidence",
                user_id="user_001",
                confidence=-0.1,
            )

    def test_memory_is_valid(self):
        """Test is_valid method."""
        active_memory = Memory(
            id="mem_009",
            content="Active memory",
            user_id="user_001",
            status=MemoryStatus.ACTIVE,
        )
        assert active_memory.is_valid() is True

        invalidated_memory = Memory(
            id="mem_010",
            content="Invalidated memory",
            user_id="user_001",
            status=MemoryStatus.INVALIDATED,
        )
        assert invalidated_memory.is_valid() is False

        expired_memory = Memory(
            id="mem_011",
            content="Expired memory",
            user_id="user_001",
            status=MemoryStatus.ACTIVE,
            valid_until=datetime.now() - timedelta(days=1),
        )
        assert expired_memory.is_valid() is False

    def test_memory_age_days(self):
        """Test age_days method."""
        memory = Memory(
            id="mem_012",
            content="Old memory",
            user_id="user_001",
            created_at=datetime.now() - timedelta(days=5),
        )
        assert memory.age_days() == 5

    def test_memory_access_tracking(self):
        """Test mark_accessed method."""
        memory = Memory(
            id="mem_013",
            content="Tracked memory",
            user_id="user_001",
        )

        assert memory.access_count == 0
        assert memory.last_accessed is None

        memory.mark_accessed()
        assert memory.access_count == 1
        assert memory.last_accessed is not None

        memory.mark_accessed()
        assert memory.access_count == 2

    def test_memory_days_since_access(self):
        """Test days_since_access method."""
        memory = Memory(
            id="mem_014",
            content="Memory without access",
            user_id="user_001",
        )
        assert memory.days_since_access() is None

        memory.mark_accessed()
        assert memory.days_since_access() == 0


class TestRelationship:
    """Tests for Relationship model."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            type=RelationshipType.SIMILAR_TO,
            target_id="mem_123",
            confidence=0.85,
            reasoning="Both discuss Python frameworks",
        )

        assert rel.type == RelationshipType.SIMILAR_TO
        assert rel.target_id == "mem_123"
        assert rel.confidence == 0.85
        assert rel.reasoning == "Both discuss Python frameworks"

    def test_relationship_all_types(self):
        """Test relationship with different types."""
        rel_types = [
            RelationshipType.UPDATES,
            RelationshipType.CONTRADICTS,
            RelationshipType.SUPPORTS,
            RelationshipType.DERIVED_FROM,
        ]

        for rel_type in rel_types:
            rel = Relationship(
                type=rel_type,
                target_id="mem_456",
                confidence=0.9,
                reasoning=f"Test reasoning for {rel_type}",
            )
            assert rel.type == rel_type

    def test_relationship_validation(self):
        """Test relationship validation for required fields."""
        with pytest.raises(ValidationError):
            Relationship(type=RelationshipType.SIMILAR_TO)

        with pytest.raises(ValidationError):
            Relationship(target_id="mem_123")


class TestRelationshipBundle:
    """Tests for RelationshipBundle model."""

    def test_bundle_creation_with_relationships(self):
        """Test bundle creation with relationships."""
        relationships = [
            Relationship(
                type=RelationshipType.SIMILAR_TO,
                target_id="mem_123",
                confidence=0.85,
                reasoning="Similar content",
            ),
            Relationship(
                type=RelationshipType.UPDATES,
                target_id="mem_124",
                confidence=0.95,
                reasoning="Updates previous info",
            ),
        ]

        bundle = RelationshipBundle(
            memory_id="mem_new",
            relationships=relationships,
        )

        assert bundle.memory_id == "mem_new"
        assert len(bundle.relationships) == 2
        assert len(bundle.derived_insights) == 0

    def test_bundle_with_derived_insights(self):
        """Test bundle with derived insights."""
        insights = [
            DerivedInsight(
                content="User is learning Python systematically",
                confidence=0.8,
                reasoning="Pattern across multiple interactions",
                source_ids=["mem_001", "mem_002"],
                type="pattern_recognition",
            )
        ]

        bundle = RelationshipBundle(
            memory_id="mem_new",
            derived_insights=insights,
        )

        assert len(bundle.derived_insights) == 1
        assert bundle.derived_insights[0].content == "User is learning Python systematically"

    def test_bundle_empty(self):
        """Test bundle with no relationships or insights."""
        bundle = RelationshipBundle(memory_id="mem_new")

        assert bundle.memory_id == "mem_new"
        assert len(bundle.relationships) == 0
        assert len(bundle.derived_insights) == 0


class TestDerivedInsight:
    """Tests for DerivedInsight model."""

    def test_insight_creation(self):
        """Test basic insight creation."""
        insight = DerivedInsight(
            content="User prefers async programming",
            confidence=0.9,
            reasoning="Consistent pattern in recent queries",
            source_ids=["mem_001", "mem_002", "mem_003"],
            type="pattern_recognition",
        )

        assert insight.content == "User prefers async programming"
        assert insight.confidence == 0.9
        assert len(insight.source_ids) == 3
        assert insight.type == "pattern_recognition"

    def test_insight_types(self):
        """Test different insight types."""
        insight_types = ["pattern_recognition", "summary", "inference", "abstraction"]

        for insight_type in insight_types:
            insight = DerivedInsight(
                content=f"Test {insight_type}",
                confidence=0.8,
                reasoning="Test reasoning",
                source_ids=["mem_001"],
                type=insight_type,
            )
            assert insight.type == insight_type


class TestEdge:
    """Tests for Edge model."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(
            source="mem_001",
            target="mem_002",
            type=RelationshipType.SIMILAR_TO,
            confidence=0.85,
        )

        assert edge.source == "mem_001"
        assert edge.target == "mem_002"
        assert edge.type == RelationshipType.SIMILAR_TO
        assert edge.confidence == 0.85
        assert edge.metadata == {}
        assert isinstance(edge.created_at, datetime)

    def test_edge_with_metadata(self):
        """Test edge with metadata."""
        metadata = {"strength": "high", "context": "technical"}
        edge = Edge(
            source="mem_001",
            target="mem_002",
            type=RelationshipType.UPDATES,
            metadata=metadata,
        )

        assert edge.metadata == metadata
        assert edge.confidence == 1.0  # Default

    def test_edge_with_source_linkage_types(self):
        """Test edge with new source linkage relationship types."""
        has_memory_edge = Edge(
            source="note_001",
            target="mem_001",
            type=RelationshipType.HAS_MEMORY,
        )
        assert has_memory_edge.type == RelationshipType.HAS_MEMORY

        has_chunk_edge = Edge(
            source="doc_001",
            target="doc_001_chunk_0",
            type=RelationshipType.HAS_CHUNK,
        )
        assert has_chunk_edge.type == RelationshipType.HAS_CHUNK

        next_chunk_edge = Edge(
            source="doc_001_chunk_0",
            target="doc_001_chunk_1",
            type=RelationshipType.NEXT_CHUNK,
        )
        assert next_chunk_edge.type == RelationshipType.NEXT_CHUNK


class TestVersionModels:
    """Tests for version tracking models."""

    def test_version_change_creation(self):
        """Test VersionChange creation."""
        change = VersionChange(
            change_type="update",
            reasoning="Updated with new information",
            description="Modified the content field",
            changed_fields=["content"],
        )

        assert change.change_type == "update"
        assert change.reasoning == "Updated with new information"
        assert change.changed_fields == ["content"]
        assert isinstance(change.timestamp, datetime)

    def test_memory_evolution_creation(self):
        """Test MemoryEvolution creation."""
        evolution = MemoryEvolution(
            current_version="mem_001",
            new_version="mem_002",
            action="update",
        )

        assert evolution.current_version == "mem_001"
        assert evolution.new_version == "mem_002"
        assert evolution.action == "update"

    def test_memory_evolution_with_change(self):
        """Test MemoryEvolution with VersionChange."""
        change = VersionChange(
            change_type="update",
            reasoning="Updated with new info",
            description="Content was updated",
            changed_fields=["content"],
        )

        evolution = MemoryEvolution(
            current_version="mem_001",
            new_version="mem_002",
            action="update",
            change=change,
        )

        assert evolution.change is not None
        assert evolution.change.change_type == "update"

    def test_version_chain_creation(self):
        """Test VersionChain creation."""
        chain = VersionChain(
            original_id="mem_001",
            current_version_id="mem_003",
            versions=[
                {"id": "mem_001", "version": 1},
                {"id": "mem_002", "version": 2},
                {"id": "mem_003", "version": 3},
            ],
            total_versions=3,
            created_at=datetime.now(),
        )

        assert chain.original_id == "mem_001"
        assert chain.current_version_id == "mem_003"
        assert len(chain.versions) == 3
        assert chain.total_versions == 3

    def test_invalidation_result_creation(self):
        """Test InvalidationResult creation."""
        result = InvalidationResult(
            memory_id="mem_001",
            status=InvalidationStatus.ACTIVE,
            reasoning="Information is still current",
            confidence=0.95,
        )

        assert result.memory_id == "mem_001"
        assert result.status == InvalidationStatus.ACTIVE
        assert result.confidence == 0.95

    def test_invalidation_result_historical(self):
        """Test InvalidationResult with HISTORICAL status."""
        result = InvalidationResult(
            memory_id="mem_001",
            status=InvalidationStatus.HISTORICAL,
            reasoning="Outdated but useful context",
            confidence=0.85,
        )

        assert result.status == InvalidationStatus.HISTORICAL


class TestContextBundle:
    """Tests for ContextBundle model."""

    def test_context_bundle_empty(self):
        """Test empty context bundle creation."""
        bundle = ContextBundle()

        assert len(bundle.vector_candidates) == 0
        assert len(bundle.temporal_context) == 0
        assert len(bundle.graph_context) == 0
        assert len(bundle.entity_context) == 0
        assert len(bundle.conversation_context) == 0
        assert len(bundle.filtered_candidates) == 0

    def test_context_bundle_with_memories(self):
        """Test context bundle with memories."""
        memory1 = Memory(id="mem_001", content="Content 1", user_id="user_001")
        memory2 = Memory(id="mem_002", content="Content 2", user_id="user_001")
        memory3 = Memory(id="mem_003", content="Content 3", user_id="user_001")

        bundle = ContextBundle(
            vector_candidates=[memory1],
            temporal_context=[memory2],
            graph_context=[memory3],
            entity_context=[memory1, memory2],
            conversation_context=[memory3],
            filtered_candidates=[memory1],
        )

        assert len(bundle.vector_candidates) == 1
        assert len(bundle.temporal_context) == 1
        assert len(bundle.graph_context) == 1
        assert len(bundle.entity_context) == 2
        assert len(bundle.conversation_context) == 1
        assert len(bundle.filtered_candidates) == 1


class TestNodeType:
    """Tests for NodeType enum."""

    def test_source_content_types(self):
        """Test source content node types."""
        assert NodeType.NOTE == "NOTE"
        assert NodeType.DOCUMENT == "DOCUMENT"
        assert NodeType.CHUNK == "CHUNK"

    def test_extracted_types(self):
        """Test extracted node types."""
        assert NodeType.MEMORY == "MEMORY"
        assert NodeType.DERIVED == "DERIVED"

    def test_entity_types(self):
        """Test entity node types."""
        assert NodeType.TOPIC == "TOPIC"
        assert NodeType.ENTITY == "ENTITY"

    def test_all_node_types_defined(self):
        """Test all expected node types are defined."""
        expected_types = ["NOTE", "DOCUMENT", "CHUNK", "MEMORY", "DERIVED", "TOPIC", "ENTITY"]

        for node_type in expected_types:
            assert hasattr(NodeType, node_type)
            assert NodeType(node_type) == node_type


class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_types(self):
        """Test source type values."""
        assert SourceType.NOTE == "NOTE"
        assert SourceType.DOCUMENT == "DOCUMENT"

    def test_all_source_types_defined(self):
        """Test all source types are defined."""
        expected_types = ["NOTE", "DOCUMENT"]

        for source_type in expected_types:
            assert hasattr(SourceType, source_type)
            assert SourceType(source_type) == source_type


class TestMemoryStatus:
    """Tests for MemoryStatus enum."""

    def test_all_status_types(self):
        """Test all memory status types."""
        expected = ["active", "historical", "superseded", "invalidated"]

        for status in expected:
            assert hasattr(MemoryStatus, status.upper())
            assert MemoryStatus(status) == status


class TestContentStatus:
    """Tests for ContentStatus enum."""

    def test_all_content_status_types(self):
        """Test all content status types."""
        assert ContentStatus.ACTIVE == "active"
        assert ContentStatus.ARCHIVED == "archived"
        assert ContentStatus.DELETED == "deleted"


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_semantic_relationships(self):
        """Test semantic relationship types."""
        assert RelationshipType.SIMILAR_TO == "SIMILAR_TO"
        assert RelationshipType.REFERENCES == "REFERENCES"

    def test_temporal_relationships(self):
        """Test temporal/causal relationship types."""
        assert RelationshipType.PRECEDES == "PRECEDES"
        assert RelationshipType.FOLLOWS == "FOLLOWS"
        assert RelationshipType.UPDATES == "UPDATES"

    def test_logical_relationships(self):
        """Test logical relationship types."""
        assert RelationshipType.CONTRADICTS == "CONTRADICTS"
        assert RelationshipType.SUPPORTS == "SUPPORTS"
        assert RelationshipType.REQUIRES == "REQUIRES"
        assert RelationshipType.DEPENDS_ON == "DEPENDS_ON"

    def test_source_linkage_relationships(self):
        """Test new source linkage relationship types."""
        assert RelationshipType.HAS_MEMORY == "HAS_MEMORY"
        assert RelationshipType.HAS_CHUNK == "HAS_CHUNK"
        assert RelationshipType.NEXT_CHUNK == "NEXT_CHUNK"

    def test_all_relationship_types_defined(self):
        """Test all expected relationship types are defined."""
        expected_types = [
            "SIMILAR_TO",
            "REFERENCES",
            "PRECEDES",
            "FOLLOWS",
            "UPDATES",
            "PART_OF",
            "BELONGS_TO",
            "PARENT_OF",
            "CONTRADICTS",
            "SUPPORTS",
            "REQUIRES",
            "DEPENDS_ON",
            "DERIVED_FROM",
            "SYNTHESIZES",
            "CO_OCCURS",
            "MENTIONS",
            "RESPONDS_TO",
            "HAS_MEMORY",
            "HAS_CHUNK",
            "NEXT_CHUNK",
        ]

        for rel_type in expected_types:
            assert hasattr(RelationshipType, rel_type)
            assert RelationshipType(rel_type) == rel_type


class TestComputeContentHash:
    """Tests for compute_content_hash utility function."""

    def test_hash_basic(self):
        """Test basic content hashing."""
        content = "Hello, World!"
        hash_result = compute_content_hash(content)

        assert hash_result.startswith("sha256:")
        assert len(hash_result) == 71  # "sha256:" (7) + 64 hex chars

    def test_hash_deterministic(self):
        """Test hash is deterministic for same content."""
        content = "Same content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2

    def test_hash_different_content(self):
        """Test different content produces different hashes."""
        hash1 = compute_content_hash("Content A")
        hash2 = compute_content_hash("Content B")

        assert hash1 != hash2

    def test_hash_normalization(self):
        """Test content is normalized (stripped) before hashing."""
        content_plain = "Hello World"
        content_with_spaces = "  Hello World  "
        content_with_newlines = "\nHello World\n"

        hash_plain = compute_content_hash(content_plain)
        hash_spaces = compute_content_hash(content_with_spaces)
        hash_newlines = compute_content_hash(content_with_newlines)

        assert hash_plain == hash_spaces
        assert hash_plain == hash_newlines

    def test_hash_empty_content(self):
        """Test hashing empty content."""
        hash_empty = compute_content_hash("")
        hash_whitespace = compute_content_hash("   ")

        assert hash_empty.startswith("sha256:")
        assert hash_empty == hash_whitespace

    def test_hash_unicode_content(self):
        """Test hashing unicode content."""
        unicode_content = "Hello, 世界! 🌍"
        hash_result = compute_content_hash(unicode_content)

        assert hash_result.startswith("sha256:")
        assert len(hash_result) == 71

    def test_hash_long_content(self):
        """Test hashing long content."""
        long_content = "A" * 100000
        hash_result = compute_content_hash(long_content)

        assert hash_result.startswith("sha256:")
        assert len(hash_result) == 71
