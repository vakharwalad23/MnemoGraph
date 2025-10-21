"""Tests for data models."""

from src.models import Chunk, Document, Edge, Memory, MemoryStatus, RelationshipType


def test_memory_creation():
    """Test Memory model creation."""
    memory = Memory(text="Python is a programming language")

    assert memory.text == "Python is a programming language"
    assert memory.status == MemoryStatus.NEW
    assert memory.access_count == 0
    assert memory.decay_score == 0.0
    assert memory.embedding is None


def test_memory_access_tracking():
    """Test memory access tracking."""
    memory = Memory(text="Test memory")

    # Initial state
    assert memory.status == MemoryStatus.NEW
    assert memory.access_count == 0
    assert memory.last_accessed is None

    # After first access
    memory.update_access()
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.access_count == 1
    assert memory.last_accessed is not None

    # After second access
    memory.update_access()
    assert memory.access_count == 2
    assert memory.status == MemoryStatus.ACTIVE


def test_document_creation():
    """Test Document model creation."""
    doc = Document(
        title="Python Tutorial",
        text="This is a comprehensive Python tutorial...",
        metadata={"author": "John Doe", "topic": "programming"},
    )

    assert doc.title == "Python Tutorial"
    assert doc.metadata["author"] == "John Doe"
    assert doc.chunk_ids == []
    assert doc.created_at is not None


def test_chunk_creation():
    """Test Chunk model creation and parent reference."""
    doc = Document(title="Test Doc", text="Test document text")

    chunk = Chunk(
        text="Functions are reusable blocks of code", parent_document_id=doc.id, chunk_index=0
    )

    assert chunk.text == "Functions are reusable blocks of code"
    assert chunk.parent_document_id == doc.id
    assert chunk.chunk_index == 0
    assert chunk.status == MemoryStatus.NEW


def test_chunk_access_tracking():
    """Test chunk access tracking."""
    chunk = Chunk(text="Test chunk", chunk_index=0)

    chunk.update_access()
    assert chunk.status == MemoryStatus.ACTIVE
    assert chunk.access_count == 1


def test_edge_creation():
    """Test Edge model creation."""
    doc = Document(title="Test", text="Test document")
    chunk = Chunk(text="Test chunk", parent_document_id=doc.id)

    edge = Edge(
        source=doc.id,
        target=chunk.id,
        type=RelationshipType.PARENT_OF,
        weight=1.0,
        metadata={"created_by": "system"},
    )

    assert edge.source == doc.id
    assert edge.target == chunk.id
    assert edge.type == RelationshipType.PARENT_OF
    assert edge.weight == 1.0
    assert edge.metadata["created_by"] == "system"


def test_relationship_types():
    """Test all relationship types are defined."""
    assert RelationshipType.SIMILAR_TO == "similar_to"
    assert RelationshipType.UPDATES == "updates"
    assert RelationshipType.FOLLOWS == "follows"
    assert RelationshipType.PARENT_OF == "parent_of"
    assert RelationshipType.CO_OCCURS == "co_occurs"
    assert RelationshipType.CAUSES == "causes"
    assert RelationshipType.REQUIRES == "requires"


def test_memory_status_enum():
    """Test memory status transitions."""
    assert MemoryStatus.NEW == "new"
    assert MemoryStatus.ACTIVE == "active"
    assert MemoryStatus.EXPIRING_SOON == "expiring_soon"
    assert MemoryStatus.FORGOTTEN == "forgotten"


def test_memory_with_embedding():
    """Test memory with embedding vector."""
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    memory = Memory(text="Test with embedding", embedding=embedding)

    assert memory.embedding == embedding
    assert len(memory.embedding) == 5
