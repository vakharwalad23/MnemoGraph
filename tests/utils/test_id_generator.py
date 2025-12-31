"""
Tests for ID generation utilities.

Tests cover:
1. Note ID generation
2. Document ID generation
3. Chunk ID generation
4. Memory ID generation
5. Job ID generation
6. Uniqueness guarantees
"""

import pytest

from src.utils import (
    generate_chunk_id,
    generate_document_id,
    generate_job_id,
    generate_memory_id,
    generate_note_id,
)


class TestGenerateNoteId:
    """Tests for Note ID generation."""

    def test_format(self):
        """Test Note ID format: note_xxx (12 hex chars)."""
        note_id = generate_note_id()

        assert note_id.startswith("note_")
        assert len(note_id) == 17  # "note_" (5) + 12 hex chars
        assert note_id[5:].isalnum()

    def test_uniqueness(self):
        """Test that generated Note IDs are unique."""
        ids = [generate_note_id() for _ in range(1000)]
        assert len(ids) == len(set(ids))

    def test_consistent_prefix(self):
        """Test all generated IDs have the same prefix."""
        ids = [generate_note_id() for _ in range(100)]
        assert all(id.startswith("note_") for id in ids)


class TestGenerateDocumentId:
    """Tests for Document ID generation."""

    def test_format(self):
        """Test Document ID format: doc_xxx (12 hex chars)."""
        doc_id = generate_document_id()

        assert doc_id.startswith("doc_")
        assert len(doc_id) == 16  # "doc_" (4) + 12 hex chars
        assert doc_id[4:].isalnum()

    def test_uniqueness(self):
        """Test that generated Document IDs are unique."""
        ids = [generate_document_id() for _ in range(1000)]
        assert len(ids) == len(set(ids))

    def test_consistent_prefix(self):
        """Test all generated IDs have the same prefix."""
        ids = [generate_document_id() for _ in range(100)]
        assert all(id.startswith("doc_") for id in ids)


class TestGenerateChunkId:
    """Tests for Chunk ID generation."""

    def test_format(self):
        """Test Chunk ID format: doc_xxx_chunk_N."""
        document_id = "doc_abc123def456"
        chunk_id = generate_chunk_id(document_id, 0)

        assert chunk_id == "doc_abc123def456_chunk_0"

    def test_sequential_chunks(self):
        """Test chunk IDs for sequential chunks."""
        document_id = "doc_testdoc12345"

        chunks = [generate_chunk_id(document_id, i) for i in range(5)]

        expected = [
            "doc_testdoc12345_chunk_0",
            "doc_testdoc12345_chunk_1",
            "doc_testdoc12345_chunk_2",
            "doc_testdoc12345_chunk_3",
            "doc_testdoc12345_chunk_4",
        ]
        assert chunks == expected

    def test_different_documents(self):
        """Test chunk IDs are unique across documents."""
        doc1_chunk = generate_chunk_id("doc_111111111111", 0)
        doc2_chunk = generate_chunk_id("doc_222222222222", 0)

        assert doc1_chunk != doc2_chunk
        assert "111111111111" in doc1_chunk
        assert "222222222222" in doc2_chunk


class TestGenerateMemoryId:
    """Tests for Memory ID generation."""

    def test_format(self):
        """Test Memory ID format: mem_xxx (12 hex chars)."""
        memory_id = generate_memory_id()

        assert memory_id.startswith("mem_")
        assert len(memory_id) == 16  # "mem_" (4) + 12 hex chars
        assert memory_id[4:].isalnum()

    def test_uniqueness(self):
        """Test that generated Memory IDs are unique."""
        ids = [generate_memory_id() for _ in range(1000)]
        assert len(ids) == len(set(ids))

    def test_consistent_prefix(self):
        """Test all generated IDs have the same prefix."""
        ids = [generate_memory_id() for _ in range(100)]
        assert all(id.startswith("mem_") for id in ids)


class TestGenerateJobId:
    """Tests for Job ID generation."""

    def test_format(self):
        """Test Job ID format: job_xxx (12 hex chars)."""
        job_id = generate_job_id()

        assert job_id.startswith("job_")
        assert len(job_id) == 16  # "job_" (4) + 12 hex chars
        assert job_id[4:].isalnum()

    def test_uniqueness(self):
        """Test that generated Job IDs are unique."""
        ids = [generate_job_id() for _ in range(1000)]
        assert len(ids) == len(set(ids))

    def test_consistent_prefix(self):
        """Test all generated IDs have the same prefix."""
        ids = [generate_job_id() for _ in range(100)]
        assert all(id.startswith("job_") for id in ids)


class TestIdTypeDistinction:
    """Tests to ensure different ID types are distinguishable."""

    def test_prefixes_are_distinct(self):
        """Test that all ID type prefixes are different."""
        note_id = generate_note_id()
        doc_id = generate_document_id()
        mem_id = generate_memory_id()
        job_id = generate_job_id()

        prefixes = [
            note_id.split("_")[0],
            doc_id.split("_")[0],
            mem_id.split("_")[0],
            job_id.split("_")[0],
        ]

        assert len(set(prefixes)) == 4  # All different

    def test_can_identify_type_from_id(self):
        """Test that ID type can be determined from the prefix."""
        note_id = generate_note_id()
        doc_id = generate_document_id()
        mem_id = generate_memory_id()
        job_id = generate_job_id()
        chunk_id = generate_chunk_id(doc_id, 5)

        assert note_id.startswith("note_")
        assert doc_id.startswith("doc_") and "_chunk_" not in doc_id
        assert mem_id.startswith("mem_")
        assert job_id.startswith("job_")
        assert "_chunk_" in chunk_id
