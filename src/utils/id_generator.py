"""
ID generation utilities for MnemoGraph.

Provides consistent ID generation for all entity types:
- Notes: note_xxx
- Documents: doc_xxx
- Chunks: doc_xxx_chunk_N
- Memories: mem_xxx
- Jobs: job_xxx
"""

from uuid import uuid4


def generate_note_id() -> str:
    """
    Generate unique Note ID.

    Returns:
        ID in format "note_xxx" where xxx is 12 hex characters
    """
    return f"note_{uuid4().hex[:12]}"


def generate_document_id() -> str:
    """
    Generate unique Document ID.

    Returns:
        ID in format "doc_xxx" where xxx is 12 hex characters
    """
    return f"doc_{uuid4().hex[:12]}"


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate Chunk ID based on parent document.

    Args:
        document_id: Parent document ID
        chunk_index: Zero-based chunk index

    Returns:
        ID in format "doc_xxx_chunk_N"
    """
    return f"{document_id}_chunk_{chunk_index}"


def generate_memory_id() -> str:
    """
    Generate unique Memory ID.

    Returns:
        ID in format "mem_xxx" where xxx is 12 hex characters
    """
    return f"mem_{uuid4().hex[:12]}"


def generate_job_id() -> str:
    """
    Generate unique async Job ID.

    Returns:
        ID in format "job_xxx" where xxx is 12 hex characters
    """
    return f"job_{uuid4().hex[:12]}"
