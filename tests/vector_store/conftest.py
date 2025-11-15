"""
Shared test fixtures for vector store tests.
"""

from datetime import datetime

import pytest

from src.core.vector_store.qdrant import QdrantStore
from src.models.memory import Memory, MemoryStatus, NodeType


@pytest.fixture
def qdrant_store():
    """Create Qdrant store for testing."""
    return QdrantStore(
        host="localhost",
        port=6333,
        collection_name="test_memories",
        vector_size=768,
        use_grpc=True,
    )


@pytest.fixture
def sample_memory():
    """Create sample memory for testing."""
    return Memory(
        id="test-memory-1",
        content="Test memory content",
        embedding=[0.1] * 768,
        user_id="user-1",
        type=NodeType.MEMORY,
        status=MemoryStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
