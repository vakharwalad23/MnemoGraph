"""
Shared test fixtures for graph store tests.
"""

from datetime import datetime

import pytest

from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.models.memory import Memory, MemoryStatus, NodeType


@pytest.fixture
def neo4j_store():
    """Create Neo4j store for testing."""
    return Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
    )


@pytest.fixture
def sample_memory():
    """Create sample memory for testing."""
    return Memory(
        id="test-memory-1",
        content="Test memory content for graph store",
        embedding=[],
        user_id="user-1",
        type=NodeType.MEMORY,
        status=MemoryStatus.ACTIVE,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
