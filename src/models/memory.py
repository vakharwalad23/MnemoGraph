"""
Memory model with versioning and lifecycle tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the memory graph."""

    MEMORY = "MEMORY"  # User-created memory
    DERIVED = "DERIVED"  # LLM-synthesized insight
    TOPIC = "TOPIC"  # Cluster/category
    ENTITY = "ENTITY"  # Extracted entity
    DOCUMENT = "DOCUMENT"  # Document node
    CHUNK = "CHUNK"  # Document chunk


class MemoryStatus(str, Enum):
    """Memory lifecycle status."""

    ACTIVE = "active"  # Currently valid and relevant
    HISTORICAL = "historical"  # Outdated but preserved as history
    SUPERSEDED = "superseded"  # Replaced by newer version
    INVALIDATED = "invalidated"  # No longer relevant


class Memory(BaseModel):
    """
    Memory node with full lifecycle tracking.

    Features:
    - Versioning: Track evolution over time
    - Status management: Active, historical, superseded, invalidated
    - Temporal tracking: Creation, updates, validity windows
    - Access patterns: Track usage for intelligent invalidation

    Storage Architecture:
    - Vector Store: Source of truth - stores ALL fields
    - Graph Store: Minimal nodes - stores only (id, content_preview, type, status, version info)
    When to use which store:
    - For full memory data: Use MemoryStore.get_memory() -> fetches from vector store
    - For graph traversal: Graph store provides minimal data for relationships
    - For search: Vector store provides semantic search with full data
    """

    # Core identity
    id: str
    content: str  # Full content in vector store, preview (200 chars) in graph store
    type: NodeType = NodeType.MEMORY
    embedding: list[float] = Field(default_factory=list)  # Vector store only

    # Versioning (minimal version info in graph store for chain traversal)
    version: int = 1
    parent_version: str | None = None  # ID of previous version
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None  # None = still valid

    # Status (stored in both stores for filtering)
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: str | None = None  # ID of newer version (in both stores)
    invalidation_reason: str | None = None  # Vector store only

    # Metadata (vector store only - NOT in graph store)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    # Access tracking (vector store only - automatic via MemoryStore)
    access_count: int = 0
    last_accessed: datetime | None = None

    # Timestamps (vector store only)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def mark_accessed(self) -> None:
        """
        Mark memory as accessed (for tracking).

        Note: This method is deprecated. Access tracking is now handled
        automatically by MemoryStore facade when using get_memory() with
        track_access=True. This method is kept for backwards compatibility
        but should not be used in new code.
        """
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.updated_at = datetime.now()

    def is_valid(self) -> bool:
        """Check if memory is currently valid."""
        if self.status != MemoryStatus.ACTIVE:
            return False

        if self.valid_until is not None and self.valid_until < datetime.now():
            return False

        return True

    def age_days(self) -> int:
        """Get age of memory in days."""
        return (datetime.now() - self.created_at).days

    def days_since_access(self) -> int | None:
        """Get days since last access."""
        if self.last_accessed is None:
            return None
        return (datetime.now() - self.last_accessed).days
