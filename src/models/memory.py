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
    """

    # Core identity
    id: str
    content: str
    type: NodeType = NodeType.MEMORY
    embedding: list[float] = Field(default_factory=list)

    # Versioning
    version: int = 1
    parent_version: str | None = None  # ID of previous version
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None  # None = still valid

    # Status
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: str | None = None  # ID of newer version
    invalidation_reason: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    # Access tracking for intelligent invalidation
    access_count: int = 0
    last_accessed: datetime | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def mark_accessed(self) -> None:
        """Mark memory as accessed (for tracking)."""
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
