"""
Memory model with versioning and lifecycle tracking.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    # Source Content Types
    NOTE = "NOTE"  # Small source content (< 2000 tokens)
    DOCUMENT = "DOCUMENT"  # Large source content (>= 2000 tokens)
    CHUNK = "CHUNK"  # Document chunk

    # Extracted Types
    MEMORY = "MEMORY"  # Extracted semantic memory
    DERIVED = "DERIVED"  # LLM-synthesized insight

    # Entity Types (future)
    TOPIC = "TOPIC"  # Topic/category
    ENTITY = "ENTITY"  # Named entity (person, place, concept)


class SourceType(str, Enum):
    """Source content type for extracted memories."""

    NOTE = "NOTE"
    DOCUMENT = "DOCUMENT"


class MemoryStatus(str, Enum):
    """Memory lifecycle status."""

    ACTIVE = "active"
    HISTORICAL = "historical"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class ContentStatus(str, Enum):
    """Status of source content (Notes, Documents)."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Memory(BaseModel):
    """
    Extracted semantic memory from source content (Notes/Documents).

    Memories are the atomic units of knowledge extracted from source content.
    They participate in relationship extraction and semantic search.

    Features:
    - Source Linkage: Every memory links back to its source (Note or Document)
    - Versioning: Track evolution over time
    - Status management: Active, historical, superseded, invalidated
    - Temporal tracking: Creation, updates, validity windows
    - Access patterns: Track usage for intelligent invalidation
    - Deduplication: content_hash for detecting duplicates

    Storage Architecture:
    - Vector Store: Source of truth - stores ALL fields including full content
    - Graph Store: Minimal nodes - stores only (id, content_preview, type, status, source info)
    """

    # Core identity
    id: str = Field(..., description="Unique memory ID (mem_xxx)")
    content: str = Field(..., description="Semantic memory content")
    content_hash: str = Field(default="", description="SHA256 hash for deduplication")
    type: NodeType = Field(default=NodeType.MEMORY, description="Node type")
    embedding: list[float] = Field(default_factory=list, description="Vector embedding")

    # Multi-user isolation
    user_id: str = Field(..., description="Owner user ID")

    # Source linkage (inks memory to its source)
    source_id: str | None = Field(
        default=None,
        description="Source note_id or document_id (None for legacy memories)",
    )
    source_type: SourceType | None = Field(
        default=None,
        description="Source type: NOTE or DOCUMENT (None for legacy memories)",
    )
    source_chunk_id: str | None = Field(
        default=None,
        description="Specific chunk ID if extracted from a document chunk",
    )

    # Versioning
    version: int = Field(default=1, description="Version number")
    parent_version: str | None = Field(default=None, description="Parent version ID")
    valid_from: datetime = Field(default_factory=datetime.now, description="Valid from timestamp")
    valid_until: datetime | None = Field(default=None, description="Valid until timestamp")

    # Status
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="Lifecycle status")
    superseded_by: str | None = Field(default=None, description="ID of superseding memory")
    invalidation_reason: str | None = Field(default=None, description="Reason for invalidation")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    # Access tracking
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: datetime | None = Field(default=None, description="Last access timestamp")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

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
        """
        Check if memory is currently valid.

        Returns:
            True if memory is active and within validity window
        """
        if self.status != MemoryStatus.ACTIVE:
            return False

        if self.valid_until is not None and self.valid_until < datetime.now():
            return False

        return True

    def age_days(self) -> int:
        """
        Get age of memory in days.

        Returns:
            Number of days since memory was created
        """
        return (datetime.now() - self.created_at).days

    def days_since_access(self) -> int | None:
        """
        Get days since last access.

        Returns:
            Number of days since last access, or None if never accessed
        """
        if self.last_accessed is None:
            return None
        return (datetime.now() - self.last_accessed).days


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of content for deduplication.

    The hash is prefixed with "sha256:" for easy identification of the algorithm used.
    Content is normalized (stripped) before hashing to avoid whitespace-only differences.

    Args:
        content: Text content to hash

    Returns:
        Hash string in format "sha256:hexdigest"
    """
    normalized = content.strip()
    hash_bytes = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{hash_bytes}"
