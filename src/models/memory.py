"""Memory models with status tracking and temporal metadata."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MemoryStatus(str, Enum):
    """Memory lifecycle status."""

    NEW = "new"
    ACTIVE = "active"
    EXPIRING_SOON = "expiring_soon"
    FORGOTTEN = "forgotten"


class Memory(BaseModel):
    """Individual memory node with metadata and access tracking."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Status tracking
    status: MemoryStatus = MemoryStatus.NEW

    # Temporal metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    access_count: int = 0

    # Decay tracking
    decay_score: float = 0.0

    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1
        if self.status == MemoryStatus.NEW:
            self.status = MemoryStatus.ACTIVE


class Document(BaseModel):
    """Document-level node."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # References to chunks
    chunk_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """Chunk-level node with parent document reference."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Hierarchical reference
    parent_document_id: str | None = None
    chunk_index: int = 0

    # Memory tracking
    status: MemoryStatus = MemoryStatus.NEW
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    access_count: int = 0
    decay_score: float = 0.0

    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1
        if self.status == MemoryStatus.NEW:
            self.status = MemoryStatus.ACTIVE
