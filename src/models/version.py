"""
Version tracking models for memory evolution.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VersionChange(BaseModel):
    """Represents a change between memory versions."""

    change_type: str  # update, replace, augment, preserve
    reasoning: str
    description: str
    changed_fields: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryEvolution(BaseModel):
    """
    Tracks the evolution of a memory through versions.

    Represents the result of evolving a memory:
    - Old version (marked historical/superseded)
    - New version (if created)
    - Relationship between versions
    """

    current_version: str  # Current memory ID
    new_version: str | None = None  # New memory ID if created
    action: str  # update, preserve, invalidate
    change: VersionChange | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class VersionChain(BaseModel):
    """
    Complete version history chain for a memory.

    Tracks all versions from original to current.
    """

    original_id: str
    versions: list[dict[str, Any]] = Field(default_factory=list)
    current_version_id: str
    total_versions: int = 0
    created_at: datetime
    last_updated: datetime = Field(default_factory=datetime.now)


class InvalidationResult(BaseModel):
    """Result of memory invalidation check."""

    memory_id: str
    status: str  # active, historical, superseded, invalidated
    reasoning: str
    confidence: float
    superseded_by: str | None = None
    preserve_as: str | None = None  # historical_context, etc.
    checked_at: datetime = Field(default_factory=datetime.now)
