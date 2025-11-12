"""
Version tracking models for memory evolution.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InvalidationStatus(str, Enum):
    """Valid statuses for memory invalidation."""

    ACTIVE = "active"
    HISTORICAL = "historical"
    INVALIDATED = "invalidated"


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
    """
    Result of memory invalidation check (LLM structured output).

    Note: This is the minimal model for LLM output. Tracking fields like checked_at
    should be added after LLM call if needed.
    """

    model_config = {"extra": "ignore"}

    memory_id: str = Field(..., description="Memory ID being checked")
    status: InvalidationStatus = Field(..., description="Determined status")
    reasoning: str = Field(..., description="Why this status was assigned with specific evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1) in this decision")
