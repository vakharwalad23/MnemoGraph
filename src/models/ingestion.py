"""
Content ingestion result models.

Models for tracking the result of content ingestion operations,
including Notes and Documents.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.memory import SourceType


class IngestionStatus(str, Enum):
    """Status of content ingestion operation."""

    COMPLETED = "completed"  # Successfully ingested
    QUEUED = "queued"  # Queued for async processing
    DUPLICATE = "duplicate"  # Content already exists
    FAILED = "failed"  # Ingestion failed
    NOT_IMPLEMENTED = "not_implemented"  # Feature not yet implemented


class ContentIngestionResult(BaseModel):
    """
    Result of content ingestion operation.

    Returned by add_content() to provide detailed information about
    what was created and any processing that occurred.
    """

    # Source identification
    source_id: str = Field(default="", description="ID of created Note or Document")
    source_type: SourceType = Field(..., description="Type of source content")
    status: IngestionStatus = Field(..., description="Ingestion status")

    # Created entities
    memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of extracted memories",
    )
    relationship_count: int = Field(
        default=0,
        ge=0,
        description="Number of relationships created",
    )

    # Document-specific
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks (documents only)")
    job_id: str | None = Field(default=None, description="Async job ID if queued")

    # Diagnostics
    token_count: int = Field(default=0, ge=0, description="Token count of input content")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time in ms")
    message: str | None = Field(default=None, description="Additional status message")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
