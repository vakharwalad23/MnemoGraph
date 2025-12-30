"""
Note model for small source content.

Notes are the simplest form of source content in MnemoGraph.
They contain small pieces of text (< 2000 tokens) from which
semantic memories are extracted.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.memory import ContentStatus, NodeType


class Note(BaseModel):
    """
    Small content unit - direct user input.

    Notes are the simplest form of source content. They contain
    the full text of user input, and semantic memories are extracted
    from them for relationship building.

    Storage Architecture:
    - Vector Store: Full content, embedding, all metadata (SOURCE OF TRUTH)
    - Graph Store: Minimal data (id, user_id, content_preview, status) for traversal
    """

    # Core identity
    id: str = Field(..., description="Unique note ID (note_xxx)")
    type: NodeType = Field(default=NodeType.NOTE, description="Node type (always NOTE)")
    user_id: str = Field(..., description="Owner user ID for multi-tenant isolation")

    # Content
    content: str = Field(..., description="Full note text")
    content_hash: str = Field(..., description="SHA256 hash for deduplication")
    embedding: list[float] = Field(default_factory=list, description="Vector embedding of content")

    # Metadata
    title: str | None = Field(default=None, description="Optional title for the note")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Extracted memories linkage
    memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memories extracted from this note",
    )

    # Status
    status: ContentStatus = Field(
        default=ContentStatus.ACTIVE,
        description="Content lifecycle status",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @property
    def content_preview(self) -> str:
        """
        Get a preview of content for graph store storage.

        Returns:
            First 200 characters of content
        """
        return self.content[:200] if len(self.content) > 200 else self.content

    def is_active(self) -> bool:
        """
        Check if note is currently active.

        Returns:
            True if note status is ACTIVE
        """
        return self.status == ContentStatus.ACTIVE
