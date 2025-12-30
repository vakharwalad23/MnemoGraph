"""
Document and Chunk models for large source content.

Documents are large content units (>= 2000 tokens) that get chunked
for efficient storage and retrieval. A summary is generated for
relationship extraction.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models.memory import ContentStatus, NodeType


class Chunk(BaseModel):
    """
    Document chunk for large content storage.

    Chunks store full content in vector store for semantic search.
    Graph store only stores chunk summaries for efficient traversal.

    Chunks are ordered within a document and may overlap with adjacent chunks
    to preserve context across boundaries.
    """

    # Core identity
    id: str = Field(..., description="Unique chunk ID (doc_xxx_chunk_0)")
    type: NodeType = Field(default=NodeType.CHUNK, description="Node type (always CHUNK)")
    user_id: str = Field(..., description="Owner user ID for multi-tenant isolation")
    document_id: str = Field(..., description="Parent document ID")

    # Content
    content: str = Field(..., description="Full chunk text (stored in vector store)")
    summary: str = Field(default="", description="Chunk summary (stored in graph store)")
    content_hash: str = Field(..., description="SHA256 hash of chunk content")
    embedding: list[float] = Field(default_factory=list, description="Vector embedding of content")

    # Position in document
    chunk_index: int = Field(..., ge=0, description="Zero-based index within document")
    chunk_total: int = Field(..., ge=1, description="Total number of chunks in document")
    start_token: int = Field(..., ge=0, description="Start token position in original document")
    end_token: int = Field(..., ge=0, description="End token position in original document")
    overlap_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of overlapping tokens with previous chunk",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    @property
    def content_preview(self) -> str:
        """
        Get a preview of content for graph store storage.

        Uses summary if available, otherwise first 200 chars of content.

        Returns:
            Summary or truncated content
        """
        if self.summary:
            return self.summary[:200] if len(self.summary) > 200 else self.summary
        return self.content[:200] if len(self.content) > 200 else self.content

    def is_first(self) -> bool:
        """
        Check if this is the first chunk in the document.

        Returns:
            True if chunk_index is 0
        """
        return self.chunk_index == 0

    def is_last(self) -> bool:
        """
        Check if this is the last chunk in the document.

        Returns:
            True if chunk_index is chunk_total - 1
        """
        return self.chunk_index == self.chunk_total - 1


class Document(BaseModel):
    """
    Large content unit with chunking support.

    Documents are automatically created when content exceeds the token threshold
    (default: 2000 tokens). They are chunked for efficient storage and retrieval,
    and a summary is generated for memory extraction and graph traversal.

    Storage Architecture:
    - Vector Store: Summary embedding + chunk embeddings (with full chunk content)
    - Graph Store: Document node (summary preview) + Chunk nodes (summaries) + relationships

    The original full content is NOT stored - only the summary and chunks.
    """

    # Core identity
    id: str = Field(..., description="Unique document ID (doc_xxx)")
    type: NodeType = Field(default=NodeType.DOCUMENT, description="Node type (always DOCUMENT)")
    user_id: str = Field(..., description="Owner user ID for multi-tenant isolation")

    # Content
    summary: str = Field(..., description="LLM-generated summary of the document")
    content_hash: str = Field(..., description="SHA256 hash of ORIGINAL full content")
    embedding: list[float] = Field(default_factory=list, description="Vector embedding of summary")

    # Chunks
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of chunk IDs belonging to this document",
    )
    chunk_count: int = Field(default=0, ge=0, description="Total number of chunks")

    # Metadata
    title: str | None = Field(default=None, description="Document title")
    source_url: str | None = Field(default=None, description="Original URL if applicable")
    total_tokens: int = Field(default=0, ge=0, description="Token count of original content")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Extracted memories linkage
    memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memories extracted from this document",
    )

    # Processing info
    chunking_strategy: str = Field(
        default="fixed",
        description="Chunking strategy used: 'fixed' or 'semantic'",
    )
    summary_model: str = Field(
        default="",
        description="Model used to generate the summary",
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

        For documents, this is the summary (truncated if needed).

        Returns:
            Summary truncated to 200 characters
        """
        return self.summary[:200] if len(self.summary) > 200 else self.summary

    def is_active(self) -> bool:
        """
        Check if document is currently active.

        Returns:
            True if document status is ACTIVE
        """
        return self.status == ContentStatus.ACTIVE

    def has_chunks(self) -> bool:
        """
        Check if document has any chunks.

        Returns:
            True if document has at least one chunk
        """
        return self.chunk_count > 0 and len(self.chunk_ids) > 0
