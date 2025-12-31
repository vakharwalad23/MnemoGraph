"""
Data models for MnemoGraph.

Three-Layer Architecture:
1. Source Content Layer (Notes, Documents, Chunks)
2. Memory Layer (extracted semantic memories)
3. Relationship Layer (connections between memories)

Core models:
- Note: Small source content (< 2000 tokens)
- Document, Chunk: Large source content with chunking
- Memory: Extracted semantic memory with source linkage
- NodeType, SourceType: Node and source type enums
- MemoryStatus, ContentStatus: Lifecycle status enums
- RelationshipType, Edge: Relationship models
- RelationshipBundle, ContextBundle: Extraction results
- ContentIngestionResult, IngestionStatus: Ingestion results
"""

from src.models.document import Chunk, Document
from src.models.ingestion import ContentIngestionResult, IngestionStatus
from src.models.memory import (
    ContentStatus,
    Memory,
    MemoryStatus,
    NodeType,
    SourceType,
    compute_content_hash,
)
from src.models.note import Note
from src.models.relationships import (
    ContextBundle,
    DerivedInsight,
    Edge,
    Relationship,
    RelationshipBundle,
    RelationshipType,
)
from src.models.version import (
    InvalidationResult,
    InvalidationStatus,
    MemoryEvolution,
    VersionChain,
    VersionChange,
)

__all__ = [
    # Source Content models
    "Note",
    "Document",
    "Chunk",
    # Memory models
    "Memory",
    "NodeType",
    "SourceType",
    "MemoryStatus",
    "ContentStatus",
    "compute_content_hash",
    # Ingestion models
    "ContentIngestionResult",
    "IngestionStatus",
    # Version models
    "MemoryEvolution",
    "VersionChange",
    "VersionChain",
    "InvalidationResult",
    "InvalidationStatus",
    # Relationship models
    "RelationshipType",
    "Edge",
    "Relationship",
    "RelationshipBundle",
    "DerivedInsight",
    "ContextBundle",
]
