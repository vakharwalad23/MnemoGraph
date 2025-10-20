"""Memory models with status tracking and temporal metadata."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class MemoryStatus(str, Enum):
    """Memory lifecycle status."""
    NEW = "new"
    ACTIVE = "active"
    EXPIRING_SOON = "expiring_soon"
    FORGOTTEN = "forgotten"


class Memory(BaseModel):
    """Individual memory node with metadata and access tracking."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    status: MemoryStatus = MemoryStatus.NEW
    
    # Temporal metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Decay tracking
    decay_score: float = 0.0
    
    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        if self.status == MemoryStatus.NEW:
            self.status = MemoryStatus.ACTIVE


class Document(BaseModel):
    """Document-level node."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # References to chunks
    chunk_ids: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """Chunk-level node with parent document reference."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Hierarchical reference
    parent_document_id: Optional[str] = None
    chunk_index: int = 0
    
    # Memory tracking
    status: MemoryStatus = MemoryStatus.NEW
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    decay_score: float = 0.0
    
    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        if self.status == MemoryStatus.NEW:
            self.status = MemoryStatus.ACTIVE