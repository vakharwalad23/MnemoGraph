"""Graph node models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4

from .memory import MemoryStatus


class NodeType(str, Enum):
    """Types of nodes in the graph."""
    MEMORY = "memory"
    DOCUMENT = "document"
    CHUNK = "chunk"
    TOPIC = "topic"  # For topic clustering


class Node(BaseModel):
    """Generic graph node representation."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NodeType
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    status: MemoryStatus = MemoryStatus.NEW
    
    # Temporal metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0