"""Graph node models."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .memory import MemoryStatus


class NodeType(str, Enum):
    """Types of nodes in the graph."""

    MEMORY = "memory"
    DOCUMENT = "document"
    CHUNK = "chunk"
    TOPIC = "topic"  # For topic clustering


class Node(BaseModel):
    """Generic graph node representation."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NodeType
    data: dict[str, Any] = Field(default_factory=dict)

    # Status tracking
    status: MemoryStatus = MemoryStatus.NEW

    # Temporal metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    access_count: int = 0
