"""Graph edge/relationship models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class RelationshipType(str, Enum):
    """Types of relationships between nodes."""
    
    # Semantic similarity
    SIMILAR_TO = "similar_to"
    
    # Temporal relationships
    UPDATES = "updates"
    FOLLOWS = "follows"
    
    # Hierarchical relationships
    PARENT_OF = "parent_of"
    PART_OF = "part_of"
    CLUSTERS_WITH = "clusters_with"
    ABSTRACTION_OF = "abstraction_of"
    
    # Entity co-occurrence
    CO_OCCURS = "co_occurs"
    
    # Causal/Sequential
    # FOLLOWS = "follows"
    REQUIRES = "requires"
    CAUSES = "causes"
    LEADS_TO = "leads_to"


class Edge(BaseModel):
    """Graph edge representation."""
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # Source node ID
    target: str  # Target node ID
    type: RelationshipType
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))