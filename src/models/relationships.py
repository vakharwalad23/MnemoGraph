"""
Relationship models and types for memory graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.memory import Memory


class RelationshipType(str, Enum):
    """Types of relationships between nodes in the knowledge graph."""

    # Semantic (Memory ↔ Memory)
    SIMILAR_TO = "SIMILAR_TO"
    REFERENCES = "REFERENCES"

    # Temporal/Causal (Memory ↔ Memory)
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    UPDATES = "UPDATES"

    # Hierarchical (Memory ↔ Memory)
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    PARENT_OF = "PARENT_OF"

    # Logical (Memory ↔ Memory)
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    REQUIRES = "REQUIRES"
    DEPENDS_ON = "DEPENDS_ON"

    # Synthesis (Memory ↔ Memory)
    DERIVED_FROM = "DERIVED_FROM"
    SYNTHESIZES = "SYNTHESIZES"

    # Entity (Memory ↔ Memory)
    CO_OCCURS = "CO_OCCURS"
    MENTIONS = "MENTIONS"

    # Conversation (Memory ↔ Memory)
    RESPONDS_TO = "RESPONDS_TO"

    # Source Linkage (Note/Document → Memory)
    HAS_MEMORY = "HAS_MEMORY"  # Note/Document to extracted Memory

    # Document Structure (Document → Chunk, Chunk → Chunk)
    HAS_CHUNK = "HAS_CHUNK"  # Document to its Chunks
    NEXT_CHUNK = "NEXT_CHUNK"  # Sequential chunk ordering


class Edge(BaseModel):
    """Relationship edge between two nodes."""

    source: str
    target: str
    type: RelationshipType
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Relationship(BaseModel):
    """Individual relationship between memories for LLM structured output."""

    model_config = {"extra": "ignore"}

    type: RelationshipType = Field(..., description="Relationship type")
    target_id: str = Field(..., description="Target memory ID")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence (0-1): 1.0=explicit, 0.7-0.9=strong, 0.5-0.7=weak",
    )
    reasoning: str = Field(..., description="Why this relationship exists with specific evidence")


class DerivedInsight(BaseModel):
    """LLM-generated insight from multiple memories for structured output."""

    model_config = {"extra": "ignore"}

    content: str = Field(..., description="Insight statement describing the pattern or conclusion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1) in this insight")
    reasoning: str = Field(..., description="How insight was derived with specific evidence")
    source_ids: list[str] = Field(..., description="Memory IDs that contributed (min 1)")
    type: str = Field(
        ..., description="Type: pattern_recognition, summary, inference, or abstraction"
    )


class RelationshipBundle(BaseModel):
    """
    Complete set of relationships extracted for a memory (LLM structured output).

    Note: This is the minimal model for LLM output. Tracking fields like extraction_time_ms
    should be added after LLM call, not included in the model schema.
    """

    model_config = {"extra": "ignore"}

    memory_id: str = Field(..., description="Memory ID for which relationships are extracted")
    relationships: list[Relationship] = Field(
        default_factory=list, description="Direct relationships to other memories (or empty)"
    )
    derived_insights: list[DerivedInsight] = Field(
        default_factory=list, description="Insights from analyzing multiple memories (or empty)"
    )


class ContextBundle(BaseModel):
    """Complete context gathered for relationship extraction."""

    model_config = {"extra": "ignore"}

    vector_candidates: list[Memory] = Field(default_factory=list)
    temporal_context: list[Memory] = Field(default_factory=list)
    graph_context: list[Memory] = Field(default_factory=list)
    entity_context: list[Memory] = Field(default_factory=list)
    conversation_context: list[Memory] = Field(default_factory=list)
    filtered_candidates: list[Memory] = Field(default_factory=list)
