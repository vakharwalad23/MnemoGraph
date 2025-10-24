"""
Relationship models and types for memory graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between memories."""

    # Semantic
    SIMILAR_TO = "SIMILAR_TO"  # Semantically similar
    REFERENCES = "REFERENCES"  # Mentions/cites

    # Temporal/Causal
    PRECEDES = "PRECEDES"  # Comes before
    FOLLOWS = "FOLLOWS"  # Comes after
    UPDATES = "UPDATES"  # Updates/corrects
    SUPERSEDES = "SUPERSEDES"  # Replaces

    # Hierarchical
    PART_OF = "PART_OF"  # Component of larger whole
    BELONGS_TO = "BELONGS_TO"  # Member of category
    PARENT_OF = "PARENT_OF"  # Parent in hierarchy

    # Logical
    CONTRADICTS = "CONTRADICTS"  # Conflicts with
    SUPPORTS = "SUPPORTS"  # Provides evidence for
    REQUIRES = "REQUIRES"  # Prerequisite
    DEPENDS_ON = "DEPENDS_ON"  # Dependency

    # Synthesis
    DERIVED_FROM = "DERIVED_FROM"  # Synthesized from
    SYNTHESIZES = "SYNTHESIZES"  # Creates synthesis

    # Entity
    CO_OCCURS = "CO_OCCURS"  # Shared entities
    MENTIONS = "MENTIONS"  # Mentions entity

    # Conversation
    RESPONDS_TO = "RESPONDS_TO"  # Response in conversation


class Edge(BaseModel):
    """
    Relationship edge between two nodes.
    """

    source: str  # Source node ID
    target: str  # Target node ID
    type: RelationshipType
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class RelationshipBundle(BaseModel):
    """
    Complete set of relationships extracted for a memory.
    """

    memory_id: str
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    derived_insights: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    overall_analysis: str = ""
    extraction_time_ms: float = 0.0


class DerivedInsight(BaseModel):
    """
    LLM-generated insight derived from multiple memories.
    """

    content: str
    confidence: float
    reasoning: str
    source_ids: list[str]
    type: str  # pattern_recognition, summary, inference, abstraction


class ContextBundle(BaseModel):
    """
    Complete context gathered for relationship extraction.
    """

    vector_candidates: list[Any] = Field(default_factory=list)  # Top vector matches
    temporal_context: list[Any] = Field(default_factory=list)  # Recent memories
    graph_context: list[Any] = Field(default_factory=list)  # Graph neighbors
    entity_context: list[Any] = Field(default_factory=list)  # Shared entities
    conversation_context: list[Any] = Field(default_factory=list)  # Thread context
    filtered_candidates: list[Any] = Field(default_factory=list)  # Final filtered set


class FilterStageResult(BaseModel):
    """Result from a filtering stage."""

    stage: str
    candidates_in: int
    candidates_out: int
    time_ms: float
    method: str
