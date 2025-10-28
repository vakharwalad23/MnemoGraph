"""
Relationship models and types for memory graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.memory import Memory


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


class Relationship(BaseModel):
    """Individual relationship between memories."""

    model_config = {"extra": "ignore"}

    type: RelationshipType = Field(
        ...,
        description="REQUIRED: The type of relationship. Must be one of the valid RelationshipType enum values (RELATES_TO, CAUSED_BY, CONFLICTS_WITH, etc.)",
    )
    target_id: str = Field(
        ..., description="REQUIRED: The ID of the target memory that this relationship points to"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="REQUIRED: Confidence score between 0.0 and 1.0. Use 1.0 for explicit relationships, 0.7-0.9 for strong inferences, 0.5-0.7 for weak inferences",
    )
    reasoning: str = Field(
        ...,
        description="REQUIRED: Explain why this relationship exists. Provide specific evidence from both memories.",
    )


class Conflict(BaseModel):
    """Conflict between memories."""

    model_config = {"extra": "ignore"}

    target_id: str = Field(..., description="REQUIRED: The ID of the conflicting memory")
    conflict_type: str = Field(
        ...,
        description="REQUIRED: Type of conflict (e.g., 'factual_contradiction', 'temporal_inconsistency', 'logical_conflict')",
    )
    resolution: str = Field(
        ...,
        description="REQUIRED: How this conflict should be resolved (e.g., 'prefer_newer', 'prefer_more_confident', 'keep_both')",
    )
    reasoning: str = Field(
        ...,
        description="REQUIRED: Explain the nature of the conflict and why this resolution approach is appropriate",
    )


class DerivedInsight(BaseModel):
    """
    LLM-generated insight derived from multiple memories.
    """

    model_config = {"extra": "ignore"}

    content: str = Field(
        ...,
        description="REQUIRED: The actual insight content. Should be a clear, concise statement of the discovered pattern or conclusion.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="REQUIRED: Confidence in this insight between 0.0 and 1.0. Higher for insights with strong supporting evidence.",
    )
    reasoning: str = Field(
        ...,
        description="REQUIRED: Explain how this insight was derived from the source memories. Include specific evidence.",
    )
    source_ids: list[str] = Field(
        ...,
        description="REQUIRED: List of memory IDs that contributed to this insight. Must include at least one ID.",
    )
    type: str = Field(
        ...,
        description="REQUIRED: Type of insight. Must be one of: 'pattern_recognition', 'summary', 'inference', 'abstraction'",
    )


class RelationshipBundle(BaseModel):
    """
    Complete set of relationships extracted for a memory.
    """

    model_config = {"extra": "ignore"}

    memory_id: str = Field(
        ...,
        description="REQUIRED: The ID of the memory for which relationships are being extracted",
    )
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="OPTIONAL: List of direct relationships to other memories. Can be empty if no relationships found.",
    )
    derived_insights: list[DerivedInsight] = Field(
        default_factory=list,
        description="OPTIONAL: List of insights derived from analyzing multiple memories together. Can be empty.",
    )
    conflicts: list[Conflict] = Field(
        default_factory=list,
        description="OPTIONAL: List of conflicts with other memories. Can be empty if no conflicts found.",
    )
    overall_analysis: str = Field(
        default="",
        description="OPTIONAL: High-level summary of the relationship analysis. Provide overview of key findings.",
    )
    extraction_time_ms: float = Field(
        default=0.0,
        description="OPTIONAL: Time taken to extract relationships in milliseconds. For internal tracking.",
    )


class ContextBundle(BaseModel):
    """
    Complete context gathered for relationship extraction.
    """

    model_config = {"extra": "ignore"}  # Ignore extra fields

    vector_candidates: list[Memory] = Field(default_factory=list)  # Top vector matches
    temporal_context: list[Memory] = Field(default_factory=list)  # Recent memories
    graph_context: list[Memory] = Field(default_factory=list)  # Graph neighbors
    entity_context: list[Memory] = Field(default_factory=list)  # Shared entities
    conversation_context: list[Memory] = Field(default_factory=list)  # Thread context
    filtered_candidates: list[Memory] = Field(default_factory=list)  # Final filtered set


class FilterStageResult(BaseModel):
    """Result from a filtering stage."""

    stage: str
    candidates_in: int
    candidates_out: int
    time_ms: float
    method: str
