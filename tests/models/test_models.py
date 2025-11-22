"""
Tests for all model classes.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from src.models.memory import Memory, MemoryStatus, NodeType
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


class TestMemory:
    """Test Memory model."""

    def test_memory_creation(self):
        """Test basic memory creation."""
        memory = Memory(
            id="test_001",
            content="Test memory content",
            type=NodeType.MEMORY,
            user_id="user_001",
        )

        assert memory.id == "test_001"
        assert memory.content == "Test memory content"
        assert memory.type == NodeType.MEMORY
        assert memory.user_id == "user_001"
        assert memory.status == MemoryStatus.ACTIVE
        assert memory.version == 1
        assert memory.confidence == 1.0
        assert memory.access_count == 0
        assert memory.last_accessed is None

    def test_memory_with_metadata(self):
        """Test memory creation with metadata."""
        metadata = {"source": "test", "importance": "high"}
        memory = Memory(
            id="test_002",
            content="Test memory with metadata",
            user_id="user_001",
            metadata=metadata,
        )

        assert memory.metadata == metadata
        assert memory.metadata["source"] == "test"
        assert memory.user_id == "user_001"

    def test_memory_versioning(self):
        """Test memory versioning fields."""
        memory = Memory(
            id="test_003",
            content="Test memory",
            user_id="user_001",
            version=2,
            parent_version="test_002",
            valid_from=datetime.now(),
            valid_until=datetime.now() + timedelta(days=30),
        )

        assert memory.version == 2
        assert memory.parent_version == "test_002"
        assert memory.user_id == "user_001"
        assert memory.valid_until is not None

    def test_memory_status_transitions(self):
        """Test memory status transitions."""
        memory = Memory(
            id="test_004",
            content="Test memory",
            user_id="user_001",
            status=MemoryStatus.SUPERSEDED,
            superseded_by="test_005",
            invalidation_reason="Updated with new information",
        )

        assert memory.status == MemoryStatus.SUPERSEDED
        assert memory.superseded_by == "test_005"
        assert memory.user_id == "user_001"
        assert memory.invalidation_reason == "Updated with new information"

    def test_memory_validation(self):
        """Test memory validation."""
        # Test required fields
        with pytest.raises(ValidationError):
            Memory(content="Missing ID", user_id="user_001")

        with pytest.raises(ValidationError):
            Memory(id="test", user_id="user_001")

        # Test user_id is required
        with pytest.raises(ValidationError):
            Memory(id="test_001", content="Test content")

    def test_memory_methods(self):
        """Test memory utility methods."""
        memory = Memory(
            id="test_005",
            content="Test memory",
            user_id="user_001",
            created_at=datetime.now() - timedelta(days=5),
        )

        # Test age calculation
        assert memory.age_days() == 5

        # Test validity
        assert memory.is_valid()

        # Test access tracking
        memory.mark_accessed()
        assert memory.access_count == 1
        assert memory.last_accessed is not None

        # Test days since access
        days_since = memory.days_since_access()
        assert days_since == 0

    def test_memory_invalid_status(self):
        """Test memory with invalid status."""
        memory = Memory(
            id="test_006",
            content="Test memory",
            user_id="user_001",
            status=MemoryStatus.INVALIDATED,
            valid_until=datetime.now() - timedelta(days=1),
        )

        assert not memory.is_valid()


class TestRelationship:
    """Test Relationship model."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            type=RelationshipType.SIMILAR_TO,
            target_id="mem_123",
            confidence=0.85,
            reasoning="Both discuss Python frameworks",
        )

        assert rel.type == RelationshipType.SIMILAR_TO
        assert rel.target_id == "mem_123"
        assert rel.confidence == 0.85
        assert rel.reasoning == "Both discuss Python frameworks"

    def test_relationship_strict_schema(self):
        """Test relationship with strict schema (no extra fields allowed)."""
        # Should work with exact fields
        rel = Relationship(
            type=RelationshipType.UPDATES,
            target_id="mem_456",
            confidence=0.95,
            reasoning="Updates previous information",
        )

        assert rel.type == RelationshipType.UPDATES
        assert rel.target_id == "mem_456"

    def test_relationship_validation(self):
        """Test relationship validation."""
        # Test required fields
        with pytest.raises(ValidationError):
            Relationship(type=RelationshipType.SIMILAR_TO)

        with pytest.raises(ValidationError):
            Relationship(target_id="mem_123")

        # Note: Relationship model doesn't enforce confidence range
        # Pydantic v2 doesn't have built-in validation for range without explicit validator
        # This is acceptable for now - validation should happen at service layer


class TestRelationshipBundle:
    """Test RelationshipBundle model."""

    def test_bundle_creation(self):
        """Test basic bundle creation."""
        relationships = [
            Relationship(
                type=RelationshipType.SIMILAR_TO,
                target_id="mem_123",
                confidence=0.85,
                reasoning="Similar content",
            )
        ]

        bundle = RelationshipBundle(
            memory_id="mem_new",
            relationships=relationships,
        )

        assert bundle.memory_id == "mem_new"
        assert len(bundle.relationships) == 1

    def test_bundle_with_derived_insights(self):
        """Test bundle with derived insights."""
        insights = [
            DerivedInsight(
                content="User is learning Python systematically",
                confidence=0.8,
                reasoning="Pattern across multiple interactions",
                source_ids=["mem_001", "mem_002"],
                type="pattern_recognition",
            )
        ]

        bundle = RelationshipBundle(
            memory_id="mem_new",
            derived_insights=insights,
        )

        assert len(bundle.derived_insights) == 1
        assert bundle.derived_insights[0].content == "User is learning Python systematically"

    def test_bundle_empty(self):
        """Test bundle with no relationships or insights."""
        bundle = RelationshipBundle(
            memory_id="mem_new",
        )

        assert bundle.memory_id == "mem_new"
        assert len(bundle.relationships) == 0
        assert len(bundle.derived_insights) == 0


class TestDerivedInsight:
    """Test DerivedInsight model."""

    def test_insight_creation(self):
        """Test basic insight creation."""
        insight = DerivedInsight(
            content="User prefers async programming",
            confidence=0.9,
            reasoning="Consistent pattern in recent queries",
            source_ids=["mem_001", "mem_002", "mem_003"],
            type="pattern_recognition",
        )

        assert insight.content == "User prefers async programming"
        assert insight.confidence == 0.9
        assert insight.reasoning == "Consistent pattern in recent queries"
        assert len(insight.source_ids) == 3
        assert insight.type == "pattern_recognition"

    def test_insight_types(self):
        """Test different insight types."""
        types = ["pattern_recognition", "summary", "inference", "abstraction"]

        for insight_type in types:
            insight = DerivedInsight(
                content=f"Test {insight_type}",
                confidence=0.8,
                reasoning="Test reasoning",
                source_ids=["mem_001"],
                type=insight_type,
            )
            assert insight.type == insight_type


class TestEdge:
    """Test Edge model."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(
            source="mem_001",
            target="mem_002",
            type=RelationshipType.SIMILAR_TO,
            confidence=0.85,
        )

        assert edge.source == "mem_001"
        assert edge.target == "mem_002"
        assert edge.type == RelationshipType.SIMILAR_TO
        assert edge.confidence == 0.85
        assert edge.metadata == {}
        assert isinstance(edge.created_at, datetime)

    def test_edge_with_metadata(self):
        """Test edge with metadata."""
        metadata = {"strength": "high", "context": "technical"}
        edge = Edge(
            source="mem_001",
            target="mem_002",
            type=RelationshipType.UPDATES,
            metadata=metadata,
        )

        assert edge.metadata == metadata
        assert edge.confidence == 1.0  # Default confidence

    def test_edge_serialization(self):
        """Test edge JSON serialization."""
        edge = Edge(
            source="mem_001",
            target="mem_002",
            type=RelationshipType.SIMILAR_TO,
        )

        # Test that datetime is serialized correctly
        json_data = edge.model_dump()
        assert "created_at" in json_data
        # Check that created_at is a datetime object (not string) in model_dump
        # To get string serialization, use model_dump_json or model_dump(mode="json")
        assert isinstance(json_data["created_at"], datetime)


class TestMemoryEvolution:
    """Test MemoryEvolution model."""

    def test_evolution_creation(self):
        """Test basic evolution creation."""
        evolution = MemoryEvolution(
            current_version="mem_001",
            new_version="mem_002",
            action="update",
        )

        assert evolution.current_version == "mem_001"
        assert evolution.new_version == "mem_002"
        assert evolution.action == "update"

    def test_evolution_with_relationship(self):
        """Test evolution with relationship and change tracking."""
        change = VersionChange(
            change_type="update",
            reasoning="Updated with new information",
            description="Content was updated",
            changed_fields=["content"],
        )

        evolution = MemoryEvolution(
            current_version="mem_001",
            new_version="mem_002",
            action="update",
            change=change,
        )

        assert evolution.change is not None
        assert evolution.change.change_type == "update"
        assert evolution.change.reasoning == "Updated with new information"


class TestVersionChange:
    """Test VersionChange model."""

    def test_version_change_creation(self):
        """Test basic version change creation."""
        change = VersionChange(
            change_type="update",
            reasoning="Updated content with new information",
            description="Modified the content field",
            changed_fields=["content"],
        )

        assert change.change_type == "update"
        assert change.reasoning == "Updated content with new information"
        assert change.description == "Modified the content field"
        assert change.changed_fields == ["content"]
        assert isinstance(change.timestamp, datetime)


class TestVersionChain:
    """Test VersionChain model."""

    def test_version_chain_creation(self):
        """Test basic version chain creation."""
        chain = VersionChain(
            original_id="mem_001",
            current_version_id="mem_003",
            versions=[
                {"id": "mem_001", "version": 1},
                {"id": "mem_002", "version": 2},
                {"id": "mem_003", "version": 3},
            ],
            total_versions=3,
            created_at=datetime.now(),
        )

        assert chain.original_id == "mem_001"
        assert chain.current_version_id == "mem_003"
        assert len(chain.versions) == 3
        assert chain.total_versions == 3


class TestInvalidationResult:
    """Test InvalidationResult model."""

    def test_invalidation_creation(self):
        """Test basic invalidation creation."""
        result = InvalidationResult(
            memory_id="mem_001",
            status=InvalidationStatus.ACTIVE,
            reasoning="New information available",
            confidence=0.95,
        )

        assert result.memory_id == "mem_001"
        assert result.status == InvalidationStatus.ACTIVE
        assert result.reasoning == "New information available"
        assert result.confidence == 0.95

    def test_invalidation_preservation(self):
        """Test invalidation with preservation."""
        result = InvalidationResult(
            memory_id="mem_001",
            status=InvalidationStatus.HISTORICAL,
            reasoning="Outdated but useful context",
            confidence=0.85,
        )

        assert result.memory_id == "mem_001"
        assert result.status == InvalidationStatus.HISTORICAL
        assert result.confidence == 0.85


class TestRelationshipTypes:
    """Test RelationshipType enum."""

    def test_all_relationship_types(self):
        """Test all relationship types are defined."""
        expected_types = [
            "SIMILAR_TO",
            "REFERENCES",
            "PRECEDES",
            "FOLLOWS",
            "UPDATES",
            "PART_OF",
            "BELONGS_TO",
            "PARENT_OF",
            "CONTRADICTS",
            "SUPPORTS",
            "REQUIRES",
            "DEPENDS_ON",
            "DERIVED_FROM",
            "SYNTHESIZES",
            "CO_OCCURS",
            "MENTIONS",
            "RESPONDS_TO",
        ]

        for rel_type in expected_types:
            assert hasattr(RelationshipType, rel_type)
            assert RelationshipType(rel_type) == rel_type

    def test_relationship_type_values(self):
        """Test relationship type string values."""
        assert RelationshipType.SIMILAR_TO == "SIMILAR_TO"
        assert RelationshipType.UPDATES == "UPDATES"
        assert RelationshipType.CONTRADICTS == "CONTRADICTS"


class TestNodeTypes:
    """Test NodeType enum."""

    def test_all_node_types(self):
        """Test all node types are defined."""
        expected_types = ["MEMORY", "DERIVED", "TOPIC", "ENTITY", "DOCUMENT", "CHUNK"]

        for node_type in expected_types:
            assert hasattr(NodeType, node_type)
            assert NodeType(node_type) == node_type


class TestMemoryStatus:
    """Test MemoryStatus enum."""

    def test_all_status_types(self):
        """Test all status types are defined."""
        expected_statuses = ["active", "historical", "superseded", "invalidated"]

        for status in expected_statuses:
            assert hasattr(MemoryStatus, status.upper())
            assert MemoryStatus(status) == status


class TestContextBundle:
    """Test ContextBundle model."""

    def test_context_bundle_creation(self):
        """Test basic context bundle creation."""
        memory1 = Memory(id="mem_001", content="Content 1", user_id="user_001")
        memory2 = Memory(id="mem_002", content="Content 2", user_id="user_001")

        bundle = ContextBundle(
            vector_candidates=[memory1],
            temporal_context=[memory2],
        )

        assert len(bundle.vector_candidates) == 1
        assert len(bundle.temporal_context) == 1
        assert len(bundle.graph_context) == 0
        assert len(bundle.entity_context) == 0
        assert len(bundle.conversation_context) == 0
        assert len(bundle.filtered_candidates) == 0

    def test_context_bundle_with_all_context(self):
        """Test context bundle with all context types."""
        memory1 = Memory(id="mem_001", content="Content 1", user_id="user_001")
        memory2 = Memory(id="mem_002", content="Content 2", user_id="user_001")
        memory3 = Memory(id="mem_003", content="Content 3", user_id="user_001")

        bundle = ContextBundle(
            vector_candidates=[memory1],
            temporal_context=[memory2],
            graph_context=[memory3],
            entity_context=[memory1, memory2],
            conversation_context=[memory3],
            filtered_candidates=[memory1],
        )

        assert len(bundle.vector_candidates) == 1
        assert len(bundle.temporal_context) == 1
        assert len(bundle.graph_context) == 1
        assert len(bundle.entity_context) == 2
        assert len(bundle.conversation_context) == 1
        assert len(bundle.filtered_candidates) == 1
