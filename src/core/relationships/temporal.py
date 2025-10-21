"""Temporal relationship inference engine - mimicking human brain's time-based memory dynamics."""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.graph_store import GraphStore
from src.core.vector_store import QdrantStore
from src.models import MemoryStatus, RelationshipType


class TemporalRelationshipEngine:
    """
    Engine for inferring temporal relationships between memories.

    Mimics human brain's temporal memory dynamics:
    - Recency bias: Recent memories are more accessible
    - Temporal decay: Memories fade over time
    - Update detection: New versions replace old information
    - Consolidation: Frequently accessed memories resist decay
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        new_memory_window_hours: int = 48,
        expiring_threshold_days: int = 30,
        update_similarity_threshold: float = 0.85,
        update_time_window_days: int = 7,
        decay_half_life_days: int = 30,
    ):
        """
        Initialize temporal relationship engine.

        Args:
            vector_store: Vector store for memory retrieval
            graph_store: Graph store for creating temporal edges
            new_memory_window_hours: Hours to consider a memory "new"
            expiring_threshold_days: Days before marking memory as expiring
            update_similarity_threshold: Similarity threshold for update detection
            update_time_window_days: Days to look back for update detection
            decay_half_life_days: Half-life for exponential decay calculation
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.new_memory_window_hours = new_memory_window_hours
        self.expiring_threshold_days = expiring_threshold_days
        self.update_similarity_threshold = update_similarity_threshold
        self.update_time_window_days = update_time_window_days
        self.decay_half_life_days = decay_half_life_days

    async def detect_new_memories(self, memory_id: str, created_at: datetime) -> bool:
        """
        Detect if a memory is "new" (recently created).

        Args:
            memory_id: Memory identifier
            created_at: Creation timestamp

        Returns:
            True if memory is new
        """
        now = datetime.now(UTC)

        # Ensure created_at is timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        age = now - created_at

        is_new = age <= timedelta(hours=self.new_memory_window_hours)

        return is_new

    async def detect_expiring_memories(
        self,
        memory_id: str,
        created_at: datetime,
        last_accessed: datetime | None = None,
        access_count: int = 0,
    ) -> dict[str, Any]:
        """
        Detect if a memory is expiring (should be forgotten soon).

        Uses a decay model similar to Ebbinghaus forgetting curve:
        - Time since creation
        - Time since last access
        - Access frequency (consolidation)

        Args:
            memory_id: Memory identifier
            created_at: Creation timestamp
            last_accessed: Last access timestamp
            access_count: Number of times accessed

        Returns:
            Dictionary with decay score and expiring status
        """
        now = datetime.now(UTC)

        # Ensure created_at is timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        # Calculate age-based decay
        age_days = (now - created_at).total_seconds() / (24 * 3600)
        age_decay = self._exponential_decay(age_days)

        # Calculate recency decay (time since last access)
        if last_accessed:
            # Ensure last_accessed is timezone-aware
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=UTC)

            days_since_access = (now - last_accessed).total_seconds() / (24 * 3600)
            recency_decay = self._exponential_decay(days_since_access)
        else:
            # Never accessed = same as age decay
            recency_decay = age_decay
            days_since_access = age_days

        # Calculate consolidation factor (resistance to forgetting)
        # More accesses = stronger memory
        consolidation = 1.0 / (1.0 + access_count)

        # Combined decay score (0 = fresh, 1 = completely decayed)
        # Weights: 40% age, 40% recency, 20% consolidation
        decay_score = 0.4 * age_decay + 0.4 * recency_decay + 0.2 * consolidation

        # Determine status
        is_expiring = age_days >= self.expiring_threshold_days
        is_forgotten = decay_score > 0.8

        return {
            "decay_score": decay_score,
            "is_expiring": is_expiring,
            "is_forgotten": is_forgotten,
            "age_days": age_days,
            "days_since_access": days_since_access,
            "access_count": access_count,
            "consolidation_strength": 1.0 - consolidation,
        }

    def _exponential_decay(self, days: float) -> float:
        """
        Calculate exponential decay based on time.

        Uses half-life formula: decay = 1 - exp(-λt)
        where λ = ln(2) / half_life

        Args:
            days: Number of days

        Returns:
            Decay value between 0 (no decay) and 1 (complete decay)
        """
        lambda_param = math.log(2) / self.decay_half_life_days
        decay = 1 - math.exp(-lambda_param * days)
        return min(decay, 1.0)  # Cap at 1.0

    async def detect_updates(
        self,
        memory_id: str,
        embedding: list[float],
        created_at: datetime,
        create_edges: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Detect if a memory is an update of older similar memories.

        Looks for semantically similar memories created within a time window
        and marks them with UPDATE relationship.

        Args:
            memory_id: Memory identifier
            embedding: Memory embedding
            created_at: Creation timestamp
            create_edges: Whether to create UPDATE edges

        Returns:
            List of older memories that this one updates
        """

        # Ensure created_at is timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        # Search for similar memories
        similar_memories = await self.vector_store.search_similar(
            query_vector=embedding, limit=20, score_threshold=self.update_similarity_threshold
        )

        # Filter for older memories within time window
        updates = []
        for mem in similar_memories:
            if mem["id"] == memory_id:
                continue

            # Get creation time from metadata
            mem_created = mem["metadata"].get("created_at")
            if not mem_created:
                continue

            try:
                mem_created_dt = datetime.fromisoformat(mem_created).replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue

            # Check if older and within time window
            time_diff = (created_at - mem_created_dt).total_seconds() / (24 * 3600)

            if 0 < time_diff <= self.update_time_window_days:
                updates.append(
                    {
                        "id": mem["id"],
                        "score": mem["score"],
                        "days_older": time_diff,
                        "metadata": mem["metadata"],
                    }
                )

        # Create UPDATE edges
        if create_edges and updates:
            for update in updates:
                await self.graph_store.add_edge(
                    source_id=memory_id,
                    target_id=update["id"],
                    edge_type=RelationshipType.UPDATES,
                    weight=update["score"],
                    metadata={
                        "similarity_score": update["score"],
                        "days_difference": update["days_older"],
                        "inference_method": "temporal_update_detection",
                    },
                )

        return updates

    async def create_temporal_sequence(
        self, memory_ids: list[str], sequence_type: str = "conversation"
    ) -> int:
        """
        Create temporal sequence edges (FOLLOWS) between memories.

        Useful for conversation threads, document sections, etc.

        Args:
            memory_ids: List of memory IDs in sequence order
            sequence_type: Type of sequence (conversation, document, etc.)

        Returns:
            Number of edges created
        """
        edges_created = 0

        for i in range(len(memory_ids) - 1):
            current_id = memory_ids[i]
            next_id = memory_ids[i + 1]

            await self.graph_store.add_edge(
                source_id=current_id,
                target_id=next_id,
                edge_type=RelationshipType.FOLLOWS,
                weight=1.0,
                metadata={
                    "sequence_type": sequence_type,
                    "sequence_index": i,
                    "total_sequence_length": len(memory_ids),
                },
            )
            edges_created += 1

        return edges_created

    async def apply_decay_to_node(self, memory_id: str, decay_info: dict[str, Any]) -> None:
        """
        Apply decay effects to a memory node.

        Updates status based on decay score:
        - NEW → ACTIVE (on first access)
        - ACTIVE → EXPIRING_SOON (moderate decay)
        - EXPIRING_SOON → FORGOTTEN (high decay)

        Args:
            memory_id: Memory identifier
            decay_info: Decay information from detect_expiring_memories
        """
        # Get current node
        node = await self.graph_store.get_node(memory_id)
        if not node:
            return

        current_status = node.status
        new_status = current_status

        # Determine new status based on decay
        if decay_info["is_forgotten"]:
            new_status = MemoryStatus.FORGOTTEN
        elif decay_info["is_expiring"]:
            if current_status != MemoryStatus.FORGOTTEN:
                new_status = MemoryStatus.EXPIRING_SOON

        # Update status if changed
        if new_status != current_status:
            await self.graph_store.update_node_status(memory_id, new_status)

    async def run_decay_cycle(self, memory_ids: list[str] | None = None) -> dict[str, Any]:
        """
        Run a decay cycle on memories (like sleep consolidation in brain).

        This simulates the brain's memory consolidation during sleep:
        - Calculate decay for all memories
        - Update statuses
        - Reduce edge weights for expiring memories

        Args:
            memory_ids: Optional list of specific memory IDs (or all if None)

        Returns:
            Summary statistics of the decay cycle
        """
        stats = {
            "processed": 0,
            "marked_expiring": 0,
            "marked_forgotten": 0,
            "average_decay_score": 0.0,
        }

        if not memory_ids:
            # In a real implementation, you'd fetch all memory IDs
            # For now, we'll work with provided IDs
            return stats

        total_decay = 0.0

        for memory_id in memory_ids:
            # Get memory data
            node = await self.graph_store.get_node(memory_id)
            if not node:
                continue

            # Calculate decay
            decay_info = await self.detect_expiring_memories(
                memory_id=memory_id,
                created_at=node.created_at,
                last_accessed=node.last_accessed,
                access_count=node.access_count,
            )

            # Apply decay effects
            await self.apply_decay_to_node(memory_id, decay_info)

            # Update statistics
            stats["processed"] += 1
            total_decay += decay_info["decay_score"]

            if decay_info["is_forgotten"]:
                stats["marked_forgotten"] += 1
            elif decay_info["is_expiring"]:
                stats["marked_expiring"] += 1

        if stats["processed"] > 0:
            stats["average_decay_score"] = total_decay / stats["processed"]

        return stats

    async def detect_temporal_clusters(
        self, memory_ids: list[str], time_window_hours: int = 24
    ) -> list[list[str]]:
        """
        Find clusters of memories created close in time.

        Similar to how brain groups related experiences from same time period.

        Args:
            memory_ids: List of memory IDs to cluster
            time_window_hours: Hours within which to cluster memories

        Returns:
            List of temporal clusters
        """
        # Get creation times for all memories
        memory_times = []

        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if node:
                memory_times.append((mem_id, node.created_at))

        # Sort by time
        memory_times.sort(key=lambda x: x[1])

        # Create clusters
        clusters = []
        current_cluster = []

        for i, (mem_id, created_at) in enumerate(memory_times):
            if not current_cluster:
                current_cluster.append(mem_id)
            else:
                # Check time difference with last memory in cluster
                last_idx = len(current_cluster) - 1
                last_time = memory_times[i - last_idx - 1][1]
                time_diff = (created_at - last_time).total_seconds() / 3600

                if time_diff <= time_window_hours:
                    current_cluster.append(mem_id)
                else:
                    # Start new cluster
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [mem_id]

        # Add last cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        return clusters

    def calculate_temporal_relevance(
        self, created_at: datetime, last_accessed: datetime | None = None, access_count: int = 0
    ) -> float:
        """
        Calculate temporal relevance score (inverse of decay).

        Higher score = more temporally relevant (recent, frequently accessed)

        Args:
            created_at: Creation timestamp
            last_accessed: Last access timestamp
            access_count: Access count

        Returns:
            Relevance score (0-1, higher is more relevant)
        """
        now = datetime.now(UTC)

        # Ensure timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        # Recency score
        if last_accessed:
            # Ensure last_accessed is timezone-aware
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=UTC)

            hours_since_access = (now - last_accessed).total_seconds() / 3600
        else:
            hours_since_access = (now - created_at).total_seconds() / 3600

        # Exponential decay of relevance
        recency_score = math.exp(-hours_since_access / (24 * self.decay_half_life_days))

        # Access frequency boost
        frequency_boost = min(access_count / 10.0, 0.5)  # Max 0.5 boost

        relevance = min(recency_score + frequency_boost, 1.0)

        return relevance
