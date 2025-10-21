"""
Relationship orchestrator - automatically infers all relationships for new memories.

When a memory is added, this orchestrator:
1. Runs semantic similarity detection
2. Detects temporal relationships (updates, sequences)
3. Identifies hierarchical relationships
4. Finds entity co-occurrences
5. Detects causal/sequential links
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from src.config import Config
from src.core.embeddings import EmbeddingProvider
from src.core.graph_store import GraphStore
from src.core.relationships import (
    CausalSequentialEngine,
    EntityCooccurrenceEngine,
    HierarchicalRelationshipEngine,
    SemanticSimilarityEngine,
    TemporalRelationshipEngine,
)
from src.core.vector_store import QdrantStore

logger = logging.getLogger(__name__)


class RelationshipOrchestrator:
    """
    Orchestrates all relationship inference engines.

    Automatically infers relationships when new memories are added,
    running all engines in parallel for efficiency.
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        embedder: EmbeddingProvider,
        config: Config,
    ):
        """
        Initialize relationship orchestrator.

        Args:
            vector_store: Vector database
            graph_store: Graph database
            embedder: Embedding provider
            config: System configuration
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.config = config

        # Initialize all relationship engines
        self.semantic_engine = SemanticSimilarityEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            similarity_threshold=config.relationships.semantic.similarity_threshold,
            max_similar_memories=config.relationships.semantic.max_similar_memories,
        )

        self.temporal_engine = TemporalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            new_memory_window_hours=config.relationships.temporal.new_memory_window_hours,
            expiring_threshold_days=config.relationships.temporal.expiring_threshold_days,
            update_similarity_threshold=config.relationships.temporal.update_similarity_threshold,
            update_time_window_days=config.relationships.temporal.update_time_window_days,
            decay_half_life_days=config.relationships.temporal.decay_half_life_days,
        )

        self.hierarchical_engine = HierarchicalRelationshipEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_cluster_size=config.relationships.hierarchical.min_cluster_size,
            num_topics=config.relationships.hierarchical.num_topics,
            abstraction_threshold=config.relationships.hierarchical.abstraction_threshold,
        )

        self.cooccurrence_engine = EntityCooccurrenceEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            min_entity_length=config.relationships.cooccurrence.min_entity_length,
            min_cooccurrence_count=config.relationships.cooccurrence.min_cooccurrence_count,
            entity_weight_threshold=config.relationships.cooccurrence.entity_weight_threshold,
            use_spacy=config.relationships.cooccurrence.use_spacy,
        )

        self.causal_engine = CausalSequentialEngine(
            vector_store=vector_store,
            graph_store=graph_store,
            max_sequence_gap=config.relationships.causal.max_sequence_gap_seconds,
            similarity_threshold=config.relationships.causal.similarity_threshold,
            topic_shift_threshold=config.relationships.causal.topic_shift_threshold,
        )

        logger.info("Relationship orchestrator initialized with all engines")

    async def process_new_memory(
        self,
        memory_id: str,
        text: str,
        embedding: list[float],
        created_at: datetime | None = None,
        context_memory_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Automatically infer all relationships for a new memory.

        Runs all relationship inference engines in parallel.

        Args:
            memory_id: Memory identifier
            text: Memory text content
            embedding: Memory embedding vector
            created_at: Creation timestamp (defaults to now)
            context_memory_ids: Optional list of recent memory IDs for context

        Returns:
            Statistics about relationships created
        """
        if created_at is None:
            created_at = datetime.now(UTC)

        logger.info(f"Processing new memory: {memory_id}")

        stats = {
            "memory_id": memory_id,
            "processed_at": datetime.now(UTC).isoformat(),
            "relationships_created": 0,
            "engines_run": {},
        }

        # Run all engines in parallel
        tasks = []

        # 1. Semantic similarity
        tasks.append(self._run_semantic_inference(memory_id, embedding, stats))

        # 2. Temporal relationships
        tasks.append(self._run_temporal_inference(memory_id, embedding, created_at, stats))

        # 3. Hierarchical relationships (needs context)
        if context_memory_ids:
            tasks.append(self._run_hierarchical_inference(memory_id, context_memory_ids, stats))

        # 4. Entity co-occurrence (needs context)
        if context_memory_ids:
            tasks.append(
                self._run_cooccurrence_inference(memory_id, text, context_memory_ids, stats)
            )

        # 5. Causal/sequential (needs context)
        if context_memory_ids:
            tasks.append(self._run_causal_inference(memory_id, context_memory_ids, stats))

        # Execute all in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"Processed memory {memory_id}: "
            f"{stats['relationships_created']} relationships created"
        )

        return stats

    async def _run_semantic_inference(
        self, memory_id: str, embedding: list[float], stats: dict[str, Any]
    ) -> None:
        """Run semantic similarity inference."""
        try:
            similar_memories = await self.semantic_engine.infer_relationships(
                memory_id=memory_id, embedding=embedding, create_edges=True
            )

            stats["engines_run"]["semantic"] = {
                "success": True,
                "edges_created": len(similar_memories),
                "similar_count": len(similar_memories),
            }
            stats["relationships_created"] += len(similar_memories)

        except Exception as e:
            logger.error(f"Semantic inference failed: {e}")
            stats["engines_run"]["semantic"] = {"success": False, "error": str(e)}

    async def _run_temporal_inference(
        self, memory_id: str, embedding: list[float], created_at: datetime, stats: dict[str, Any]
    ) -> None:
        """Run temporal relationship inference."""
        try:
            # Detect updates
            updates = await self.temporal_engine.detect_updates(
                memory_id=memory_id, embedding=embedding, created_at=created_at, create_edges=True
            )

            stats["engines_run"]["temporal"] = {
                "success": True,
                "updates_found": len(updates),
            }
            stats["relationships_created"] += len(updates)

        except Exception as e:
            logger.error(f"Temporal inference failed: {e}")
            stats["engines_run"]["temporal"] = {"success": False, "error": str(e)}

    async def _run_hierarchical_inference(
        self, memory_id: str, context_memory_ids: list[str], stats: dict[str, Any]
    ) -> None:
        """Run hierarchical relationship inference."""
        try:
            # Get all memories including new one
            all_ids = context_memory_ids + [memory_id]

            # Create topic clusters
            result = await self.hierarchical_engine.create_topic_clusters(memory_ids=all_ids)

            stats["engines_run"]["hierarchical"] = {
                "success": True,
                "clusters_found": result.get("num_clusters", 0),
                "edges_created": result.get("edges_created", 0),
            }
            stats["relationships_created"] += result.get("edges_created", 0)

        except Exception as e:
            logger.error(f"Hierarchical inference failed: {e}")
            stats["engines_run"]["hierarchical"] = {"success": False, "error": str(e)}

    async def _run_cooccurrence_inference(
        self, memory_id: str, text: str, context_memory_ids: list[str], stats: dict[str, Any]
    ) -> None:
        """Run entity co-occurrence inference."""
        try:
            # Extract entities from new memory
            entities = self.cooccurrence_engine.extract_entities(text)

            if entities:
                # Find co-occurrences with context memories
                all_ids = context_memory_ids + [memory_id]
                result = await self.cooccurrence_engine.create_cooccurrence_edges(
                    memory_ids=all_ids
                )

                stats["engines_run"]["cooccurrence"] = {
                    "success": True,
                    "entities_found": len(entities),
                    "edges_created": result["edges_created"],
                }
                stats["relationships_created"] += result["edges_created"]
            else:
                stats["engines_run"]["cooccurrence"] = {
                    "success": True,
                    "entities_found": 0,
                    "edges_created": 0,
                }

        except Exception as e:
            logger.error(f"Co-occurrence inference failed: {e}")
            stats["engines_run"]["cooccurrence"] = {"success": False, "error": str(e)}

    async def _run_causal_inference(
        self, memory_id: str, context_memory_ids: list[str], stats: dict[str, Any]
    ) -> None:
        """Run causal/sequential inference."""
        try:
            # Detect prerequisites (if memory depends on earlier knowledge)
            result = await self.causal_engine.create_prerequisite_edges(
                memory_id=memory_id, candidate_ids=context_memory_ids, max_prerequisites=3
            )

            stats["engines_run"]["causal"] = {
                "success": True,
                "prerequisites_found": len(result["prerequisites"]),
                "edges_created": result["edges_created"],
            }
            stats["relationships_created"] += result["edges_created"]

        except Exception as e:
            logger.error(f"Causal inference failed: {e}")
            stats["engines_run"]["causal"] = {"success": False, "error": str(e)}

    async def batch_process_memories(
        self, memory_ids: list[str], batch_size: int = 50
    ) -> dict[str, Any]:
        """
        Process multiple memories in batches.

        Useful for reindexing or initial bulk processing.

        Args:
            memory_ids: List of memory IDs to process
            batch_size: Memories per batch

        Returns:
            Aggregated statistics
        """
        logger.info(f"Batch processing {len(memory_ids)} memories")

        total_stats = {
            "total_memories": len(memory_ids),
            "total_relationships": 0,
            "batches_processed": 0,
            "engine_stats": {},
        }

        # Process in batches
        for i in range(0, len(memory_ids), batch_size):
            batch = memory_ids[i : i + batch_size]

            # Run batch inference engines
            try:
                # Get batch memories with embeddings
                batch_memories = []
                for mem_id in batch:
                    mem_data = await self.vector_store.get_memory(mem_id)
                    if mem_data and mem_data.get("vector"):
                        batch_memories.append({"id": mem_id, "embedding": mem_data["vector"]})

                # Semantic similarity for batch
                semantic_result = await self.semantic_engine.batch_infer_relationships(
                    memories=batch_memories, create_edges=True
                )
                edges_count = sum(len(similar) for similar in semantic_result.values())
                total_stats["total_relationships"] += edges_count

                # Hierarchical clustering for batch
                hierarchical_result = await self.hierarchical_engine.create_topic_clusters(
                    memory_ids=batch
                )
                total_stats["total_relationships"] += hierarchical_result.get("edges_created", 0)

                # Entity co-occurrence for batch
                cooccurrence_result = await self.cooccurrence_engine.create_cooccurrence_edges(
                    memory_ids=batch
                )
                total_stats["total_relationships"] += cooccurrence_result.get("edges_created", 0)

                total_stats["batches_processed"] += 1

            except Exception as e:
                logger.error(f"Batch processing failed for batch {i//batch_size}: {e}")

        logger.info(
            f"Batch processing complete: {total_stats['total_relationships']} "
            f"relationships created across {total_stats['batches_processed']} batches"
        )

        return total_stats

    async def create_conversation_thread(
        self, memory_ids: list[str], thread_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a conversation thread with sequential FOLLOWS edges.

        Args:
            memory_ids: Ordered list of memory IDs in the conversation
            thread_metadata: Optional metadata for the thread

        Returns:
            Thread creation statistics
        """
        logger.info(f"Creating conversation thread with {len(memory_ids)} messages")

        result = await self.causal_engine.create_sequential_chain(
            memory_ids=memory_ids, chain_type="conversation", metadata=thread_metadata
        )

        logger.info(f"Created conversation thread with {result['edges_created']} edges")

        return result

    async def create_document_hierarchy(
        self, document_id: str, chunk_ids: list[str]
    ) -> dict[str, Any]:
        """
        Create hierarchical parent-child relationships for a document.

        Args:
            document_id: Parent document node ID
            chunk_ids: List of chunk node IDs

        Returns:
            Hierarchy creation statistics
        """
        logger.info(f"Creating document hierarchy: {document_id} with {len(chunk_ids)} chunks")

        edges_created = 0
        for child_id in chunk_ids:
            await self.hierarchical_engine.create_parent_child_relationship(
                parent_id=document_id, child_id=child_id
            )
            edges_created += 1

        result = {
            "document_id": document_id,
            "chunks": len(chunk_ids),
            "edges_created": edges_created,
        }

        logger.info(f"Created document hierarchy with {edges_created} edges")

        return result
