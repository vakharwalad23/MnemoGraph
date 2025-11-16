"""
Multi-stage filtering pipeline to narrow millions of memories to relevant context.

Pipeline:
1M memories → 100 (vector) → 50 (hybrid) → 20 (LLM filter) → extraction
"""

import asyncio
import time
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from src.config import Config
from src.core.llm.base import LLMProvider
from src.core.memory_store import MemoryStore
from src.models.memory import Memory
from src.models.relationships import ContextBundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RelevanceScore(BaseModel):
    """Relevance score from LLM pre-filter (structured output)."""

    model_config = {"extra": "ignore"}

    id: str = Field(..., description="Memory ID being scored")
    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance (0-1): 1.0=high, 0.7-0.9=relevant, 0.4-0.7=some, <0.4=not",
    )
    reason: str = Field(..., description="Brief why this score with specific aspects")


class EntityList(BaseModel):
    """List of entities extracted from text (structured output)."""

    model_config = {"extra": "ignore"}

    entities: list[str] = Field(
        ..., description="Key entities (people, places, orgs, concepts), max 5", max_length=5
    )


class MultiStageFilter:
    """
    Efficient multi-stage filtering to narrow context for LLM.

    Pipeline:
    1M memories → 100 (vector) → 50 (hybrid) → 20 (LLM filter) → extraction
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_provider: LLMProvider,
        config: Config,
    ):
        """
        Initialize multi-stage filter.

        Args:
            memory_store: Unified memory store facade
            llm_provider: LLM for pre-filtering
            config: Configuration object
        """
        self.memory_store = memory_store
        self.llm = llm_provider
        self.config = config

    async def gather_context(self, new_memory: Memory) -> ContextBundle:
        """
        Main entry point: Gather all relevant context for a new memory.

        Args:
            new_memory: Memory to gather context for

        Returns:
            Complete context bundle
        """
        start = time.time()
        vector_candidates = await self._stage1_vector_search(new_memory)
        stage1_time = (time.time() - start) * 1000

        start = time.time()
        context_results = await self._stage2_hybrid_filtering(new_memory)
        stage2_time = (time.time() - start) * 1000

        start = time.time()
        combined_candidates = self._combine_and_deduplicate(vector_candidates, context_results)

        context_window = getattr(
            getattr(self.config, "llm_relationships", None), "context_window", 50
        )

        if len(combined_candidates) > context_window:
            filtered = await self._stage3_llm_prefilter(
                new_memory, combined_candidates, context_window
            )
        else:
            filtered = combined_candidates

        stage3_time = (time.time() - start) * 1000

        logger.info(
            f"Context gathering: Stage1={stage1_time:.1f}ms ({len(vector_candidates)} candidates), "
            f"Stage2={stage2_time:.1f}ms ({len(combined_candidates)} total), "
            f"Stage3={stage3_time:.1f}ms ({len(filtered)} final), "
            f"Total={stage1_time + stage2_time + stage3_time:.1f}ms"
        )

        return ContextBundle(
            vector_candidates=vector_candidates[:10],
            temporal_context=context_results["temporal"],
            graph_context=context_results["graph"],
            entity_context=[],
            conversation_context=[],
            # entity_context=context_results["entity"],
            # conversation_context=context_results["conversation"],
            filtered_candidates=filtered,
        )

    async def _stage1_vector_search(self, memory: Memory) -> list[Memory]:
        """
        Stage 1: Vector similarity search.
        Uses Qdrant HNSW index for fast approximate search.

        Args:
            memory: Memory to search for

        Returns:
            List of candidate memories
        """
        try:
            results = await self.memory_store.search_similar(
                query_embedding=memory.embedding,
                limit=100,
                filters={
                    "user_id": memory.user_id,
                    "status": ["active", "historical"],
                    "created_after": (datetime.now() - timedelta(days=90)).isoformat(),
                },
                track_access=False,
                score_threshold=0.3,
            )

            return [r[0] for r in results] if results else []

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _stage2_hybrid_filtering(self, memory: Memory) -> dict[str, list[Memory]]:
        """
        Stage 2: Gather context from multiple dimensions in parallel.

        Args:
            memory: Memory to get context for

        Returns:
            Dictionary of context from different sources
        """
        # Run all filters in parallel
        results = await asyncio.gather(
            self._temporal_filter(memory),
            self._graph_filter(memory),
            # self._entity_filter(memory),
            # self._conversation_filter(memory),
            return_exceptions=True,
        )

        return {
            "temporal": results[0] if not isinstance(results[0], Exception) else [],
            "graph": results[1] if not isinstance(results[1], Exception) else [],
            "entity": [],
            "conversation": [],
            # "entity": results[2] if not isinstance(results[2], Exception) else [],
            # "conversation": results[3] if not isinstance(results[3], Exception) else [],
        }

    async def _temporal_filter(self, memory: Memory, window_days: int = 30) -> list[Memory]:
        """
        Get recent memories that are semantically similar to the new memory.

        Instead of just grabbing recent memories (pollution risk), this finds
        recent memories that are ALSO similar to the new memory.

        Args:
            memory: Memory to get context for
            window_days: Time window in days

        Returns:
            List of recent, relevant memories
        """
        try:
            cutoff = datetime.now() - timedelta(days=window_days)

            results = await self.memory_store.search_similar(
                query_embedding=memory.embedding,
                limit=20,
                filters={
                    "user_id": memory.user_id,
                    "status": ["active"],
                    "created_after": cutoff.isoformat(),
                },
                track_access=False,
                score_threshold=0.4,
            )

            return [r[0] for r in results] if results else []

        except Exception:
            return []

    async def _graph_filter(self, memory: Memory, depth: int = 2) -> list[Memory]:
        """
        Get graph neighbors for structural context.

        Args:
            memory: Memory to get neighbors for
            depth: Graph traversal depth

        Returns:
            List of neighboring memories
        """
        try:
            neighbors = await self.memory_store.get_neighbors(
                memory_id=memory.id,
                user_id=memory.user_id,
                depth=depth,
                relationship_types=[
                    "SIMILAR_TO",
                    "UPDATES",
                    "FOLLOWS",
                    "PART_OF",
                    "REFERENCES",
                ],
                limit=15,
            )

            return [n[0] for n in neighbors] if neighbors else []

        except Exception:
            return []

    async def _entity_filter(self, memory: Memory) -> list[Memory]:
        """
        Find memories with shared entities.
        Uses LLM or lightweight model for entity extraction.

        Args:
            memory: Memory to find related memories for

        Returns:
            List of memories with shared entities
        """
        try:
            entity_prompt = f"""
Extract key entities from this text. Identify important people, places, organizations, concepts, or technical terms.

Text: {memory.content}

Return a JSON object with a list of up to 5 key entities.
"""

            result = await self.llm.complete(
                entity_prompt, response_format=EntityList, max_tokens=100, temperature=0.0
            )

            if not result.entities:
                return []

            candidates = []
            for entity in result.entities[:5]:
                try:
                    related = await self.memory_store.vector_store.search_by_payload(
                        filter={"entities": {"$contains": entity}}, limit=5
                    )
                    candidates.extend([r.memory for r in related])
                except Exception:
                    continue

            return self._deduplicate(candidates)[:15]

        except Exception:
            return []

    async def _conversation_filter(self, memory: Memory) -> list[Memory]:
        """
        Get conversation thread context if applicable.

        Args:
            memory: Memory to get conversation for

        Returns:
            List of memories in same conversation
        """
        if "conversation_id" not in memory.metadata:
            return []

        try:
            conversation_id = memory.metadata["conversation_id"]

            results = await self.memory_store.graph_store.query_memories(
                user_id=memory.user_id,
                filters={"metadata.conversation_id": conversation_id},
                order_by="created_at DESC",
                limit=10,
            )

            return results

        except Exception:
            return []

    async def _stage3_llm_prefilter(
        self, new_memory: Memory, candidates: list[Memory], target_count: int
    ) -> list[Memory]:
        """
        Stage 3: Use fast/cheap LLM to rank candidates by relevance.

        This is MUCH cheaper than full relationship extraction:
        - Uses smaller/faster model
        - Shorter prompts and responses
        - Saves 80-90% of LLM costs

        Args:
            new_memory: New memory
            candidates: Candidate memories
            target_count: Target number of candidates

        Returns:
            Filtered list of most relevant candidates
        """
        target = target_count // 3  # Get top 1/3

        prompt = f"""
You are a memory relevance filter. Rate how relevant each candidate is to the new memory.

New Memory: {new_memory.content}

Rate each candidate (0.0-1.0 for relevance):

{self._format_candidates_compact(candidates)}

Consider:
- Semantic similarity (topic overlap)
- Temporal relevance (updates, sequences)
- Entity overlap (shared concepts/people/things)
- Potential for meaningful relationships

Return JSON array of top {target} most relevant:
[
    {{"id": "mem_123", "relevance": 0.85, "reason": "brief reason"}},
    ...
]
"""

        try:
            scores = await self.llm.complete(
                prompt,
                response_format=list[RelevanceScore],
                max_tokens=500,
                temperature=0.0,
            )

            sorted_scores = sorted(scores, key=lambda x: x.relevance, reverse=True)

            top_ids = {s.id for s in sorted_scores[:target]}

            return [c for c in candidates if c.id in top_ids]

        except Exception as e:
            logger.warning(f"LLM pre-filtering failed: {e}, using all candidates")
            return candidates[:target]

    def _combine_and_deduplicate(
        self, vector_candidates: list[Memory], context_results: dict[str, list[Memory]]
    ) -> list[Memory]:
        """
        Combine results from all stages and remove duplicates.

        Args:
            vector_candidates: Candidates from vector search
            context_results: Results from hybrid filtering

        Returns:
            Combined unique candidates
        """
        all_candidates = []

        # Extract Memory objects from vector_candidates (should already be Memory objects)
        for candidate in vector_candidates[:50]:
            if isinstance(candidate, tuple):
                # Handle tuple (Memory, score) if somehow passed
                all_candidates.append(candidate[0])
            elif hasattr(candidate, "id"):
                all_candidates.append(candidate)
            else:
                logger.warning(f"Unexpected candidate type: {type(candidate)}")
                continue

        # Extract Memory objects from context_results
        for _context_type, memories in context_results.items():
            for memory in memories:
                if isinstance(memory, tuple):
                    # Handle tuple (Memory, score) or (Memory, Edge) if somehow passed
                    all_candidates.append(memory[0])
                elif hasattr(memory, "id"):
                    all_candidates.append(memory)
                else:
                    logger.warning(f"Unexpected memory type in {_context_type}: {type(memory)}")
                    continue

        seen = set()
        unique_candidates = []

        for memory in all_candidates:
            if memory.id not in seen:
                seen.add(memory.id)
                unique_candidates.append(memory)

        return unique_candidates

    def _deduplicate(self, memories: list[Memory]) -> list[Memory]:
        """
        Remove duplicate memories by ID.

        Args:
            memories: List of memories

        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []

        for memory in memories:
            if memory.id not in seen:
                seen.add(memory.id)
                unique.append(memory)

        return unique

    def _format_candidates_compact(self, candidates: list[Memory]) -> str:
        """
        Format candidates compactly for LLM prompt.

        Args:
            candidates: Candidate memories

        Returns:
            Formatted string
        """
        lines = []
        for i, mem in enumerate(candidates, 1):
            content_preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            lines.append(f"{i}. [{mem.id}] {content_preview}")

        return "\n".join(lines)
