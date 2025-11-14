"""
Invalidation Manager - Intelligent memory relevance checking.

Replaces mathematical decay with LLM-based semantic evaluation.

Three strategies:
1. On-demand (lazy): Check when accessing memory
2. Proactive (background): Periodic validation worker
3. Event-driven: Validate when related memories change
"""

import asyncio
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from src.core.llm.base import LLMProvider
from src.core.memory_store import MemoryStore
from src.models.memory import Memory, MemoryStatus
from src.models.version import InvalidationResult, InvalidationStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationCheck(BaseModel):
    """Result of whether memory needs validation."""

    memory_id: str
    should_validate: bool
    reason: str
    priority: int  # 1=high, 2=medium, 3=low


class LLMInvalidationCheck(BaseModel):
    """LLM's invalidation analysis."""

    status: str  # active, historical, superseded, invalidated
    reasoning: str
    confidence: float
    superseded_by: str | None = None


class InvalidationManager:
    """
    Manages memory invalidation across the system.

    Uses LLM to evaluate semantic relevance instead of
    mathematical decay formulas.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory_store: MemoryStore,
    ):
        """
        Initialize invalidation manager.

        Args:
            llm: LLM provider for semantic evaluation
            memory_store: MemoryStore facade for unified memory operations
        """
        self.llm = llm
        self.memory_store = memory_store

        self._worker_task: asyncio.Task | None = None

    async def validate_on_access(self, memory: Memory, current_context: str = "") -> Memory:
        """
        Check memory validity when it's accessed.

        Only validates periodically, not every access.

        Note: Access tracking is now handled automatically by MemoryStore.get_memory(),
        so this method focuses only on validation logic.

        Args:
            memory: Memory to validate
            current_context: Optional context for validation

        Returns:
            Updated memory (may have changed status)
        """
        if memory.status != MemoryStatus.ACTIVE:
            await self.memory_store.update_memory(memory, memory.user_id)
            return memory

        if not self._should_validate(memory):
            return memory

        result = await self.check_invalidation(memory, current_context)

        if result.status != InvalidationStatus.ACTIVE:
            memory = await self._mark_invalidated(memory, result)
        else:
            memory.metadata["last_validated"] = datetime.now().isoformat()
            await self.memory_store.update_memory(memory, memory.user_id)

        return memory

    def _should_validate(self, memory: Memory) -> bool:
        """
        Decide if memory needs re-validation.

        Based on:
        - Age (older = more frequent validation)
        - Access pattern (rarely accessed = more frequent)
        - Last validation time

        Args:
            memory: Memory to check

        Returns:
            True if validation needed
        """
        last_validated = memory.metadata.get("last_validated")

        if not last_validated:
            return True

        last_validated_dt = datetime.fromisoformat(last_validated)
        days_since_validation = (datetime.now() - last_validated_dt).days

        age_days = memory.age_days()

        if age_days < 30:
            return days_since_validation > 90

        if age_days > 180:
            return days_since_validation > 14

        return days_since_validation > 30

    def start_background_worker(self, interval_hours: int = 24):
        """
        Start background invalidation worker.

        Args:
            interval_hours: Hours between validation runs
        """
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._invalidation_worker(interval_hours))

    def stop_background_worker(self):
        """Stop background invalidation worker."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    async def _invalidation_worker(self, interval_hours: int):
        """
        Background worker that periodically validates memories across all users.

        This worker uses internal methods that bypass user_id filtering to
        validate memories system-wide. It's intended for maintenance operations.

        Args:
            interval_hours: Hours between runs
        """
        while True:
            try:
                logger.info("Starting periodic memory validation")

                candidates = await self._find_validation_candidates()

                logger.info(f"Found {len(candidates)} memories to validate")

                batch_size = 10
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i : i + batch_size]

                    tasks = [self._validate_candidate(mem) for mem in batch]

                    await asyncio.gather(*tasks, return_exceptions=True)

                    await asyncio.sleep(1)

                logger.info("Periodic validation complete")

            except asyncio.CancelledError:
                logger.info("Background validation worker stopped")
                break
            except Exception as e:
                logger.error(f"Error in validation worker: {e}")

            await asyncio.sleep(interval_hours * 3600)

    async def _find_validation_candidates(self) -> list[Memory]:
        """
        INTERNAL: Find memories that should be validated across all users.

        This method uses internal vector store methods (source of truth) that bypass
        user_id filtering. It's intended for background worker operations that need
        to validate memories across the entire system.

        Priority:
        1. Old, rarely accessed memories
        2. Memories with recently invalidated neighbors
        3. Random sample for maintenance

        Returns:
            List of memories to validate (from all users)
        """
        candidates = []

        try:
            # Use vector store as source of truth
            filters = {
                "status": MemoryStatus.ACTIVE.value,
                "created_before": (datetime.now() - timedelta(days=180)).isoformat(),
            }
            # Use internal method that queries across all users from vector store
            old_inactive = await self.memory_store.vector_store._search_by_payload_all_users(
                filter=filters,
                order_by="created_at ASC",
                limit=50,
            )

            # Filter by access_count in Python (if Qdrant doesn't support it)
            if old_inactive:
                old_inactive = [m for m in old_inactive if m.access_count < 5]

            candidates.extend(old_inactive)
        except Exception as e:
            logger.warning(f"Error finding validation candidates: {e}")

        return candidates[:100]

    async def _validate_candidate(self, memory: Memory):
        """
        Validate a single candidate memory.

        Args:
            memory: Memory to validate
        """
        try:
            context = await self._get_memory_context(memory)

            result = await self.check_invalidation(memory, context)

            if result.status != InvalidationStatus.ACTIVE:
                await self._mark_invalidated(memory, result)
            else:
                memory.metadata["last_validated"] = datetime.now().isoformat()
                await self.memory_store.update_memory(memory, memory.user_id)

        except Exception as e:
            logger.error(f"Error validating {memory.id}: {e}")

    async def check_supersession(
        self, new_memory: Memory, similar_memories: list[Memory]
    ) -> list[Memory]:
        """
        When a new memory is added, check if it supersedes existing ones.

        Args:
            new_memory: Newly added memory
            similar_memories: Similar existing memories

        Returns:
            List of superseded memories
        """
        superseded = []

        for candidate in similar_memories:
            similarity = self._calculate_similarity(new_memory.embedding, candidate.embedding)

            if similarity < 0.7:
                continue

            current_dt = datetime.now()

            prompt = f"""[CURRENT DATE/TIME: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}]

Analyze if a new memory supersedes an existing one.

Existing Memory:
Content: {candidate.content}
Created: {candidate.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Status: {candidate.status.value}

New Memory:
Content: {new_memory.content}
Created: {new_memory.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Important: Use the CURRENT DATE/TIME above as your reference point for temporal ordering.

Does the new memory:
1. Update/correct the existing memory? (action: "supersede")
2. Complement without replacing? (action: "none")

Respond with JSON:
- action: supersede|none
- reasoning: Brief explanation
"""

            class SupersessionAnalysis(BaseModel):
                model_config = {"extra": "ignore"}
                action: str = Field(
                    ...,
                    description="REQUIRED: Must be exactly 'supersede' if the new memory replaces the candidate, or 'none' if both should remain active",
                )
                reasoning: str = Field(
                    ...,
                    description="REQUIRED: Clear explanation of why this decision was made. Explain the relationship between the memories.",
                )

            try:
                result = await self.llm.complete(
                    prompt, response_format=SupersessionAnalysis, temperature=0.0
                )

                if result.action == "supersede":
                    candidate.status = MemoryStatus.SUPERSEDED
                    candidate.valid_until = datetime.now()
                    candidate.superseded_by = new_memory.id
                    candidate.invalidation_reason = result.reasoning

                    await self.memory_store.update_memory(candidate, candidate.user_id)

                    superseded.append(candidate)

            except Exception as e:
                logger.error(f"Error checking supersession for {candidate.id}: {e}")

        return superseded

    async def check_invalidation(self, memory: Memory, context: str = "") -> InvalidationResult:
        """
        Core invalidation check using LLM.

        Args:
            memory: Memory to check
            context: Optional context about related memories

        Returns:
            Invalidation result with status and reasoning
        """
        current_dt = datetime.now()

        prompt = f"""[CURRENT DATE/TIME: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}]

Analyze this memory for relevance and validity.

Memory:
Content: {memory.content}
Created: {memory.created_at.strftime('%Y-%m-%d %H:%M:%S')} ({memory.age_days()} days ago)
Last accessed: {memory.last_accessed.strftime('%Y-%m-%d %H:%M:%S') if memory.last_accessed else 'Never'}
Access count: {memory.access_count}
Age: {memory.age_days()} days

{f"Context: {context}" if context else ""}

Important: Use the CURRENT DATE/TIME above as your reference point for all temporal reasoning.

Determine if this memory is:
1. "active" - Still accurate and relevant
2. "historical" - Outdated but useful as historical context
3. "invalidated" - No longer needed

Respond with JSON:
- status: active|historical|invalidated
- reasoning: Explain your decision
- confidence: 0.0-1.0 confidence in this assessment
"""

        result = await self.llm.complete(
            prompt, response_format=InvalidationResult, temperature=0.0
        )

        return InvalidationResult(
            memory_id=memory.id,
            status=InvalidationStatus(result.status),
            reasoning=result.reasoning,
            confidence=result.confidence,
        )

    async def _mark_invalidated(self, memory: Memory, result: InvalidationResult) -> Memory:
        """
        Mark memory as invalidated with reasoning.

        Args:
            memory: Memory to invalidate
            result: Invalidation result

        Returns:
            Updated memory
        """
        memory.status = MemoryStatus[result.status.upper()]
        memory.invalidation_reason = result.reasoning
        memory.valid_until = datetime.now()
        memory.metadata["invalidation_confidence"] = result.confidence
        memory.metadata["invalidated_at"] = datetime.now().isoformat()
        memory.updated_at = datetime.now()

        # Update via MemoryStore (handles both stores)
        await self.memory_store.update_memory(memory, memory.user_id)

        return memory

    async def _get_memory_context(self, memory: Memory) -> str:
        """
        Get brief context about memory's neighborhood.

        Args:
            memory: Memory to get context for

        Returns:
            Context string
        """
        try:
            neighbors = await self.memory_store.get_neighbors(
                memory_id=memory.id,
                user_id=memory.user_id,
                direction="both",
                depth=1,
                limit=5,
                track_access=False,
            )

            if not neighbors:
                return "No related memories found"

            context_lines = [f"Related to {len(neighbors)} other memories:"]
            for neighbor_memory, _ in neighbors[:3]:
                content_preview = neighbor_memory.content[:50]
                context_lines.append(f"- {content_preview}...")

            return "\n".join(context_lines)

        except Exception:
            return "Unable to retrieve context"

    def _calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0.0-1.0)
        """
        import math

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)
