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

from pydantic import BaseModel

from src.core.graph_store.base import GraphStore
from src.core.llm.base import LLMProvider
from src.core.vector_store.base import VectorStore
from src.models.memory import Memory, MemoryStatus
from src.models.version import InvalidationResult


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
        graph_store: GraphStore,
        vector_store: VectorStore,
    ):
        """
        Initialize invalidation manager.

        Args:
            llm: LLM provider for semantic evaluation
            graph_store: Graph database
            vector_store: Vector database
        """
        self.llm = llm
        self.graph_store = graph_store
        self.vector_store = vector_store

        # Background worker task
        self._worker_task: asyncio.Task | None = None

    # ═══════════════════════════════════════════════════════════
    # 1. ON-DEMAND INVALIDATION (Lazy Evaluation)
    # ═══════════════════════════════════════════════════════════

    async def validate_on_access(self, memory: Memory, current_context: str = "") -> Memory:
        """
        Check memory validity when it's accessed.

        Only validates periodically, not every access.

        Args:
            memory: Memory to validate
            current_context: Optional context for validation

        Returns:
            Updated memory (may have changed status)
        """
        # Mark as accessed
        memory.mark_accessed()

        # Check if already invalidated
        if memory.status != MemoryStatus.ACTIVE:
            return memory

        # Decide if validation is needed
        if not self._should_validate(memory):
            return memory

        # Perform validation
        result = await self.check_invalidation(memory, current_context)

        if result.status != "active":
            # Mark as invalidated
            memory = await self._mark_invalidated(memory, result)

        else:
            # Update last_validated timestamp
            memory.metadata["last_validated"] = datetime.now().isoformat()
            await self.graph_store.update_node(memory)

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
            return True  # Never validated

        last_validated_dt = datetime.fromisoformat(last_validated)
        days_since_validation = (datetime.now() - last_validated_dt).days

        age_days = memory.age_days()

        # Recent memories (< 30 days): validate every 90 days
        if age_days < 30:
            return days_since_validation > 90

        # Old memories (> 180 days): validate every 14 days
        if age_days > 180:
            return days_since_validation > 14

        # Medium age: validate every 30 days
        return days_since_validation > 30

    # ═══════════════════════════════════════════════════════════
    # 2. PROACTIVE INVALIDATION (Background Worker)
    # ═══════════════════════════════════════════════════════════

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
        if self._worker_task:
            self._worker_task.cancel()

    async def _invalidation_worker(self, interval_hours: int):
        """
        Background worker that periodically validates memories.

        Args:
            interval_hours: Hours between runs
        """
        while True:
            try:
                print("🔄 Starting periodic memory validation...")

                # Find memories that need validation
                candidates = await self._find_validation_candidates()

                print(f"   Found {len(candidates)} memories to validate")

                # Validate in batches
                batch_size = 10
                for i in range(0, len(candidates), batch_size):
                    batch = candidates[i : i + batch_size]

                    # Validate batch in parallel
                    tasks = [self._validate_candidate(mem) for mem in batch]

                    await asyncio.gather(*tasks, return_exceptions=True)

                    # Small delay between batches
                    await asyncio.sleep(1)

                print("✅ Periodic validation complete")

            except asyncio.CancelledError:
                print("🛑 Background validation worker stopped")
                break
            except Exception as e:
                print(f"❌ Error in validation worker: {e}")

            # Wait for next run
            await asyncio.sleep(interval_hours * 3600)

    async def _find_validation_candidates(self) -> list[Memory]:
        """
        Find memories that should be validated.

        Priority:
        1. Old, rarely accessed memories
        2. Memories with recently invalidated neighbors
        3. Random sample for maintenance

        Returns:
            List of memories to validate
        """
        candidates = []

        # Priority 1: Old, rarely accessed
        try:
            old_inactive = await self.graph_store.query_memories(
                filters={
                    "status": MemoryStatus.ACTIVE.value,
                    "created_before": (datetime.now() - timedelta(days=180)).isoformat(),
                    "access_count_lt": 5,
                },
                limit=50,
            )
            candidates.extend(old_inactive)
        except Exception:
            pass  # Graph store may not support these queries yet

        # Priority 2: Random sample (fallback)
        try:
            random_sample = await self.graph_store.get_random_memories(
                filters={"status": MemoryStatus.ACTIVE.value}, limit=50
            )
            candidates.extend(random_sample)
        except Exception:
            pass

        # Deduplicate
        seen = set()
        unique_candidates = []
        for mem in candidates:
            if mem.id not in seen:
                seen.add(mem.id)
                unique_candidates.append(mem)

        return unique_candidates[:100]  # Limit to 100 per run

    async def _validate_candidate(self, memory: Memory):
        """
        Validate a single candidate memory.

        Args:
            memory: Memory to validate
        """
        try:
            # Get context
            context = await self._get_memory_context(memory)

            # Check invalidation
            result = await self.check_invalidation(memory, context)

            if result.status != "active":
                await self._mark_invalidated(memory, result)
            else:
                # Update last_validated
                memory.metadata["last_validated"] = datetime.now().isoformat()
                await self.graph_store.update_node(memory)

        except Exception as e:
            print(f"Error validating {memory.id}: {e}")

    # ═══════════════════════════════════════════════════════════
    # 3. EVENT-DRIVEN INVALIDATION
    # ═══════════════════════════════════════════════════════════

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
            # Only check highly similar memories
            similarity = self._calculate_similarity(new_memory.embedding, candidate.embedding)

            if similarity < 0.7:
                continue

            # Ask LLM: Does new memory supersede this one?
            prompt = f"""
Analyze if a new memory supersedes an existing one.

Existing Memory:
Content: {candidate.content}
Created: {candidate.created_at}
Status: {candidate.status.value}

New Memory:
Content: {new_memory.content}
Created: {new_memory.created_at}

Does the new memory:
1. Update/correct the existing memory? (action: "supersede")
2. Contradict but both are valid? (action: "mark_conflict")
3. Complement without replacing? (action: "none")

Respond with JSON:
- action: supersede|mark_conflict|none
- reasoning: Brief explanation
"""

            class SupersessionAnalysis(BaseModel):
                action: str
                reasoning: str

            try:
                result = await self.llm.complete(
                    prompt, response_format=SupersessionAnalysis, temperature=0.0
                )

                if result.action == "supersede":
                    # Mark as superseded
                    candidate.status = MemoryStatus.SUPERSEDED
                    candidate.valid_until = datetime.now()
                    candidate.superseded_by = new_memory.id
                    candidate.invalidation_reason = result.reasoning

                    await self.graph_store.update_node(candidate)

                    superseded.append(candidate)

            except Exception as e:
                print(f"Error checking supersession for {candidate.id}: {e}")

        return superseded

    # ═══════════════════════════════════════════════════════════
    # SHARED UTILITIES
    # ═══════════════════════════════════════════════════════════

    async def check_invalidation(self, memory: Memory, context: str = "") -> InvalidationResult:
        """
        Core invalidation check using LLM.

        Args:
            memory: Memory to check
            context: Optional context about related memories

        Returns:
            Invalidation result with status and reasoning
        """
        prompt = f"""
Analyze this memory for relevance and validity.

Memory:
Content: {memory.content}
Created: {memory.created_at}
Last accessed: {memory.last_accessed}
Access count: {memory.access_count}
Age: {memory.age_days()} days

{f"Context: {context}" if context else ""}

Determine if this memory is:
1. "active" - Still accurate and relevant
2. "historical" - Outdated but useful as historical context
3. "superseded" - Replaced by newer information
4. "invalidated" - No longer needed

Respond with JSON:
- status: active|historical|superseded|invalidated
- reasoning: Explain your decision
- confidence: 0.0-1.0 confidence in this assessment
"""

        result = await self.llm.complete(
            prompt, response_format=InvalidationResult, temperature=0.0
        )

        return InvalidationResult(
            memory_id=memory.id,
            status=result.status,
            reasoning=result.reasoning,
            confidence=result.confidence,
            superseded_by=result.superseded_by,
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

        if result.superseded_by:
            memory.superseded_by = result.superseded_by

        await self.graph_store.update_node(memory)

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
            neighbors = await self.graph_store.get_neighbors(memory.id, limit=5)

            if not neighbors:
                return "No related memories found"

            context_lines = [f"Related to {len(neighbors)} other memories:"]
            for n in neighbors[:3]:
                content_preview = n["node"].content[:50]
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

        # Dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)
