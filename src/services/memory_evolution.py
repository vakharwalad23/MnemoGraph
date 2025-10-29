"""
Memory Evolution Service - Tracks memory versions and lifecycle.

Handles:
- Version creation and tracking
- Memory evolution (updates, replacements)
- Version history queries
- Time-travel queries (point-in-time snapshots)
"""

from datetime import datetime

from pydantic import BaseModel, Field

from src.core.embeddings.base import Embedder
from src.core.graph_store.base import GraphStore
from src.core.llm.base import LLMProvider
from src.models.memory import Memory, MemoryStatus
from src.models.version import MemoryEvolution, VersionChain, VersionChange


class EvolutionAnalysis(BaseModel):
    """LLM analysis of how to evolve a memory."""

    model_config = {"extra": "ignore"}  # Ignore extra fields from LLM

    action: str = Field(
        ...,
        description="REQUIRED: The action to take. Must be exactly one of: 'update', 'augment', 'replace', or 'preserve'",
    )
    reasoning: str = Field(
        ...,
        description="REQUIRED: Detailed explanation of why this action is appropriate. Include specific details about what changed and why.",
    )
    change_description: str = Field(
        ...,
        description="REQUIRED: Brief description of what changed (1-2 sentences). Example: 'Updated location from Paris to London' or 'Added new skill: Python programming'",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="REQUIRED: Confidence score between 0.0 and 1.0. Use 1.0 for certain changes, 0.5-0.8 for probable changes, <0.5 for uncertain changes.",
    )


class MemoryEvolutionService:
    """
    Manages memory lifecycle and evolution.

    Features:
    - Create new versions while preserving history
    - Track version chains
    - LLM-based evolution analysis
    - Time-travel queries
    """

    def __init__(
        self,
        llm: LLMProvider,
        graph_store: GraphStore,  # Will be properly typed in integration
        embedder: Embedder,  # Will be properly typed in integration
    ):
        """
        Initialize memory evolution service.

        Args:
            llm: LLM provider for evolution analysis
            graph_store: Graph database for storing memory nodes
            embedder: Embedder for generating embeddings
        """
        self.llm = llm
        self.graph_store = graph_store
        self.embedder = embedder

    async def evolve_memory(self, current: Memory, new_info: str) -> MemoryEvolution:
        """
        Evolve a memory with new information.

        Determines whether to:
        - Update/correct (create new version, mark old as superseded)
        - Augment (add details without replacing)
        - Replace entirely
        - Preserve both (if conflicting)

        Args:
            current: Current memory
            new_info: New information to incorporate

        Returns:
            MemoryEvolution tracking the change
        """
        # Ask LLM how to handle the evolution
        analysis = await self._analyze_evolution(current, new_info)

        if analysis.action in ["update", "replace"]:
            # Create new version
            new_version = await self._create_new_version(current, new_info, analysis)

            # Mark current as superseded
            current.status = MemoryStatus.SUPERSEDED
            current.valid_until = datetime.now()
            current.superseded_by = new_version.id
            current.invalidation_reason = f"Superseded: {analysis.change_description}"
            current.updated_at = datetime.now()

            # Save both versions
            await self.graph_store.update_node(current)
            await self.graph_store.add_node(new_version)

            return MemoryEvolution(
                current_version=current.id,
                new_version=new_version.id,
                action=analysis.action,
                change=VersionChange(
                    change_type=analysis.action,
                    reasoning=analysis.reasoning,
                    description=analysis.change_description,
                    changed_fields=["content"],  # Track what changed
                ),
            )

        elif analysis.action == "augment":
            # Add to existing memory without versioning
            current.content = f"{current.content}\n\nUpdate: {new_info}"
            current.metadata["augmented"] = current.metadata.get("augmented", 0) + 1
            current.metadata["last_augment"] = datetime.now().isoformat()
            current.updated_at = datetime.now()

            await self.graph_store.update_node(current)

            return MemoryEvolution(
                current_version=current.id,
                new_version=None,
                action="augment",
                change=VersionChange(
                    change_type="augment",
                    reasoning=analysis.reasoning,
                    description=analysis.change_description,
                    changed_fields=["content"],
                ),
            )

        else:  # preserve
            # Keep both as separate memories (handled outside this method)
            return MemoryEvolution(
                current_version=current.id,
                new_version=None,
                action="preserve",
                change=VersionChange(
                    change_type="preserve",
                    reasoning=analysis.reasoning,
                    description="Both memories preserved as separate entities",
                    changed_fields=[],
                ),
            )

    async def _analyze_evolution(self, current: Memory, new_info: str) -> EvolutionAnalysis:
        """
        Use LLM to analyze how to evolve the memory.

        Args:
            current: Current memory
            new_info: New information

        Returns:
            Evolution analysis from LLM
        """
        prompt = f"""
Analyze how to incorporate new information into an existing memory.

Current Memory:
Content: {current.content}
Created: {current.created_at}
Version: {current.version}

New Information:
{new_info}

Determine the best action:
1. "update" - New info corrects/updates the current memory (create new version)
2. "augment" - New info adds details without contradicting (append to existing)
3. "replace" - New info completely replaces the memory (create new version)
4. "preserve" - New info contradicts; both should be kept as separate memories

You MUST respond with valid JSON containing ALL required fields:
{{
  "action": "update|augment|replace|preserve",
  "reasoning": "Detailed explanation of why this action is appropriate",
  "change_description": "Brief description of what changed (1-2 sentences)",
  "confidence": 0.8
}}

All fields are REQUIRED. The confidence must be a number between 0.0 and 1.0.
"""

        result = await self.llm.complete(prompt, response_format=EvolutionAnalysis, temperature=0.0)

        return result

    async def _create_new_version(
        self, current: Memory, new_content: str, analysis: EvolutionAnalysis
    ) -> Memory:
        """
        Create a new version of a memory.

        Args:
            current: Current memory
            new_content: New content
            analysis: Evolution analysis

        Returns:
            New memory version
        """
        # Generate new embedding
        embedding = await self.embedder.embed(new_content)

        # Create new version
        new_version = Memory(
            id=f"{current.id}_v{current.version + 1}",
            content=new_content,
            type=current.type,
            embedding=embedding,
            version=current.version + 1,
            parent_version=current.id,
            valid_from=datetime.now(),
            valid_until=None,
            status=MemoryStatus.ACTIVE,
            metadata={
                **current.metadata,
                "evolution_reason": analysis.reasoning,
                "change_description": analysis.change_description,
                "evolution_confidence": analysis.confidence,
            },
            confidence=current.confidence * analysis.confidence,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return new_version

    async def get_version_history(self, memory_id: str) -> VersionChain:
        """
        Get complete version history for a memory.

        Walks back through parent_version links to build full chain.

        Args:
            memory_id: Memory ID to get history for

        Returns:
            Complete version chain
        """
        versions = []
        current_id = memory_id
        original_id = memory_id

        # Walk backwards through versions
        while current_id:
            memory = await self.graph_store.get_node(current_id)
            versions.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "version": memory.version,
                    "status": memory.status.value,
                    "valid_from": memory.valid_from.isoformat(),
                    "valid_until": (memory.valid_until.isoformat() if memory.valid_until else None),
                    "created_at": memory.created_at.isoformat(),
                }
            )

            # Move to parent
            if memory.parent_version:
                current_id = memory.parent_version
                original_id = current_id  # Track original
            else:
                current_id = None

        # Reverse to get chronological order
        versions.reverse()

        return VersionChain(
            original_id=original_id,
            versions=versions,
            current_version_id=memory_id,
            total_versions=len(versions),
            created_at=datetime.fromisoformat(versions[0]["created_at"]),
        )

    async def time_travel_query(
        self, query_embedding: list[float], as_of: datetime, limit: int = 10
    ) -> list[Memory]:
        """
        Query memories as they existed at a specific point in time.

        Returns only memories that were valid at the specified time.

        Args:
            query_embedding: Query vector
            as_of: Point in time to query
            limit: Maximum results

        Returns:
            List of memories valid at that time
        """
        # Get candidates from graph query
        candidates = await self.graph_store.query_memories(limit=limit * 2)

        # Filter to memories valid at specified time
        valid_memories = []
        for memory in candidates:
            # Check temporal validity
            if memory.valid_from <= as_of and (
                memory.valid_until is None or memory.valid_until > as_of
            ):
                valid_memories.append(memory)

                if len(valid_memories) >= limit:
                    break

        return valid_memories

    async def get_current_version(self, memory_id: str) -> Memory:
        """
        Get the current (latest) version of a memory.

        If the memory has been superseded, follows the chain to find
        the current version.

        Args:
            memory_id: Any version of the memory

        Returns:
            Current active version
        """
        memory = await self.graph_store.get_node(memory_id)

        # Follow supersession chain
        while memory.superseded_by:
            memory = await self.graph_store.get_node(memory.superseded_by)

        return memory

    async def rollback_to_version(self, target_version_id: str) -> Memory:
        """
        Rollback to a previous version.

        Creates a new version with the content of the target version.

        Args:
            target_version_id: Version to rollback to

        Returns:
            New memory with rolled back content
        """
        # Get target version
        target = await self.graph_store.get_node(target_version_id)

        # Get current version
        current = await self.get_current_version(target_version_id)

        # Create new version with target's content
        new_version = Memory(
            id=f"{current.id}_rollback_{datetime.now().timestamp()}",
            content=target.content,
            type=target.type,
            embedding=target.embedding,  # Reuse embedding
            version=current.version + 1,
            parent_version=current.id,
            valid_from=datetime.now(),
            status=MemoryStatus.ACTIVE,
            metadata={
                **target.metadata,
                "rollback_from": current.id,
                "rollback_to": target_version_id,
                "rollback_reason": "Manual rollback",
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Mark current as superseded
        current.status = MemoryStatus.SUPERSEDED
        current.valid_until = datetime.now()
        current.superseded_by = new_version.id
        current.invalidation_reason = f"Rolled back to version {target.version}"

        # Save both
        await self.graph_store.update_node(current)
        await self.graph_store.add_node(new_version)

        return new_version
