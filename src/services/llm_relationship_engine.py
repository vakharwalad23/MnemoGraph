"""
LLM-native relationship extraction engine.
Uses multi-stage filtering + single LLM call for all relationship types.
"""

import asyncio
import time

from src.config import Config
from src.core.embeddings.base import Embedder
from src.core.llm.base import LLMProvider
from src.core.memory_store import MemoryStore
from src.models.memory import Memory, NodeType
from src.models.relationships import (
    ContextBundle,
    DerivedInsight,
    Edge,
    Relationship,
    RelationshipBundle,
)
from src.services.context_filter import MultiStageFilter
from src.services.invalidation_manager import InvalidationManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMRelationshipEngine:
    """
    Production-ready relationship extraction engine.

    Features:
    - Multi-stage filtering for scalability
    - Single LLM call for all relationships
    - Parallel execution
    - Intelligent invalidation
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedder: Embedder,
        memory_store: MemoryStore,
        config: Config,
    ):
        """
        Initialize LLM relationship engine.

        Args:
            llm_provider: LLM for relationship extraction
            embedder: Embedder for generating embeddings
            memory_store: MemoryStore facade for unified memory operations
            config: Configuration object
        """
        self.llm = llm_provider
        self.embedder = embedder
        self.memory_store = memory_store
        self.config = config

        self.filter = MultiStageFilter(memory_store, llm_provider, config)

        self.invalidation = InvalidationManager(llm_provider, memory_store)

    async def process_new_memory(self, memory: Memory) -> RelationshipBundle:
        """
        Main entry point: Process new memory and extract all relationships.

        Flow:
        1. Multi-stage filtering to get context
        2. Single LLM call for relationship extraction
        3. Parallel edge creation
        4. Event-driven invalidation check

        Args:
            memory: New memory to process

        Returns:
            Complete relationship extraction result
        """
        start_time = time.time()

        logger.info(f"Gathering context for memory: {memory.id[:8]}")
        context = await self.filter.gather_context(memory)

        prompt = self._build_extraction_prompt(memory, context)

        logger.debug("Extracting relationships with LLM")

        extraction_task = self.llm.complete(
            prompt, response_format=RelationshipBundle, max_tokens=2000, temperature=0.0
        )

        memory_creation_task = self.memory_store.create_memory(memory)

        results = await asyncio.gather(
            extraction_task, memory_creation_task, return_exceptions=True
        )

        extraction = results[0] if not isinstance(results[0], Exception) else None

        if not extraction:
            extraction_time = (time.time() - start_time) * 1000
            logger.error(f"LLM extraction failed (took {extraction_time:.0f}ms)")
            return RelationshipBundle(memory_id=memory.id, relationships=[], derived_insights=[])

        logger.debug(f"Creating {len(extraction.relationships)} relationship edges")
        edge_tasks = []

        min_confidence = getattr(
            getattr(self.config, "llm_relationships", None), "min_confidence", 0.5
        )

        for rel in extraction.relationships:
            if rel.confidence >= min_confidence:
                edge = self._create_edge_from_relationship(memory.id, rel)
                edge_tasks.append(self.memory_store.add_relationship(edge))

        if edge_tasks:
            await asyncio.gather(*edge_tasks, return_exceptions=True)

        if extraction.derived_insights:
            logger.debug(f"Creating {len(extraction.derived_insights)} derived memories")
            await self._create_derived_memories(extraction.derived_insights, memory)

        if context.vector_candidates:
            logger.debug("Checking for supersession")
            superseded = await self.invalidation.check_supersession(
                memory, context.vector_candidates[:10]
            )
            if superseded:
                logger.info(f"Superseded {len(superseded)} existing memories")

        extraction_time = (time.time() - start_time) * 1000
        logger.info(f"Extraction complete in {extraction_time:.0f}ms")

        return extraction

    def _build_extraction_prompt(self, memory: Memory, context: ContextBundle) -> str:
        """
        Build comprehensive relationship extraction prompt.

        Args:
            memory: New memory
            context: Gathered context

        Returns:
            LLM prompt
        """
        return f"""
You are a memory relationship extraction system. Analyze this new memory and identify ALL relevant relationships.

## New Memory
Content: {memory.content}
Created: {memory.created_at}
Type: {memory.type.value}
Metadata: {memory.metadata}

## Context

### Most Relevant Candidates (Filtered)
{self._format_memories_detailed(context.filtered_candidates[:15])}

### Recent Temporal Context
{self._format_memories_compact(context.temporal_context[:5])}

### Graph Context (Connected)
{self._format_memories_compact(context.graph_context[:5])}

## Task

Extract ALL relevant relationships. For each relationship, choose EXACTLY ONE type from the list below.

IMPORTANT: Do NOT combine types with slashes (e.g., "REFERENCES/MENTIONS"). Use only the exact type names listed.

### Valid Relationship Types:

1. **SIMILAR_TO**: Semantically similar memories
   - Use when: Memories share similar topics, concepts, or themes
   - Confidence: 0.7-1.0 for strong similarity, 0.5-0.7 for moderate

2. **UPDATES**: One memory updates information in another
   - Use when: New information modifies or corrects previous content
   - Note the specific changes made

3. **CONTRADICTS**: Conflicting information between memories
   - Use when: Memories contain incompatible facts or statements
   - Note if both could be valid in different contexts

4. **FOLLOWS**: Temporal or logical sequence (this comes after that)
   - Use when: This memory follows another in time or causality
   - Explain why ordering matters

5. **PRECEDES**: Temporal or logical sequence (this comes before that)
   - Use when: This memory comes before another in time or causality
   - Note the time gap significance

6. **PART_OF**: Hierarchical containment (this is part of that)
   - Use when: Memory belongs to a larger category or topic
   - Specify the containment relationship

7. **BELONGS_TO**: Category membership (this belongs to that group)
   - Use when: Memory is a member of a broader category
   - Identify the abstraction level

8. **REQUIRES**: Prerequisite dependency (this needs that first)
   - Use when: Understanding this memory requires knowledge from another
   - Explain the dependency

9. **DEPENDS_ON**: Depends on another memory for context
    - Use when: This memory relies on context from another
    - Clarify the dependency relationship

10. **REFERENCES**: Direct reference or citation
    - Use when: Memory explicitly references another memory
    - Note the context of reference

11. **MENTIONS**: Casual mention of related content
    - Use when: Memory briefly mentions related information
    - Describe how it's mentioned

12. **DERIVED_FROM**: Synthesized or inferred from other memories
    - Use when: This memory is a synthesis of multiple sources
    - List all source memories

## Output Format

Return JSON:
{{
    "memory_id": "{memory.id}",
    "relationships": [
        {{
            "type": "SIMILAR_TO",
            "target_id": "mem_123",
            "confidence": 0.85,
            "reasoning": "Both discuss Python async patterns",
            "metadata": {{
                "aspect": "technical_content",
                "strength": 0.85
            }}
        }}
    ],
    "derived_insights": [
        {{
            "content": "User is learning async Python web development",
            "confidence": 0.75,
            "reasoning": "Pattern across recent queries",
            "source_ids": ["mem_120", "mem_121"],
            "type": "pattern_recognition"
        }}
    ]
}}

## Guidelines

1. **Be selective**: Only include relationships with confidence > 0.5
2. **Be specific**: Explain WHY relationships exist
3. **Consider context**: Same content might mean different things
4. **Detect patterns**: Look for emerging patterns
5. **Preserve history**: Don't just replace - track evolution

Begin extraction:
"""

    def _format_memories_detailed(self, memories: list[Memory]) -> str:
        """
        Format memories with full detail for LLM.

        Args:
            memories: List of memories

        Returns:
            Formatted string
        """
        if not memories:
            return "No memories available"

        lines = []
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"""
{i}. ID: {mem.id}
   Content: {mem.content}
   Created: {mem.created_at}
   Type: {mem.type.value}
"""
            )
        return "\n".join(lines)

    def _format_memories_compact(self, memories: list[Memory]) -> str:
        """
        Format memories compactly.

        Args:
            memories: List of memories

        Returns:
            Formatted string
        """
        if not memories:
            return "No memories available"

        lines = []
        for mem in memories:
            preview = mem.content[:80] + "..." if len(mem.content) > 80 else mem.content
            lines.append(f"- [{mem.id}] {preview}")
        return "\n".join(lines)

    def _create_edge_from_relationship(self, source_id: str, rel: Relationship) -> Edge:
        """
        Create edge dict from relationship.

        Args:
            source_id: Source memory ID
            rel: Relationship data

        Returns:
            Edge dictionary
        """
        return {
            "source": source_id,
            "target": rel.target_id,
            "type": rel.type,
            "metadata": {
                "confidence": rel.confidence,
                "reasoning": rel.reasoning,
            },
        }

    async def _create_derived_memories(self, insights: list[DerivedInsight], source_memory: Memory):
        """
        Create derived memory nodes from LLM insights.

        Args:
            insights: List of derived insights
            source_memory: Source memory that triggered insights
        """
        min_derived_confidence = getattr(
            getattr(self.config, "llm_relationships", None),
            "min_derived_confidence",
            0.7,
        )

        for insight in insights:
            if insight.confidence < min_derived_confidence:
                continue

            try:
                derived_id = f"derived_{source_memory.id}_{hash(insight.content) % 10000}"

                embedding = await self.embedder.embed(insight.content)

                derived = Memory(
                    id=derived_id,
                    content=insight.content,
                    type=NodeType.DERIVED,
                    embedding=embedding,
                    metadata={
                        "confidence": insight.confidence,
                        "reasoning": insight.reasoning,
                        "synthesis_type": insight.type,
                        "derived_from": insight.source_ids,
                        "triggered_by": source_memory.id,
                    },
                    confidence=insight.confidence,
                )

                await self.memory_store.create_memory(derived)

                for source_id in insight.source_ids:
                    await self.memory_store.add_relationship(
                        Edge(
                            source=derived.id,
                            target=source_id,
                            type="DERIVED_FROM",
                            confidence=insight.confidence,
                            metadata={"confidence": insight.confidence},
                        )
                    )

                logger.debug(f"Created derived memory: {derived_id[:8]}")

            except Exception as e:
                logger.error(f"Error creating derived memory: {e}")

    async def close(self):
        """Close resources."""
        await self.llm.close()
        await self.embedder.close()
