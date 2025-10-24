"""
LLM-native relationship extraction engine.
Uses multi-stage filtering + single LLM call for all relationship types.
"""
import asyncio
import time
from typing import Any

from src.core.embeddings.base import Embedder
from src.core.llm.base import LLMProvider
from src.models.memory import Memory, NodeType
from src.models.relationships import ContextBundle, RelationshipBundle
from src.services.context_filter import MultiStageFilter
from src.services.invalidation_manager import InvalidationManager


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
        vector_store: Any,
        graph_store: Any,
        config: Any,
    ):
        """
        Initialize LLM relationship engine.

        Args:
            llm_provider: LLM for relationship extraction
            embedder: Embedder for generating embeddings
            vector_store: Vector database
            graph_store: Graph database
            config: Configuration object
        """
        self.llm = llm_provider
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config

        # Initialize sub-systems
        self.filter = MultiStageFilter(
            vector_store, graph_store, llm_provider, config
        )

        self.invalidation = InvalidationManager(llm_provider, graph_store, vector_store)

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

        # Step 1: Gather context (multi-stage filtering)
        print(f"\n🔍 Gathering context for memory: {memory.id[:8]}...")
        context = await self.filter.gather_context(memory)

        # Step 2: Build comprehensive extraction prompt
        prompt = self._build_extraction_prompt(memory, context)

        # Step 3: Parallel operations
        # - LLM extraction
        # - Vector store upsert
        # - Graph node creation
        print(f"🤖 Extracting relationships with LLM...")

        extraction_task = self.llm.complete(
            prompt, response_format=RelationshipBundle, max_tokens=2000, temperature=0.0
        )

        upsert_task = self.vector_store.upsert_memory(memory)

        node_task = self.graph_store.add_node(memory)

        results = await asyncio.gather(
            extraction_task, upsert_task, node_task, return_exceptions=True
        )

        extraction = results[0] if not isinstance(results[0], Exception) else None

        if not extraction:
            print(f"❌ LLM extraction failed")
            return RelationshipBundle(
                memory_id=memory.id,
                relationships=[],
                derived_insights=[],
                conflicts=[],
                overall_analysis="Extraction failed",
                extraction_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 4: Create edges in parallel
        print(f"🔗 Creating {len(extraction.relationships)} relationship edges...")
        edge_tasks = []

        min_confidence = getattr(
            getattr(self.config, "llm_relationships", None), "min_confidence", 0.5
        )

        for rel in extraction.relationships:
            if rel.get("confidence", 0) >= min_confidence:
                edge = self._create_edge_from_relationship(memory.id, rel)
                edge_tasks.append(self.graph_store.add_edge(edge))

        if edge_tasks:
            await asyncio.gather(*edge_tasks, return_exceptions=True)

        # Step 5: Handle derived memories
        if extraction.derived_insights:
            print(
                f"💡 Creating {len(extraction.derived_insights)} derived memories..."
            )
            await self._create_derived_memories(extraction.derived_insights, memory)

        # Step 6: Event-driven invalidation check
        # Check if this new memory supersedes any existing ones
        if context.vector_candidates:
            print(f"🔄 Checking for supersession...")
            superseded = await self.invalidation.check_supersession(
                memory, context.vector_candidates[:10]
            )
            if superseded:
                print(f"   ✓ Superseded {len(superseded)} existing memories")

        extraction_time = (time.time() - start_time) * 1000
        extraction.extraction_time_ms = extraction_time

        print(f"✅ Extraction complete in {extraction_time:.0f}ms")

        return extraction

    def _build_extraction_prompt(
        self, memory: Memory, context: ContextBundle
    ) -> str:
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

Extract ALL relevant relationships. For each relationship:

### Relationship Types:

1. **SIMILAR_TO**: Semantically similar memories
   - Strength: 0.0-1.0
   - Aspect: what makes them similar

2. **UPDATES/SUPERSEDES**: Memory updates/corrections
   - What changed
   - Should old version be preserved as historical context?

3. **CONTRADICTS**: Conflicting information
   - Nature of contradiction
   - Are both valid in different contexts?

4. **FOLLOWS/PRECEDES**: Temporal/causal sequences
   - Why ordering matters
   - Time gap significance

5. **PART_OF/BELONGS_TO**: Hierarchical containment
   - Topic/category membership
   - Abstraction levels

6. **REQUIRES/DEPENDS_ON**: Prerequisites
   - Knowledge dependencies

7. **REFERENCES/MENTIONS**: Direct references
   - Context of mention

8. **DERIVES_FROM**: Synthesized from other memories
   - Source memories
   - Synthesis reasoning

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
    ],
    "conflicts": [
        {{
            "target_id": "mem_100",
            "conflict_type": "update",
            "resolution": "preserve_both_as_history",
            "reasoning": "New version supersedes old"
        }}
    ],
    "overall_analysis": "Brief summary"
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

    def _create_edge_from_relationship(
        self, source_id: str, rel: dict[str, Any]
    ) -> dict[str, Any]:
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
            "target": rel["target_id"],
            "type": rel["type"],
            "metadata": {
                "confidence": rel.get("confidence", 0.0),
                "reasoning": rel.get("reasoning", ""),
                **rel.get("metadata", {}),
            },
        }

    async def _create_derived_memories(
        self, insights: list[dict[str, Any]], source_memory: Memory
    ):
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
            if insight.get("confidence", 0) < min_derived_confidence:
                continue

            try:
                # Generate unique ID
                derived_id = f"derived_{source_memory.id}_{hash(insight['content']) % 10000}"

                # Generate embedding
                embedding = await self.embedder.embed(insight["content"])

                # Create derived memory
                derived = Memory(
                    id=derived_id,
                    content=insight["content"],
                    type=NodeType.DERIVED,
                    embedding=embedding,
                    metadata={
                        "confidence": insight["confidence"],
                        "reasoning": insight["reasoning"],
                        "synthesis_type": insight["type"],
                        "derived_from": insight["source_ids"],
                        "triggered_by": source_memory.id,
                    },
                    confidence=insight["confidence"],
                )

                # Add to stores
                await self.graph_store.add_node(derived)
                await self.vector_store.upsert_memory(derived)

                # Create DERIVED_FROM edges
                for source_id in insight["source_ids"]:
                    await self.graph_store.add_edge(
                        {
                            "source": derived.id,
                            "target": source_id,
                            "type": "DERIVED_FROM",
                            "metadata": {"confidence": insight["confidence"]},
                        }
                    )

                print(f"   ✓ Created derived memory: {derived_id[:8]}...")

            except Exception as e:
                print(f"   ✗ Error creating derived memory: {e}")

    async def close(self):
        """Close resources."""
        await self.llm.close()
        await self.embedder.close()

