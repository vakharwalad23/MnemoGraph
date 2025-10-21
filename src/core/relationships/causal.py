"""Causal and sequential relationship inference engine."""

from typing import List, Dict, Any, Optional, Tuple

from src.models import RelationshipType
from src.core.vector_store import QdrantStore
from src.core.graph_store import GraphStore


class CausalSequentialEngine:
    """
    Engine for inferring causal and sequential relationships.
    
    Detects and creates:
    - Sequential ordering in conversation threads (FOLLOWS)
    - Version history and update chains (UPDATES)
    - Prerequisite knowledge dependencies (REQUIRES)
    - Causal relationships between events (CAUSES)
    """
    
    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        max_sequence_gap: int = 3600,  # Max seconds between sequential items
        similarity_threshold: float = 0.6,  # For prerequisite detection
        topic_shift_threshold: float = 0.4,  # Lower similarity = topic shift
    ):
        """
        Initialize causal/sequential engine.
        
        Args:
            vector_store: Vector store for similarity search
            graph_store: Graph store for creating relationships
            max_sequence_gap: Maximum time gap (seconds) for sequential linking
            similarity_threshold: Minimum similarity for prerequisite detection
            topic_shift_threshold: Threshold below which indicates topic shift
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.max_sequence_gap = max_sequence_gap
        self.similarity_threshold = similarity_threshold
        self.topic_shift_threshold = topic_shift_threshold
    
    async def create_sequential_chain(
        self,
        memory_ids: List[str],
        chain_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create sequential FOLLOWS relationships between ordered memories.
        
        Creates a chain: mem1 -> mem2 -> mem3 -> ...
        
        Args:
            memory_ids: Ordered list of memory IDs
            chain_type: Type of chain (conversation, steps, timeline, etc.)
            metadata: Additional metadata for all edges
            
        Returns:
            Statistics about the chain created
        """
        if len(memory_ids) < 2:
            return {
                "edges_created": 0,
                "chain_length": len(memory_ids),
                "chain_type": chain_type
            }
        
        edges_created = 0
        
        for i in range(len(memory_ids) - 1):
            current_id = memory_ids[i]
            next_id = memory_ids[i + 1]
            
            # Get timestamps to calculate gap
            current_node = await self.graph_store.get_node(current_id)
            next_node = await self.graph_store.get_node(next_id)
            
            if not (current_node and next_node):
                continue
            
            time_gap = (next_node.created_at - current_node.created_at).total_seconds()
            
            edge_metadata = {
                "chain_type": chain_type,
                "sequence_position": i,
                "time_gap_seconds": time_gap,
                **(metadata or {})
            }
            
            await self.graph_store.add_edge(
                source_id=current_id,
                target_id=next_id,
                edge_type=RelationshipType.FOLLOWS,
                weight=1.0,  # Sequential edges have uniform weight
                metadata=edge_metadata
            )
            
            edges_created += 1
        
        return {
            "edges_created": edges_created,
            "chain_length": len(memory_ids),
            "chain_type": chain_type
        }
    
    async def detect_conversation_threads(
        self,
        memory_ids: List[str],
        max_gap_seconds: Optional[int] = None
    ) -> List[List[str]]:
        """
        Detect conversation threads based on temporal proximity.
        
        Groups memories into threads where consecutive items are within
        max_gap_seconds of each other.
        
        Args:
            memory_ids: List of memory IDs to analyze
            max_gap_seconds: Maximum gap to stay in same thread
            
        Returns:
            List of threads, each thread is a list of memory IDs
        """
        if not memory_ids:
            return []
        
        max_gap = max_gap_seconds or self.max_sequence_gap
        
        # Get all nodes with timestamps
        nodes_with_time = []
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if node:
                nodes_with_time.append((mem_id, node.created_at))
        
        # Sort by timestamp
        nodes_with_time.sort(key=lambda x: x[1])
        
        # Group into threads
        threads = []
        current_thread = [nodes_with_time[0][0]]
        
        for i in range(1, len(nodes_with_time)):
            prev_time = nodes_with_time[i - 1][1]
            curr_time = nodes_with_time[i][1]
            gap = (curr_time - prev_time).total_seconds()
            
            if gap <= max_gap:
                # Continue current thread
                current_thread.append(nodes_with_time[i][0])
            else:
                # Start new thread
                threads.append(current_thread)
                current_thread = [nodes_with_time[i][0]]
        
        # Add last thread
        if current_thread:
            threads.append(current_thread)
        
        return threads
    
    async def create_conversation_threads(
        self,
        memory_ids: List[str],
        max_gap_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect and create FOLLOWS edges for conversation threads.
        
        Args:
            memory_ids: List of memory IDs to analyze
            max_gap_seconds: Maximum gap to stay in same thread
            
        Returns:
            Statistics about threads created
        """
        threads = await self.detect_conversation_threads(memory_ids, max_gap_seconds)
        
        total_edges = 0
        thread_stats = []
        
        for thread_idx, thread in enumerate(threads):
            result = await self.create_sequential_chain(
                memory_ids=thread,
                chain_type="conversation",
                metadata={"thread_id": thread_idx}
            )
            total_edges += result["edges_created"]
            thread_stats.append({
                "thread_id": thread_idx,
                "length": len(thread),
                "edges": result["edges_created"]
            })
        
        return {
            "threads_detected": len(threads),
            "total_edges_created": total_edges,
            "thread_stats": thread_stats
        }
    
    async def create_version_history(
        self,
        memory_id: str,
        previous_version_id: Optional[str] = None,
        version_number: Optional[int] = None
    ) -> str:
        """
        Create version history link (UPDATES relationship).
        
        Links a memory to its previous version, creating a version chain.
        
        Args:
            memory_id: Current version memory ID
            previous_version_id: Previous version memory ID
            version_number: Version number for this memory
            
        Returns:
            Edge ID created
        """
        if not previous_version_id:
            # This is the first version, no edge to create
            return ""
        
        # Get both nodes to verify they exist
        current = await self.graph_store.get_node(memory_id)
        previous = await self.graph_store.get_node(previous_version_id)
        
        if not (current and previous):
            return ""
        
        # Calculate time between versions
        time_delta = (current.created_at - previous.created_at).total_seconds()
        
        edge_id = await self.graph_store.add_edge(
            source_id=previous_version_id,
            target_id=memory_id,
            edge_type=RelationshipType.UPDATES,
            weight=1.0,
            metadata={
                "version_number": version_number,
                "time_delta_seconds": time_delta,
                "update_type": "version"
            }
        )
        
        return edge_id
    
    async def get_version_chain(
        self,
        memory_id: str,
        direction: str = "forward"
    ) -> List[Dict[str, Any]]:
        """
        Get the complete version chain for a memory.
        
        Args:
            memory_id: Starting memory ID
            direction: "forward" (newer versions) or "backward" (older versions)
            
        Returns:
            List of versions in chronological order
        """
        versions = []
        visited = set()
        current_id = memory_id
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = await self.graph_store.get_node(current_id)
            
            if not node:
                break
            
            versions.append({
                "id": current_id,
                "created_at": node.created_at,
                "data": node.data
            })
            
            # Get next/previous version
            if direction == "forward":
                neighbors = await self.graph_store.get_neighbors(
                    current_id,
                    edge_types=[RelationshipType.UPDATES],
                    direction="outgoing"
                )
            else:
                neighbors = await self.graph_store.get_neighbors(
                    current_id,
                    edge_types=[RelationshipType.UPDATES],
                    direction="incoming"
                )
            
            # Move to next version
            if neighbors:
                current_id = neighbors[0]["node"].id
            else:
                current_id = None
        
        # Sort chronologically
        versions.sort(key=lambda x: x["created_at"])
        
        if direction == "forward":
            # Find position of starting memory
            start_idx = next(
                (i for i, v in enumerate(versions) if v["id"] == memory_id),
                0
            )
            return versions[start_idx:]
        else:
            # Return up to and including starting memory
            start_idx = next(
                (i for i, v in enumerate(versions) if v["id"] == memory_id),
                len(versions) - 1
            )
            return versions[:start_idx + 1]
    
    async def detect_prerequisites(
        self,
        memory_id: str,
        candidate_ids: List[str],
        max_candidates: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Detect potential prerequisite knowledge for a memory.
        
        Looks for memories that:
        1. Are semantically similar (share concepts)
        2. Were created earlier
        3. Might be foundational knowledge
        
        Args:
            memory_id: Memory to find prerequisites for
            candidate_ids: Potential prerequisite memories
            max_candidates: Maximum prerequisites to return
            
        Returns:
            List of (prerequisite_id, relevance_score) tuples
        """
        node = await self.graph_store.get_node(memory_id)
        if not node:
            return []
        
        # Get embedding for current memory
        text = node.data.get("text", "")
        if not text:
            return []
        
        # Get embedding for the text (we need the actual embedding stored)
        # Note: In production, you'd get the embedding from the memory in vector store
        # For now, we need to search by getting the stored embedding first
        memory_data = await self.vector_store.get_memory(memory_id)
        if not memory_data:
            return []
        
        query_vector = memory_data.get("vector")
        if not query_vector:
            return []
        
        # Search for similar memories
        similar = await self.vector_store.search_similar(
            query_vector=query_vector,
            limit=max_candidates * 2,  # Get more to filter
            score_threshold=self.similarity_threshold
        )
        
        prerequisites = []
        
        for result in similar:
            candidate_id = result["id"]
            similarity = result["score"]
            
            # Skip self and non-candidates
            if candidate_id == memory_id or candidate_id not in candidate_ids:
                continue
            
            candidate_node = await self.graph_store.get_node(candidate_id)
            if not candidate_node:
                continue
            
            # Must be older to be a prerequisite
            if candidate_node.created_at >= node.created_at:
                continue
            
            # Calculate time-weighted relevance
            age_hours = (node.created_at - candidate_node.created_at).total_seconds() / 3600
            time_decay = 1.0 / (1.0 + age_hours / 168)  # Decay over weeks
            
            relevance = similarity * (0.7 + 0.3 * time_decay)
            prerequisites.append((candidate_id, relevance))
        
        # Sort by relevance and return top candidates
        prerequisites.sort(key=lambda x: x[1], reverse=True)
        return prerequisites[:max_candidates]
    
    async def create_prerequisite_edges(
        self,
        memory_id: str,
        candidate_ids: List[str],
        max_prerequisites: int = 3
    ) -> Dict[str, Any]:
        """
        Create REQUIRES edges for prerequisite knowledge.
        
        Args:
            memory_id: Memory that requires prerequisites
            candidate_ids: Potential prerequisite memories
            max_prerequisites: Maximum prerequisite edges to create
            
        Returns:
            Statistics about prerequisites created
        """
        prerequisites = await self.detect_prerequisites(
            memory_id,
            candidate_ids,
            max_candidates=max_prerequisites
        )
        
        edges_created = 0
        prerequisite_info = []
        
        for prereq_id, relevance in prerequisites:
            edge_id = await self.graph_store.add_edge(
                source_id=memory_id,
                target_id=prereq_id,
                edge_type=RelationshipType.REQUIRES,
                weight=relevance,
                metadata={
                    "relevance_score": relevance,
                    "relationship_type": "prerequisite"
                }
            )
            
            edges_created += 1
            prerequisite_info.append({
                "prerequisite_id": prereq_id,
                "relevance": relevance,
                "edge_id": edge_id
            })
        
        return {
            "edges_created": edges_created,
            "prerequisites": prerequisite_info,
            "memory_id": memory_id
        }
    
    async def detect_topic_shifts(
        self,
        memory_ids: List[str]
    ) -> List[int]:
        """
        Detect topic shifts in a sequence of memories.
        
        Identifies positions where the topic significantly changes,
        useful for segmenting conversations or documents.
        
        Args:
            memory_ids: Ordered list of memory IDs
            
        Returns:
            List of indices where topic shifts occur
        """
        if len(memory_ids) < 2:
            return []
        
        shift_positions = []
        
        for i in range(len(memory_ids) - 1):
            curr_node = await self.graph_store.get_node(memory_ids[i])
            next_node = await self.graph_store.get_node(memory_ids[i + 1])
            
            if not (curr_node and next_node):
                continue
            
            curr_text = curr_node.data.get("text", "")
            next_text = next_node.data.get("text", "")
            
            if not (curr_text and next_text):
                continue
            
            # Get similarity between consecutive memories
            # Get current memory's embedding
            curr_memory_data = await self.vector_store.get_memory(memory_ids[i])
            if not curr_memory_data or not curr_memory_data.get("vector"):
                continue
            
            results = await self.vector_store.search_similar(
                query_vector=curr_memory_data["vector"],
                limit=10
            )
            
            # Find next memory in results
            next_similarity = 0.0
            for result in results:
                if result["id"] == memory_ids[i + 1]:
                    next_similarity = result["score"]
                    break
            
            # If similarity drops below threshold, it's a topic shift
            if next_similarity < self.topic_shift_threshold:
                shift_positions.append(i + 1)
        
        return shift_positions
    
    async def create_causal_relationship(
        self,
        cause_id: str,
        effect_id: str,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an explicit causal relationship (CAUSES edge).
        
        Args:
            cause_id: Memory representing the cause
            effect_id: Memory representing the effect
            confidence: Confidence in the causal relationship (0-1)
            metadata: Additional metadata about the relationship
            
        Returns:
            Edge ID created
        """
        edge_metadata = {
            "confidence": confidence,
            "relationship_type": "causal",
            **(metadata or {})
        }
        
        edge_id = await self.graph_store.add_edge(
            source_id=cause_id,
            target_id=effect_id,
            edge_type=RelationshipType.CAUSES,
            weight=confidence,
            metadata=edge_metadata
        )
        
        return edge_id
    
    async def get_knowledge_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find a knowledge path from start to end memory.
        
        Useful for understanding how knowledge builds from
        prerequisites to advanced concepts.
        
        Args:
            start_id: Starting memory
            end_id: Target memory
            max_depth: Maximum path length to search
            
        Returns:
            Path as list of memory info, or None if no path found
        """
        # Use graph store's path finding
        path_ids = await self.graph_store.find_path(start_id, end_id, max_depth)
        
        if not path_ids:
            return None
        
        # Enrich with node information
        path = []
        for node_id in path_ids:
            node = await self.graph_store.get_node(node_id)
            if node:
                path.append({
                    "id": node_id,
                    "type": node.type,
                    "created_at": node.created_at,
                    "data": node.data
                })
        
        return path
    
    async def analyze_sequence(
        self,
        memory_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a sequence of memories.
        
        Args:
            memory_ids: Ordered list of memory IDs
            
        Returns:
            Analysis including threads, topic shifts, and statistics
        """
        # Detect conversation threads
        threads = await self.detect_conversation_threads(memory_ids)
        
        # Detect topic shifts
        topic_shifts = await self.detect_topic_shifts(memory_ids)
        
        # Calculate temporal statistics
        nodes_with_time = []
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if node:
                nodes_with_time.append(node.created_at)
        
        nodes_with_time.sort()
        
        if len(nodes_with_time) >= 2:
            total_duration = (nodes_with_time[-1] - nodes_with_time[0]).total_seconds()
            avg_gap = total_duration / (len(nodes_with_time) - 1) if len(nodes_with_time) > 1 else 0
        else:
            total_duration = 0
            avg_gap = 0
        
        return {
            "total_memories": len(memory_ids),
            "threads_detected": len(threads),
            "thread_lengths": [len(t) for t in threads],
            "topic_shifts": len(topic_shifts),
            "topic_shift_positions": topic_shifts,
            "temporal_stats": {
                "total_duration_seconds": total_duration,
                "average_gap_seconds": avg_gap,
                "start_time": nodes_with_time[0] if nodes_with_time else None,
                "end_time": nodes_with_time[-1] if nodes_with_time else None
            }
        }