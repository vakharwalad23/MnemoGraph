"""Hierarchical relationship inference engine - organizing memories in tree structures."""

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from src.core.graph_store import GraphStore
from src.core.vector_store import QdrantStore
from src.models import NodeType, RelationshipType


class HierarchicalRelationshipEngine:
    """
    Engine for inferring hierarchical relationships between memories.

    Mimics brain's hierarchical memory organization:
    - Parent-child: Documents → Chunks (compositional hierarchy)
    - Topic clustering: Grouping related memories by semantic topic
    - Abstraction levels: Specific → General (conceptual hierarchy)
    - Category formation: Creating topic nodes for semantic clusters
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        min_cluster_size: int = 3,
        num_topics: int = 5,
        abstraction_threshold: float = 0.7,
    ):
        """
        Initialize hierarchical relationship engine.

        Args:
            vector_store: Vector store for embeddings
            graph_store: Graph store for creating hierarchical edges
            min_cluster_size: Minimum memories to form a topic cluster
            num_topics: Number of topic clusters to create
            abstraction_threshold: Similarity threshold for abstraction detection
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.min_cluster_size = min_cluster_size
        self.num_topics = num_topics
        self.abstraction_threshold = abstraction_threshold

    async def create_parent_child_relationship(
        self, parent_id: str, child_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Create a parent-child relationship (e.g., Document → Chunk).

        Args:
            parent_id: Parent memory/document ID
            child_id: Child memory/chunk ID
            metadata: Additional metadata for the relationship

        Returns:
            Edge ID
        """
        edge_metadata = metadata or {}
        edge_metadata["hierarchy_type"] = "parent_child"

        edge_id = await self.graph_store.add_edge(
            source_id=parent_id,
            target_id=child_id,
            edge_type=RelationshipType.PARENT_OF,
            weight=1.0,
            metadata=edge_metadata,
        )

        return edge_id

    async def chunk_document(self, document_id: str, chunk_ids: list[str]) -> int:
        """
        Create parent-child relationships for document chunks.

        Args:
            document_id: Parent document ID
            chunk_ids: List of chunk IDs

        Returns:
            Number of relationships created
        """
        edges_created = 0

        for i, chunk_id in enumerate(chunk_ids):
            await self.create_parent_child_relationship(
                parent_id=document_id,
                child_id=chunk_id,
                metadata={"chunk_index": i, "total_chunks": len(chunk_ids)},
            )
            edges_created += 1

        return edges_created

    async def create_topic_clusters(
        self, memory_ids: list[str], create_topic_nodes: bool = True
    ) -> dict[str, Any]:
        """
        Cluster memories by semantic topic and create topic nodes.

        Uses K-means clustering on embeddings to group similar memories.
        Creates virtual "topic" nodes and links memories to them.

        Args:
            memory_ids: List of memory IDs to cluster
            create_topic_nodes: Whether to create topic nodes in graph

        Returns:
            Dictionary with cluster assignments and topic nodes
        """
        # Gather embeddings for all memories
        embeddings = []
        valid_ids = []

        for mem_id in memory_ids:
            memory_data = await self.vector_store.get_memory(mem_id)
            if memory_data and memory_data.get("vector"):
                embeddings.append(memory_data["vector"])
                valid_ids.append(mem_id)

        if len(embeddings) < self.min_cluster_size:
            return {"clusters": [], "topic_nodes": [], "assignments": {}}

        # Determine optimal number of clusters (not more than memories/min_cluster_size)
        max_clusters = min(self.num_topics, len(valid_ids) // self.min_cluster_size)
        if max_clusters < 1:
            max_clusters = 1

        # K-means clustering
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Organize clusters
        clusters = defaultdict(list)
        for mem_id, label in zip(valid_ids, cluster_labels, strict=False):
            clusters[int(label)].append(mem_id)

        # Filter out small clusters
        valid_clusters = {
            label: members
            for label, members in clusters.items()
            if len(members) >= self.min_cluster_size
        }

        # Create topic nodes and relationships
        topic_nodes = []
        assignments = {}

        if create_topic_nodes:
            for cluster_id, member_ids in valid_clusters.items():
                # Create topic node
                topic_node_id = f"topic-{cluster_id}"

                # Calculate cluster centroid
                cluster_embeddings = [embeddings_array[valid_ids.index(mid)] for mid in member_ids]
                centroid = np.mean(cluster_embeddings, axis=0).tolist()

                # Create topic node in graph
                await self.graph_store.add_node(
                    node_id=topic_node_id,
                    node_type=NodeType.TOPIC,
                    data={
                        "cluster_id": cluster_id,
                        "member_count": len(member_ids),
                        "centroid": centroid[:10],  # Store first 10 dimensions as preview
                    },
                )

                # Link memories to topic
                for mem_id in member_ids:
                    await self.graph_store.add_edge(
                        source_id=mem_id,
                        target_id=topic_node_id,
                        edge_type=RelationshipType.CLUSTERS_WITH,
                        weight=1.0,
                        metadata={"cluster_id": cluster_id, "cluster_size": len(member_ids)},
                    )

                    assignments[mem_id] = topic_node_id

                topic_nodes.append(
                    {
                        "topic_id": topic_node_id,
                        "cluster_id": cluster_id,
                        "members": member_ids,
                        "size": len(member_ids),
                    }
                )

        return {
            "clusters": list(valid_clusters.values()),
            "topic_nodes": topic_nodes,
            "assignments": assignments,
            "num_clusters": len(valid_clusters),
        }

    async def detect_abstraction_levels(
        self, memory_ids: list[str], create_edges: bool = True
    ) -> list[tuple[str, str, float]]:
        """
        Detect abstraction relationships (specific → general).

        Analyzes text length, complexity, and semantic similarity to determine
        if one memory is a more general/abstract version of another.

        Args:
            memory_ids: List of memory IDs to analyze
            create_edges: Whether to create ABSTRACTION_OF edges

        Returns:
            List of (specific_id, general_id, confidence) tuples
        """
        abstractions = []

        # Get all memories with their data
        memories = []
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            mem_data = await self.vector_store.get_memory(mem_id)

            if node and mem_data:
                text = node.data.get("text", "")
                memories.append(
                    {
                        "id": mem_id,
                        "text": text,
                        "embedding": mem_data.get("vector", []),
                        "length": len(text),
                        "word_count": len(text.split()),
                    }
                )

        # Compare pairs to find abstraction relationships
        for i, mem1 in enumerate(memories):
            for j, mem2 in enumerate(memories):
                if i >= j:  # Skip self and duplicates
                    continue

                # Check if one is significantly shorter (potential abstraction)
                length_ratio = min(mem1["length"], mem2["length"]) / max(
                    mem1["length"], mem2["length"]
                )

                # If one is much shorter and they're semantically similar
                if length_ratio < 0.5 and mem1["embedding"] and mem2["embedding"]:
                    # Calculate semantic similarity
                    from sklearn.metrics.pairwise import cosine_similarity

                    similarity = cosine_similarity([mem1["embedding"]], [mem2["embedding"]])[0][0]

                    if similarity >= self.abstraction_threshold:
                        # Shorter one is more general/abstract
                        if mem1["length"] < mem2["length"]:
                            general_id, specific_id = mem1["id"], mem2["id"]
                        else:
                            general_id, specific_id = mem2["id"], mem1["id"]

                        abstractions.append((specific_id, general_id, float(similarity)))

                        if create_edges:
                            await self.graph_store.add_edge(
                                source_id=specific_id,
                                target_id=general_id,
                                edge_type=RelationshipType.ABSTRACTION_OF,
                                weight=similarity,
                                metadata={
                                    "similarity": similarity,
                                    "length_ratio": length_ratio,
                                    "abstraction_type": "length_based",
                                },
                            )

        return abstractions

    async def create_hierarchical_tree(
        self, root_id: str, child_ids: list[str], levels: list[list[str]] | None = None
    ) -> dict[str, Any]:
        """
        Create a multi-level hierarchical tree structure.

        Useful for organizing information in nested hierarchies like:
        Book → Chapters → Sections → Paragraphs

        Args:
            root_id: Root node ID
            child_ids: Direct children of root
            levels: Optional list of levels for deeper hierarchies

        Returns:
            Tree structure information
        """
        # Create edges from root to direct children
        edges_created = 0

        for child_id in child_ids:
            await self.graph_store.add_edge(
                source_id=root_id,
                target_id=child_id,
                edge_type=RelationshipType.PARENT_OF,
                weight=1.0,
                metadata={"level": 1},
            )
            edges_created += 1

        # Create deeper levels if provided
        if levels:
            for level_num, level_nodes in enumerate(levels, start=2):
                # Each node in this level connects to previous level
                # (simplified - you might want more sophisticated logic)
                for node_id in level_nodes:
                    # Connect to parent from previous level
                    # This is a simple example - adapt based on your needs
                    parent_id = child_ids[0] if child_ids else root_id

                    await self.graph_store.add_edge(
                        source_id=parent_id,
                        target_id=node_id,
                        edge_type=RelationshipType.PARENT_OF,
                        weight=1.0,
                        metadata={"level": level_num},
                    )
                    edges_created += 1

        return {
            "root_id": root_id,
            "direct_children": len(child_ids),
            "total_edges": edges_created,
            "max_depth": 1 + (len(levels) if levels else 0),
        }

    async def get_hierarchy_path(self, node_id: str, max_depth: int = 10) -> list[str]:
        """
        Get the hierarchical path from a node to its root.

        Traverses PARENT_OF edges upward to find the ancestry chain.

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse

        Returns:
            List of node IDs from node to root
        """
        path = [node_id]
        current_id = node_id

        for _ in range(max_depth):
            # Get incoming PARENT_OF edges (parents of current node)
            neighbors = await self.graph_store.get_neighbors(
                current_id, edge_types=[RelationshipType.PARENT_OF], direction="incoming"
            )

            if not neighbors:
                break

            # Take first parent (assuming single parent hierarchy)
            parent_id = neighbors[0]["node"].id
            path.append(parent_id)
            current_id = parent_id

        return path

    async def get_descendants(self, node_id: str, max_depth: int = 10) -> list[str]:
        """
        Get all descendants of a node in the hierarchy.

        Recursively traverses PARENT_OF edges downward.

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse

        Returns:
            List of all descendant node IDs
        """
        descendants = []
        to_visit = [(node_id, 0)]
        visited = {node_id}

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if depth >= max_depth:
                continue

            # Get children (outgoing PARENT_OF edges)
            neighbors = await self.graph_store.get_neighbors(
                current_id, edge_types=[RelationshipType.PARENT_OF], direction="outgoing"
            )

            for neighbor_info in neighbors:
                child_id = neighbor_info["node"].id
                if child_id not in visited:
                    descendants.append(child_id)
                    visited.add(child_id)
                    to_visit.append((child_id, depth + 1))

        return descendants
