"""Semantic similarity relationship inference engine."""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.graph_store import GraphStore
from src.core.vector_store import QdrantStore
from src.models import RelationshipType


class SemanticSimilarityEngine:
    """Engine for inferring semantic similarity relationships between memories."""

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        similarity_threshold: float = 0.75,
        max_similar_memories: int = 10,
    ):
        """
        Initialize semantic similarity engine.

        Args:
            vector_store: Vector store for similarity search
            graph_store: Graph store for creating edges
            similarity_threshold: Minimum similarity score to create edge (0-1)
            max_similar_memories: Maximum number of similar memories to link
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.similarity_threshold = similarity_threshold
        self.max_similar_memories = max_similar_memories

    async def infer_relationships(
        self, memory_id: str, embedding: list[float], create_edges: bool = True
    ) -> list[dict[str, Any]]:
        """
        Infer semantic similarity relationships for a memory.

        Args:
            memory_id: ID of the memory to find relationships for
            embedding: Embedding vector of the memory
            create_edges: Whether to create edges in graph store

        Returns:
            List of similar memories with scores
        """
        # Search for similar memories in vector store
        similar_memories = await self.vector_store.search_similar(
            query_vector=embedding,
            limit=self.max_similar_memories + 1,  # +1 to exclude self
            score_threshold=self.similarity_threshold,
        )

        # Filter out the memory itself and those below threshold
        similar_memories = [
            mem
            for mem in similar_memories
            if mem["id"] != memory_id and mem["score"] >= self.similarity_threshold
        ]

        # Create edges in graph store if requested
        if create_edges:
            for similar_mem in similar_memories:
                await self.graph_store.add_edge(
                    source_id=memory_id,
                    target_id=similar_mem["id"],
                    edge_type=RelationshipType.SIMILAR_TO,
                    weight=similar_mem["score"],
                    metadata={
                        "similarity_score": similar_mem["score"],
                        "inference_method": "cosine_similarity",
                    },
                )

        return similar_memories

    async def batch_infer_relationships(
        self, memories: list[dict[str, Any]], create_edges: bool = True
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Infer relationships for multiple memories in batch.

        Args:
            memories: List of memory dictionaries with 'id' and 'embedding'
            create_edges: Whether to create edges in graph store

        Returns:
            Dictionary mapping memory IDs to their similar memories
        """
        results = {}

        for memory in memories:
            memory_id = memory["id"]
            embedding = memory["embedding"]

            similar = await self.infer_relationships(
                memory_id=memory_id, embedding=embedding, create_edges=create_edges
            )

            results[memory_id] = similar

        return results

    @staticmethod
    def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(vec1, vec2)[0][0]

        return float(similarity)

    @staticmethod
    def compute_batch_similarity(
        query_embedding: list[float], embeddings: list[list[float]]
    ) -> list[float]:
        """
        Compute cosine similarity between a query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors to compare against

        Returns:
            List of similarity scores
        """
        query_vec = np.array(query_embedding).reshape(1, -1)
        embedding_matrix = np.array(embeddings)

        similarities = cosine_similarity(query_vec, embedding_matrix)[0]

        return similarities.tolist()

    async def update_similarity_edges(self, memory_id: str, embedding: list[float]) -> int:
        """
        Update similarity edges for a memory (useful after embedding updates).

        Args:
            memory_id: ID of the memory
            embedding: Updated embedding vector

        Returns:
            Number of edges updated/created
        """
        # Get existing neighbors
        existing_neighbors = await self.graph_store.get_neighbors(
            node_id=memory_id, edge_types=[RelationshipType.SIMILAR_TO], direction="both"
        )

        # Remove old similarity edges
        for neighbor in existing_neighbors:
            edge_id = neighbor.get("edge_id")
            if edge_id:
                # We don't have delete_edge in base interface, so we update weight to 0
                # or we could extend the interface
                await self.graph_store.update_edge_weight(edge_id, 0.0)

        # Create new edges
        similar_memories = await self.infer_relationships(
            memory_id=memory_id, embedding=embedding, create_edges=True
        )

        return len(similar_memories)

    async def find_clusters(
        self, memory_ids: list[str], min_cluster_size: int = 2
    ) -> list[list[str]]:
        """
        Find clusters of semantically similar memories.

        Args:
            memory_ids: List of memory IDs to cluster
            min_cluster_size: Minimum size for a cluster

        Returns:
            List of clusters (each cluster is a list of memory IDs)
        """
        # Get embeddings for all memories
        embeddings = []
        valid_ids = []

        for mem_id in memory_ids:
            memory_data = await self.vector_store.get_memory(mem_id)
            if memory_data and memory_data.get("vector"):
                embeddings.append(memory_data["vector"])
                valid_ids.append(mem_id)

        if len(embeddings) < min_cluster_size:
            return []

        # Compute pairwise similarity matrix
        similarity_matrix = cosine_similarity(np.array(embeddings))

        # Simple clustering: group memories above threshold
        clusters = []
        visited = set()

        for i, mem_id in enumerate(valid_ids):
            if mem_id in visited:
                continue

            # Find all similar memories
            cluster = [mem_id]
            visited.add(mem_id)

            for j, other_id in enumerate(valid_ids):
                if i != j and other_id not in visited:
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        cluster.append(other_id)
                        visited.add(other_id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters
