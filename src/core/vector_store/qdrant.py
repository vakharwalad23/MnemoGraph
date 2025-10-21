"""Qdrant vector store implementation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantStore:
    """Async Qdrant vector store for memory embeddings."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "memories",
        vector_size: int = 768,
    ):
        """
        Initialize Qdrant store.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection to use
            vector_size: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client: Optional[AsyncQdrantClient] = None
    
    def _to_uuid(self, id_str: str) -> str:
        """
        Convert string ID to UUID format.
        
        Args:
            id_str: String identifier (can be UUID or regular string)
            
        Returns:
            UUID string
        """
        try:
            # If it's already a valid UUID, return it
            UUID(id_str)
            return id_str
        except ValueError:
            # If not, create a UUID from the string
            # Use UUID5 with a namespace for consistent conversion
            from uuid import uuid5, NAMESPACE_DNS
            return str(uuid5(NAMESPACE_DNS, id_str))
    
    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self.client is None:
            self.client = AsyncQdrantClient(host=self.host, port=self.port)
    
    async def initialize(self) -> None:
        """Initialize the collection if it doesn't exist."""
        await self.connect()
        
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
    
    async def upsert_memory(
        self,
        memory_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store or update a memory vector.
        
        Args:
            memory_id: Unique memory identifier
            embedding: Embedding vector
            metadata: Additional metadata to store
        """
        await self.connect()
        
        payload = metadata or {}
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["original_id"] = memory_id  # Store original ID in payload
        
        # Convert to UUID for Qdrant
        uuid_id = self._to_uuid(memory_id)
        
        point = PointStruct(
            id=uuid_id,
            vector=embedding,
            payload=payload
        )
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar memories with scores
        """
        await self.connect()
        
        query_filter = None
        if filter_dict:
            # Simple filter implementation
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True
        )
        
        return [
        {
            "id": point.payload.get("original_id", str(point.id)),  # Return original ID
            "score": point.score,
            "metadata": point.payload
        }
        for point in response.points
    ]
    
    async def update_access_metadata(
        self,
        memory_id: str,
        access_count: int,
        last_accessed: datetime
    ) -> None:
        """
        Update access tracking metadata.
        
        Args:
            memory_id: Memory identifier
            access_count: Updated access count
            last_accessed: Last access timestamp
        """
        await self.connect()
        
        uuid_id = self._to_uuid(memory_id)
        
        await self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "access_count": access_count,
                "last_accessed": last_accessed.isoformat()
            },
            points=[uuid_id]
        )
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory data or None if not found
        """
        await self.connect()
        
        uuid_id = self._to_uuid(memory_id)
        
        results = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[uuid_id],
            with_vectors=True
        )
        
        if not results:
            return None
        
        point = results[0]
        return {
            "id": point.payload.get("original_id", memory_id),  # Return original ID
            "vector": point.vector,
            "metadata": point.payload
        }
    
    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from the store.
        
        Args:
            memory_id: Memory identifier
        """
        await self.connect()
        
        uuid_id = self._to_uuid(memory_id)
        
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=[uuid_id]
        )
    
    async def count_memories(self) -> int:
        """
        Get the total number of memories in the collection.
        
        Returns:
            Number of memories
        """
        await self.connect()
        
        collection_info = await self.client.get_collection(self.collection_name)
        return collection_info.points_count
    
    async def close(self) -> None:
        """Close the connection to Qdrant."""
        if self.client is not None:
            await self.client.close()
            self.client = None