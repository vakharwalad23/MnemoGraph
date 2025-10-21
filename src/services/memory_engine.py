"""
Memory Engine - Main service for MnemoGraph.

This is the primary API that users interact with. It provides high-level
operations for managing memories with automatic relationship inference.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from src.config import Config
from src.core.embeddings import EmbeddingProvider
from src.core.graph_store import GraphStore
from src.core.vector_store import QdrantStore
from src.models import NodeType
from src.services.relationship_orchestrator import RelationshipOrchestrator

logger = logging.getLogger(__name__)


class MemoryEngine:
    """
    Main memory management engine.

    Provides high-level API for:
    - Adding memories (with automatic relationship inference)
    - Querying memories (semantic search)
    - Updating memories
    - Deleting memories
    - Managing conversations and documents
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        embedder: EmbeddingProvider,
        config: Config,
    ):
        """
        Initialize memory engine.

        Args:
            vector_store: Vector database for embeddings
            graph_store: Graph database for relationships
            embedder: Embedding provider
            config: System configuration
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.config = config

        # Initialize relationship orchestrator
        self.orchestrator = RelationshipOrchestrator(
            vector_store=vector_store, graph_store=graph_store, embedder=embedder, config=config
        )

        logger.info("Memory engine initialized")

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.vector_store.initialize()
        await self.graph_store.initialize()
        logger.info("Memory engine components initialized")

    async def close(self) -> None:
        """Close all connections."""
        await self.vector_store.close()
        await self.graph_store.close()
        logger.info("Memory engine closed")

    async def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        auto_infer_relationships: bool | None = None,
        context_window_size: int = 50,
    ) -> dict[str, Any]:
        """
        Add a new memory with automatic relationship inference.

        Args:
            text: Memory text content
            metadata: Optional metadata
            memory_id: Optional custom ID (generated if not provided)
            auto_infer_relationships: Whether to automatically infer relationships
                (defaults to config.relationships.auto_infer_on_add)
            context_window_size: Number of recent memories to consider for relationships

        Returns:
            Dictionary with memory ID and relationship statistics
        """
        # Generate ID if not provided
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        # Determine if we should auto-infer
        if auto_infer_relationships is None:
            auto_infer_relationships = self.config.relationships.auto_infer_on_add

        # Generate embedding
        logger.info(f"Adding memory: {memory_id}")
        embedding = await self.embedder.embed(text)

        # Prepare metadata
        created_at = datetime.now(UTC)
        mem_metadata = metadata or {}
        mem_metadata["text"] = text
        mem_metadata["created_at"] = created_at.isoformat()

        # Add to vector store
        await self.vector_store.upsert_memory(
            memory_id=memory_id, embedding=embedding, metadata=mem_metadata
        )

        # Add to graph store
        await self.graph_store.add_node(
            node_id=memory_id, node_type=NodeType.MEMORY, data=mem_metadata
        )

        result = {
            "memory_id": memory_id,
            "created_at": created_at.isoformat(),
            "text": text,
            "relationships_created": 0,
            "auto_infer_enabled": auto_infer_relationships,
        }

        # Automatically infer relationships if enabled
        if auto_infer_relationships:
            # Get recent similar memories for context using semantic search
            similar_memories = await self.vector_store.search_similar(
                query_vector=embedding,
                limit=context_window_size,
                score_threshold=None,  # Get all to provide context
            )

            # Extract IDs, excluding the current memory
            context_ids = [mem["id"] for mem in similar_memories if mem["id"] != memory_id][
                :context_window_size
            ]

            # Run orchestrator
            stats = await self.orchestrator.process_new_memory(
                memory_id=memory_id,
                text=text,
                embedding=embedding,
                created_at=created_at,
                context_memory_ids=context_ids,
            )

            result["relationships_created"] = stats["relationships_created"]
            result["engines_run"] = stats["engines_run"]

            logger.info(
                f"Added memory {memory_id} with " f"{stats['relationships_created']} relationships"
            )
        else:
            logger.info(f"Added memory {memory_id} (no automatic relationships)")

        return result

    async def query_memories(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        include_relationships: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query memories using semantic search.

        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            include_relationships: Whether to include relationship info

        Returns:
            List of matching memories with scores
        """
        logger.info(f"Querying memories: '{query}' (limit={limit})")

        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Search vector store
        results = await self.vector_store.search_similar(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=similarity_threshold,
            filter_dict=filters,
        )

        # Optionally enrich with relationship data
        if include_relationships:
            for result in results:
                mem_id = result["id"]
                neighbors = await self.graph_store.get_neighbors(mem_id)
                result["relationships"] = {
                    "count": len(neighbors),
                    "types": list({n.get("edge_type") for n in neighbors}),
                }

        logger.info(f"Found {len(results)} matching memories")
        return results

    async def get_memory(
        self, memory_id: str, include_relationships: bool = False
    ) -> dict[str, Any] | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory identifier
            include_relationships: Whether to include relationship info

        Returns:
            Memory data or None if not found
        """
        # Get from vector store
        mem_data = await self.vector_store.get_memory(memory_id)

        if not mem_data:
            return None

        # Get from graph store for full properties
        node = await self.graph_store.get_node(memory_id)

        if node:
            mem_data["properties"] = node.data

        # Optionally include relationships
        if include_relationships:
            neighbors = await self.graph_store.get_neighbors(memory_id)
            mem_data["relationships"] = {
                "count": len(neighbors),
                "neighbors": [
                    {"id": n["node"].id, "type": n.get("edge_type"), "weight": n.get("edge_weight")}
                    for n in neighbors
                ],
            }

        return mem_data

    async def update_memory(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        reindex_relationships: bool = True,
    ) -> dict[str, Any]:
        """
        Update an existing memory.

        Args:
            memory_id: Memory identifier
            text: New text content (if changing)
            metadata: New/updated metadata
            reindex_relationships: Whether to re-infer relationships

        Returns:
            Update statistics
        """
        logger.info(f"Updating memory: {memory_id}")

        # Get existing memory
        existing = await self.vector_store.get_memory(memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # Update text and embedding if provided
        if text is not None:
            embedding = await self.embedder.embed(text)
            updated_metadata = existing.get("metadata", {})
            updated_metadata["text"] = text
            updated_metadata["updated_at"] = datetime.now(UTC).isoformat()

            if metadata:
                updated_metadata.update(metadata)

            # Update vector store
            await self.vector_store.upsert_memory(
                memory_id=memory_id, embedding=embedding, metadata=updated_metadata
            )

            # Update graph store
            await self.graph_store.add_node(
                node_id=memory_id, node_type=NodeType.MEMORY, data=updated_metadata
            )
        elif metadata:
            # Update only metadata
            existing_metadata = existing.get("metadata", {})
            existing_metadata.update(metadata)
            existing_metadata["updated_at"] = datetime.now(UTC).isoformat()

            await self.graph_store.add_node(
                node_id=memory_id, node_type=NodeType.MEMORY, data=existing_metadata
            )

        result = {
            "memory_id": memory_id,
            "updated": True,
            "text_changed": text is not None,
            "relationships_reindexed": False,
        }

        # Optionally reindex relationships
        if reindex_relationships and text is not None:
            # Delete old edges (except structural ones like PARENT_OF)
            # Then re-run orchestrator
            # TODO: Implement selective edge deletion
            logger.info(f"Relationship reindexing for {memory_id} (not yet implemented)")

        logger.info(f"Updated memory {memory_id}")
        return result

    async def delete_memory(
        self, memory_id: str, delete_relationships: bool = True
    ) -> dict[str, Any]:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier
            delete_relationships: Whether to delete associated relationships

        Returns:
            Deletion statistics
        """
        logger.info(f"Deleting memory: {memory_id}")

        # Get relationship count before deletion
        neighbors = await self.graph_store.get_neighbors(memory_id)
        relationship_count = len(neighbors)

        # Delete from vector store
        await self.vector_store.delete_memory(memory_id)

        # Delete from graph store
        if delete_relationships:
            await self.graph_store.delete_node(memory_id)
        else:
            # Just remove the node, keep orphaned edges
            # (graph store should handle this automatically)
            await self.graph_store.delete_node(memory_id)

        logger.info(f"Deleted memory {memory_id} " f"(removed {relationship_count} relationships)")

        return {
            "memory_id": memory_id,
            "deleted": True,
            "relationships_removed": relationship_count,
        }

    async def add_conversation(
        self,
        messages: list[dict[str, str]],
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a conversation as a sequence of messages.

        Args:
            messages: List of message dicts with 'text' and optional 'role'
            conversation_id: Optional conversation identifier
            metadata: Optional conversation-level metadata

        Returns:
            Conversation creation statistics
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        logger.info(f"Adding conversation {conversation_id} with {len(messages)} messages")

        message_ids = []

        # Add each message as a memory
        for i, msg in enumerate(messages):
            msg_id = f"{conversation_id}-msg-{i}"
            msg_metadata = {
                "conversation_id": conversation_id,
                "message_index": i,
                "role": msg.get("role", "user"),
            }
            if metadata:
                msg_metadata.update(metadata)

            await self.add_memory(
                text=msg["text"],
                metadata=msg_metadata,
                memory_id=msg_id,
                auto_infer_relationships=False,  # We'll create thread manually
            )
            message_ids.append(msg_id)

        # Create conversation thread
        thread_result = await self.orchestrator.create_conversation_thread(
            memory_ids=message_ids, thread_metadata={"conversation_id": conversation_id}
        )

        logger.info(
            f"Created conversation {conversation_id} with "
            f"{len(message_ids)} messages and {thread_result['edges_created']} edges"
        )

        return {
            "conversation_id": conversation_id,
            "message_count": len(message_ids),
            "message_ids": message_ids,
            "edges_created": thread_result["edges_created"],
        }

    async def add_document(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a document by chunking it into memories.

        Args:
            text: Full document text
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            document_id: Optional document identifier
            metadata: Optional document-level metadata

        Returns:
            Document creation statistics
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        logger.info(f"Adding document {document_id}")

        # Create document node
        doc_metadata = metadata or {}
        doc_metadata["type"] = "document"
        doc_metadata["created_at"] = datetime.now(UTC).isoformat()

        await self.graph_store.add_node(
            node_id=document_id, node_type=NodeType.DOCUMENT, data=doc_metadata
        )

        # Chunk the document
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        chunk_ids = []

        # Add each chunk as a memory
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}-chunk-{i}"
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if metadata:
                chunk_metadata.update(metadata)

            # Add chunk as memory
            embedding = await self.embedder.embed(chunk_text)
            await self.vector_store.upsert_memory(
                memory_id=chunk_id, embedding=embedding, metadata=chunk_metadata
            )

            await self.graph_store.add_node(
                node_id=chunk_id, node_type=NodeType.CHUNK, data=chunk_metadata
            )

            chunk_ids.append(chunk_id)

        # Create document hierarchy
        hierarchy_result = await self.orchestrator.create_document_hierarchy(
            document_id=document_id, chunk_ids=chunk_ids
        )

        # Infer relationships between chunks
        if self.config.relationships.auto_infer_on_add:
            batch_result = await self.orchestrator.batch_process_memories(
                memory_ids=chunk_ids, batch_size=20
            )
            total_relationships = (
                hierarchy_result["edges_created"] + batch_result["total_relationships"]
            )
        else:
            total_relationships = hierarchy_result["edges_created"]

        logger.info(
            f"Created document {document_id} with {len(chunks)} chunks "
            f"and {total_relationships} relationships"
        )

        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
            "relationships_created": total_relationships,
        }

    def _chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings
                last_period = chunk.rfind(". ")
                last_question = chunk.rfind("? ")
                last_exclaim = chunk.rfind("! ")

                boundary = max(last_period, last_question, last_exclaim)
                if boundary > chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[: boundary + 2]
                    end = start + boundary + 2

            chunks.append(chunk.strip())
            start = end - chunk_overlap

        return chunks
