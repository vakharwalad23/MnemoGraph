"""
MnemoGraph FastAPI Application

A REST API server for the MnemoGraph memory engine.
Provides endpoints for adding, querying, updating, and deleting memories.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config
from src.core.factory import EmbedderFactory, GraphStoreFactory, LLMFactory, VectorStoreFactory
from src.services.memory_engine import MemoryEngine
from src.utils.logger import get_logger, setup_logging

# Global engine instance
engine: MemoryEngine | None = None
logger = get_logger(__name__)


# Pydantic models for API
class AddMemoryRequest(BaseModel):
    """Request model for adding a memory."""

    content: str = Field(..., description="Memory content")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")


class AddMemoryResponse(BaseModel):
    """Response model for add memory."""

    memory_id: str
    created_at: str
    content: str
    relationships_created: int
    derived_insights_created: int
    extraction_time_ms: float


class QueryMemoriesRequest(BaseModel):
    """Request model for querying memories."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None
    include_relationships: bool = Field(default=False)


class MemoryResult(BaseModel):
    """Memory search result."""

    id: str
    score: float
    content: str
    type: str
    status: str
    version: int
    created_at: str
    updated_at: str
    confidence: float
    metadata: dict[str, Any]
    relationships: dict[str, Any] | None = None


class UpdateMemoryRequest(BaseModel):
    """Request model for updating a memory."""

    content: str | None = None
    metadata: dict[str, Any] | None = None


class UpdateMemoryResponse(BaseModel):
    """Response model for update memory."""

    memory_id: str
    version: int
    action: str  # "update", "augment", "replace", or "preserve"
    change_description: str
    reasoning: str
    previous_version_id: str | None = None
    new_version_id: str | None = None
    confidence: float


class AddConversationMessage(BaseModel):
    """Single conversation message."""

    text: str
    role: str = "user"


class AddConversationRequest(BaseModel):
    """Request model for adding a conversation."""

    messages: list[AddConversationMessage]
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None


class AddDocumentRequest(BaseModel):
    """Request model for adding a document."""

    text: str
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    document_id: str | None = None
    metadata: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    engine_initialized: bool
    vector_store: str
    graph_store: str
    embedding_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global engine

    # Load configuration from environment or use defaults
    config = Config.from_env()

    # Initialize logging with config
    setup_logging(
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_dir=config.logging.log_dir,
        file_rotation=config.logging.file_rotation,
        file_retention=config.logging.file_retention,
        compression=config.logging.compression,
        serialize=config.logging.serialize,
    )

    logger.info("Starting MnemoGraph server")
    logger.info(
        f"Configuration: LLM={config.llm.provider}/{config.llm.model}, "
        f"Embedder={config.embedder.provider}/{config.embedder.model}, "
        f"Graph={config.graph_backend}"
    )

    # Create components using factories
    logger.info("Creating LLM provider")
    llm = LLMFactory.create(config.llm)

    logger.info("Creating embedder")
    embedder = EmbedderFactory.create(config.embedder)

    logger.info("Creating graph store")
    graph_store = GraphStoreFactory.create(config)

    logger.info("Detecting embedding dimension")
    vector_size = await EmbedderFactory.get_dimension(embedder, config.embedder)
    logger.info(f"Embedding dimension: {vector_size}")

    logger.info("Creating vector store")
    vector_store = VectorStoreFactory.create(config.qdrant, vector_size)

    # Create and initialize engine
    engine = MemoryEngine(
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        vector_store=vector_store,
        config=config,
    )

    await engine.initialize()
    logger.info("MnemoGraph engine initialized")

    yield

    # Cleanup
    logger.info("Shutting down MnemoGraph server")
    await engine.close()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="MnemoGraph API",
    description="Memory management system with automatic relationship inference",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if engine else "initializing",
        engine_initialized=engine is not None,
        vector_store="Qdrant (http://localhost:6333)",
        graph_store="Neo4j (bolt://localhost:7687)",
        embedding_model="nomic-embed-text (Ollama)",
    )


# Memory endpoints
@app.post("/memories", response_model=AddMemoryResponse)
async def add_memory(request: AddMemoryRequest):
    """
    Add a new memory with automatic relationship inference.

    The memory will be embedded and stored in both vector and graph databases.
    Relationships are automatically inferred using LLM-based analysis including:
    - Semantic similarity
    - Temporal relationships (updates, follows, precedes)
    - Hierarchical relationships (part_of, belongs_to)
    - Logical relationships (contradicts, requires, depends_on)
    - Derived insights from patterns across memories
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memory, extraction = await engine.add_memory(
            content=request.content,
            metadata=request.metadata or {},
        )

        return AddMemoryResponse(
            memory_id=memory.id,
            created_at=memory.created_at.isoformat(),
            content=memory.content,
            relationships_created=len(extraction.relationships),
            derived_insights_created=len(extraction.derived_insights),
            extraction_time_ms=extraction.extraction_time_ms,
        )
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/memories/search", response_model=list[MemoryResult])
async def query_memories(request: QueryMemoriesRequest):
    """
    Search memories using semantic similarity.

    Performs vector-based similarity search using embeddings to find memories
    most relevant to the query. Optionally includes relationship information
    showing how memories are connected in the knowledge graph.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        results = await engine.search_similar(
            query=request.query,
            limit=request.limit,
            score_threshold=request.similarity_threshold,
            filters=request.filters,
        )

        memory_results = []
        for memory, score in results:
            result_data = {
                "id": memory.id,
                "score": score,
                "content": memory.content,
                "type": memory.type.value,
                "status": memory.status.value,
                "version": memory.version,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "confidence": memory.confidence,
                "metadata": memory.metadata,
            }

            # Include relationships if requested
            if request.include_relationships:
                neighbors = await engine.get_neighbors(memory.id, limit=10)
                result_data["relationships"] = {
                    "count": len(neighbors),
                    "edges": [
                        {
                            "target_id": edge.target,
                            "type": edge.type.value,
                            "confidence": edge.confidence,
                            "metadata": edge.metadata,
                        }
                        for _, edge in neighbors
                    ],
                }

            memory_results.append(MemoryResult(**result_data))

        return memory_results
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str, include_relationships: bool = Query(default=False)):
    """
    Retrieve a specific memory by ID.

    Returns the memory with its full content, metadata, and version information.
    Optionally includes relationship information showing connected memories.
    Access tracking is automatically updated (access_count, last_accessed).
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memory = await engine.get_memory(memory_id, validate=True)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        result = {
            "id": memory.id,
            "content": memory.content,
            "type": memory.type.value,
            "status": memory.status.value,
            "version": memory.version,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "access_count": memory.access_count,
            "confidence": memory.confidence,
            "metadata": memory.metadata,
        }

        if include_relationships:
            neighbors = await engine.get_neighbors(memory_id, limit=20)
            result["relationships"] = {
                "count": len(neighbors),
                "edges": [
                    {
                        "target_id": edge.target,
                        "type": edge.type.value,
                        "confidence": edge.confidence,
                        "reasoning": edge.metadata.get("reasoning", ""),
                    }
                    for _, edge in neighbors
                ],
            }

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put("/memories/{memory_id}", response_model=UpdateMemoryResponse)
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """
    Update an existing memory with intelligent versioning.

    The LLM analyzes the update and determines the appropriate action:
    - "update": Creates new version, marks old as superseded
    - "augment": Adds information without creating new version
    - "replace": Completely replaces the memory
    - "preserve": Keeps both as separate memories (for conflicts)

    Optionally triggers relationship reindexing for the new/updated memory.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        if request.content:
            # Update with LLM-guided versioning
            updated_memory, evolution = await engine.update_memory(memory_id, request.content)

            # Get confidence from memory metadata (set during evolution)
            confidence = updated_memory.metadata.get("evolution_confidence", 1.0)

            return UpdateMemoryResponse(
                memory_id=updated_memory.id,
                version=updated_memory.version,
                action=evolution.action,
                change_description=evolution.change.description if evolution.change else "",
                reasoning=evolution.change.reasoning if evolution.change else "",
                previous_version_id=evolution.current_version,
                new_version_id=evolution.new_version,
                confidence=confidence,
            )
        else:
            # Metadata-only update
            memory = await engine.update_memory_metadata(memory_id, request.metadata or {})

            return UpdateMemoryResponse(
                memory_id=memory.id,
                version=memory.version,
                action="metadata_update",
                change_description="Metadata updated",
                reasoning="User requested metadata-only update",
                confidence=1.0,
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """
    Delete a memory and its relationships.

    Removes the memory from both vector and graph stores, along with
    all associated relationship edges.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        await engine.delete_memory(memory_id)
        return {"id": memory_id, "deleted": True}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Conversation endpoints (Coming soon)
@app.post("/conversations")
async def add_conversation(request: AddConversationRequest):
    """
    Add a conversation with sequential message linking.

    Creates a thread of messages with FOLLOWS relationships preserving
    the conversation order.
    Note: This endpoint is under development for the new architecture.
    """
    raise HTTPException(status_code=501, detail="Conversation endpoints coming soon")


# Document endpoints (Coming soon)
@app.post("/documents")
async def add_document(request: AddDocumentRequest):
    """
    Add a document with automatic chunking and indexing.

    Splits long documents into chunks, creates hierarchical relationships,
    and infers semantic connections between chunks.
    Note: This endpoint is under development for the new architecture.
    """
    raise HTTPException(status_code=501, detail="Document endpoints coming soon")


# Statistics endpoint
@app.get("/stats")
async def get_stats():
    """
    Get system statistics.

    Returns information about the number of memories, relationships, etc.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        stats = await engine.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MnemoGraph API",
        "version": "1.0.0",
        "description": "Memory management system with automatic relationship inference",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
