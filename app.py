"""
MnemoGraph FastAPI Application

A REST API server for the MnemoGraph memory engine.
Provides endpoints for adding, querying, updating, and deleting memories.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import Config
from src.core.embeddings.ollama import OllamaEmbedder
from src.core.graph_store.neo4j_store import Neo4jGraphStore
from src.core.llm.ollama import OllamaLLM
from src.core.vector_store.qdrant import QdrantStore
from src.services.memory_engine import MemoryEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: MemoryEngine | None = None


# Pydantic models for API
class AddMemoryRequest(BaseModel):
    """Request model for adding a memory."""

    text: str = Field(..., description="Memory text content")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")
    memory_id: str | None = Field(default=None, description="Custom memory ID")
    auto_infer_relationships: bool | None = Field(
        default=None, description="Auto-infer relationships"
    )


class AddMemoryResponse(BaseModel):
    """Response model for add memory."""

    memory_id: str
    created_at: str
    text: str
    relationships_created: int
    auto_infer_enabled: bool
    engines_run: dict[str, Any] | None = None


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
    metadata: dict[str, Any]
    relationships: dict[str, Any] | None = None


class UpdateMemoryRequest(BaseModel):
    """Request model for updating a memory."""

    text: str | None = None
    metadata: dict[str, Any] | None = None
    reindex_relationships: bool = False


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

    logger.info("🚀 Starting MnemoGraph server...")

    # Initialize components
    config = Config(
        graph_backend="neo4j",
    )

    # LLM provider
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        timeout=120.0,
    )

    # Embedder
    embedder = OllamaEmbedder(
        host="http://localhost:11434",
        model="nomic-embed-text",
        timeout=120.0,
    )

    # Graph store (using SQLite for simplicity, can switch to Neo4j)
    graph_store = Neo4jGraphStore(
        uri="bolt://localhost:7687", username="neo4j", password="password"
    )

    # Vector store
    vector_store = QdrantStore(
        collection_name="mnemograph_memories",
        vector_size=768,  # nomic-embed-text dimension
    )

    # Create and initialize engine
    engine = MemoryEngine(
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        vector_store=vector_store,
        config=config,
    )

    await engine.initialize()
    logger.info("✅ MnemoGraph engine initialized")

    yield

    # Cleanup
    logger.info("🛑 Shutting down MnemoGraph server...")
    await engine.close()
    logger.info("✅ Cleanup complete")


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
        graph_store="SQLite (data/mnemograph.db)",
        embedding_model="nomic-embed-text (Ollama)",
    )


# Memory endpoints
@app.post("/memories", response_model=AddMemoryResponse)
async def add_memory(request: AddMemoryRequest):
    """
    Add a new memory with automatic relationship inference.

    The memory will be embedded and stored in both vector and graph databases.
    Relationships will be automatically inferred based on semantic similarity,
    temporal proximity, entity co-occurrence, and causal patterns.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memory, extraction = await engine.add_memory(
            content=request.text,
            metadata=request.metadata or {},
        )

        return AddMemoryResponse(
            memory_id=memory.id,
            created_at=memory.created_at.isoformat(),
            text=memory.content,
            relationships_created=len(extraction.relationships),
            auto_infer_enabled=True,
            engines_run={
                "relationships": len(extraction.relationships),
                "derived_insights": len(extraction.derived_insights),
                "conflicts": len(extraction.conflicts),
            },
        )
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/memories/search", response_model=list[MemoryResult])
async def query_memories(request: QueryMemoriesRequest):
    """
    Search memories using semantic similarity.

    Performs vector-based similarity search to find memories most relevant
    to the query. Optionally includes relationship information.
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
                "metadata": {
                    "content": memory.content,
                    "type": memory.type.value,
                    "status": memory.status.value,
                    "created_at": memory.created_at.isoformat(),
                    **memory.metadata,
                },
            }

            # Include relationships if requested
            if request.include_relationships:
                neighbors = await engine.get_neighbors(memory.id, limit=10)
                result_data["relationships"] = {
                    "count": len(neighbors),
                    "edges": [
                        {
                            "target_id": n[0].id,
                            "type": n[1].type.value,
                            "confidence": n[1].confidence,
                        }
                        for n in neighbors
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

    Optionally includes relationship information showing connected memories.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memory = await engine.get_memory(memory_id, validate=False)

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
            "metadata": memory.metadata,
            "confidence": memory.confidence,
        }

        if include_relationships:
            neighbors = await engine.get_neighbors(memory.id, limit=20)
            result["relationships"] = [
                {
                    "target_id": n[0].id,
                    "target_content": n[0].content[:100],
                    "type": n[1].type.value,
                    "confidence": n[1].confidence,
                }
                for n in neighbors
            ]

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """
    Update an existing memory.

    Can update text content, metadata, or both. Optionally triggers
    relationship reindexing.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        if request.text:
            # Update with versioning
            updated_memory, evolution = await engine.update_memory(memory_id, request.text)

            return {
                "id": updated_memory.id,
                "version": updated_memory.version,
                "action": evolution.action_taken,
                "analysis": evolution.analysis,
                "previous_version": (
                    evolution.current_memory.id if evolution.current_memory else None
                ),
            }
        else:
            # Just metadata update
            memory = await engine.get_memory(memory_id, validate=False)
            if not memory:
                raise HTTPException(status_code=404, detail="Memory not found")

            if request.metadata:
                memory.metadata.update(request.metadata)
                await engine.graph_store.update_memory(memory)

            return {"id": memory.id, "updated": True}
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
