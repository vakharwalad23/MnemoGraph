"""
MnemoGraph FastAPI Application

A REST API server for the MnemoGraph memory engine.
Provides endpoints for adding, querying, updating, and deleting memories.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

from src.services import MemoryEngine
from src.core.vector_store import QdrantStore
from src.core.graph_store import Neo4jGraphStore
from src.core.embeddings import OllamaEmbedding
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[MemoryEngine] = None


# Pydantic models for API
class AddMemoryRequest(BaseModel):
    """Request model for adding a memory."""
    text: str = Field(..., description="Memory text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    memory_id: Optional[str] = Field(default=None, description="Custom memory ID")
    auto_infer_relationships: Optional[bool] = Field(default=None, description="Auto-infer relationships")


class AddMemoryResponse(BaseModel):
    """Response model for add memory."""
    memory_id: str
    created_at: str
    text: str
    relationships_created: int
    auto_infer_enabled: bool
    engines_run: Optional[Dict[str, Any]] = None


class QueryMemoriesRequest(BaseModel):
    """Request model for querying memories."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    include_relationships: bool = Field(default=False)


class MemoryResult(BaseModel):
    """Memory search result."""
    id: str
    score: float
    metadata: Dict[str, Any]
    relationships: Optional[Dict[str, Any]] = None


class UpdateMemoryRequest(BaseModel):
    """Request model for updating a memory."""
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    reindex_relationships: bool = False


class AddConversationMessage(BaseModel):
    """Single conversation message."""
    text: str
    role: str = "user"


class AddConversationRequest(BaseModel):
    """Request model for adding a conversation."""
    messages: List[AddConversationMessage]
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddDocumentRequest(BaseModel):
    """Request model for adding a document."""
    text: str
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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
        relationships={
            "auto_infer_on_add": True,
            "semantic": {"similarity_threshold": 0.5},
        }
    )
    
    embedder = OllamaEmbedding(
        model="nomic-embed-text",
        host="http://localhost:11434"
    )
    
    vector_store = QdrantStore(
        host="localhost",
        port=6333,
        collection_name="mnemograph_prod",
        vector_size=768
    )
    
    graph_store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="mnemograph123"
    )
    
    # Create and initialize engine
    engine = MemoryEngine(
        vector_store=vector_store,
        graph_store=graph_store,
        embedder=embedder,
        config=config
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
    lifespan=lifespan
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
        vector_store="Qdrant (localhost:6333)",
        graph_store="Neo4j (bolt://localhost:7687)",
        embedding_model="nomic-embed-text (Ollama)"
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
        result = await engine.add_memory(
            text=request.text,
            metadata=request.metadata,
            memory_id=request.memory_id,
            auto_infer_relationships=request.auto_infer_relationships
        )
        return AddMemoryResponse(**result)
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search", response_model=List[MemoryResult])
async def query_memories(request: QueryMemoriesRequest):
    """
    Search memories using semantic similarity.
    
    Performs vector-based similarity search to find memories most relevant
    to the query. Optionally includes relationship information.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        results = await engine.query_memories(
            query=request.query,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters,
            include_relationships=request.include_relationships
        )
        return [MemoryResult(**r) for r in results]
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    include_relationships: bool = Query(default=False)
):
    """
    Retrieve a specific memory by ID.
    
    Optionally includes relationship information showing connected memories.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        memory = await engine.get_memory(
            memory_id=memory_id,
            include_relationships=include_relationships
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        result = await engine.update_memory(
            memory_id=memory_id,
            text=request.text,
            metadata=request.metadata,
            reindex_relationships=request.reindex_relationships
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        result = await engine.delete_memory(memory_id=memory_id)
        return result
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation endpoints
@app.post("/conversations")
async def add_conversation(request: AddConversationRequest):
    """
    Add a conversation with sequential message linking.
    
    Creates a thread of messages with FOLLOWS relationships preserving
    the conversation order.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        messages = [{"text": msg.text, "role": msg.role} for msg in request.messages]
        result = await engine.add_conversation(
            messages=messages,
            conversation_id=request.conversation_id,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error adding conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document endpoints
@app.post("/documents")
async def add_document(request: AddDocumentRequest):
    """
    Add a document with automatic chunking and indexing.
    
    Splits long documents into chunks, creates hierarchical relationships,
    and infers semantic connections between chunks.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = await engine.add_document(
            text=request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            document_id=request.document_id,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        # This is a placeholder - you'd implement actual stats gathering
        return {
            "status": "operational",
            "message": "Statistics endpoint - to be implemented"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MnemoGraph API",
        "version": "1.0.0",
        "description": "Memory management system with automatic relationship inference",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

