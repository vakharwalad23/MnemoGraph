"""Configuration management for MnemoGraph."""

from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import os


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""
    
    provider: Literal["ollama", "openai", "sentence-transformer"] = "ollama"
    model: str = "nomic-embed-text"
    host: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = None
    vector_size: int = 768


class VectorStoreConfig(BaseModel):
    """Vector database configuration."""
    
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "memories"
    vector_size: int = 768


class GraphStoreConfig(BaseModel):
    """Graph database configuration."""
    
    backend: Literal["sqlite", "neo4j"] = "sqlite"
    
    # SQLite options
    db_path: str = "mnemograph.db"
    
    # Neo4j options
    uri: Optional[str] = "bolt://localhost:7687"
    user: Optional[str] = "neo4j"
    password: Optional[str] = None


class SemanticConfig(BaseModel):
    """Semantic similarity configuration."""
    
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_similar_memories: int = 10


class TemporalConfig(BaseModel):
    """Temporal relationship configuration."""
    
    new_memory_window_hours: int = 48
    update_similarity_threshold: float = 0.65
    update_time_window_days: int = 7
    decay_half_life_days: int = 30
    expiring_threshold_days: int = 30


class HierarchicalConfig(BaseModel):
    """Hierarchical relationship configuration."""
    
    min_cluster_size: int = 2
    num_topics: int = 3
    abstraction_threshold: float = 0.6


class CooccurrenceConfig(BaseModel):
    """Entity co-occurrence configuration."""
    
    min_entity_length: int = 2
    min_cooccurrence_count: int = 1
    entity_weight_threshold: float = 0.1
    use_spacy: bool = True


class CausalConfig(BaseModel):
    """Causal/sequential relationship configuration."""
    
    max_sequence_gap_seconds: int = 3600
    similarity_threshold: float = 0.6
    topic_shift_threshold: float = 0.4


class DecayConfig(BaseModel):
    """Memory decay configuration."""
    
    enabled: bool = True
    decay_half_life_days: float = 30.0
    new_memory_threshold_hours: float = 48.0
    expiring_threshold: float = 0.6
    forgotten_threshold: float = 0.9
    time_weight: float = 0.4
    access_weight: float = 0.3
    connectivity_weight: float = 0.3


class RelationshipConfig(BaseModel):
    """All relationship inference configurations."""
    
    auto_infer_on_add: bool = True
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)
    cooccurrence: CooccurrenceConfig = Field(default_factory=CooccurrenceConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)


class BackgroundWorkerConfig(BaseModel):
    """Background worker configuration."""
    
    enable_decay_worker: bool = True
    decay_interval_hours: int = 24
    
    enable_reindexing_worker: bool = False
    reindexing_interval_days: int = 7
    
    enable_cleanup_worker: bool = True
    cleanup_interval_hours: int = 168  # Weekly
    cleanup_grace_period_days: int = 90


class Config(BaseModel):
    """Main MnemoGraph configuration."""
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    relationships: RelationshipConfig = Field(default_factory=RelationshipConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    workers: BackgroundWorkerConfig = Field(default_factory=BackgroundWorkerConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            embedding=EmbeddingConfig(
                provider=os.getenv("MNEMOGRAPH_EMBEDDING_PROVIDER", "ollama"),
                model=os.getenv("MNEMOGRAPH_EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("MNEMOGRAPH_OLLAMA_HOST", "http://localhost:11434"),
            ),
            vector_store=VectorStoreConfig(
                host=os.getenv("MNEMOGRAPH_QDRANT_HOST", "localhost"),
                port=int(os.getenv("MNEMOGRAPH_QDRANT_PORT", "6333")),
                collection_name=os.getenv("MNEMOGRAPH_COLLECTION", "memories"),
            ),
            graph_store=GraphStoreConfig(
                backend=os.getenv("MNEMOGRAPH_GRAPH_BACKEND", "sqlite"),
                db_path=os.getenv("MNEMOGRAPH_SQLITE_PATH", "mnemograph.db"),
                uri=os.getenv("MNEMOGRAPH_NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("MNEMOGRAPH_NEO4J_USER", "neo4j"),
                password=os.getenv("MNEMOGRAPH_NEO4J_PASSWORD"),
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return self.model_dump()


# Default configuration instance
default_config = Config()