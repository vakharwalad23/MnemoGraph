"""
Configuration for MnemoGraph V2.
"""

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"  # ollama, openai
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: float = 120.0


class EmbedderConfig(BaseModel):
    """Embedder configuration."""

    provider: str = "ollama"  # ollama, openai
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    timeout: float = 120.0


class LLMRelationshipConfig(BaseModel):
    """LLM relationship extraction configuration."""

    min_confidence: float = 0.5
    min_derived_confidence: float = 0.7
    context_window: int = 50
    recent_window_days: int = 30
    graph_depth: int = 2
    enable_derived_memories: bool = True
    enable_auto_invalidation: bool = True


class MemoryEvolutionConfig(BaseModel):
    """Memory versioning configuration."""

    preserve_history: bool = True
    auto_detect_updates: bool = True
    max_version_history: int = 100
    enable_time_travel: bool = True


class QdrantConfig(BaseModel):
    """Optimized Qdrant configuration."""

    url: str = "http://localhost:6333"
    collection_name: str = "memories"
    use_grpc: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    use_quantization: bool = True
    quantization_type: str = "int8"
    on_disk: bool = False
    batch_size: int = 100
    timeout: int = 30


class Neo4jConfig(BaseModel):
    """Neo4j graph database configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class SQLiteConfig(BaseModel):
    """SQLite graph database configuration."""

    db_path: str = "data/mnemo_graph.db"


class ClusteringConfig(BaseModel):
    """Adaptive clustering configuration."""

    algorithm: str = "hdbscan"  # hdbscan, llm
    min_cluster_size: int = 3
    min_samples: int = 2
    use_llm_naming: bool = True
    max_clusters: int | None = None


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig = LLMConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    llm_relationships: LLMRelationshipConfig = LLMRelationshipConfig()
    memory_evolution: MemoryEvolutionConfig = MemoryEvolutionConfig()
    qdrant: QdrantConfig = QdrantConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    sqlite: SQLiteConfig = SQLiteConfig()
    clustering: ClusteringConfig = ClusteringConfig()

    # Graph store backend
    graph_backend: str = "sqlite"  # sqlite, neo4j

    # Debug and logging
    debug: bool = False
    log_level: str = "INFO"


# Default config instance
default_config = Config()
