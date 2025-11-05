"""
Configuration for MnemoGraph V2.

Supports loading from:
1. Environment variables (highest priority)
2. YAML config file
3. Default values (fallback)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"  # ollama, openai
    model: str = "llama3.1:8b"
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
    # Optional: embedding dimension (fallback to auto-detect)
    dimension: int | None = None


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


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    file_rotation: str = "10 MB"
    file_retention: str = "7 days"
    compression: str = "zip"
    serialize: bool = True


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


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    llm_relationships: LLMRelationshipConfig = Field(default_factory=LLMRelationshipConfig)
    memory_evolution: MemoryEvolutionConfig = Field(default_factory=MemoryEvolutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)

    # Graph store backend
    graph_backend: str = "neo4j"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Config":
        """
        Load configuration from environment variables.

        Priority: .env file -> system environment variables -> defaults

        Args:
            env_file: Optional path to .env file (default: .env in project root)

        Returns:
            Config instance

        Environment variables:
            MNEMO_LLM_PROVIDER: LLM provider (ollama, openai)
            MNEMO_LLM_MODEL: LLM model name
            MNEMO_LLM_BASE_URL: LLM base URL
            MNEMO_LLM_API_KEY: LLM API key (for OpenAI)
            MNEMO_EMBEDDER_PROVIDER: Embedder provider
            MNEMO_EMBEDDER_MODEL: Embedder model name
            MNEMO_EMBEDDER_API_KEY: Embedder API key (for OpenAI)
            MNEMO_EMBEDDER_DIMENSION: Embedding dimension (optional)
            MNEMO_GRAPH_BACKEND: Graph backend (neo4j)
            MNEMO_NEO4J_URI: Neo4j URI
            MNEMO_NEO4J_USERNAME: Neo4j username
            MNEMO_NEO4J_PASSWORD: Neo4j password
            MNEMO_QDRANT_URL: Qdrant URL
            MNEMO_QDRANT_COLLECTION: Qdrant collection name
        """
        # Load .env file if provided or exists
        if env_file:
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv()

        def get_env(key: str, default: Any = None) -> Any:
            """Get environment variable with type conversion."""
            value = os.getenv(key)
            if value is None:
                return default
            # If value is empty string, return default
            if value == "":
                return default
            # Convert boolean strings
            if isinstance(default, bool):
                return str(value).lower() in ("true", "1", "yes")
            # Convert numeric strings
            if isinstance(default, int):
                return int(value)
            if isinstance(default, float):
                return float(value)
            return value

        return cls(
            llm=LLMConfig(
                provider=get_env("MNEMO_LLM_PROVIDER", "ollama"),
                model=get_env("MNEMO_LLM_MODEL", "llama3.1:8b"),
                base_url=get_env("MNEMO_LLM_BASE_URL", "http://localhost:11434"),
                api_key=get_env("MNEMO_LLM_API_KEY"),
                temperature=get_env("MNEMO_LLM_TEMPERATURE", 0.0),
                max_tokens=get_env("MNEMO_LLM_MAX_TOKENS", 2000),
                timeout=get_env("MNEMO_LLM_TIMEOUT", 120.0),
            ),
            embedder=EmbedderConfig(
                provider=get_env("MNEMO_EMBEDDER_PROVIDER", "ollama"),
                model=get_env("MNEMO_EMBEDDER_MODEL", "nomic-embed-text"),
                base_url=get_env("MNEMO_EMBEDDER_BASE_URL", "http://localhost:11434"),
                api_key=get_env("MNEMO_EMBEDDER_API_KEY"),
                timeout=get_env("MNEMO_EMBEDDER_TIMEOUT", 120.0),
                dimension=get_env("MNEMO_EMBEDDER_DIMENSION"),
            ),
            graph_backend=get_env("MNEMO_GRAPH_BACKEND", "neo4j"),
            neo4j=Neo4jConfig(
                uri=get_env("MNEMO_NEO4J_URI", "bolt://localhost:7687"),
                username=get_env("MNEMO_NEO4J_USERNAME", "neo4j"),
                password=get_env("MNEMO_NEO4J_PASSWORD", "password"),
                database=get_env("MNEMO_NEO4J_DATABASE", "neo4j"),
            ),
            qdrant=QdrantConfig(
                url=get_env("MNEMO_QDRANT_URL", "http://localhost:6333"),
                collection_name=get_env("MNEMO_QDRANT_COLLECTION", "memories"),
                use_grpc=get_env("MNEMO_QDRANT_USE_GRPC", True),
                hnsw_m=get_env("MNEMO_QDRANT_HNSW_M", 16),
                hnsw_ef_construct=get_env("MNEMO_QDRANT_HNSW_EF_CONSTRUCT", 100),
                use_quantization=get_env("MNEMO_QDRANT_USE_QUANTIZATION", True),
                quantization_type=get_env("MNEMO_QDRANT_QUANTIZATION_TYPE", "int8"),
                on_disk=get_env("MNEMO_QDRANT_ON_DISK", False),
            ),
            logging=LoggingConfig(
                level=get_env("MNEMO_LOG_LEVEL", "INFO"),
                log_to_file=get_env("MNEMO_LOG_TO_FILE", True),
                log_dir=get_env("MNEMO_LOG_DIR", "logs"),
                file_rotation=get_env("MNEMO_LOG_FILE_ROTATION", "10 MB"),
                file_retention=get_env("MNEMO_LOG_FILE_RETENTION", "7 days"),
                compression=get_env("MNEMO_LOG_COMPRESSION", "zip"),
                serialize=get_env("MNEMO_LOG_SERIALIZE", True),
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_env_or_yaml(
        cls, yaml_path: str | Path | None = None, env_file: str | Path | None = None
    ) -> "Config":
        """
        Load configuration with priority: env vars > YAML > defaults.

        Args:
            yaml_path: Optional path to YAML config
            env_file: Optional path to .env file

        Returns:
            Config instance
        """
        # Start with YAML if provided
        if yaml_path and Path(yaml_path).exists():
            config_dict = yaml.safe_load(open(yaml_path))
        else:
            config_dict = {}

        # Load env vars (overrides YAML)
        if env_file:
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv()

        # Override with env vars if present
        env_config = cls.from_env()

        # Merge: env vars override YAML
        final_dict = {**config_dict}

        # Apply env overrides (non-default values)
        default = cls()
        if env_config.llm != default.llm:
            final_dict["llm"] = env_config.llm.model_dump()
        if env_config.embedder != default.embedder:
            final_dict["embedder"] = env_config.embedder.model_dump()
        if env_config.neo4j != default.neo4j:
            final_dict["neo4j"] = env_config.neo4j.model_dump()
        if env_config.qdrant != default.qdrant:
            final_dict["qdrant"] = env_config.qdrant.model_dump()
        if env_config.logging != default.logging:
            final_dict["logging"] = env_config.logging.model_dump()

        # Also check top-level fields
        if env_config.graph_backend != default.graph_backend:
            final_dict["graph_backend"] = env_config.graph_backend

        return cls(**final_dict) if final_dict else env_config


# Default config instance
default_config = Config()
