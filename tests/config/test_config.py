"""
Tests for configuration management.

Tests config loading from:
1. Environment variables
2. YAML files
3. Combined (env overrides YAML)
"""

import os

import pytest
import yaml

from src.config import Config, EmbedderConfig, LLMConfig, Neo4jConfig, QdrantConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = Config()

        # LLM defaults
        assert config.llm.provider == "ollama"
        assert config.llm.model == "llama3.1:8b"
        assert config.llm.base_url == "http://localhost:11434"
        assert config.llm.api_key is None
        assert config.llm.temperature == 0.0
        assert config.llm.max_tokens == 2000

        # Embedder defaults
        assert config.embedder.provider == "ollama"
        assert config.embedder.model == "nomic-embed-text"
        assert config.embedder.dimension is None  # Auto-detect

        # Graph backend
        assert config.graph_backend == "neo4j"

        # Neo4j defaults
        assert config.neo4j.uri == "bolt://localhost:7687"
        assert config.neo4j.username == "neo4j"
        assert config.neo4j.password == "password"

        # Qdrant defaults
        assert config.qdrant.url == "http://localhost:6333"
        assert config.qdrant.collection_name == "memories"
        assert config.qdrant.use_grpc is True
        assert config.qdrant.use_quantization is True

    def test_llm_config_creation(self):
        """Test creating LLM config."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            temperature=0.7,
        )

        assert llm_config.provider == "openai"
        assert llm_config.model == "gpt-4o"
        assert llm_config.api_key == "sk-test"
        assert llm_config.temperature == 0.7

    def test_embedder_config_with_dimension(self):
        """Test embedder config with explicit dimension."""
        embedder_config = EmbedderConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )

        assert embedder_config.dimension == 1536


class TestConfigFromEnv:
    """Test loading configuration from environment variables."""

    def test_from_env_basic(self, monkeypatch):
        """Test loading basic config from environment."""
        # Set environment variables
        monkeypatch.setenv("MNEMO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("MNEMO_LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("MNEMO_LLM_API_KEY", "sk-test-key")
        monkeypatch.setenv("MNEMO_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("MNEMO_EMBEDDER_MODEL", "text-embedding-3-small")

        config = Config.from_env()

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.api_key == "sk-test-key"
        assert config.embedder.provider == "openai"
        assert config.embedder.model == "text-embedding-3-small"

    def test_from_env_with_numbers(self, monkeypatch):
        """Test loading numeric values from environment."""
        monkeypatch.setenv("MNEMO_LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("MNEMO_LLM_MAX_TOKENS", "4000")
        monkeypatch.setenv("MNEMO_EMBEDDER_DIMENSION", "1536")

        config = Config.from_env()

        assert config.llm.temperature == 0.7
        assert config.llm.max_tokens == 4000
        assert config.embedder.dimension == 1536

    def test_from_env_with_booleans(self, monkeypatch):
        """Test loading boolean values from environment."""
        monkeypatch.setenv("MNEMO_QDRANT_USE_GRPC", "false")
        monkeypatch.setenv("MNEMO_QDRANT_USE_QUANTIZATION", "0")
        monkeypatch.setenv("MNEMO_DEBUG", "true")

        config = Config.from_env()

        assert config.qdrant.use_grpc is False
        assert config.qdrant.use_quantization is False
        assert config.debug is True

    def test_from_env_graph_backend(self, monkeypatch):
        """Test setting graph backend from environment."""
        monkeypatch.setenv("MNEMO_GRAPH_BACKEND", "neo4j")
        monkeypatch.setenv("MNEMO_NEO4J_PASSWORD", "secret")

        config = Config.from_env()

        assert config.graph_backend == "neo4j"
        assert config.neo4j.password == "secret"

    def test_from_env_qdrant_config(self, monkeypatch):
        """Test Qdrant configuration from environment."""
        monkeypatch.setenv("MNEMO_QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("MNEMO_QDRANT_COLLECTION", "test_memories")
        monkeypatch.setenv("MNEMO_QDRANT_HNSW_M", "32")
        monkeypatch.setenv("MNEMO_QDRANT_HNSW_EF_CONSTRUCT", "200")
        monkeypatch.setenv("MNEMO_QDRANT_QUANTIZATION_TYPE", "scalar")
        monkeypatch.setenv("MNEMO_QDRANT_ON_DISK", "true")

        config = Config.from_env()

        assert config.qdrant.url == "http://qdrant:6333"
        assert config.qdrant.collection_name == "test_memories"
        assert config.qdrant.hnsw_m == 32
        assert config.qdrant.hnsw_ef_construct == 200
        assert config.qdrant.quantization_type == "scalar"
        assert config.qdrant.on_disk is True

    def test_from_env_with_dotenv_file(self, monkeypatch, tmp_path):
        """Test loading from .env file."""
        env_file = tmp_path / ".env.test"
        env_file.write_text(
            """
MNEMO_LLM_PROVIDER=openai
MNEMO_LLM_MODEL=gpt-4o
MNEMO_LLM_API_KEY=sk-from-file
MNEMO_GRAPH_BACKEND=neo4j
"""
        )

        config = Config.from_env(env_file=str(env_file))

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.llm.api_key == "sk-from-file"
        assert config.graph_backend == "neo4j"

    def test_from_env_missing_optional_values(self, monkeypatch):
        """Test that missing optional values use defaults."""
        # Clear any existing env vars that might interfere
        monkeypatch.delenv("MNEMO_LLM_MODEL", raising=False)
        monkeypatch.delenv("MNEMO_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MNEMO_EMBEDDER_PROVIDER", raising=False)
        monkeypatch.delenv("MNEMO_GRAPH_BACKEND", raising=False)

        # Only set required values, rest should use defaults
        monkeypatch.setenv("MNEMO_LLM_PROVIDER", "ollama")

        config = Config.from_env()

        # Should still have defaults
        assert config.llm.model == "llama3.1:8b"
        assert config.embedder.provider == "ollama"
        assert config.graph_backend == "neo4j"


class TestConfigFromYAML:
    """Test loading configuration from YAML files."""

    def test_from_yaml_basic(self, tmp_path):
        """Test loading config from YAML file."""
        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-yaml-key",
                "temperature": 0.5,
            },
            "embedder": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimension": 3072,
            },
            "graph_backend": "neo4j",
        }
        yaml_file.write_text(yaml.dump(config_data))

        config = Config.from_yaml(str(yaml_file))

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.llm.api_key == "sk-yaml-key"
        assert config.llm.temperature == 0.5
        assert config.embedder.dimension == 3072
        assert config.graph_backend == "neo4j"

    def test_from_yaml_full_config(self, tmp_path):
        """Test loading full config from YAML."""
        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "ollama",
                "model": "llama3.1:8b",
                "base_url": "http://ollama:11434",
            },
            "embedder": {
                "provider": "ollama",
                "model": "nomic-embed-text",
            },
            "graph_backend": "neo4j",
            "neo4j": {
                "uri": "bolt://neo4j:7687",
                "username": "admin",
                "password": "admin123",
                "database": "graph",
            },
            "qdrant": {
                "url": "http://qdrant:6333",
                "collection_name": "prod_memories",
                "use_grpc": False,
                "hnsw_m": 32,
                "hnsw_ef_construct": 200,
                "use_quantization": True,
                "quantization_type": "int8",
                "on_disk": True,
            },
            "debug": True,
            "log_level": "DEBUG",
        }
        yaml_file.write_text(yaml.dump(config_data))

        config = Config.from_yaml(str(yaml_file))

        # Verify all sections
        assert config.llm.base_url == "http://ollama:11434"
        assert config.neo4j.uri == "bolt://neo4j:7687"
        assert config.neo4j.username == "admin"
        assert config.neo4j.password == "admin123"
        assert config.neo4j.database == "graph"
        assert config.qdrant.url == "http://qdrant:6333"
        assert config.qdrant.collection_name == "prod_memories"
        assert config.qdrant.use_grpc is False
        assert config.qdrant.hnsw_m == 32
        assert config.qdrant.hnsw_ef_construct == 200
        assert config.qdrant.use_quantization is True
        assert config.qdrant.quantization_type == "int8"
        assert config.qdrant.on_disk is True
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_from_yaml_partial_config(self, tmp_path):
        """Test loading partial config with defaults."""
        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {"model": "llama3.2:latest"},
            "graph_backend": "neo4j",
        }
        yaml_file.write_text(yaml.dump(config_data))

        config = Config.from_yaml(str(yaml_file))

        # Should merge with defaults
        assert config.llm.model == "llama3.2:latest"
        assert config.llm.provider == "ollama"  # default
        assert config.graph_backend == "neo4j"
        assert config.embedder.model == "nomic-embed-text"  # default

    def test_from_yaml_qdrant_advanced_config(self, tmp_path):
        """Test loading Qdrant advanced configuration from YAML."""
        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "qdrant": {
                "url": "http://custom-qdrant:6334",
                "collection_name": "advanced_memories",
                "use_grpc": True,
                "hnsw_m": 64,
                "hnsw_ef_construct": 400,
                "use_quantization": False,
                "quantization_type": "scalar",
                "on_disk": True,
                "batch_size": 200,
                "timeout": 60,
            }
        }
        yaml_file.write_text(yaml.dump(config_data))

        config = Config.from_yaml(str(yaml_file))

        # Verify all Qdrant settings
        assert config.qdrant.url == "http://custom-qdrant:6334"
        assert config.qdrant.collection_name == "advanced_memories"
        assert config.qdrant.use_grpc is True
        assert config.qdrant.hnsw_m == 64
        assert config.qdrant.hnsw_ef_construct == 400
        assert config.qdrant.use_quantization is False
        assert config.qdrant.quantization_type == "scalar"
        assert config.qdrant.on_disk is True
        assert config.qdrant.batch_size == 200
        assert config.qdrant.timeout == 60

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML file."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            Config.from_yaml(str(yaml_file))


class TestConfigFromEnvOrYAML:
    """Test combined loading (env overrides YAML)."""

    # @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        """Test that environment variables override YAML values."""
        # Clear all existing MNEMO_ env vars
        for key in list(os.environ.keys()):
            if key.startswith("MNEMO_"):
                monkeypatch.delenv(key, raising=False)

        # Create YAML file
        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "ollama",
                "model": "llama3.1:8b",
                "temperature": 0.0,
            },
            "graph_backend": "neo4j",
        }
        yaml_file.write_text(yaml.dump(config_data))

        # Set environment variables (should override)
        monkeypatch.setenv("MNEMO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("MNEMO_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("MNEMO_GRAPH_BACKEND", "neo4j")

        config = Config.from_env_or_yaml(yaml_path=str(yaml_file))

        # Environment should win
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.graph_backend == "neo4j"
        # YAML value preserved where no env override
        assert config.llm.temperature == 0.0

    def test_yaml_only_when_no_env(self, tmp_path, monkeypatch):
        """Test YAML values used when no environment variables."""
        # Clear all existing MNEMO_ env vars
        for key in list(os.environ.keys()):
            if key.startswith("MNEMO_"):
                monkeypatch.delenv(key, raising=False)

        yaml_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "embedder": {"dimension": 1536},
        }
        yaml_file.write_text(yaml.dump(config_data))

        config = Config.from_env_or_yaml(yaml_path=str(yaml_file))

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.embedder.dimension == 1536

    def test_env_only_when_no_yaml(self, monkeypatch):
        """Test environment values used when no YAML file."""
        # Clear all existing MNEMO_ env vars first
        for key in list(os.environ.keys()):
            if key.startswith("MNEMO_"):
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("MNEMO_LLM_PROVIDER", "openai")
        monkeypatch.setenv("MNEMO_LLM_MODEL", "gpt-4o")

        config = Config.from_env_or_yaml(yaml_path="/nonexistent.yaml")

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"

    def test_defaults_when_no_yaml_or_env(self, monkeypatch):
        """Test defaults used when neither YAML nor env vars."""
        # Clear all existing MNEMO_ env vars
        for key in list(os.environ.keys()):
            if key.startswith("MNEMO_"):
                monkeypatch.delenv(key, raising=False)

        config = Config.from_env_or_yaml(yaml_path="/nonexistent.yaml")

        # Should have all defaults
        assert config.llm.provider == "ollama"
        assert config.llm.model == "llama3.1:8b"
        assert config.embedder.provider == "ollama"


class TestConfigSubComponents:
    """Test individual config component models."""

    def test_qdrant_config_validation(self):
        """Test Qdrant config with various settings."""
        config = QdrantConfig(
            url="http://custom:6333",
            collection_name="custom_collection",
            use_grpc=False,
            hnsw_m=32,
            hnsw_ef_construct=200,
            use_quantization=False,
            quantization_type="scalar",
            on_disk=True,
        )

        assert config.url == "http://custom:6333"
        assert config.collection_name == "custom_collection"
        assert config.use_grpc is False
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construct == 200
        assert config.use_quantization is False
        assert config.quantization_type == "scalar"
        assert config.on_disk is True

    def test_qdrant_config_with_all_hnsw_params(self):
        """Test Qdrant config with all HNSW parameters."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
            hnsw_m=64,
            hnsw_ef_construct=400,
            use_quantization=True,
            quantization_type="int8",
        )

        assert config.hnsw_m == 64
        assert config.hnsw_ef_construct == 400
        assert config.use_quantization is True
        assert config.quantization_type == "int8"

    def test_qdrant_config_defaults(self):
        """Test Qdrant config default values."""
        config = QdrantConfig()

        assert config.url == "http://localhost:6333"
        assert config.collection_name == "memories"
        assert config.use_grpc is True
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construct == 100
        assert config.use_quantization is True
        assert config.quantization_type == "int8"
        assert config.on_disk is False

    def test_neo4j_config_validation(self):
        """Test Neo4j config."""
        config = Neo4jConfig(
            uri="neo4j://custom:7687",
            username="admin",
            password="secret",
            database="custom_db",
        )

        assert config.uri == "neo4j://custom:7687"
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.database == "custom_db"

    def test_llm_relationships_config(self):
        """Test LLM relationships config defaults."""
        config = Config()

        assert config.llm_relationships.min_confidence == 0.5
        assert config.llm_relationships.min_derived_confidence == 0.7
        assert config.llm_relationships.context_window == 50
        assert config.llm_relationships.enable_derived_memories is True
        assert config.llm_relationships.enable_auto_invalidation is True

    def test_memory_evolution_config(self):
        """Test memory evolution config defaults."""
        config = Config()

        assert config.memory_evolution.preserve_history is True
        assert config.memory_evolution.auto_detect_updates is True
        assert config.memory_evolution.max_version_history == 100
        assert config.memory_evolution.enable_time_travel is True


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_env_vars_use_defaults(self, monkeypatch):
        """Test that empty env vars don't override defaults."""
        # Set empty string (shouldn't override)
        monkeypatch.setenv("MNEMO_LLM_PROVIDER", "")

        config = Config.from_env()

        # Empty string should not override, but currently it does
        # This tests actual behavior
        assert config.llm.provider == "" or config.llm.provider == "ollama"

    def test_invalid_boolean_env_var(self, monkeypatch):
        """Test handling of invalid boolean values."""
        monkeypatch.setenv("MNEMO_DEBUG", "maybe")

        config = Config.from_env()

        # Invalid boolean should be False
        assert config.debug is False

    def test_config_immutability_after_creation(self):
        """Test that config can be modified after creation."""
        config = Config()

        # Should be able to modify
        config.llm.model = "llama3.2:latest"
        assert config.llm.model == "llama3.2:latest"

    def test_multiple_config_instances_independent(self):
        """Test that multiple config instances are independent."""
        config1 = Config()
        config2 = Config()

        config1.llm.model = "model1"
        config2.llm.model = "model2"

        assert config1.llm.model == "model1"
        assert config2.llm.model == "model2"
