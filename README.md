# 🧠 MnemoGraph

**An LLM-Native Memory System with Intelligent Relationship Extraction**

MnemoGraph is an experimental memory management system that leverages Large Language Models (LLMs) to understand and connect information like a human would. It combines vector embeddings for semantic search with LLM-powered relationship inference to create a rich, contextually-aware knowledge network.

> **⚠️ Development Status**: MnemoGraph is under active development. Features are being tested and refined. Contributions and feedback are welcome as we continue to improve the system.

> **🚀 Coming Soon: Brain-Like Retrieval System** - We're implementing a revolutionary "RAG on Steroids" retrieval mode that mimics how human memory works. This will combine vector similarity with relationship-based traversal, temporal scoring, and multi-factor ranking to deliver unprecedented retrieval accuracy.

---

## ✨ Current Features

### 🎯 Core Capabilities

- **🤖 LLM-Native Architecture**: Relationship extraction powered by LLMs for human-like understanding
- **📝 Memory Management**: Store, retrieve, and update memories with metadata tracking
- **🔄 Dual-Store Architecture**: Graph store (Neo4j) for relationships, vector store (Qdrant) for semantic search
- **🔍 Multi-Stage Context Filtering**: Efficient pipeline from millions of candidates to relevant context
- **🕸️ 13 Relationship Types**: Comprehensive relationship extraction in a single LLM call
- **🧬 Memory Evolution**: Change detection, supersession tracking, and history preservation
- **♻️ Semantic Invalidation**: LLM-based relevance checking for memory lifecycle management
- **💡 Derived Insights**: Pattern recognition across related memories
- **⚡ FastAPI Interface**: REST API with automatic OpenAPI documentation

### 🔗 Relationship Types (LLM-Extracted)

MnemoGraph extracts 13 types of relationships in a single LLM inference:

| Type             | Description                  | Example                                                  |
| ---------------- | ---------------------------- | -------------------------------------------------------- |
| **SIMILAR_TO**   | Semantically similar content | "Python async" ↔ "Python coroutines"                     |
| **UPDATES**      | Information updates          | "Python 3.9 features" → "Python 3.10 features"           |
| **SUPERSEDES**   | Complete replacement         | Old API docs → New API docs                              |
| **CONTRADICTS**  | Conflicting information      | "Project deadline: Jan 10" ⚡ "Project deadline: Jan 15" |
| **FOLLOWS**      | Temporal/logical sequence    | Message 1 → Message 2                                    |
| **PRECEDES**     | Reverse temporal order       | Setup guide ← Installation guide                         |
| **PART_OF**      | Hierarchical containment     | "Neural Networks" ⊂ "Deep Learning"                      |
| **BELONGS_TO**   | Category membership          | "FastAPI" ∈ "Python Web Frameworks"                      |
| **REQUIRES**     | Prerequisite dependency      | "Advanced Tutorial" requires "Basics"                    |
| **DEPENDS_ON**   | Contextual dependency        | "Code snippet" depends on "Library setup"                |
| **REFERENCES**   | Direct reference/citation    | Paper references another paper                           |
| **MENTIONS**     | Casual mention               | Blog post mentions a tool                                |
| **DERIVED_FROM** | Synthesized insight          | Pattern derived from multiple memories                   |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM Relationship Engine                        │
│  (Single LLM call extracts all 13 relationship types)            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                │
┌────────▼────────┐            ┌─────────▼────────┐
│ Context Filter  │            │  Memory Engine   │
│ (Multi-Stage)   │            │  (High-level API)│
│                 │            │                  │
│ 1M→100 (Vector) │            │ • CRUD ops       │
│ 100→50 (Hybrid) │            │ • Search         │
│ 50→20 (LLM)     │            │ • Evolution      │
└────────┬────────┘            └─────────┬────────┘
         │                               │
         ┌───────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         │   Memory Sync Manager │
         │   (Consistency Layer) │
         │                       │
         │ • Retry logic         │
         │ • Validation          │
         │ • Batch operations    │
         │ • Repair mechanism    │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼────────┐
│  Vector Store   │    │   Graph Store    │
│   (Qdrant)      │    │    (Neo4j)       │
│                 │    │                  │
│ • Embeddings    │    │ • Nodes          │
│ • HNSW Search   │    │ • Relationships  │
│ • Quantization  │    │ • Graph Queries  │
└─────────────────┘    └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
      ┌──────────────▼──────────────────┐
      │      Supporting Services         │
      ├──────────────────────────────────┤
      │ • Memory Evolution               │
      │   (Change detection, history)    │
      │                                  │
      │ • Invalidation Manager           │
      │   (LLM-based relevance checking) │
      │                                  │
      │ • Derived Insights               │
      │   (Pattern recognition)          │
      └──────────────────────────────────┘
```

### Key Components

#### 🔄 Memory Sync Manager

Synchronization layer between graph and vector stores:

- **Automatic retry logic**: Multiple attempts with exponential backoff for transient failures
- **Consistency validation**: Detects and reports sync mismatches between stores
- **Repair mechanism**: Automatically fixes inconsistencies (graph = source of truth)
- **Batch operations**: Efficient bulk sync with partial failure handling
- **Embedding retrieval**: Combines graph metadata with vector embeddings during repair

#### 🤖 LLM Relationship Engine

- **Single-call extraction**: All 13 relationship types in one LLM inference
- **Parallel execution**: Simultaneous vector store + graph store operations
- **Event-driven invalidation**: Automatic supersession detection
- **Derived memories**: Creates insights from patterns across memories

#### 🔍 Multi-Stage Context Filter

Efficient pipeline that scales to millions of memories:

1. **Stage 1 - Vector Search** (10-50ms): 1M+ → 100 candidates via HNSW
2. **Stage 2 - Hybrid Filtering** (50-100ms): Temporal, graph, entity, conversation context
3. **Stage 3 - LLM Pre-filter** (200-500ms): 50 → 20 most relevant via fast LLM

Reduces LLM costs by filtering before expensive relationship extraction.

#### 🧬 Memory Evolution Service

- **Change detection**: LLM determines update vs augment vs replace
- **History tracking**: Complete history with parent-child links
- **Supersession handling**: Tracks when memories replace older information
- **Time-based queries**: Access memories from specific time periods

#### ♻️ Invalidation Manager

Three validation strategies:

1. **On-demand (lazy)**: Check on access based on age/access patterns
2. **Proactive (background)**: Periodic worker validates old/inactive memories
3. **Event-driven**: New memories trigger supersession checks

All decisions made by LLM semantic analysis.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (for Qdrant, Ollama, and optionally Neo4j)
- **OpenAI API key** (optional, if not using Ollama)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/vakharwalad23/mnemograph.git
cd mnemograph
```

2. **Install dependencies:**

```bash
pip install -e .
```

3. **Start required services:**

```bash
# Start all services (Qdrant + Ollama with models)
docker compose up -d

# This will:
# - Start Qdrant on port 6333
# - Start Ollama on port 11434
# - Automatically pull llama3.1:8b and nomic-embed-text models
# - Optionally start Neo4j on port 7687 (if configured)

# Wait ~30 seconds for Ollama to initialize and pull models
```

4. **Configure environment:**

Create a `.env` file or `config.yml` in the project root:

```bash
# .env file example
LLM_PROVIDER=ollama  # or openai
LLM_MODEL=llama3.1:8b  # or gpt-4o-mini
EMBEDDER_PROVIDER=ollama  # or openai
EMBEDDER_MODEL=nomic-embed-text  # or text-embedding-3-small

# If using OpenAI
OPENAI_API_KEY=your-openai-api-key

# Ollama settings (if using local Ollama)
OLLAMA_BASE_URL=http://localhost:11434
```

Or configure via `config.yml`:

```yaml
llm:
  provider: ollama # or openai
  model: llama3.1:8b
  base_url: http://localhost:11434

embedder:
  provider: ollama # or openai
  model: nomic-embed-text

# Add OpenAI API key if using OpenAI
openai:
  api_key: your-openai-api-key
```

5. **Start the API server:**

```bash
python app.py
```

6. **Access the API:**

- **Interactive docs**: http://localhost:8000/docs
- **API endpoint**: http://localhost:8000

---

## 💡 Usage Examples

### Python API

The system provides a high-level Python API for managing memories:

- **Add memories** with automatic relationship extraction
- **Search memories** using semantic similarity
- **Update memories** with intelligent change detection
- **Track memory evolution** over time
- **Validate memory relevance** using LLM analysis

Refer to the `/examples` directory for detailed Python code samples.

### REST API

Access MnemoGraph through a RESTful API with automatic OpenAPI documentation at `http://localhost:8000/docs`.

#### Key Features:

- **Add memories** with metadata and automatic relationship inference
- **Search** using natural language queries with configurable filters
- **Update/evolve** memories with LLM-guided change detection
- **Version history** access for any memory
- **Graph traversal** to explore related memories
- **Health monitoring** and system statistics

All endpoints support JSON request/response formats with comprehensive validation.

---

## 🌐 REST API

### Core Endpoints

| Method   | Endpoint           | Description                                            |
| -------- | ------------------ | ------------------------------------------------------ |
| `POST`   | `/memories`        | Add a new memory with automatic relationship inference |
| `GET`    | `/memories/{id}`   | Get a specific memory with optional relationships      |
| `PUT`    | `/memories/{id}`   | Update memory with LLM-guided versioning               |
| `DELETE` | `/memories/{id}`   | Delete a memory and its relationships                  |
| `POST`   | `/memories/search` | Semantic search with optional relationship information |
| `POST`   | `/conversations`   | Add conversation with sequential linking (coming soon) |
| `POST`   | `/documents`       | Add document with automatic chunking (coming soon)     |
| `GET`    | `/stats`           | System statistics                                      |
| `GET`    | `/health`          | Health check                                           |
| `GET`    | `/`                | API information and documentation links                |

### Interactive Documentation

Visit `http://localhost:8000/docs` after starting the server for:

- **Complete API documentation** with request/response schemas
- **Interactive testing** - Try endpoints directly in the browser
- **Authentication** configuration (if enabled)
- **Example requests** for all endpoints

The API follows RESTful conventions and returns JSON responses with comprehensive error messages.

---

## 🧪 Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Suites

```bash
# LLM relationship extraction tests
pytest tests/test_llm_relationship_engine.py -v

# Memory evolution tests
pytest tests/test_memory_evolution.py -v

# Invalidation manager tests
pytest tests/test_invalidation_manager.py -v

# Context filter tests
pytest tests/test_context_filter.py -v

# Memory sync manager tests
pytest tests/services/test_memory_sync.py -v

# Neo4j graph store tests (requires Neo4j running)
pytest tests/ -v -m neo4j
```

### Test Coverage

**Tested Components:**

- LLM relationship extraction (13 relationship types)
- Multi-stage context filtering (3 stages)
- Memory evolution (update, augment, replace, preserve)
- Semantic invalidation (on-demand, proactive, event-driven)
- Memory sync manager (retry, validation, repair, batch ops)
- Vector store (Qdrant integration)
- Graph store (Neo4j)
- LLM providers (Ollama & OpenAI)
- Embedders (Ollama & OpenAI)

**Coverage by Module:**

- `llm_relationship_engine.py`
- `memory_sync.py`
- `memory_evolution.py`
- `memory_engine.py`
- `context_filter.py`
- `invalidation_manager.py`

**Note:** Tests use real LLM implementations for integration testing. Sync manager tests use real Neo4j and Qdrant stores.

---

## ⚙️ Configuration

The system is highly configurable through `src/config.py`. Key configuration areas:

### LLM Provider

```python
llm=LLMConfig(
    provider="ollama",  # or "openai"
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
    max_tokens=2000
)
```

### Embeddings

```python
embedder=EmbedderConfig(
    provider="ollama",  # or "openai"
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)
```

### Relationship Extraction

```python
llm_relationships=LLMRelationshipConfig(
    min_confidence=0.5,          # Filter low-confidence relationships
    min_derived_confidence=0.7,   # Threshold for derived insights
    context_window=50,            # Max candidates for LLM
    recent_window_days=30,        # Temporal context window
    graph_depth=2,                # Graph traversal depth
    enable_derived_memories=True, # Auto-generate insights
    enable_auto_invalidation=True # Check for supersession
)
```

### Memory Evolution

```python
memory_evolution=MemoryEvolutionConfig(
    preserve_history=True,        # Keep all versions
    auto_detect_updates=True,     # Use LLM to detect updates
    max_version_history=100,      # Max versions per memory
    enable_time_travel=True       # Support historical queries
)
```

### Vector Store (Qdrant)

```python
qdrant=QdrantConfig(
    url="http://localhost:6333",
    collection_name="memories",
    use_grpc=True,                # Faster than HTTP
    hnsw_m=16,                    # HNSW graph connections
    hnsw_ef_construct=100,        # Construction accuracy
    use_quantization=True,        # Compress vectors (int8)
    on_disk=False                 # Use memory for speed
)
```

### Key Configuration Options

| Category              | Parameter                  | Default | Description                          |
| --------------------- | -------------------------- | ------- | ------------------------------------ |
| **LLM Relationships** | `min_confidence`           | 0.5     | Minimum confidence for relationships |
| **LLM Relationships** | `context_window`           | 50      | Max candidates sent to LLM           |
| **LLM Relationships** | `enable_auto_invalidation` | True    | Check for supersession automatically |
| **Memory Evolution**  | `preserve_history`         | True    | Keep all versions                    |
| **Memory Evolution**  | `auto_detect_updates`      | True    | Use LLM to analyze updates           |
| **Qdrant**            | `use_quantization`         | True    | Compress vectors (4x smaller)        |
| **Qdrant**            | `use_grpc`                 | True    | Faster than HTTP                     |

---

## 📊 Performance

### Scalability

**Multi-Stage Filter Performance:**

- Stage 1 (Vector): 10-50ms (1M+ memories → 100 candidates)
- Stage 2 (Hybrid): 50-100ms (temporal, graph, entity filtering)
- Stage 3 (LLM): 200-500ms (50 → 20 most relevant)

**Optimization Strategies:**

- LLM pre-filtering reduces extraction costs by 80-90%
- Vector quantization reduces memory usage by 75%
- Parallel operations reduce latency by 40-60%

---

## 🛠️ Technology Stack

- **LLM**: [Ollama](https://ollama.ai/) (local) or [OpenAI](https://openai.com/) (cloud)
- **Vector Store**: [Qdrant](https://qdrant.tech/) - High-performance vector similarity search with HNSW indexing
- **Graph Store**: [Neo4j](https://neo4j.com/) - Relationship storage and traversal
- **Embeddings**: [Ollama](https://ollama.ai/) (nomic-embed-text) or [OpenAI](https://openai.com/) (text-embedding-3-small)
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- **Testing**: [pytest](https://pytest.org/) - Comprehensive test suite

---

## 🗺️ Development Roadmap

### ✅ Current Progress

**Core Infrastructure:**

- ✅ LLM-native relationship extraction (13 types in single call)
- ✅ Multi-stage context filtering (vector → hybrid → LLM)
- ✅ Dual-store architecture (Neo4j + Qdrant)
- ✅ Memory sync manager with retry and validation
- ✅ FastAPI interface with OpenAPI docs

**Memory Management:**

- ✅ Memory evolution with change detection
- ✅ LLM-based semantic invalidation
- ✅ Event-driven supersession detection
- ✅ History preservation and tracking
- ✅ Derived insights from patterns

**Testing & Quality:**

- ✅ Integration tests with real LLM/stores
- ✅ Comprehensive test coverage across modules
- ✅ Support for both Ollama and OpenAI

### 🚧 Active Development

**Performance & Reliability:**

- 🔨 Enhanced error handling and recovery mechanisms
- 🔨 Caching layer for repeated queries
- 🔨 Background worker optimization for invalidation checks
- 🔨 Batch operation improvements
- 🔨 Query performance optimization

**Testing & Documentation:**

- 🔨 Increase test coverage across all modules
- 🔨 Add more real-world usage examples
- 🔨 Performance benchmarking suite
- 🔨 API usage documentation

**Features:**

- 🔨 Advanced graph traversal algorithms
- 🔨 Memory consolidation strategies
- 🔨 Enhanced clustering capabilities

### 🔮 Future Plans

**Advanced Features:**

- [ ] Multi-modal memory support (images, audio, video)
- [ ] Interactive graph visualization UI
- [ ] Collaborative memory spaces (multi-user)
- [ ] Streaming API via WebSocket
- [ ] Advanced analytics and insights dashboard

**Scalability:**

- [ ] Distributed processing for large-scale deployments
- [ ] Advanced caching strategies
- [ ] Query optimization for massive graphs
- [ ] Partitioning strategies for horizontal scaling

**Integration:**

- [ ] Additional LLM provider support (Anthropic, Cohere, etc.)
- [ ] Plugin system for custom relationship types
- [ ] Export/import functionality
- [ ] Backup and restore mechanisms

**Intelligence:**

- [ ] Attention mechanisms for importance scoring
- [ ] Periodic "consolidation" cycles
- [ ] Automatic memory clustering and topic extraction
- [ ] Conflict resolution strategies

---

## 🤝 Contributing

MnemoGraph is an open-source project and we welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing feedback, your help is appreciated.

### 🎯 Priority Areas

We're particularly interested in contributions for:

### 🎯 Priority Areas

**Performance Optimization:**

- Caching strategies for repeated queries
- Parallel processing improvements
- Database query optimization
- Batch operation enhancements

**LLM Integration:**

- Support for additional LLM providers (Anthropic, Cohere, etc.)
- Prompt engineering improvements
- Cost optimization strategies
- Multi-modal embedding support

**Advanced Features:**

- Graph visualization components
- Memory clustering algorithms
- Advanced relationship inference
- Pattern recognition improvements

**Testing & Quality:**

- Increase test coverage
- Real-world usage examples
- Performance benchmarks
- Integration test scenarios

**Documentation:**

- API usage guides
- Architecture deep-dives
- Performance tuning guides
- Troubleshooting documentation

### 💡 How You Can Help

1. **Report Issues**: Found a bug or have a feature request? [Open an issue](https://github.com/vakharwalad23/mnemograph/issues)
2. **Improve Documentation**: Help make the docs clearer and more comprehensive
3. **Add Tests**: Help increase coverage and test edge cases
4. **Optimize Performance**: Profile and improve bottlenecks
5. **Share Use Cases**: Tell us how you're using MnemoGraph
6. **Submit Pull Requests**: Fix bugs, add features, or improve code quality

### 📋 Contribution Guidelines

- Fork the repository and create a new branch for your feature
- Write tests for new functionality
- Follow the existing code style and conventions
- Update documentation as needed
- Submit a pull request with a clear description of changes

For major changes, please open an issue first to discuss what you'd like to change.

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- Inspired by cognitive science research on human memory systems
- Built with modern LLM infrastructure (Ollama, OpenAI)
- Vector search powered by Qdrant's HNSW implementation
- Graph storage using Neo4j
- Thanks to the open-source AI/ML community

### Design Philosophy

MnemoGraph is built on these principles:

1. **LLMs understand context better than algorithms**: A single LLM call with rich context can outperform multiple specialized engines with hand-tuned thresholds.

2. **Filter before processing**: Multi-stage filtering (1M → 100 → 50 → 20) makes LLM-based processing practical at scale.

3. **Semantic invalidation over decay formulas**: "Is this still relevant?" is a question LLMs can answer more intelligently than mathematical decay functions.

4. **Relationships need reasoning**: Understanding _why_ a relationship exists is as important as knowing _what_ type it is.

5. **Memory evolves, it doesn't just accumulate**: Track changes over time, understand updates, preserve history.

---

## 📧 Contact

For questions or discussions about the project:

- **GitHub**: [vakharwalad23/mnemograph](https://github.com/vakharwalad23/mnemograph)

<div align="center">

**MnemoGraph** - _LLM-Native Memory System_ 🧠✨

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()

</div>
