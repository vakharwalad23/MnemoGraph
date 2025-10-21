# 🧠 MnemoGraph

**A Human-Like Memory System with Graph-Based Relationship Inference**

MnemoGraph is a sophisticated memory management system that mimics how the human brain stores, retrieves, and connects information. It combines vector embeddings for semantic search with graph-based relationship inference to create a rich, interconnected knowledge network.

---

## ✨ Features

### 🎯 Core Capabilities

- **📝 Memory Management**: Store, retrieve, update, and delete memories with automatic embedding generation
- **🔍 Semantic Search**: Find relevant memories using natural language queries
- **🕸️ Automatic Relationship Inference**: Multiple relationship types inferred automatically
- **💬 Conversation Threading**: Track message sequences with temporal ordering
- **📚 Document Chunking**: Intelligently split and index large documents
- **⚡ FastAPI Interface**: REST API with automatic OpenAPI documentation

### 🔗 Relationship Types

MnemoGraph automatically infers multiple types of relationships between memories:

| Type                     | Description                         | Example                                      |
| ------------------------ | ----------------------------------- | -------------------------------------------- |
| **Semantic Similarity**  | Content-based connections           | "Python programming" ↔ "Python data science" |
| **Temporal**             | Time-based updates and sequences    | Python 3.9 → Python 3.10 → Python 3.11       |
| **Hierarchical**         | Topic clusters and parent-child     | "Machine Learning" ⊃ "Neural Networks"       |
| **Entity Co-occurrence** | Shared entity relationships         | Documents mentioning "Tesla" and "SpaceX"    |
| **Causal/Sequential**    | Conversation threads, prerequisites | Message 1 → Message 2 → Message 3            |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory Engine                           │
│  (High-level API for memory operations)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼────────┐
│  Vector Store   │    │   Graph Store    │
│   (Qdrant)      │    │ (SQLite/Neo4j)   │
│                 │    │                  │
│ • Embeddings    │    │ • Nodes          │
│ • Similarity    │    │ • Relationships  │
│ • Metadata      │    │ • Graph Queries  │
└─────────────────┘    └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
      ┌──────────────▼──────────────┐
      │  Relationship Orchestrator   │
      └──────────────┬──────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │ Semantic │          │ Temporal │
    │  Engine  │          │  Engine  │
    └──────────┘          └──────────┘
         │                       │
    ┌────▼─────┐          ┌─────▼────┐
    │Hierarchi-│          │  Entity  │
    │   cal    │          │Co-occur. │
    └──────────┘          └──────────┘
         │
    ┌────▼─────┐
    │  Causal  │
    │Sequential│
    └──────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker** (for Qdrant and Neo4j)
- **Ollama** (for embeddings)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/mnemograph.git
cd mnemograph
```

2. **Install dependencies:**

```bash
pip install -e .
```

3. **Start required services:**

```bash
# Start all services (Qdrant + Neo4j + Ollama)
docker compose up -d

# The embedding model (nomic-embed-text) is automatically pulled on first startup
# Wait ~30 seconds for Ollama to initialize and pull the model
```

4. **Start the API server:**

```bash
python app.py
```

5. **Visit the interactive docs:**

```
http://localhost:8000/docs
```

---

## 💡 Usage Examples

### Adding Memories

```python
from src.services import MemoryEngine
from src.core.vector_store import QdrantStore
from src.core.graph_store import SQLiteGraphStore
from src.core.embeddings import OllamaEmbedding
from src.config import Config

# Initialize
vector_store = QdrantStore(collection_name="my_memories")
graph_store = SQLiteGraphStore(db_path="memories.db")
embedder = OllamaEmbedding()
config = Config()

engine = MemoryEngine(vector_store, graph_store, embedder, config)
await engine.initialize()

# Add a memory
result = await engine.add_memory(
    text="Python is a high-level programming language",
    metadata={"source": "tutorial", "topic": "python"}
)

print(f"Created memory: {result['memory_id']}")
print(f"Relationships created: {result['relationships_created']}")
```

### Semantic Search

```python
# Search for relevant memories
results = await engine.query_memories(
    query="programming languages",
    limit=5,
    include_relationships=True
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['metadata']['text']}")
    print(f"Relationships: {result['relationships']['count']}\n")
```

### Adding Conversations

```python
# Add a conversation thread
conversation = await engine.add_conversation(
    messages=[
        {"text": "What is Python?", "role": "user"},
        {"text": "Python is a programming language.", "role": "assistant"},
        {"text": "What is it used for?", "role": "user"},
        {"text": "Web dev, data science, automation.", "role": "assistant"}
    ]
)

print(f"Created {conversation['message_count']} messages")
print(f"Created {conversation['edges_created']} sequential links")
```

### Document Processing

```python
# Add a document with automatic chunking
document = await engine.add_document(
    text=long_text,
    chunk_size=500,
    chunk_overlap=50,
    metadata={"title": "Python Guide", "author": "Developer"}
)

print(f"Created {document['chunk_count']} chunks")
print(f"Created {document['relationships_created']} relationships")
```

---

## 🌐 REST API

### Endpoints

| Method   | Endpoint           | Description             |
| -------- | ------------------ | ----------------------- |
| `POST`   | `/memories`        | Add a new memory        |
| `GET`    | `/memories/{id}`   | Get a specific memory   |
| `PUT`    | `/memories/{id}`   | Update a memory         |
| `DELETE` | `/memories/{id}`   | Delete a memory         |
| `POST`   | `/memories/search` | Semantic search         |
| `POST`   | `/conversations`   | Add conversation thread |
| `POST`   | `/documents`       | Add and chunk document  |
| `GET`    | `/stats`           | System statistics       |
| `GET`    | `/health`          | Health check            |

### Example API Calls

**Add Memory:**

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python is great for data science",
    "metadata": {"topic": "python"}
  }'
```

**Search Memories:**

```bash
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming",
    "limit": 5,
    "include_relationships": true
  }'
```

**Add Conversation:**

```bash
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"text": "Hello!", "role": "user"},
      {"text": "Hi there!", "role": "assistant"}
    ]
  }'
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# SQLite tests only
pytest tests/ -v -m sqlite

# Neo4j tests only
pytest tests/ -v -m neo4j

# Specific test file
pytest tests/test_memory_engine.py -v
```

### Test Coverage

- ✅ Vector store (Qdrant)
- ✅ Graph stores (SQLite & Neo4j)
- ✅ All relationship engines
- ✅ Memory engine operations
- ✅ Relationship orchestrator
- ✅ Embeddings (Ollama)

---

## ⚙️ Configuration

Edit `src/config.py` to customize behavior:

```python
config = Config(
    relationships={
        "auto_infer_on_add": True,
        "semantic": {
            "similarity_threshold": 0.5,
            "max_similar_memories": 10
        },
        "temporal": {
            "update_similarity_threshold": 0.65,
            "update_time_window_days": 7
        },
        "hierarchical": {
            "min_cluster_size": 3,
            "num_topics": 10
        }
    }
)
```

### Key Configuration Options

| Category         | Parameter                     | Default | Description                          |
| ---------------- | ----------------------------- | ------- | ------------------------------------ |
| **Semantic**     | `similarity_threshold`        | 0.5     | Minimum similarity for relationships |
| **Temporal**     | `update_similarity_threshold` | 0.65    | Threshold for update detection       |
| **Temporal**     | `update_time_window_days`     | 7       | Time window for updates              |
| **Hierarchical** | `num_topics`                  | 10      | Number of topic clusters             |
| **Entity**       | `min_entity_length`           | 3       | Minimum entity name length           |

---

## 📊 Performance

Typical response times with Ollama embeddings:

| Operation        | Time      | Notes                           |
| ---------------- | --------- | ------------------------------- |
| Add Memory       | 200-500ms | Including embedding generation  |
| Semantic Search  | 100-300ms | With relationship info          |
| Get Memory       | 50-150ms  | Single memory retrieval         |
| Add Conversation | 400-800ms | 4-message thread                |
| Add Document     | 1-3s      | 500-word document with chunking |

---

## 🛠️ Technology Stack

- **Vector Store**: [Qdrant](https://qdrant.tech/) - High-performance vector similarity search
- **Graph Store**: SQLite (default) or [Neo4j](https://neo4j.com/) - Relationship storage
- **Embeddings**: [Ollama](https://ollama.ai/) - Local embedding generation
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework
- **Testing**: [pytest](https://pytest.org/) - Comprehensive test suite

---

## 🗺️ Roadmap

### ✅ Completed

- ✅ Core memory management system
- ✅ Vector-based semantic search
- ✅ Graph-based relationship storage
- ✅ Semantic similarity relationships
- ✅ Temporal relationships (updates, sequences)
- ✅ Hierarchical topic clustering
- ✅ Entity co-occurrence detection
- ✅ Causal/sequential relationships
- ✅ Relationship orchestration
- ✅ Memory engine API
- ✅ FastAPI REST interface
- ✅ Comprehensive test suite

### 🚧 In Progress

- [ ] **Background Workers**: Automated memory decay and cleanup processes

### 🔮 Future Plans

- [ ] Multi-modal memories (images, audio)
- [ ] Advanced graph queries (path finding, community detection)
- [ ] Memory consolidation during "sleep" cycles
- [ ] Attention mechanisms for memory retrieval
- [ ] Knowledge graph visualization
- [ ] Memory importance scoring
- [ ] Batch processing optimizations
- [ ] Memory export/import functionality

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### 🐛 Known Issues & Improvement Areas

This is a **Proof of Concept (PoC)** and there are known areas that need improvement:

#### Relationship Engine Issues

1. **Hierarchical Clustering**

   - Topic clustering may not trigger with small datasets (< 5 memories)
   - Abstraction level detection needs refinement
   - K-means clustering could be replaced with more adaptive algorithms

2. **Entity Co-occurrence**

   - spaCy dependency adds overhead; consider lighter NER models
   - Entity extraction misses domain-specific terms
   - Co-occurrence weights could be more sophisticated

3. **Temporal Relationships**

   - Update detection threshold may need per-domain tuning
   - Time windows are fixed; could benefit from adaptive windowing
   - Decay model is simplified; real forgetting curves are more complex

4. **Semantic Similarity**

   - Threshold (0.5) works for general use but may need domain tuning
   - No negative example filtering
   - Could benefit from learned similarity metrics

5. **General**
   - Relationship inference happens sequentially; could be parallelized
   - No conflict resolution when multiple engines suggest different relationships
   - Missing relationship confidence scores

**Want to help?** Pick any of these issues and submit a PR! We especially welcome:

- Performance optimizations
- Better algorithm implementations
- Additional relationship types
- Improved configuration auto-tuning
- Bug fixes and test improvements

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- Inspired by human memory systems and neuroscience research
- Built with modern AI/ML infrastructure
- Thanks to the open-source community

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

<div align="center">

**MnemoGraph** - _Because memories are better connected_ 🧠✨

</div>
