"""
SQLite graph store implementation - redesigned for Phase 2 & 3.

Clean, efficient implementation using aiosqlite.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from src.core.graph_store.base import GraphStore
from src.models.memory import Memory, MemoryStatus, NodeType
from src.models.relationships import Edge, RelationshipType


class SQLiteGraphStore(GraphStore):
    """
    SQLite-based graph store for memories and relationships.

    Features:
    - Fast local storage
    - JSON support for metadata
    - Full-text search capability
    - Transaction support
    """

    def __init__(self, db_path: str = "data/mnemo_graph.db"):
        """
        Initialize SQLite graph store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> None:
        """Establish connection to SQLite."""
        if self.connection is None:
            self.connection = await aiosqlite.connect(self.db_path)
            # Enable foreign keys
            await self.connection.execute("PRAGMA foreign_keys = ON")
            # Enable JSON support
            await self.connection.execute("PRAGMA journal_mode = WAL")
            await self.connection.commit()

    async def initialize(self) -> None:
        """Initialize database schema."""
        await self.connect()

        # Create nodes table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                embedding BLOB,
                version INTEGER DEFAULT 1,
                parent_version TEXT,
                valid_from TEXT NOT NULL,
                valid_until TEXT,
                status TEXT NOT NULL,
                superseded_by TEXT,
                invalidation_reason TEXT,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """
        )

        # Create edges table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """
        )

        # Create indices
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)"
        )
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")

        await self.connection.commit()

    # ═══════════════════════════════════════════════════════════
    # NODE OPERATIONS
    # ═══════════════════════════════════════════════════════════

    async def add_node(self, memory: Memory) -> None:
        """Add a memory node to the graph."""
        await self.connect()

        # Serialize embedding as JSON (or None)
        embedding_json = json.dumps(memory.embedding) if memory.embedding else None

        await self.connection.execute(
            """
            INSERT OR REPLACE INTO nodes (
                id, content, type, embedding, version, parent_version,
                valid_from, valid_until, status, superseded_by, invalidation_reason,
                confidence, access_count, last_accessed, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.type.value,
                embedding_json,
                memory.version,
                memory.parent_version,
                memory.valid_from.isoformat(),
                memory.valid_until.isoformat() if memory.valid_until else None,
                memory.status.value,
                memory.superseded_by,
                memory.invalidation_reason,
                memory.confidence,
                memory.access_count,
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                json.dumps(memory.metadata),
            ),
        )

        await self.connection.commit()

    async def get_node(self, node_id: str) -> Memory | None:
        """Retrieve a memory node by ID."""
        await self.connect()

        cursor = await self.connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    async def update_node(self, memory: Memory) -> None:
        """Update an existing memory node."""
        await self.add_node(memory)  # INSERT OR REPLACE handles this

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        await self.connect()

        await self.connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        await self.connection.commit()

    async def query_memories(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Query memories with filters."""
        await self.connect()

        query = "SELECT * FROM nodes WHERE 1=1"
        params = []

        if filters:
            if "status" in filters:
                query += " AND status = ?"
                params.append(filters["status"])

            if "type" in filters:
                query += " AND type = ?"
                params.append(filters["type"])

            if "created_after" in filters:
                query += " AND created_at > ?"
                params.append(filters["created_after"])

            if "created_before" in filters:
                query += " AND created_at < ?"
                params.append(filters["created_before"])

            if "access_count_lt" in filters:
                query += " AND access_count < ?"
                params.append(filters["access_count_lt"])

        if order_by:
            query += f" ORDER BY {order_by}"

        query += f" LIMIT {limit}"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def get_random_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[Memory]:
        """Get random memories for sampling."""
        await self.connect()

        query = "SELECT * FROM nodes WHERE 1=1"
        params = []

        if filters:
            if "status" in filters:
                query += " AND status = ?"
                params.append(filters["status"])

        query += " ORDER BY RANDOM() LIMIT ?"
        params.append(limit)

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    # ═══════════════════════════════════════════════════════════
    # EDGE OPERATIONS
    # ═══════════════════════════════════════════════════════════

    async def add_edge(self, edge: dict[str, Any] | Edge) -> str:
        """Add an edge between two nodes."""
        await self.connect()

        # Handle both dict and Edge object
        if isinstance(edge, Edge):
            source_id = edge.source
            target_id = edge.target
            edge_type = edge.type.value if isinstance(edge.type, RelationshipType) else edge.type
            confidence = edge.confidence
            metadata = edge.metadata
            created_at = edge.created_at.isoformat()
        else:
            source_id = edge["source"]
            target_id = edge["target"]
            edge_type = edge["type"]
            confidence = edge.get("metadata", {}).get("confidence", 1.0)
            metadata = edge.get("metadata", {})
            created_at = datetime.now().isoformat()

        cursor = await self.connection.execute(
            """
            INSERT INTO edges (source_id, target_id, type, confidence, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                target_id,
                edge_type,
                confidence,
                created_at,
                json.dumps(metadata),
            ),
        )

        await self.connection.commit()

        return str(cursor.lastrowid)

    async def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        """Get edge by ID."""
        await self.connect()

        cursor = await self.connection.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_edge(row)

    async def get_edge_between(
        self, source_id: str, target_id: str, relationship_type: str | None = None
    ) -> dict[str, Any] | None:
        """Find edge between two nodes."""
        await self.connect()

        query = "SELECT * FROM edges WHERE source_id = ? AND target_id = ?"
        params = [source_id, target_id]

        if relationship_type:
            query += " AND type = ?"
            params.append(relationship_type)

        query += " LIMIT 1"

        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()

        if not row:
            # Check reverse direction for undirected relationships
            if not relationship_type or relationship_type in ["CO_OCCURS", "SIMILAR_TO"]:
                cursor = await self.connection.execute(
                    "SELECT * FROM edges WHERE source_id = ? AND target_id = ? LIMIT 1",
                    (target_id, source_id),
                )
                row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_edge(row)

    async def update_edge(self, edge: dict[str, Any]) -> None:
        """Update edge metadata."""
        await self.connect()

        await self.connection.execute(
            """
            UPDATE edges
            SET confidence = ?, metadata = ?
            WHERE id = ?
            """,
            (
                edge.get("confidence", 1.0),
                json.dumps(edge.get("metadata", {})),
                edge["id"],
            ),
        )

        await self.connection.commit()

    async def delete_edge(self, edge_id: str) -> None:
        """Delete an edge."""
        await self.connect()

        await self.connection.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        await self.connection.commit()

    # ═══════════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════════════════

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = 100,
    ) -> list[tuple[Memory, Edge]]:
        """Get neighboring nodes."""
        await self.connect()

        if depth > 1:
            # Recursive traversal (simplified for now)
            return await self._get_neighbors_recursive(
                node_id, relationship_types, direction, depth, limit
            )

        # Single depth
        if direction == "outgoing":
            query = """
                SELECT n.*, e.id as edge_id, e.type as edge_type, e.metadata as edge_metadata,
                       e.source_id, e.confidence, e.created_at as edge_created
                FROM edges e
                JOIN nodes n ON e.target_id = n.id
                WHERE e.source_id = ?
            """
        elif direction == "incoming":
            query = """
                SELECT n.*, e.id as edge_id, e.type as edge_type, e.metadata as edge_metadata,
                       e.target_id, e.confidence, e.created_at as edge_created
                FROM edges e
                JOIN nodes n ON e.source_id = n.id
                WHERE e.target_id = ?
            """
        else:  # both
            query = """
                SELECT n.*, e.id as edge_id, e.type as edge_type, e.metadata as edge_metadata,
                       e.source_id, e.confidence, e.created_at as edge_created
                FROM edges e
                JOIN nodes n ON (e.target_id = n.id OR e.source_id = n.id)
                WHERE (e.source_id = ? OR e.target_id = ?) AND n.id != ?
            """

        params = [node_id]
        if direction == "both":
            params = [node_id, node_id, node_id]

        if relationship_types:
            placeholders = ",".join("?" * len(relationship_types))
            query += f" AND e.type IN ({placeholders})"
            params.extend(relationship_types)

        query += f" LIMIT {limit}"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            # Extract node columns
            node = self._row_to_memory(row[:17])  # First 17 columns are node data

            # Create proper Edge object
            edge = Edge(
                source=row[20] if direction == "outgoing" else node.id,  # source_id
                target=node.id if direction == "outgoing" else row[20],  # target_id
                type=RelationshipType(row[18]),  # edge_type
                confidence=row[21] if len(row) > 21 else 1.0,  # confidence
                created_at=(
                    datetime.fromisoformat(row[22]) if len(row) > 22 and row[22] else datetime.now()
                ),
                metadata=json.loads(row[19]) if row[19] else {},  # edge_metadata
            )

            results.append((node, edge))

        return results

    async def _get_neighbors_recursive(
        self,
        node_id: str,
        relationship_types: list[str] | None,
        direction: str,
        depth: int,
        limit: int,
    ) -> list[tuple[Memory, Edge]]:
        """Recursive neighbor traversal (simplified BFS)."""
        visited = set()
        results = []
        queue = [(node_id, 0)]

        while queue and len(results) < limit:
            current_id, current_depth = queue.pop(0)

            if current_id in visited or current_depth >= depth:
                continue

            visited.add(current_id)

            # Get immediate neighbors
            neighbors = await self.get_neighbors(
                current_id, relationship_types, direction, depth=1, limit=limit
            )

            for neighbor in neighbors:
                if neighbor["node"].id not in visited:
                    results.append(neighbor)
                    if current_depth + 1 < depth:
                        queue.append((neighbor["node"].id, current_depth + 1))

                if len(results) >= limit:
                    break

        return results

    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[str] | None:
        """Find shortest path between two nodes (BFS)."""
        await self.connect()

        if start_id == end_id:
            return [start_id]

        visited = set()
        queue = [(start_id, [start_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id in visited:
                continue

            visited.add(current_id)

            # Get neighbors
            neighbors = await self.get_neighbors(
                current_id, direction="outgoing", depth=1, limit=100
            )

            for neighbor in neighbors:
                neighbor_id = neighbor["node"].id

                if neighbor_id == end_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════

    async def count_nodes(self, filters: dict[str, Any] | None = None) -> int:
        """Count nodes matching filters."""
        await self.connect()

        query = "SELECT COUNT(*) FROM nodes WHERE 1=1"
        params = []

        if filters:
            if "status" in filters:
                query += " AND status = ?"
                params.append(filters["status"])

        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()

        return row[0] if row else 0

    async def count_edges(self, relationship_type: str | None = None) -> int:
        """Count edges."""
        await self.connect()

        query = "SELECT COUNT(*) FROM edges"
        params = []

        if relationship_type:
            query += " WHERE type = ?"
            params.append(relationship_type)

        cursor = await self.connection.execute(query, params)
        row = await cursor.fetchone()

        return row[0] if row else 0

    async def close(self) -> None:
        """Close the connection."""
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════

    def _row_to_memory(self, row: tuple) -> Memory:
        """Convert database row to Memory object."""
        embedding = json.loads(row[3]) if row[3] else []

        return Memory(
            id=row[0],
            content=row[1],
            type=NodeType(row[2]),
            embedding=embedding,
            version=row[4],
            parent_version=row[5],
            valid_from=datetime.fromisoformat(row[6]),
            valid_until=datetime.fromisoformat(row[7]) if row[7] else None,
            status=MemoryStatus(row[8]),
            superseded_by=row[9],
            invalidation_reason=row[10],
            confidence=row[11],
            access_count=row[12],
            last_accessed=datetime.fromisoformat(row[13]) if row[13] else None,
            created_at=datetime.fromisoformat(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
            metadata=json.loads(row[16]) if row[16] else {},
        )

    def _row_to_edge(self, row: tuple) -> dict[str, Any]:
        """Convert database row to edge dict."""
        return {
            "id": str(row[0]),
            "source": row[1],
            "target": row[2],
            "type": row[3],
            "confidence": row[4],
            "created_at": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }
