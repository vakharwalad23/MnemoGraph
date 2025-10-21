"""SQLite-based graph store implementation."""

import aiosqlite
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.models import Node, Edge, NodeType, RelationshipType, MemoryStatus
from .base import GraphStore


class SQLiteGraphStore(GraphStore):
    """SQLite-based graph storage with optimized indexes."""
    
    def __init__(self, db_path: str = "mnemograph.db"):
        """
        Initialize SQLite graph store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
        return self.conn
    
    async def initialize(self) -> None:
        """Create tables and indexes."""
        conn = await self._get_connection()
        
        # Create nodes table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Create edges table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for fast lookups
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source, type)"
        )
        
        await conn.commit()
    
    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        data: Dict[str, Any],
        status: MemoryStatus = MemoryStatus.NEW
    ) -> None:
        """Add a node to the graph."""
        conn = await self._get_connection()
        
        await conn.execute("""
            INSERT OR REPLACE INTO nodes (id, type, data, status, created_at, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (
            node_id,
            node_type.value,
            json.dumps(data),
            status.value,
            datetime.now(timezone.utc).isoformat()
        ))
        
        await conn.commit()
    
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: RelationshipType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an edge between two nodes."""
        from uuid import uuid4
        
        conn = await self._get_connection()
        edge_id = str(uuid4())
        
        await conn.execute("""
            INSERT INTO edges (id, source, target, type, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            edge_id,
            source_id,
            target_id,
            edge_type.value,
            weight,
            json.dumps(metadata) if metadata else None,
            datetime.now(timezone.utc).isoformat()
        ))
        
        await conn.commit()
        return edge_id
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        conn = await self._get_connection()
        
        async with conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ) as cursor:
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return Node(
                id=row["id"],
                type=NodeType(row["type"]),
                data=json.loads(row["data"]),
                status=MemoryStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
                last_accessed=datetime.fromisoformat(row["last_accessed"]).replace(tzinfo=timezone.utc) if row["last_accessed"] else None,
                access_count=row["access_count"]
            )
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[RelationshipType]] = None,
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        conn = await self._get_connection()
        
        if direction == "outgoing":
            query = """
                SELECT n.*, e.type as edge_type, e.weight, e.id as edge_id
                FROM edges e
                JOIN nodes n ON e.target = n.id
                WHERE e.source = ?
            """
            params = [node_id]
        elif direction == "incoming":
            query = """
                SELECT n.*, e.type as edge_type, e.weight, e.id as edge_id
                FROM edges e
                JOIN nodes n ON e.source = n.id
                WHERE e.target = ?
            """
            params = [node_id]
        else:  # both
            query = """
                SELECT n.*, e.type as edge_type, e.weight, e.id as edge_id
                FROM edges e
                JOIN nodes n ON (e.target = n.id OR e.source = n.id)
                WHERE (e.source = ? OR e.target = ?) AND n.id != ?
            """
            params = [node_id, node_id, node_id]
        
        if edge_types:
            type_values = [et.value for et in edge_types]
            placeholders = ",".join("?" * len(type_values))
            query += f" AND e.type IN ({placeholders})"
            params.extend(type_values)
        
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            
            return [
                {
                    "node": Node(
                        id=row["id"],
                        type=NodeType(row["type"]),
                        data=json.loads(row["data"]),
                        status=MemoryStatus(row["status"]),
                        created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
                        last_accessed=datetime.fromisoformat(row["last_accessed"]).replace(tzinfo=timezone.utc) if row["last_accessed"] else None,
                        access_count=row["access_count"]
                    ),
                    "edge_type": RelationshipType(row["edge_type"]),
                    "edge_weight": row["weight"],
                    "edge_id": row["edge_id"]
                }
                for row in rows
            ]
    
    async def update_node_status(self, node_id: str, status: MemoryStatus) -> None:
        """Update node status."""
        conn = await self._get_connection()
        
        await conn.execute(
            "UPDATE nodes SET status = ? WHERE id = ?",
            (status.value, node_id)
        )
        
        await conn.commit()
    
    async def update_node_access(
        self,
        node_id: str,
        access_count: int,
        last_accessed: datetime
    ) -> None:
        """Update node access tracking."""
        conn = await self._get_connection()
        
        await conn.execute("""
            UPDATE nodes 
            SET access_count = ?, last_accessed = ?
            WHERE id = ?
        """, (access_count, last_accessed.replace(tzinfo=timezone.utc).isoformat(), node_id))
        
        await conn.commit()
    
    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update edge weight."""
        conn = await self._get_connection()
        
        await conn.execute(
            "UPDATE edges SET weight = ? WHERE id = ?",
            (weight, edge_id)
        )
        
        await conn.commit()
    
    async def mark_forgotten(self, node_id: str) -> None:
        """Mark a node as forgotten."""
        await self.update_node_status(node_id, MemoryStatus.FORGOTTEN)
    
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """Find path between two nodes using BFS."""
        conn = await self._get_connection()
        
        # Simple BFS implementation
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current_id == end_id:
                return path
            
            # Get neighbors
            async with conn.execute(
                "SELECT target FROM edges WHERE source = ?",
                (current_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                
                for row in rows:
                    neighbor_id = row["target"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        conn = await self._get_connection()
        
        # Delete edges first (due to foreign key constraints)
        await conn.execute(
            "DELETE FROM edges WHERE source = ? OR target = ?",
            (node_id, node_id)
        )
        
        # Delete node
        await conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        
        await conn.commit()
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None