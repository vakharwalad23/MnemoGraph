"""Neo4j-based graph store implementation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver

from src.models import Node, Edge, NodeType, RelationshipType, MemoryStatus
from .base import GraphStore


class Neo4jGraphStore(GraphStore):
    """Neo4j-based graph storage."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "mnemograph123"
    ):
        """
        Initialize Neo4j graph store.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[AsyncDriver] = None
    
    async def _get_driver(self) -> AsyncDriver:
        """Get or create Neo4j driver."""
        if self.driver is None:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self.driver
    
    async def initialize(self) -> None:
        """Create indexes and constraints."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            # Create constraints and indexes
            await session.run(
                "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX node_type IF NOT EXISTS FOR (n:Node) ON (n.type)"
            )
            await session.run(
                "CREATE INDEX node_status IF NOT EXISTS FOR (n:Node) ON (n.status)"
            )
    
    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        data: Dict[str, Any],
        status: MemoryStatus = MemoryStatus.NEW
    ) -> None:
        """Add a node to the graph."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            await session.run("""
                MERGE (n:Node {id: $id})
                SET n.type = $type,
                    n.data = $data,
                    n.status = $status,
                    n.created_at = $created_at,
                    n.access_count = COALESCE(n.access_count, 0)
            """, {
                "id": node_id,
                "type": node_type.value,
                "data": str(data),
                "status": status.value,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
    
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
        
        driver = await self._get_driver()
        edge_id = str(uuid4())
        
        async with driver.session() as session:
            # Use dynamic relationship type based on edge_type
            await session.run(f"""
                MATCH (a:Node {{id: $source_id}})
                MATCH (b:Node {{id: $target_id}})
                CREATE (a)-[r:{edge_type.value.upper()} {{
                    id: $edge_id,
                    weight: $weight,
                    metadata: $metadata,
                    created_at: $created_at
                }}]->(b)
            """, {
                "source_id": source_id,
                "target_id": target_id,
                "edge_id": edge_id,
                "weight": weight,
                "metadata": str(metadata) if metadata else None,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
        
        return edge_id
    
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            result = await session.run(
                "MATCH (n:Node {id: $id}) RETURN n",
                {"id": node_id}
            )
            
            record = await result.single()
            if not record:
                return None
            
            node_data = dict(record["n"])
            
            import ast
            return Node(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                data=ast.literal_eval(node_data["data"]),
                status=MemoryStatus(node_data["status"]),
                created_at=datetime.fromisoformat(node_data["created_at"]).replace(tzinfo=timezone.utc),
                last_accessed=datetime.fromisoformat(node_data["last_accessed"]) if node_data.get("last_accessed") else None,
                access_count=node_data.get("access_count", 0)
            )
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[RelationshipType]] = None,
        direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        driver = await self._get_driver()
        
        if direction == "outgoing":
            pattern = "(n)-[r]->(neighbor)"
        elif direction == "incoming":
            pattern = "(n)<-[r]-(neighbor)"
        else:  # both
            pattern = "(n)-[r]-(neighbor)"
        
        query = f"MATCH {pattern} WHERE n.id = $node_id"
        
        if edge_types:
            type_filter = " OR ".join([f"type(r) = '{et.value.upper()}'" for et in edge_types])
            query += f" AND ({type_filter})"
        
        query += " RETURN neighbor, type(r) as edge_type, r.weight as weight, r.id as edge_id, r.metadata as metadata"
        
        async with driver.session() as session:
            result = await session.run(query, {"node_id": node_id})
            
            neighbors = []
            async for record in result:
                node_data = dict(record["neighbor"])
                
                import ast
                neighbors.append({
                    "node": Node(
                        id=node_data["id"],
                        type=NodeType(node_data["type"]),
                        data=ast.literal_eval(node_data["data"]),
                        status=MemoryStatus(node_data["status"]),
                        created_at=datetime.fromisoformat(node_data["created_at"]).replace(tzinfo=timezone.utc),
                        last_accessed=datetime.fromisoformat(node_data["last_accessed"]) if node_data.get("last_accessed") else None,
                        access_count=node_data.get("access_count", 0)
                    ),
                    "edge_type": RelationshipType(record["edge_type"].lower()),
                    "edge_weight": record["weight"],
                    "edge_id": record["edge_id"],
                    "edge_metadata": ast.literal_eval(record["metadata"]) if record["metadata"] else None
                })
            
            return neighbors
    
    async def update_node_status(self, node_id: str, status: MemoryStatus) -> None:
        """Update node status."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            await session.run(
                "MATCH (n:Node {id: $id}) SET n.status = $status",
                {"id": node_id, "status": status.value}
            )
    
    async def update_node_access(
        self,
        node_id: str,
        access_count: int,
        last_accessed: datetime
    ) -> None:
        """Update node access tracking."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            await session.run("""
                MATCH (n:Node {id: $id})
                SET n.access_count = $access_count,
                    n.last_accessed = $last_accessed
            """, {
                "id": node_id,
                "access_count": access_count,
                "last_accessed": last_accessed.replace(tzinfo=timezone.utc).isoformat()
            })
    
    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update edge weight."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            await session.run(
                "MATCH ()-[r {id: $edge_id}]->() SET r.weight = $weight",
                {"edge_id": edge_id, "weight": weight}
            )
    
    async def mark_forgotten(self, node_id: str) -> None:
        """Mark a node as forgotten."""
        await self.update_node_status(node_id, MemoryStatus.FORGOTTEN)
    
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """Find path between two nodes."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            # Build query with max_depth properly
            query = f"""
                MATCH path = shortestPath(
                    (start:Node {{id: $start_id}})-[*..{max_depth}]->(end:Node {{id: $end_id}})
                )
                RETURN [node in nodes(path) | node.id] as node_ids
            """
            
            result = await session.run(query, {
                "start_id": start_id,
                "end_id": end_id
            })
            
            record = await result.single()
            if not record:
                return None
            
            return record["node_ids"]
    
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        driver = await self._get_driver()
        
        async with driver.session() as session:
            await session.run(
                "MATCH (n:Node {id: $id}) DETACH DELETE n",
                {"id": node_id}
            )
    
    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self.driver = None