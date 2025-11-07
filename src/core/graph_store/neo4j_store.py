"""
Neo4j graph store implementation - redesigned for Phase 2 & 3.

Production-grade graph database for complex relationship queries and graph analytics.
"""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.core.graph_store.base import GraphStore
from src.models.memory import Memory, MemoryStatus, NodeType
from src.models.relationships import Edge, RelationshipType
from src.utils.exceptions import GraphStoreError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jGraphStore(GraphStore):
    """
    Neo4j-based graph store for memories and relationships.

    Features:
    - Production-grade graph database
    - Native graph traversal
    - Cypher query language
    - ACID transactions
    - Distributed graph analytics
    - Fast path finding
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j graph store.

        Args:
            uri: Neo4j connection URI
            username: Username
            password: Password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """
        Establish connection to Neo4j.

        Raises:
            GraphStoreError: If connection fails
        """
        if self.driver is None:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to Neo4j: {e}",
                    extra={"uri": self.uri, "error": str(e)},
                )
                raise GraphStoreError(f"Failed to connect to Neo4j: {e}") from e

    async def initialize(self) -> None:
        """
        Create indexes and constraints.

        Raises:
            GraphStoreError: If initialization fails
        """
        try:
            await self.connect()

            async with self.driver.session(database=self.database) as session:
                # Create constraint on Memory ID (unique)
                await session.run(
                    "CREATE CONSTRAINT memory_id IF NOT EXISTS "
                    "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
                )

                # Create indices for fast lookups
                await session.run(
                    "CREATE INDEX memory_status IF NOT EXISTS " "FOR (m:Memory) ON (m.status)"
                )

                await session.run(
                    "CREATE INDEX memory_type IF NOT EXISTS " "FOR (m:Memory) ON (m.type)"
                )

                await session.run(
                    "CREATE INDEX memory_created IF NOT EXISTS " "FOR (m:Memory) ON (m.created_at)"
                )

                await session.run(
                    "CREATE INDEX memory_version IF NOT EXISTS " "FOR (m:Memory) ON (m.version)"
                )
        except Exception as e:
            logger.error(
                f"Failed to initialize Neo4j: {e}",
                extra={"database": self.database, "error": str(e)},
            )
            raise GraphStoreError(f"Failed to initialize Neo4j: {e}") from e

    # NODE OPERATIONS

    async def add_node(self, memory: Memory) -> None:
        """
        Add a memory node to the graph.

        Args:
            memory: Memory object to store

        Raises:
            ValidationError: If memory is invalid
            GraphStoreError: If add operation fails
        """
        if not memory:
            raise ValidationError("Memory cannot be None")
        if not memory.id:
            raise ValidationError("Memory ID cannot be empty")

        try:
            await self.connect()

            async with self.driver.session(database=self.database) as session:
                await session.run(
                    """
                    MERGE (m:Memory {id: $id})
                    SET m.content = $content,
                        m.type = $type,
                        m.version = $version,
                        m.parent_version = $parent_version,
                        m.valid_from = $valid_from,
                        m.valid_until = $valid_until,
                        m.status = $status,
                        m.superseded_by = $superseded_by,
                        m.invalidation_reason = $invalidation_reason,
                        m.confidence = $confidence,
                        m.access_count = $access_count,
                        m.last_accessed = $last_accessed,
                        m.created_at = $created_at,
                        m.updated_at = $updated_at,
                        m.metadata = $metadata
                    """,
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "type": memory.type.value,
                        "version": memory.version,
                        "parent_version": memory.parent_version,
                        "valid_from": memory.valid_from.isoformat(),
                        "valid_until": (
                            memory.valid_until.isoformat() if memory.valid_until else None
                        ),
                        "status": memory.status.value,
                        "superseded_by": memory.superseded_by,
                        "invalidation_reason": memory.invalidation_reason,
                        "confidence": memory.confidence,
                        "access_count": memory.access_count,
                        "last_accessed": (
                            memory.last_accessed.isoformat() if memory.last_accessed else None
                        ),
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat(),
                        "metadata": json.dumps(memory.metadata),
                    },
                )
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to add node {memory.id}: {e}",
                extra={"memory_id": memory.id, "error": str(e)},
            )
            raise GraphStoreError(f"Failed to add node: {e}") from e

    async def get_node(self, node_id: str) -> Memory | None:
        """
        Retrieve a memory node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Memory or None if not found

        Raises:
            ValidationError: If node_id is invalid
            GraphStoreError: If retrieval operation fails
        """
        if not node_id or not node_id.strip():
            raise ValidationError("Node ID cannot be empty")

        try:
            await self.connect()

            async with self.driver.session(database=self.database) as session:
                result = await session.run("MATCH (m:Memory {id: $id}) RETURN m", {"id": node_id})

                record = await result.single()
                if not record:
                    return None

                node = record["m"]
                return self._node_to_memory(node)
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get node {node_id}: {e}",
                extra={"node_id": node_id, "error": str(e)},
            )
            raise GraphStoreError(f"Failed to get node: {e}") from e

    async def update_node(self, memory: Memory) -> None:
        """Update an existing memory node."""
        await self.add_node(memory)  # MERGE handles updates

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        await self.connect()

        async with self.driver.session(database=self.database) as session:
            await session.run("MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": node_id})

    async def query_memories(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Query memories with filters."""
        await self.connect()

        query = "MATCH (m:Memory) WHERE 1=1"
        params = {}

        if filters:
            if "status" in filters:
                query += " AND m.status = $status"
                params["status"] = filters["status"]

            if "type" in filters:
                query += " AND m.type = $type"
                params["type"] = filters["type"]

            if "created_after" in filters:
                query += " AND m.created_at > $created_after"
                params["created_after"] = filters["created_after"]

            if "created_before" in filters:
                query += " AND m.created_at < $created_before"
                params["created_before"] = filters["created_before"]

            if "access_count_lt" in filters:
                query += " AND m.access_count < $access_count_lt"
                params["access_count_lt"] = filters["access_count_lt"]

        query += " RETURN m"

        if order_by:
            # Parse order_by (e.g., "created_at DESC")
            query += f" ORDER BY m.{order_by}"

        query += f" LIMIT {limit}"

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)

            memories = []
            async for record in result:
                memories.append(self._node_to_memory(record["m"]))

            return memories

    async def get_random_memories(
        self, filters: dict[str, Any] | None = None, limit: int = 10
    ) -> list[Memory]:
        """Get random memories for sampling."""
        await self.connect()

        query = "MATCH (m:Memory) WHERE 1=1"
        params = {}

        if filters:
            if "status" in filters:
                query += " AND m.status = $status"
                params["status"] = filters["status"]

        query += f" RETURN m ORDER BY rand() LIMIT {limit}"

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)

            memories = []
            async for record in result:
                memories.append(self._node_to_memory(record["m"]))

            return memories

    # EDGE OPERATIONS

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

        edge_id = str(uuid4())

        async with self.driver.session(database=self.database) as session:
            # Use dynamic relationship type
            await session.run(
                f"""
                MATCH (a:Memory {{id: $source_id}})
                MATCH (b:Memory {{id: $target_id}})
                CREATE (a)-[r:{edge_type.upper()} {{
                    id: $edge_id,
                    confidence: $confidence,
                    created_at: $created_at,
                    metadata: $metadata
                }}]->(b)
                """,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_id": edge_id,
                    "confidence": confidence,
                    "created_at": created_at,
                    "metadata": json.dumps(metadata),
                },
            )

        return edge_id

    async def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        """Get edge by ID."""
        await self.connect()

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (a)-[r {id: $edge_id}]->(b)
                RETURN r, type(r) as edge_type, a.id as source, b.id as target
                """,
                {"edge_id": edge_id},
            )

            record = await result.single()
            if not record:
                return None

            rel = record["r"]
            return {
                "id": rel["id"],
                "source": record["source"],
                "target": record["target"],
                "type": record["edge_type"],
                "confidence": rel.get("confidence", 1.0),
                "created_at": rel["created_at"],
                "metadata": json.loads(rel.get("metadata", "{}")),
            }

    async def get_edge_between(
        self, source_id: str, target_id: str, relationship_type: str | None = None
    ) -> dict[str, Any] | None:
        """Find edge between two nodes."""
        await self.connect()

        query = """
        MATCH (a:Memory {id: $source_id})-[r]->(b:Memory {id: $target_id})
        """

        params = {"source_id": source_id, "target_id": target_id}

        if relationship_type:
            query = f"""
            MATCH (a:Memory {{id: $source_id}})-[r:{relationship_type.upper()}]->(b:Memory {{id: $target_id}})
            """

        query += " RETURN r, type(r) as edge_type LIMIT 1"

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)

            record = await result.single()
            if not record:
                # Check reverse for undirected relationships
                if not relationship_type or relationship_type in ["CO_OCCURS", "SIMILAR_TO"]:
                    result = await session.run(
                        """
                        MATCH (a:Memory {id: $target_id})-[r]->(b:Memory {id: $source_id})
                        RETURN r, type(r) as edge_type LIMIT 1
                        """,
                        params,
                    )
                    record = await result.single()

            if not record:
                return None

            rel = record["r"]
            return {
                "id": rel["id"],
                "source": source_id,
                "target": target_id,
                "type": record["edge_type"],
                "confidence": rel.get("confidence", 1.0),
                "created_at": rel["created_at"],
                "metadata": json.loads(rel.get("metadata", "{}")),
            }

    async def update_edge(self, edge: dict[str, Any]) -> None:
        """Update edge metadata."""
        await self.connect()

        async with self.driver.session(database=self.database) as session:
            await session.run(
                """
                MATCH ()-[r {id: $edge_id}]->()
                SET r.confidence = $confidence,
                    r.metadata = $metadata
                """,
                {
                    "edge_id": edge["id"],
                    "confidence": edge.get("confidence", 1.0),
                    "metadata": json.dumps(edge.get("metadata", {})),
                },
            )

    async def delete_edge(self, edge_id: str) -> None:
        """Delete an edge."""
        await self.connect()

        async with self.driver.session(database=self.database) as session:
            await session.run("MATCH ()-[r {id: $edge_id}]->() DELETE r", {"edge_id": edge_id})

    # GRAPH TRAVERSAL

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

        # Build pattern based on direction
        if direction == "outgoing":
            pattern = "(m)-[r]->(n)"
        elif direction == "incoming":
            pattern = "(m)<-[r]-(n)"
        else:  # both
            pattern = "(m)-[r]-(n)"

        # Build relationship type filter
        if relationship_types:
            rel_types = "|".join([rt.upper() for rt in relationship_types])
            pattern = pattern.replace("-[r]-", f"-[r:{rel_types}]-")

        # Build query with depth
        if depth > 1:
            pattern = pattern.replace("[r]", f"[r*1..{depth}]")

        # Include type(r) in the return to get the relationship type
        query = (
            f"MATCH {pattern} WHERE m.id = $node_id RETURN n, r, type(r) as rel_type LIMIT {limit}"
        )

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, {"node_id": node_id})

            neighbors = []
            async for record in result:
                node = record["n"]
                rel = record["r"]
                rel_type = record["rel_type"]

                # Handle both single relationship and path
                if isinstance(rel, list):
                    # Path with multiple relationships - use first relationship type
                    edge_info = Edge(
                        source=node_id,
                        target=node["id"],
                        type=RelationshipType(rel_type),
                        confidence=rel[0].get("confidence", 1.0) if rel else 1.0,
                        created_at=datetime.fromisoformat(
                            rel[0].get("created_at", datetime.now().isoformat())
                            if rel
                            else datetime.now().isoformat()
                        ),
                        metadata={},
                    )
                else:
                    # Single relationship
                    edge_info = Edge(
                        source=node_id,
                        target=node["id"],
                        type=RelationshipType(rel_type),
                        confidence=rel.get("confidence", 1.0),
                        created_at=datetime.fromisoformat(
                            rel.get("created_at", datetime.now().isoformat())
                        ),
                        metadata=json.loads(rel.get("metadata", "{}")),
                    )

                neighbors.append((self._node_to_memory(node), edge_info))

            return neighbors

    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list[str] | None:
        """Find shortest path between two nodes."""
        await self.connect()

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                f"""
                MATCH path = shortestPath(
                    (start:Memory {{id: $start_id}})-[*..{max_depth}]-(end:Memory {{id: $end_id}})
                )
                RETURN [node in nodes(path) | node.id] as node_ids
                """,
                {"start_id": start_id, "end_id": end_id},
            )

            record = await result.single()
            if not record:
                return None

            return record["node_ids"]

    # UTILITY METHODS

    async def count_nodes(self, filters: dict[str, Any] | None = None) -> int:
        """Count nodes matching filters."""
        await self.connect()

        query = "MATCH (m:Memory) WHERE 1=1"
        params = {}

        if filters:
            if "status" in filters:
                query += " AND m.status = $status"
                params["status"] = filters["status"]

        query += " RETURN count(m) as count"

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)
            record = await result.single()
            return record["count"] if record else 0

    async def count_edges(self, relationship_type: str | None = None) -> int:
        """Count edges."""
        await self.connect()

        if relationship_type:
            query = f"MATCH ()-[r:{relationship_type.upper()}]->() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) as count"

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["count"] if record else 0

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver is not None:
            await self.driver.close()
            self.driver = None

    # HELPER METHODS

    def _node_to_memory(self, node) -> Memory:
        """Convert Neo4j node to Memory object."""
        return Memory(
            id=node["id"],
            content=node["content"],
            type=NodeType(node["type"]),
            embedding=[],  # Embeddings stored in vector store
            version=node.get("version", 1),
            parent_version=node.get("parent_version"),
            valid_from=datetime.fromisoformat(node["valid_from"]),
            valid_until=(
                datetime.fromisoformat(node["valid_until"]) if node.get("valid_until") else None
            ),
            status=MemoryStatus(node["status"]),
            superseded_by=node.get("superseded_by"),
            invalidation_reason=node.get("invalidation_reason"),
            confidence=node.get("confidence", 1.0),
            access_count=node.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(node["last_accessed"]) if node.get("last_accessed") else None
            ),
            created_at=datetime.fromisoformat(node["created_at"]),
            updated_at=datetime.fromisoformat(node["updated_at"]),
            metadata=json.loads(node.get("metadata", "{}")),
        )
