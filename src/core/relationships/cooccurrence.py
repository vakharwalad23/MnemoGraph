"""Entity co-occurrence relationship inference engine."""

import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter

from src.models import RelationshipType
from src.core.vector_store import QdrantStore
from src.core.graph_store import GraphStore


class EntityCooccurrenceEngine:
    """
    Engine for inferring co-occurrence relationships between memories.
    
    Links memories that share common entities (people, places, concepts).
    Mimics brain's associative memory - connecting experiences through shared elements.
    """
    
    def __init__(
        self,
        vector_store: QdrantStore,
        graph_store: GraphStore,
        min_entity_length: int = 3,
        min_cooccurrence_count: int = 2,
        entity_weight_threshold: float = 0.3,
        use_spacy: bool = True,
    ):
        """
        Initialize entity co-occurrence engine.
        
        Args:
            vector_store: Vector store for memory retrieval
            graph_store: Graph store for creating co-occurrence edges
            min_entity_length: Minimum characters for entity extraction
            min_cooccurrence_count: Minimum shared entities to create edge
            entity_weight_threshold: Weight threshold for edge creation
            use_spacy: Whether to use spaCy NER (recommended for production)
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.min_entity_length = min_entity_length
        self.min_cooccurrence_count = min_cooccurrence_count
        self.entity_weight_threshold = entity_weight_threshold
        self.use_spacy = use_spacy
        
        # Initialize spaCy if requested
        self.nlp = None
        if self.use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available, falling back to simple extraction")
                self.use_spacy = False
    
    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities from text using NER.
        
        Uses spaCy for Named Entity Recognition if available,
        otherwise falls back to simple pattern matching.
        
        Extracts:
        - People (PERSON)
        - Organizations (ORG)
        - Locations (GPE, LOC)
        - Products (PRODUCT)
        - Technologies and concepts (via noun chunks)
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted entities
        """
        entities = set()
        
        if self.use_spacy and self.nlp:
            # Use spaCy NER - much more accurate!
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'NORP', 'FAC']:
                    entity_text = ent.text.lower().strip()
                    if len(entity_text) >= self.min_entity_length:
                        entities.add(entity_text)
                        # Also add individual words from multi-word entities
                        words = entity_text.split()
                        for word in words:
                            if len(word) >= self.min_entity_length:
                                entities.add(word)
            
            # Extract noun chunks (potential concepts/technologies)
            for chunk in doc.noun_chunks:
                # Filter for meaningful noun phrases
                if len(chunk.text) >= self.min_entity_length:
                    # Skip if starts with determiner
                    if chunk.root.pos_ in ['NOUN', 'PROPN']:
                        entity_text = chunk.text.lower().strip()
                        entities.add(entity_text)
                        # Also add individual important words from noun chunks
                        # Extract proper nouns and nouns from the chunk
                        for token in chunk:
                            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) >= self.min_entity_length:
                                entities.add(token.text.lower())
        else:
            # Fallback: Simple capitalized word extraction
            capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            capitalized_matches = re.findall(capitalized_pattern, text)
            
            for match in capitalized_matches:
                if len(match) >= self.min_entity_length:
                    entities.add(match.lower())
                    # Also add individual words from multi-word matches
                    words = match.split()
                    if len(words) > 1:
                        for word in words:
                            if len(word) >= self.min_entity_length:
                                entities.add(word.lower())
        
        return entities
    
    async def detect_cooccurrence(
        self,
        memory_id: str,
        entities: Set[str],
        create_edges: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect memories that share entities with the given memory.
        
        Args:
            memory_id: Memory identifier
            entities: Set of entities in this memory
            create_edges: Whether to create CO_OCCURS edges
            
        Returns:
            List of co-occurring memories with shared entities
        """
        if not entities:
            return []
        
        # Get all memories and their entities
        # In production, we should maintain an entity index for efficiency
        cooccurring = []
        
        # For now, we'll check other memories by fetching their text
        # This is simplified - in production we should use an entity index
        node = await self.graph_store.get_node(memory_id)
        if not node:
            return []
        
        # Get neighbors to find other memories
        # Note: This is a simplified approach
        # In production, maintain an inverted index: entity -> [memory_ids]
        
        return cooccurring
    
    async def build_entity_index(
        self,
        memory_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Build an inverted index of entities to memories.
        
        Args:
            memory_ids: List of memory IDs to index
            
        Returns:
            Dictionary mapping entities to memory IDs
        """
        entity_index = defaultdict(list)
        
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if not node:
                continue
            
            text = node.data.get("text", "")
            entities = self.extract_entities(text)
            
            for entity in entities:
                entity_index[entity].append(mem_id)
        
        return dict(entity_index)
    
    async def create_cooccurrence_edges(
        self,
        memory_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Create co-occurrence edges between memories sharing entities.
        
        Builds entity index and creates edges between memories
        that share significant entity overlap.
        
        Args:
            memory_ids: List of memory IDs to process
            
        Returns:
            Statistics about edges created
        """
        # Build entity index
        entity_index = await self.build_entity_index(memory_ids)
        
        # Track which memory pairs we've already connected
        connected_pairs = set()
        edges_created = 0
        entity_stats = defaultdict(int)
        
        # For each entity, connect memories that share it
        for entity, mem_ids in entity_index.items():
            if len(mem_ids) < 2:
                continue
            
            entity_stats[entity] = len(mem_ids)
            
            # Create edges between all pairs sharing this entity
            for i, mem_id1 in enumerate(mem_ids):
                for mem_id2 in mem_ids[i + 1:]:
                    pair = tuple(sorted([mem_id1, mem_id2]))
                    
                    if pair in connected_pairs:
                        continue
                    
                    # Count shared entities between this pair
                    node1 = await self.graph_store.get_node(mem_id1)
                    node2 = await self.graph_store.get_node(mem_id2)
                    
                    if not (node1 and node2):
                        continue
                    
                    entities1 = self.extract_entities(node1.data.get("text", ""))
                    entities2 = self.extract_entities(node2.data.get("text", ""))
                    
                    shared_entities = entities1 & entities2
                    
                    if len(shared_entities) >= self.min_cooccurrence_count:
                        # Calculate edge weight based on Jaccard similarity
                        weight = len(shared_entities) / len(entities1 | entities2)
                        
                        if weight >= self.entity_weight_threshold:
                            await self.graph_store.add_edge(
                                source_id=mem_id1,
                                target_id=mem_id2,
                                edge_type=RelationshipType.CO_OCCURS,
                                weight=weight,
                                metadata={
                                    "shared_entities": list(shared_entities),
                                    "num_shared": len(shared_entities),
                                    "jaccard_similarity": weight
                                }
                            )
                            
                            connected_pairs.add(pair)
                            edges_created += 1
        
        return {
            "edges_created": edges_created,
            "unique_entities": len(entity_index),
            "entity_stats": dict(entity_stats),
            "memory_pairs": len(connected_pairs)
        }
    
    async def find_memories_by_entity(
        self,
        entity: str,
        memory_ids: List[str]
    ) -> List[str]:
        """
        Find all memories that mention a specific entity.
        
        Args:
            entity: Entity to search for
            memory_ids: List of memory IDs to search
            
        Returns:
            List of memory IDs containing the entity
        """
        matching_memories = []
        entity_lower = entity.lower()
        
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if not node:
                continue
            
            text = node.data.get("text", "")
            entities = self.extract_entities(text)
            
            if entity_lower in entities:
                matching_memories.append(mem_id)
        
        return matching_memories
    
    def calculate_entity_importance(
        self,
        entity_stats: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate importance scores for entities based on frequency.
        
        Uses TF-IDF-like scoring: entities that appear in few memories
        but multiple times are more important than ubiquitous entities.
        
        Args:
            entity_stats: Dictionary of entity -> occurrence count
            
        Returns:
            Dictionary of entity -> importance score
        """
        total_memories = sum(entity_stats.values())
        importance = {}
        
        for entity, count in entity_stats.items():
            # Simple importance: inverse frequency
            # Less common entities are more distinctive
            importance[entity] = 1.0 / (1.0 + count / total_memories)
        
        return importance
    
    async def get_entity_graph(
        self,
        memory_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Build a graph representation of entity relationships.
        
        Args:
            memory_ids: List of memory IDs
            
        Returns:
            Entity graph with nodes and edges
        """
        entity_index = await self.build_entity_index(memory_ids)
        
        # Build entity co-occurrence matrix
        # Entities that appear together in memories are related
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for mem_id in memory_ids:
            node = await self.graph_store.get_node(mem_id)
            if not node:
                continue
            
            entities = list(self.extract_entities(node.data.get("text", "")))
            
            # Count co-occurrences
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1:]:
                    entity_cooccurrence[entity1][entity2] += 1
                    entity_cooccurrence[entity2][entity1] += 1
        
        # Build graph structure
        nodes = [
            {
                "entity": entity,
                "memory_count": len(mem_ids),
                "importance": 1.0 / (1.0 + len(mem_ids))
            }
            for entity, mem_ids in entity_index.items()
        ]
        
        edges = []
        for entity1, related in entity_cooccurrence.items():
            for entity2, count in related.items():
                if entity1 < entity2:  # Avoid duplicates
                    edges.append({
                        "source": entity1,
                        "target": entity2,
                        "weight": count
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "num_entities": len(nodes),
            "num_relationships": len(edges)
        }
    
    async def find_entity_clusters(
        self,
        memory_ids: List[str],
        min_cluster_size: int = 3
    ) -> List[Set[str]]:
        """
        Find clusters of related entities.
        
        Groups entities that frequently co-occur together.
        
        Args:
            memory_ids: List of memory IDs
            min_cluster_size: Minimum entities per cluster
            
        Returns:
            List of entity clusters (sets of related entities)
        """
        entity_graph = await self.get_entity_graph(memory_ids)
        
        # Simple clustering: connected components in entity graph
        # In production, use more sophisticated community detection
        
        adjacency = defaultdict(set)
        for edge in entity_graph["edges"]:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])
        
        # Find connected components
        visited = set()
        clusters = []
        
        for entity in adjacency.keys():
            if entity in visited:
                continue
            
            # BFS to find connected component
            cluster = set()
            queue = [entity]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters