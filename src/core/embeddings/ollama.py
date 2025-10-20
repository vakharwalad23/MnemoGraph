"""Ollama embedding provider using official Ollama SDK."""

from typing import List, Optional
import ollama
from .base import EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Ollama-based embedding provider using official SDK."""
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model: Name of the Ollama model to use
            host: Host URL of the Ollama service
        """
        self.model = model
        self.host = host
        self._dimension: Optional[int] = None
        # Initialize client with custom host
        self.client = ollama.Client(host=host)
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        embedding = response["embedding"]
        
        # Cache dimension
        if self._dimension is None:
            self._dimension = len(embedding)
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            embedding = response["embedding"]
            embeddings.append(embedding)
            
            # Cache dimension from first embedding
            if self._dimension is None and embedding:
                self._dimension = len(embedding)
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        if self._dimension is None:
            # Default dimension for nomic-embed-text
            return 768 if "nomic" in self.model else 384
        return self._dimension