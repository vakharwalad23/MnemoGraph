"""
Custom exception hierarchy for MnemoGraph.

Provides structured error types for better error handling and debugging.
All exceptions inherit from MnemoGraphError for easy catching.
"""


class MnemoGraphError(Exception):
    """
    Base exception for all MnemoGraph errors.
    All custom exceptions should inherit from this class.
    """

    def __init__(self, message: str, context: dict | None = None):
        """
        Initialize MnemoGraph error.
        Args:
            message: Error message
            context: Optional context dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}


class StoreError(MnemoGraphError):
    """
    Base exception for store operations.
    Used for errors related to data storage operations.
    """

    pass


class VectorStoreError(StoreError):
    """
    Vector store operation errors.
    Raised when vector database operations fail.
    """

    pass


class GraphStoreError(StoreError):
    """
    Graph store operation errors.
    Raised when graph database operations fail.
    """

    pass


class SyncError(StoreError):
    """
    Synchronization errors between stores.
    Raised when synchronization between vector and graph stores fails.
    """

    pass


class ValidationError(MnemoGraphError):
    """
    Validation errors.
    Raised when input validation fails or data is invalid.
    """

    pass


class NotFoundError(MnemoGraphError):
    """
    Resource not found errors.
    Raised when a requested resource (memory, edge, etc.) doesn't exist.
    """

    pass


class ConfigurationError(MnemoGraphError):
    """
    Configuration errors.
    Raised when configuration is invalid or missing required values.
    """

    pass


class EmbeddingError(MnemoGraphError):
    """
    Embedding generation errors.
    Raised when embedding generation fails.
    """

    pass


class LLMError(MnemoGraphError):
    """
    LLM operation errors.
    Raised when LLM operations fail (API errors, timeouts, etc.).
    """

    pass


class MemoryError(MnemoGraphError):
    """
    Memory-specific errors.
    Raised for memory-related operation failures.
    """

    pass
