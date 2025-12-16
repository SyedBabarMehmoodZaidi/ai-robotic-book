"""
Base exception classes for the application.
"""


class RagQdrantException(Exception):
    """Base exception for the RAG Qdrant pipeline."""
    pass


class ContentExtractionError(RagQdrantException):
    """Raised when content extraction fails."""
    pass


class ChunkingError(RagQdrantException):
    """Raised when content chunking fails."""
    pass


class EmbeddingGenerationError(RagQdrantException):
    """Raised when embedding generation fails."""
    pass


class StorageError(RagQdrantException):
    """Raised when storage operations fail."""
    pass


class ValidationError(RagQdrantException):
    """Raised when validation fails."""
    pass


class ConfigurationError(RagQdrantException):
    """Raised when configuration is invalid."""
    pass


class APIClientError(RagQdrantException):
    """Raised when API client operations fail."""
    pass