from typing import Optional


class RAGException(Exception):
    """
    Base exception class for RAG retrieval application.
    All custom exceptions should inherit from this class.
    """
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "RAG_ERROR"
        self.details = details or {}

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class RetrievalException(RAGException):
    """
    Exception raised when there are issues with the retrieval process.
    """
    def __init__(self, message: str, error_code: Optional[str] = "RETRIEVAL_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class ValidationException(RAGException):
    """
    Exception raised when there are issues with the validation process.
    """
    def __init__(self, message: str, error_code: Optional[str] = "VALIDATION_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class ConfigurationException(RAGException):
    """
    Exception raised when there are issues with application configuration.
    """
    def __init__(self, message: str, error_code: Optional[str] = "CONFIG_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class QdrantConnectionException(RAGException):
    """
    Exception raised when there are connection issues with Qdrant.
    """
    def __init__(self, message: str, error_code: Optional[str] = "QDRANT_CONNECTION_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class CohereAPIException(RAGException):
    """
    Exception raised when there are issues with the Cohere API.
    """
    def __init__(self, message: str, error_code: Optional[str] = "COHERE_API_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class InvalidQueryException(RAGException):
    """
    Exception raised when a query is invalid or malformed.
    """
    def __init__(self, message: str, error_code: Optional[str] = "INVALID_QUERY_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)