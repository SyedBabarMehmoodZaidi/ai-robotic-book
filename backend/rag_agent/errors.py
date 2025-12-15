from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ErrorCode(str, Enum):
    """
    Enum for standardized error codes.
    """
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    OPENAI_API_ERROR = "OPENAI_API_ERROR"
    CONTEXT_TOO_LONG = "CONTEXT_TOO_LONG"
    NO_CONTEXT_FOUND = "NO_CONTEXT_FOUND"
    HALLUCINATION_DETECTED = "HALLUCINATION_DETECTED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIError(BaseModel):
    """
    Standardized error response model.
    """
    error_code: ErrorCode
    message: str
    details: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None

    class Config:
        use_enum_values = True


class RAGAgentError(Exception):
    """
    Base exception class for RAG agent errors.
    """
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details


class ValidationError(RAGAgentError):
    """
    Exception raised for validation errors.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)


class RetrievalError(RAGAgentError):
    """
    Exception raised for retrieval pipeline errors.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.RETRIEVAL_ERROR, details)


class AgentError(RAGAgentError):
    """
    Exception raised for agent processing errors.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.AGENT_ERROR, details)


class OpenAIError(RAGAgentError):
    """
    Exception raised for OpenAI API errors.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.OPENAI_API_ERROR, details)


class ContextError(RAGAgentError):
    """
    Exception raised for context-related errors.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.CONTEXT_TOO_LONG if "too long" in message.lower() else ErrorCode.NO_CONTEXT_FOUND, details)


class HallucinationError(RAGAgentError):
    """
    Exception raised when hallucination is detected.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.HALLUCINATION_DETECTED, details)


class RateLimitError(RAGAgentError):
    """
    Exception raised when rate limit is exceeded.
    """
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorCode.RATE_LIMIT_EXCEEDED, details)