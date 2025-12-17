from typing import Optional


class AgentException(Exception):
    """
    Base exception class for RAG agent application.
    All custom agent exceptions should inherit from this class.
    """
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AGENT_ERROR"
        self.details = details or {}

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class AgentInitializationException(AgentException):
    """
    Exception raised when there are issues initializing the AI agent.
    """
    def __init__(self, message: str, error_code: Optional[str] = "AGENT_INIT_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class RetrievalIntegrationException(AgentException):
    """
    Exception raised when there are issues with the retrieval pipeline integration.
    """
    def __init__(self, message: str, error_code: Optional[str] = "RETRIEVAL_INTEGRATION_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class QueryProcessingException(AgentException):
    """
    Exception raised when there are issues processing agent queries.
    """
    def __init__(self, message: str, error_code: Optional[str] = "QUERY_PROCESSING_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class ResponseGenerationException(AgentException):
    """
    Exception raised when there are issues generating agent responses.
    """
    def __init__(self, message: str, error_code: Optional[str] = "RESPONSE_GENERATION_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class HallucinationPreventionException(AgentException):
    """
    Exception raised when hallucination prevention mechanisms detect invalid responses.
    """
    def __init__(self, message: str, error_code: Optional[str] = "HALLUCINATION_PREVENTION_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class ConfigurationException(AgentException):
    """
    Exception raised when there are issues with application configuration.
    """
    def __init__(self, message: str, error_code: Optional[str] = "CONFIG_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class OpenAIServiceException(AgentException):
    """
    Exception raised when there are issues with the OpenAI service.
    """
    def __init__(self, message: str, error_code: Optional[str] = "OPENAI_SERVICE_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class RetrievalServiceException(AgentException):
    """
    Exception raised when there are issues with the retrieval service.
    """
    def __init__(self, message: str, error_code: Optional[str] = "RETRIEVAL_SERVICE_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)


class InvalidQueryException(AgentException):
    """
    Exception raised when a query is invalid or malformed.
    """
    def __init__(self, message: str, error_code: Optional[str] = "INVALID_QUERY_ERROR", details: Optional[dict] = None):
        super().__init__(message, error_code, details)