from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class QueryRequest(BaseModel):
    """
    Represents a user's request to the AI agent, containing the question and optional parameters.
    """
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The main question or query from the user"
    )
    selected_text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=5000,
        description="Specific text segment the user wants to focus on"
    )
    context_window: Optional[int] = Field(
        None,
        ge=100,
        le=4000,
        description="Size of context window (default: system determined)"
    )
    user_id: Optional[str] = Field(
        None,
        description="Identifier for the requesting user (for future expansion)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional query metadata including timestamp, source, etc."
    )

    @field_validator('query_text')
    @classmethod
    def validate_query_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty or just whitespace')
        if len(v.strip()) < 3:
            raise ValueError('Query text must be at least 3 characters long when stripped of whitespace')
        return v.strip()

    @field_validator('selected_text')
    @classmethod
    def validate_selected_text(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Selected text cannot be empty or just whitespace')
            if len(v.strip()) < 10:
                raise ValueError('Selected text should be at least 10 characters long when stripped of whitespace')
        return v.strip() if v else v

class RetrievedContext(BaseModel):
    """
    Represents the context retrieved from the book content based on the user's query.
    """
    content: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The retrieved book content"
    )
    source: str = Field(
        ...,
        description="Source document/section identifier"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from retrieval (0.0-1.0)"
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier for the content chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata including url, section, etc."
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score to the original query"
    )

class SourceReference(BaseModel):
    """
    Represents a reference to a specific source used in the response.
    """
    source: str = Field(
        ...,
        description="Source document/section identifier"
    )
    content_preview: str = Field(
        ...,
        max_length=500,
        description="Preview of the content that was used"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score of this source to the query"
    )
    chunk_id: str = Field(
        ...,
        description="Unique identifier for the content chunk"
    )


class AgentResponse(BaseModel):
    """
    Represents the AI-generated response to the user's query based on the retrieved context.
    """
    response_text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The AI-generated response"
    )
    source_context: List[str] = Field(
        ...,
        description="References to source material used"
    )
    detailed_source_references: List[SourceReference] = Field(
        default_factory=list,
        description="Detailed references to specific sources with content previews and relevance scores"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in response accuracy (0.0-1.0)"
    )
    tokens_used: int = Field(
        ...,
        gt=0,
        description="Number of tokens in response"
    )
    processing_time: float = Field(
        ...,
        gt=0,
        description="Time taken to generate response in seconds"
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query"
    )
    is_hallucination_detected: bool = Field(
        ...,
        description="Flag if hallucination was detected"
    )

class APIRequest(BaseModel):
    """
    Represents the HTTP request to the FastAPI endpoint containing the query parameters.
    """
    query: QueryRequest = Field(
        ...,
        description="The query object as defined above"
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of the request"
    )
    client_ip: Optional[str] = Field(
        None,
        description="IP address of the requesting client"
    )
    api_key_hash: Optional[str] = Field(
        None,
        description="Hash of the API key used (for rate limiting)"
    )

class APIResponse(BaseModel):
    """
    Represents the HTTP response from the FastAPI endpoint containing the agent's answer.
    """
    response: AgentResponse = Field(
        ...,
        description="The agent response as defined above"
    )
    request_id: str = Field(
        ...,
        description="Reference to the original request"
    )
    status_code: int = Field(
        ...,
        description="HTTP status code"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of the response"
    )
    processing_time: float = Field(
        ...,
        gt=0,
        description="Total processing time in seconds"
    )

class AgentSession(BaseModel):
    """
    Represents a session for multi-turn conversations with the agent (optional for future use).
    """
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the session"
    )
    user_id: Optional[str] = Field(
        None,
        description="Identifier for the user"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of session creation"
    )
    last_activity: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of last activity"
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="History of conversation turns"
    )
    context_window: Optional[List[RetrievedContext]] = Field(
        default_factory=list,
        description="Maintained context for the session"
    )