from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid


class AgentQuery(BaseModel):
    """
    Model representing a query submitted to the AI agent.

    Attributes:
        query_text: The user's question or query text (1-2000 characters)
        context_text: Optional selected text provided as context
        query_id: Unique identifier for the query (auto-generated)
        created_at: Timestamp when the query was created (auto-generated)
        user_id: Identifier for the user making the query
        query_type: Type of query ("general" | "context-specific")
    """
    query_text: str = Field(..., min_length=1, max_length=2000)
    context_text: Optional[str] = None
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    query_type: str = Field(default="general", pattern=r"^(general|context-specific)$")

    class Config:
        # Allow extra fields in case additional parameters are added
        extra = "allow"
        # Enable ORM mode for potential database integration
        orm_mode = True