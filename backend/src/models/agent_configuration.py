from pydantic import BaseModel, Field
from typing import Optional
import uuid


class AgentConfiguration(BaseModel):
    """
    Model representing configuration settings for the AI agent.

    Attributes:
        agent_id: Unique identifier for the agent (auto-generated if not provided)
        model_name: Name of the OpenAI model to use
        temperature: Temperature setting for response creativity (0-1)
        max_tokens: Maximum tokens in response (1-4096)
        retrieval_threshold: Minimum similarity score for retrieval (0-1)
        context_window: Maximum context window size
    """
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = "gpt-4-turbo"
    temperature: float = Field(ge=0.0, le=1.0, default=0.1)
    max_tokens: int = Field(ge=1, le=4096, default=1000)
    retrieval_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    context_window: int = 4096

    class Config:
        # Allow extra fields in case additional configuration parameters are added
        extra = "allow"
        # Enable ORM mode for potential database integration
        orm_mode = True