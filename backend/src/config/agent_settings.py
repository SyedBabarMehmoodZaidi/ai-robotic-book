import os
from pydantic import Field
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """
    Application settings for the RAG agent loaded from environment variables.
    """
    # OpenAI settings
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model_name: str = Field(default_factory=lambda: os.getenv("AGENT_MODEL_NAME", "gpt-4-turbo"))
    openai_temperature: float = Field(default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.1")))
    openai_max_tokens: int = Field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "1000")))

    # Agent settings
    agent_retrieval_threshold: float = Field(default_factory=lambda: float(os.getenv("RETRIEVAL_THRESHOLD", "0.5")))
    agent_context_window: int = Field(default_factory=lambda: int(os.getenv("AGENT_CONTEXT_WINDOW", "4096")))

    # Qdrant settings for retrieval pipeline integration
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection_name: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings"))

    # Application settings
    app_name: str = "RAG Agent Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")

    # Default top-k for retrieval
    default_top_k: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5")))

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables that are not defined in the model
    }


# Create a single instance of settings
agent_settings = AgentSettings()