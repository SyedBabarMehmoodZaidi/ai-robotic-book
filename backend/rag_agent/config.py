from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # OpenAI Configuration
    openai_api_key: str
    agent_model: str = "gpt-4-turbo"

    # Retrieval Pipeline Configuration
    retrieval_endpoint: str = "http://localhost:8000"

    # API Configuration
    rate_limit_requests: int = 100
    context_size_limit: int = 4000

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging Configuration
    log_level: str = "INFO"

    # Additional settings that might be needed
    debug: bool = False
    allowed_origins: str = "*"  # In production, specify actual origins

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create a single instance of settings
settings = Settings()