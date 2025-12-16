import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # Qdrant settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")

    # Cohere settings
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    cohere_model: str = os.getenv("COHERE_MODEL", "embed-english-v3.0")

    # Application settings
    app_name: str = "RAG Retrieval Service"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Retrieval settings
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    default_similarity_threshold: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.0"))

    # Validation settings
    validation_accuracy_threshold: float = float(os.getenv("VALIDATION_ACCURACY_THRESHOLD", "0.8"))

    class Config:
        # Load from .env file if present
        env_file = ".env"
        case_sensitive = False


# Create a single instance of settings
settings = Settings()