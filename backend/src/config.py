"""
Configuration module for API keys and application settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class to manage application settings."""

    # Cohere configuration
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY environment variable is required")

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL environment variable is required")

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    # Note: QDRANT_API_KEY is optional for local instances

    # Application settings
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")
    COHERE_MODEL_NAME = os.getenv("COHERE_MODEL_NAME", "embed-multilingual-v3.0")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))  # words
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # words
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # for processing batches

    # Validation
    @classmethod
    def validate(cls):
        """Validate that all required configuration values are present."""
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is required")
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL is required")