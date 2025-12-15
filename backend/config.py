import os
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Constants and Configuration Variables
class Config:
    """Configuration class containing all application settings"""

    # API Keys and Connection Settings
    COHERE_API_KEY: str = os.getenv('COHERE_API_KEY', '')
    QDRANT_URL: Optional[str] = os.getenv('QDRANT_URL')  # For remote instances
    QDRANT_API_KEY: Optional[str] = os.getenv('QDRANT_API_KEY')  # For remote instances
    LOCAL_QDRANT_PATH: str = os.getenv('LOCAL_QDRANT_PATH', './qdrant_data')  # For local storage

    # Database settings
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'book_embeddings')

    # Validation thresholds and settings
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '5'))
    VALIDATION_TIMEOUT: int = int(os.getenv('VALIDATION_TIMEOUT', '30'))  # seconds
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))

    # Quality validation thresholds
    QUALITY_THRESHOLD: float = float(os.getenv('QUALITY_THRESHOLD', '0.9'))  # 90% threshold for quality
    METADATA_ACCURACY_THRESHOLD: float = float(os.getenv('METADATA_ACCURACY_THRESHOLD', '1.0'))  # 100% for metadata
    RETRIEVAL_SUCCESS_RATE_THRESHOLD: float = float(os.getenv('RETRIEVAL_SUCCESS_RATE_THRESHOLD', '0.95'))  # 95% success rate

    # Validation parameters
    MIN_CHUNK_LENGTH: int = int(os.getenv('MIN_CHUNK_LENGTH', '50'))  # Minimum content length
    MAX_CHUNK_LENGTH: int = int(os.getenv('MAX_CHUNK_LENGTH', '2000'))  # Maximum content length
    MIN_SENTENCES_FOR_QUALITY: int = int(os.getenv('MIN_SENTENCES_FOR_QUALITY', '5'))  # Minimum sentences for quality assessment

    # Cohere settings
    COHERE_MODEL: str = os.getenv('COHERE_MODEL', 'embed-multilingual-v3.0')
    COHERE_INPUT_TYPE: str = os.getenv('COHERE_INPUT_TYPE', 'search_query')  # Default input type for queries

    # Performance settings
    MAX_QUERY_LENGTH: int = int(os.getenv('MAX_QUERY_LENGTH', '1000'))  # Maximum query length
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))  # Maximum retries for API calls
    RETRY_DELAY: float = float(os.getenv('RETRY_DELAY', '1.0'))  # Delay between retries in seconds


class ValidationConfig:
    """Specific configuration for validation parameters"""

    @staticmethod
    def get_quality_weights():
        """Get weights for different quality assessment components"""
        return {
            'length_weight': float(os.getenv('LENGTH_WEIGHT', '0.3')),
            'keyword_weight': float(os.getenv('KEYWORD_WEIGHT', '0.5')),
            'sentence_quality_weight': float(os.getenv('SENTENCE_QUALITY_WEIGHT', '0.2'))
        }

    @staticmethod
    def get_validation_rules():
        """Get validation rules for content quality"""
        return {
            'min_length': Config.MIN_CHUNK_LENGTH,
            'max_length': Config.MAX_CHUNK_LENGTH,
            'min_sentences': Config.MIN_SENTENCES_FOR_QUALITY,
            'similarity_threshold': Config.SIMILARITY_THRESHOLD,
            'quality_threshold': Config.QUALITY_THRESHOLD,
            'metadata_accuracy_threshold': Config.METADATA_ACCURACY_THRESHOLD
        }

    @staticmethod
    def get_thresholds():
        """Get all validation thresholds"""
        return {
            'quality': Config.QUALITY_THRESHOLD,
            'metadata_accuracy': Config.METADATA_ACCURACY_THRESHOLD,
            'retrieval_success_rate': Config.RETRIEVAL_SUCCESS_RATE_THRESHOLD
        }


# Error handling and logging utilities
def setup_logging():
    """Set up logging configuration for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('retrieval_validation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_environment_variables():
    """Validate that required environment variables are set"""
    required_vars = ['COHERE_API_KEY']

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Check if either remote or local Qdrant configuration is provided
    has_remote = bool(Config.QDRANT_URL and Config.QDRANT_API_KEY)
    has_local = bool(Config.LOCAL_QDRANT_PATH)

    if not (has_remote or has_local):
        raise ValueError("Either QDRANT_URL and QDRANT_API_KEY (remote) or LOCAL_QDRANT_PATH (local) must be provided")


# Client initialization functions
def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client based on configuration"""
    validate_environment_variables()

    try:
        if Config.QDRANT_URL:
            # Connect to remote Qdrant instance
            logger.info(f"Connecting to remote Qdrant at {Config.QDRANT_URL}")
            client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
                timeout=Config.VALIDATION_TIMEOUT
            )
        else:
            # Connect to local Qdrant instance
            logger.info(f"Connecting to local Qdrant at {Config.LOCAL_QDRANT_PATH}")
            client = QdrantClient(path=Config.LOCAL_QDRANT_PATH)

        # Test connection
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise Exception(f"Qdrant connection failed: {str(e)}")


def get_cohere_client() -> cohere.Client:
    """Initialize and return Cohere client based on configuration"""
    validate_environment_variables()

    if not Config.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY environment variable is required")

    try:
        logger.info("Initializing Cohere client")
        client = cohere.Client(api_key=Config.COHERE_API_KEY)
        logger.info("Successfully initialized Cohere client")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Cohere client: {str(e)}")
        raise Exception(f"Cohere client initialization failed: {str(e)}")


# Helper functions for data validation
def validate_query_text(query_text: str) -> bool:
    """Validate that the query text is not empty and meets minimum requirements"""
    if not query_text or not isinstance(query_text, str):
        return False

    query_text = query_text.strip()
    if len(query_text) == 0:
        return False

    if len(query_text) > 1000:  # Reasonable upper limit
        return False

    return True


def validate_top_k(top_k: int) -> bool:
    """Validate that top_k is a reasonable value"""
    if not isinstance(top_k, int):
        return False

    if top_k <= 0 or top_k > 100:  # Reasonable range
        return False

    return True


def validate_retrieved_chunks(chunks: list) -> bool:
    """Validate that retrieved chunks have the expected structure"""
    if not isinstance(chunks, list):
        return False

    for chunk in chunks:
        if not isinstance(chunk, dict):
            return False

        # Check for required keys in each chunk
        required_keys = ['content', 'metadata', 'score']
        for key in required_keys:
            if key not in chunk:
                return False

    return True


# Initialize logger
logger = setup_logging()


__all__ = [
    'Config',
    'get_qdrant_client',
    'get_cohere_client',
    'validate_environment_variables',
    'setup_logging',
    'validate_query_text',
    'validate_top_k',
    'validate_retrieved_chunks',
    'logger'
]