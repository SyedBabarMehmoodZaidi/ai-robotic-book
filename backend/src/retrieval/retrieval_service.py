import cohere
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..models.retrieved_chunk import RetrievedChunk
from ..models.search_query import SearchQuery
from ..config.settings import settings
from ..exceptions.base import (
    RetrievalException, QdrantConnectionException,
    CohereAPIException, ConfigurationException, InvalidQueryException
)


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service class for performing semantic search in Qdrant using Cohere embeddings.
    """
    def __init__(self):
        """
        Initialize the retrieval service with Qdrant and Cohere clients.
        """
        try:
            # Initialize Qdrant client
            if settings.qdrant_api_key:
                self.qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
            else:
                self.qdrant_client = QdrantClient(url=settings.qdrant_url)
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise QdrantConnectionException(
                f"Failed to connect to Qdrant at {settings.qdrant_url}",
                details={"error": str(e)}
            )

        # Initialize Cohere client
        if not settings.cohere_api_key:
            raise ConfigurationException(
                "COHERE_API_KEY environment variable is required",
                details={"missing_key": "COHERE_API_KEY"}
            )

        try:
            self.cohere_client = cohere.Client(settings.cohere_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise CohereAPIException(
                "Failed to initialize Cohere client",
                details={"error": str(e)}
            )

        logger.info("RetrievalService initialized successfully")

    def search(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[RetrievedChunk]:
        """
        Perform semantic search against the Qdrant vector database.

        Args:
            query_text: The text to search for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of RetrievedChunk objects ranked by similarity score
        """
        try:
            # Validate inputs
            if not query_text or not query_text.strip():
                logger.warning("Empty query text received")
                raise InvalidQueryException("Query text cannot be empty")

            if top_k <= 0:
                logger.warning(f"Invalid top_k value: {top_k}")
                raise InvalidQueryException("top_k must be greater than 0")

            if similarity_threshold < 0 or similarity_threshold > 1:
                logger.warning(f"Invalid similarity_threshold value: {similarity_threshold}")
                raise InvalidQueryException("similarity_threshold must be between 0 and 1")

            logger.info(f"Performing semantic search for query: '{query_text[:50]}...' with top_k={top_k}")

            # Convert query text to vector using Cohere
            response = self.cohere_client.embed(
                texts=[query_text],
                model=settings.cohere_model,
                input_type="search_query"
            )

            query_vector = response.embeddings[0]

            # Perform similarity search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=similarity_threshold
            )

            # Convert search results to RetrievedChunk objects
            retrieved_chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    content=result.payload.get("content", ""),
                    similarity_score=result.score,
                    chunk_id=result.id,
                    metadata=result.payload,
                    position=result.payload.get("position", 0)
                )
                retrieved_chunks.append(chunk)

            logger.info(f"Found {len(retrieved_chunks)} relevant chunks for query")

            return retrieved_chunks

        except cohere.CohereError as e:
            logger.error(f"Cohere API error during search: {str(e)}")
            raise CohereAPIException(
                "Error occurred during Cohere embedding generation",
                details={"error": str(e), "query_text": query_text}
            )
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            raise RetrievalException(
                "Error occurred during semantic search",
                details={"error": str(e), "query_text": query_text}
            )

    def search_with_model(self, search_query: SearchQuery) -> List[RetrievedChunk]:
        """
        Perform semantic search using a SearchQuery model.

        Args:
            search_query: SearchQuery object containing query parameters

        Returns:
            List of RetrievedChunk objects ranked by similarity score
        """
        return self.search(
            query_text=search_query.query_text,
            top_k=search_query.top_k,
            similarity_threshold=search_query.similarity_threshold
        )

    def health_check(self) -> bool:
        """
        Check if the retrieval service is healthy by testing connections.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test Qdrant connection
            self.qdrant_client.get_collection(settings.qdrant_collection_name)

            # Test Cohere connection by generating a simple embedding
            test_response = self.cohere_client.embed(
                texts=["test"],
                model=settings.cohere_model,
                input_type="search_query"
            )

            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False