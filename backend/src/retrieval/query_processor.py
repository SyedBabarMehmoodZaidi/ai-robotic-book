import logging
from typing import List
from ..models.retrieved_chunk import RetrievedChunk
from ..models.search_query import SearchQuery
from .retrieval_service import RetrievalService
from ..exceptions.base import RetrievalException


logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Manages the query workflow including validation, processing, and result formatting.
    """
    def __init__(self, retrieval_service: RetrievalService):
        """
        Initialize the query processor with a retrieval service.

        Args:
            retrieval_service: Instance of RetrievalService to use for searches
        """
        self.retrieval_service = retrieval_service

    def process_search(self, search_query: SearchQuery) -> List[RetrievedChunk]:
        """
        Process a search query through the retrieval pipeline.

        Args:
            search_query: SearchQuery object containing query parameters

        Returns:
            List of RetrievedChunk objects
        """
        try:
            logger.info(f"Processing search query: '{search_query.query_text[:50]}...'")

            # Validate the search query
            self._validate_search_query(search_query)

            # Perform the search
            results = self.retrieval_service.search_with_model(search_query)

            logger.info(f"Successfully processed search query, returned {len(results)} results")

            return results

        except RetrievalException:
            # Re-raise retrieval exceptions as they're already properly formatted
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {str(e)}")
            raise RetrievalException(
                "Unexpected error occurred during query processing",
                details={"error": str(e), "query_id": search_query.query_id}
            )

    def _validate_search_query(self, search_query: SearchQuery) -> None:
        """
        Validate the search query parameters.

        Args:
            search_query: SearchQuery object to validate
        """
        # Check query text
        if not search_query.query_text or not search_query.query_text.strip():
            raise RetrievalException(
                "Query text is required and cannot be empty",
                details={"query_id": search_query.query_id}
            )

        # Check top_k value
        if search_query.top_k <= 0:
            raise RetrievalException(
                "top_k must be greater than 0",
                details={"query_id": search_query.query_id, "top_k": search_query.top_k}
            )

        if search_query.top_k > 100:  # Reasonable upper limit
            raise RetrievalException(
                "top_k cannot exceed 100",
                details={"query_id": search_query.query_id, "top_k": search_query.top_k}
            )

        # Check similarity threshold
        if search_query.similarity_threshold < 0 or search_query.similarity_threshold > 1:
            raise RetrievalException(
                "similarity_threshold must be between 0 and 1",
                details={
                    "query_id": search_query.query_id,
                    "similarity_threshold": search_query.similarity_threshold
                }
            )

        logger.debug(f"Search query validation passed for query_id: {search_query.query_id}")

    def process_search_text(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[RetrievedChunk]:
        """
        Process a search query given as text with parameters.

        Args:
            query_text: The text to search for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of RetrievedChunk objects
        """
        search_query = SearchQuery(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        return self.process_search(search_query)