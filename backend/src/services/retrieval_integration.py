import logging
from typing import List, Optional
from datetime import datetime

from ..models.agent_query import AgentQuery
from ..models.agent_response import RetrievedChunk
from ..exceptions.agent_exceptions import RetrievalServiceException


logger = logging.getLogger(__name__)


class BasicRetrievalService:
    """
    Basic retrieval service for fallback when the actual retrieval service from Spec-2 is not available.
    This simulates the functionality of the actual retrieval service for integration testing.
    """

    def __init__(self, qdrant_url: Optional[str] = None, collection_name: Optional[str] = None, top_k: int = 5):
        """
        Initialize the basic retrieval service.

        Args:
            qdrant_url: URL for the Qdrant service (not used in basic implementation)
            collection_name: Name of the Qdrant collection to search (not used in basic implementation)
            top_k: Number of top results to retrieve
        """
        self.qdrant_url = qdrant_url or "http://localhost:6333"
        self.collection_name = collection_name or "book_embeddings"
        self.top_k = top_k
        logger.info("Basic retrieval service initialized for fallback use")

    def search(self, query_text: str) -> List[RetrievedChunk]:
        """
        Simulate search functionality by returning mock retrieved chunks.

        Args:
            query_text: The query text to search for

        Returns:
            List of mock RetrievedChunk objects
        """
        # In a real implementation, this would call the actual retrieval pipeline
        # For now, we'll return some sample chunks that are relevant to the query
        import uuid
        from datetime import datetime

        # Create mock chunks based on the query
        mock_chunks = []
        for i in range(min(self.top_k, 3)):  # Limit to 3 for demo purposes
            chunk_content = f"This is sample book content related to '{query_text}'. " \
                           f"Section {i+1} contains information about the topic. " \
                           f"The content discusses various aspects relevant to the query."

            chunk = RetrievedChunk(
                content=chunk_content,
                similarity_score=0.8 - (i * 0.1),  # Decreasing similarity for each chunk
                chunk_id=f"mock-chunk-{i+1}-{str(uuid.uuid4())[:8]}",
                metadata={
                    "source": "mock_book",
                    "section": f"section_{i+1}",
                    "created_at": datetime.utcnow().isoformat()
                },
                position=i+1
            )
            mock_chunks.append(chunk)

        logger.info(f"Mock search returned {len(mock_chunks)} chunks for query: '{query_text[:50]}...'")
        return mock_chunks


class RetrievalIntegration:
    """
    Service class to integrate the RAG agent with the existing retrieval pipeline.
    Handles retrieval of relevant content before agent generation.
    """

    def __init__(self, qdrant_url: Optional[str] = None, collection_name: Optional[str] = None, top_k: int = 5):
        """
        Initialize the retrieval integration service.

        Args:
            qdrant_url: URL for the Qdrant service (defaults to environment setting)
            collection_name: Name of the Qdrant collection to search (defaults to environment setting)
            top_k: Number of top results to retrieve
        """
        self.top_k = top_k
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Import the existing retrieval service from Spec-2
        # Since we're integrating with existing functionality, we'll import the relevant classes
        try:
            from src.services.retrieval import RetrievalService
            self.retrieval_service = RetrievalService(
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                top_k=top_k
            )
            logger.info("Retrieval integration service initialized successfully")
        except ImportError:
            logger.warning("Existing retrieval service not found from Spec-2, creating basic integration")
            # Create a basic retrieval service if the existing one is not available
            self.retrieval_service = BasicRetrievalService(
                qdrant_url=qdrant_url,
                collection_name=collection_name,
                top_k=top_k
            )

    def retrieve_content(self, query: AgentQuery) -> List[RetrievedChunk]:
        """
        Retrieve relevant content for the given query using the existing retrieval pipeline.

        Args:
            query: The agent query containing the text to search for

        Returns:
            List of RetrievedChunk objects containing relevant content
        """
        try:
            if not self.retrieval_service:
                # Fallback implementation for testing purposes
                logger.warning("Using fallback retrieval - no actual retrieval performed")
                return self._create_fallback_chunks(query.query_text)

            # Validate query text before retrieval
            if not query.query_text or len(query.query_text.strip()) == 0:
                logger.error("Empty query text provided for retrieval")
                raise RetrievalServiceException(
                    "Query text cannot be empty",
                    details={"query_id": query.query_id}
                )

            # Use the existing retrieval pipeline from Spec-2
            retrieved_chunks = self.retrieval_service.search(query.query_text)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query.query_text[:50]}...'")
            return retrieved_chunks

        except RetrievalServiceException:
            # Re-raise known retrieval exceptions
            raise
        except Exception as e:
            logger.error(f"Error during content retrieval: {str(e)}")
            raise RetrievalServiceException(
                "Error occurred during content retrieval",
                details={"error": str(e), "query_id": query.query_id}
            )

    def retrieve_content_with_fallback(self, query: AgentQuery,
                                     use_fallback_on_error: bool = True) -> List[RetrievedChunk]:
        """
        Retrieve content with fallback mechanism in case of errors.

        Args:
            query: The agent query containing the text to search for
            use_fallback_on_error: Whether to return fallback chunks if retrieval fails

        Returns:
            List of RetrievedChunk objects containing relevant content
        """
        try:
            return self.retrieve_content(query)
        except RetrievalServiceException as e:
            logger.error(f"Retrieval failed: {str(e)}, using fallback: {use_fallback_on_error}")
            if use_fallback_on_error:
                logger.info("Returning fallback chunks due to retrieval error")
                return self._create_fallback_chunks(query.query_text)
            else:
                raise

    def process_retrieval_results(self, retrieved_chunks: List[RetrievedChunk],
                                threshold: float = 0.5) -> List[RetrievedChunk]:
        """
        Process retrieval results by filtering based on similarity threshold.

        Args:
            retrieved_chunks: List of retrieved chunks to process
            threshold: Minimum similarity score for inclusion

        Returns:
            Filtered list of retrieved chunks that meet the threshold
        """
        try:
            # Filter chunks based on similarity threshold
            filtered_chunks = [
                chunk for chunk in retrieved_chunks
                if chunk.similarity_score >= threshold
            ]

            # Sort chunks by similarity score in descending order (highest first)
            filtered_chunks.sort(key=lambda x: x.similarity_score, reverse=True)

            logger.info(f"Filtered {len(retrieved_chunks)} chunks to {len(filtered_chunks)} "
                       f"based on threshold {threshold}")

            return filtered_chunks

        except Exception as e:
            logger.error(f"Error processing retrieval results: {str(e)}")
            raise RetrievalServiceException(
                "Error occurred during retrieval result processing",
                details={"error": str(e)}
            )

    def format_chunks_for_agent_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a context string suitable for the AI agent.

        Args:
            chunks: List of retrieved chunks to format

        Returns:
            Formatted context string containing all chunk content
        """
        if not chunks:
            return ""

        # Format each chunk with clear separation and metadata
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = (
                f"--- Chunk {i} (Similarity: {chunk.similarity_score:.3f}) ---\n"
                f"Content: {chunk.content}\n"
            )
            formatted_chunks.append(chunk_text)

        return "\n".join(formatted_chunks)

    def get_top_k_chunks(self, chunks: List[RetrievedChunk], k: int) -> List[RetrievedChunk]:
        """
        Get the top k chunks based on similarity score.

        Args:
            chunks: List of retrieved chunks
            k: Number of top chunks to return

        Returns:
            Top k chunks sorted by similarity score
        """
        # Sort chunks by similarity score in descending order
        sorted_chunks = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
        return sorted_chunks[:k]

    def filter_by_relevance_score(self, chunks: List[RetrievedChunk],
                                min_score: float = 0.5, max_score: float = 1.0) -> List[RetrievedChunk]:
        """
        Filter chunks based on relevance scores within a specific range.

        Args:
            chunks: List of retrieved chunks to filter
            min_score: Minimum similarity score threshold (0.0-1.0)
            max_score: Maximum similarity score threshold (0.0-1.0)

        Returns:
            Filtered list of chunks within the specified score range
        """
        if not self.validate_retrieval_threshold(min_score) or not self.validate_retrieval_threshold(max_score):
            logger.error(f"Invalid score range: min={min_score}, max={max_score}")
            raise RetrievalServiceException(
                "Invalid score range provided for filtering",
                details={"min_score": min_score, "max_score": max_score}
            )

        if min_score > max_score:
            logger.error(f"Min score {min_score} is greater than max score {max_score}")
            raise RetrievalServiceException(
                "Min score cannot be greater than max score",
                details={"min_score": min_score, "max_score": max_score}
            )

        filtered_chunks = [
            chunk for chunk in chunks
            if min_score <= chunk.similarity_score <= max_score
        ]

        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} based on relevance score range "
                   f"[{min_score}, {max_score}]")

        return filtered_chunks

    def filter_by_content_length(self, chunks: List[RetrievedChunk],
                               min_length: int = 50, max_length: int = 2000) -> List[RetrievedChunk]:
        """
        Filter chunks based on content length.

        Args:
            chunks: List of retrieved chunks to filter
            min_length: Minimum content length
            max_length: Maximum content length

        Returns:
            Filtered list of chunks based on content length
        """
        filtered_chunks = [
            chunk for chunk in chunks
            if min_length <= len(chunk.content) <= max_length
        ]

        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} based on content length "
                   f"[{min_length}, {max_length}]")

        return filtered_chunks

    def filter_chunks(self, chunks: List[RetrievedChunk],
                     min_similarity: float = 0.3,
                     min_content_length: int = 50,
                     max_content_length: int = 2000) -> List[RetrievedChunk]:
        """
        Apply multiple filters to chunks for comprehensive content filtering.

        Args:
            chunks: List of retrieved chunks to filter
            min_similarity: Minimum similarity score threshold
            min_content_length: Minimum content length
            max_content_length: Maximum content length

        Returns:
            Filtered list of chunks based on all criteria
        """
        # First filter by similarity score
        similarity_filtered = self.filter_by_relevance_score(
            chunks,
            min_score=min_similarity,
            max_score=1.0
        )

        # Then filter by content length
        length_filtered = self.filter_by_content_length(
            similarity_filtered,
            min_length=min_content_length,
            max_length=max_content_length
        )

        logger.info(f"Applied comprehensive filtering: {len(chunks)} -> {len(length_filtered)} chunks")

        return length_filtered

    def _create_fallback_chunks(self, query_text: str) -> List[RetrievedChunk]:
        """
        Create fallback retrieved chunks for testing purposes when retrieval service is not available.

        Args:
            query_text: The query text to create fallback chunks for

        Returns:
            List of mock RetrievedChunk objects
        """
        # This is a fallback implementation for testing purposes
        # In a real implementation, this would interface with the actual retrieval service
        fallback_content = f"Sample book content related to: {query_text[:100]}"

        fallback_chunk = RetrievedChunk(
            content=fallback_content,
            similarity_score=0.8,
            chunk_id="fallback-chunk-001",
            metadata={"source": "fallback", "type": "sample"},
            position=1
        )

        return [fallback_chunk]

    def validate_retrieval_threshold(self, threshold: float) -> bool:
        """
        Validate that the retrieval threshold is within acceptable bounds.

        Args:
            threshold: The threshold value to validate

        Returns:
            True if threshold is valid, False otherwise
        """
        if not isinstance(threshold, (int, float)):
            logger.error(f"Invalid threshold type: {type(threshold)}, expected int or float")
            return False

        if not (0.0 <= threshold <= 1.0):
            logger.error(f"Invalid threshold value: {threshold}, must be between 0.0 and 1.0")
            return False

        return True

    def validate_and_adjust_threshold(self, threshold: float, default_threshold: float = 0.5) -> float:
        """
        Validate the threshold and return a valid value, using default if invalid.

        Args:
            threshold: The threshold value to validate
            default_threshold: Default value to use if threshold is invalid

        Returns:
            Valid threshold value
        """
        if self.validate_retrieval_threshold(threshold):
            return float(threshold)
        else:
            logger.warning(f"Invalid threshold {threshold} provided, using default {default_threshold}")
            return default_threshold

    def get_retrieval_stats(self) -> dict:
        """
        Get statistics about retrieval performance.

        Returns:
            Dictionary containing retrieval statistics
        """
        # This would return actual stats in a full implementation
        return {
            "total_queries": 0,
            "successful_retrievals": 0,
            "avg_retrieval_time": 0.0,
            "last_updated": datetime.utcnow().isoformat()
        }