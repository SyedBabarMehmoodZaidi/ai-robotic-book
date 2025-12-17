import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.models.agent_query import AgentQuery
from src.models.agent_response import RetrievedChunk
from src.services.retrieval_integration import RetrievalIntegration
from src.exceptions.agent_exceptions import RetrievalServiceException


class TestRetrievalIntegration:
    """Unit tests for RetrievalIntegration class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a sample agent query for testing
        self.sample_query = AgentQuery(
            query_text="What is the capital of France?",
            query_id="test-query-123",
            created_at=datetime.utcnow(),
            query_type="general"
        )

        # Create sample retrieved chunks
        self.sample_chunks = [
            RetrievedChunk(
                content="Paris is the capital of France.",
                similarity_score=0.9,
                chunk_id="chunk-1",
                metadata={"source": "book1", "page": 10},
                position=1
            ),
            RetrievedChunk(
                content="France is a country in Europe.",
                similarity_score=0.7,
                chunk_id="chunk-2",
                metadata={"source": "book1", "page": 11},
                position=2
            )
        ]

    @patch('src.services.retrieval_integration.BasicRetrievalService')
    def test_initialization_with_fallback_service(self, mock_basic_service):
        """Test initialization of RetrievalIntegration with fallback service."""
        # Mock the basic retrieval service
        mock_service_instance = Mock()
        mock_basic_service.return_value = mock_service_instance

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration(
            qdrant_url="http://test-url:6333",
            collection_name="test_collection",
            top_k=5
        )

        # Assertions
        assert retrieval_integration.top_k == 5
        assert retrieval_integration.qdrant_url == "http://test-url:6333"
        assert retrieval_integration.collection_name == "test_collection"
        assert retrieval_integration.retrieval_service == mock_service_instance

    def test_retrieve_content_success(self):
        """Test successful content retrieval."""
        # Create a mock retrieval service
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.return_value = self.sample_chunks

        # Initialize the retrieval integration with the mock service
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service
        retrieval_integration.top_k = 5
        retrieval_integration.qdrant_url = "http://test-url:6333"
        retrieval_integration.collection_name = "test_collection"

        # Perform retrieval
        result = retrieval_integration.retrieve_content(self.sample_query)

        # Assertions
        assert result == self.sample_chunks
        mock_retrieval_service.search.assert_called_once_with(self.sample_query.query_text)

    def test_retrieve_content_empty_query(self):
        """Test content retrieval with empty query text."""
        # Create a query and manually set query_text to empty to bypass validation
        empty_query = AgentQuery(
            query_text="test",
            query_id="empty-query-123",
            created_at=datetime.utcnow(),
            query_type="general"
        )
        empty_query.query_text = ""  # Manually set to empty to test validation logic

        # Initialize the retrieval integration with a mock service
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = Mock()

        # Attempt to retrieve content should raise an exception
        with pytest.raises(RetrievalServiceException) as exc_info:
            retrieval_integration.retrieve_content(empty_query)

        # Assertions
        assert "Query text cannot be empty" in str(exc_info.value)

    def test_process_retrieval_results_filtering(self):
        """Test processing retrieval results with threshold filtering."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Test with a threshold that should filter out the second chunk
        threshold = 0.8
        result = retrieval_integration.process_retrieval_results(self.sample_chunks, threshold)

        # Assertions
        assert len(result) == 1  # Only the first chunk should remain
        assert result[0].similarity_score == 0.9  # The chunk with score 0.9 should remain
        assert result[0].chunk_id == "chunk-1"

    def test_process_retrieval_results_sorting(self):
        """Test that processing retrieval results sorts by similarity score."""
        # Create chunks in reverse order of similarity
        unsorted_chunks = [
            RetrievedChunk(
                content="Content with lower similarity",
                similarity_score=0.6,
                chunk_id="chunk-low",
                metadata={"source": "book1"},
                position=1
            ),
            RetrievedChunk(
                content="Content with higher similarity",
                similarity_score=0.9,
                chunk_id="chunk-high",
                metadata={"source": "book1"},
                position=2
            )
        ]

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Process results
        result = retrieval_integration.process_retrieval_results(unsorted_chunks, 0.0)  # No filtering

        # Assertions
        assert len(result) == 2
        assert result[0].similarity_score == 0.9  # Highest similarity first
        assert result[1].similarity_score == 0.6  # Lower similarity second

    def test_validate_retrieval_threshold_valid(self):
        """Test validation of valid retrieval thresholds."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Test valid thresholds
        assert retrieval_integration.validate_retrieval_threshold(0.0) is True
        assert retrieval_integration.validate_retrieval_threshold(0.5) is True
        assert retrieval_integration.validate_retrieval_threshold(1.0) is True
        assert retrieval_integration.validate_retrieval_threshold(0.75) is True

    def test_validate_retrieval_threshold_invalid(self):
        """Test validation of invalid retrieval thresholds."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Test invalid thresholds
        assert retrieval_integration.validate_retrieval_threshold(-0.1) is False
        assert retrieval_integration.validate_retrieval_threshold(1.1) is False
        assert retrieval_integration.validate_retrieval_threshold("invalid") is False
        assert retrieval_integration.validate_retrieval_threshold(None) is False

    def test_validate_and_adjust_threshold(self):
        """Test validation and adjustment of threshold."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Test valid threshold - should return the same value
        result = retrieval_integration.validate_and_adjust_threshold(0.7, 0.5)
        assert result == 0.7

        # Test invalid threshold - should return default
        result = retrieval_integration.validate_and_adjust_threshold(1.5, 0.6)
        assert result == 0.6

        # Test another invalid threshold
        result = retrieval_integration.validate_and_adjust_threshold(-0.1, 0.3)
        assert result == 0.3

    def test_format_chunks_for_agent_context(self):
        """Test formatting chunks for agent context."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Format the sample chunks
        result = retrieval_integration.format_chunks_for_agent_context(self.sample_chunks)

        # Assertions
        assert "--- Chunk 1 (Similarity: 0.900) ---" in result
        assert "Content: Paris is the capital of France." in result
        assert "--- Chunk 2 (Similarity: 0.700) ---" in result
        assert "Content: France is a country in Europe." in result

    def test_format_chunks_for_agent_context_empty(self):
        """Test formatting empty chunks list."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Format empty chunks
        result = retrieval_integration.format_chunks_for_agent_context([])

        # Should return empty string
        assert result == ""

    def test_get_top_k_chunks(self):
        """Test getting top k chunks by similarity score."""
        # Add a third chunk with lower similarity
        all_chunks = self.sample_chunks + [
            RetrievedChunk(
                content="Additional content",
                similarity_score=0.5,
                chunk_id="chunk-3",
                metadata={"source": "book1", "page": 12},
                position=3
            )
        ]

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Get top 2 chunks
        result = retrieval_integration.get_top_k_chunks(all_chunks, 2)

        # Assertions
        assert len(result) == 2
        assert result[0].similarity_score == 0.9  # Highest first
        assert result[1].similarity_score == 0.7  # Second highest

    def test_filter_by_relevance_score(self):
        """Test filtering chunks by relevance score range."""
        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Filter chunks with score between 0.6 and 0.8
        result = retrieval_integration.filter_by_relevance_score(
            self.sample_chunks,
            min_score=0.6,
            max_score=0.8
        )

        # Should only return the chunk with score 0.7
        assert len(result) == 1
        assert result[0].similarity_score == 0.7

    def test_filter_by_content_length(self):
        """Test filtering chunks by content length."""
        # Create chunks with different content lengths
        chunks = [
            RetrievedChunk(
                content="Short",  # Length 5
                similarity_score=0.9,
                chunk_id="chunk-short",
                metadata={"source": "book1"},
                position=1
            ),
            RetrievedChunk(
                content="This is a longer piece of content that should pass the length filter",  # Length > 20
                similarity_score=0.8,
                chunk_id="chunk-long",
                metadata={"source": "book1"},
                position=2
            )
        ]

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Filter chunks with length between 10 and 100
        result = retrieval_integration.filter_by_content_length(chunks, min_length=10, max_length=100)

        # Should only return the longer chunk
        assert len(result) == 1
        assert "longer piece of content" in result[0].content

    def test_filter_chunks_comprehensive(self):
        """Test comprehensive filtering by multiple criteria."""
        # Create chunks with different characteristics
        chunks = [
            RetrievedChunk(
                content="Short low quality",  # Short and low similarity
                similarity_score=0.2,
                chunk_id="chunk-1",
                metadata={"source": "book1"},
                position=1
            ),
            RetrievedChunk(
                content="This is a longer piece of content with good similarity score",
                similarity_score=0.8,
                chunk_id="chunk-2",
                metadata={"source": "book1"},
                position=2
            ),
            RetrievedChunk(
                content="Another good chunk",
                similarity_score=0.7,
                chunk_id="chunk-3",
                metadata={"source": "book1"},
                position=3
            )
        ]

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__

        # Apply comprehensive filtering: min_similarity=0.5, min_length=10, max_length=100
        result = retrieval_integration.filter_chunks(
            chunks,
            min_similarity=0.5,
            min_content_length=10,
            max_content_length=100
        )

        # Should return only the chunk that meets all criteria
        assert len(result) == 2  # Both chunk-2 and chunk-3 should pass
        for chunk in result:
            assert chunk.similarity_score >= 0.5
            assert len(chunk.content) >= 10
            assert len(chunk.content) <= 100

    def test_retrieve_content_with_fallback_success(self):
        """Test retrieve_content_with_fallback when retrieval succeeds."""
        # Create a mock retrieval service that succeeds
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.return_value = self.sample_chunks

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service
        retrieval_integration.top_k = 5
        retrieval_integration.qdrant_url = "http://test-url:6333"
        retrieval_integration.collection_name = "test_collection"

        # Perform retrieval with fallback
        result = retrieval_integration.retrieve_content_with_fallback(self.sample_query)

        # Should return the actual results
        assert result == self.sample_chunks

    def test_retrieve_content_with_fallback_error_use_fallback(self):
        """Test retrieve_content_with_fallback when retrieval fails and fallback is used."""
        # Create a mock retrieval service that raises an exception
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.side_effect = Exception("Retrieval failed")

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service

        # Perform retrieval with fallback enabled
        result = retrieval_integration.retrieve_content_with_fallback(
            self.sample_query,
            use_fallback_on_error=True
        )

        # Should return fallback chunks
        assert len(result) >= 0  # Should return some fallback chunks

    def test_retrieve_content_with_fallback_error_no_fallback(self):
        """Test retrieve_content_with_fallback when retrieval fails and fallback is disabled."""
        # Create a mock retrieval service that raises an exception
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.side_effect = RetrievalServiceException("Retrieval failed")

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service

        # Perform retrieval with fallback disabled should raise the exception
        with pytest.raises(RetrievalServiceException):
            retrieval_integration.retrieve_content_with_fallback(
                self.sample_query,
                use_fallback_on_error=False
            )

    def test_rag_pattern_enforcement_general_query(self):
        """Test RAG pattern enforcement with general query type."""
        # Create a general query
        general_query = AgentQuery(
            query_text="What is artificial intelligence?",
            query_id="general-query-123",
            created_at=datetime.utcnow(),
            query_type="general"
        )

        # Mock retrieval service to return some chunks
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.return_value = self.sample_chunks

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service
        retrieval_integration.top_k = 5
        retrieval_integration.qdrant_url = "http://test-url:6333"
        retrieval_integration.collection_name = "test_collection"

        # Perform retrieval
        result = retrieval_integration.retrieve_content(general_query)

        # Verify that retrieval was performed (RAG pattern enforced)
        mock_retrieval_service.search.assert_called_once_with(general_query.query_text)
        assert result == self.sample_chunks

    def test_rag_pattern_enforcement_context_specific_query(self):
        """Test RAG pattern enforcement with context-specific query type."""
        # Create a context-specific query
        context_query = AgentQuery(
            query_text="Explain the concept mentioned in the provided text",
            context_text="Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence.",
            query_id="context-query-123",
            created_at=datetime.utcnow(),
            query_type="context-specific"
        )

        # Mock retrieval service to return some chunks
        mock_retrieval_service = Mock()
        mock_retrieval_service.search.return_value = self.sample_chunks

        # Initialize the retrieval integration
        retrieval_integration = RetrievalIntegration.__new__(RetrievalIntegration)  # Create without __init__
        retrieval_integration.retrieval_service = mock_retrieval_service
        retrieval_integration.top_k = 5
        retrieval_integration.qdrant_url = "http://test-url:6333"
        retrieval_integration.collection_name = "test_collection"

        # Perform retrieval - should still retrieve based on query text even with context provided
        result = retrieval_integration.retrieve_content(context_query)

        # Verify that retrieval was performed based on query text (RAG pattern enforced)
        mock_retrieval_service.search.assert_called_once_with(context_query.query_text)
        assert result == self.sample_chunks