import pytest
from unittest.mock import Mock, patch
from src.models.search_query import SearchQuery
from src.retrieval.retrieval_service import RetrievalService


class TestRetrievalService:
    """
    Unit tests for the RetrievalService class.
    """

    @patch('src.retrieval.retrieval_service.QdrantClient')
    @patch('src.retrieval.retrieval_service.cohere.Client')
    def test_initialization(self, mock_cohere_client, mock_qdrant_client):
        """
        Test that the retrieval service initializes correctly with Qdrant and Cohere clients.
        """
        # Mock the clients
        mock_qdrant_client_instance = Mock()
        mock_cohere_client_instance = Mock()
        mock_qdrant_client.return_value = mock_qdrant_client_instance
        mock_cohere_client.return_value = mock_cohere_client_instance

        # Initialize the service
        service = RetrievalService()

        # Verify that the clients were created
        assert service.qdrant_client is mock_qdrant_client_instance
        assert service.cohere_client is mock_cohere_client_instance

    @patch('src.retrieval.retrieval_service.QdrantClient')
    @patch('src.retrieval.retrieval_service.cohere.Client')
    def test_search_method(self, mock_cohere_client, mock_qdrant_client):
        """
        Test the search method with valid parameters.
        """
        # Mock the clients and their methods
        mock_qdrant_client_instance = Mock()
        mock_cohere_client_instance = Mock()

        # Mock the embed response
        mock_embed_response = Mock()
        mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere_client_instance.embed.return_value = mock_embed_response

        # Mock the search response
        mock_search_result = Mock()
        mock_search_result.payload = {"content": "test content", "position": 1}
        mock_search_result.score = 0.8
        mock_search_result.id = "chunk_123"
        mock_qdrant_client_instance.search.return_value = [mock_search_result]

        mock_qdrant_client.return_value = mock_qdrant_client_instance
        mock_cohere_client.return_value = mock_cohere_client_instance

        # Initialize the service
        service = RetrievalService()

        # Perform the search
        results = service.search("test query", top_k=5, similarity_threshold=0.5)

        # Verify the results
        assert len(results) == 1
        assert results[0].content == "test content"
        assert results[0].similarity_score == 0.8
        assert results[0].chunk_id == "chunk_123"
        assert results[0].position == 1

        # Verify that the methods were called correctly
        mock_cohere_client_instance.embed.assert_called_once()
        mock_qdrant_client_instance.search.assert_called_once()

    @patch('src.retrieval.retrieval_service.QdrantClient')
    @patch('src.retrieval.retrieval_service.cohere.Client')
    def test_search_with_model(self, mock_cohere_client, mock_qdrant_client):
        """
        Test the search_with_model method.
        """
        # Mock the clients and their methods
        mock_qdrant_client_instance = Mock()
        mock_cohere_client_instance = Mock()

        # Mock the embed response
        mock_embed_response = Mock()
        mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere_client_instance.embed.return_value = mock_embed_response

        # Mock the search response
        mock_search_result = Mock()
        mock_search_result.payload = {"content": "test content", "position": 1}
        mock_search_result.score = 0.8
        mock_search_result.id = "chunk_123"
        mock_qdrant_client_instance.search.return_value = [mock_search_result]

        mock_qdrant_client.return_value = mock_qdrant_client_instance
        mock_cohere_client.return_value = mock_cohere_client_instance

        # Initialize the service
        service = RetrievalService()

        # Create a search query model
        search_query = SearchQuery(
            query_text="test query",
            top_k=3,
            similarity_threshold=0.6
        )

        # Perform the search with model
        results = service.search_with_model(search_query)

        # Verify the results
        assert len(results) == 1
        assert results[0].content == "test content"

    @patch('src.retrieval.retrieval_service.QdrantClient')
    @patch('src.retrieval.retrieval_service.cohere.Client')
    def test_health_check(self, mock_cohere_client, mock_qdrant_client):
        """
        Test the health check method.
        """
        # Mock the clients
        mock_qdrant_client_instance = Mock()
        mock_cohere_client_instance = Mock()

        # Mock the get_collection and embed methods to return successfully
        mock_qdrant_client_instance.get_collection.return_value = True
        mock_embed_response = Mock()
        mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere_client_instance.embed.return_value = mock_embed_response

        mock_qdrant_client.return_value = mock_qdrant_client_instance
        mock_cohere_client.return_value = mock_cohere_client_instance

        # Initialize the service
        service = RetrievalService()

        # Perform health check
        is_healthy = service.health_check()

        # Verify the result
        assert is_healthy is True
        mock_qdrant_client_instance.get_collection.assert_called_once()
        mock_cohere_client_instance.embed.assert_called_once()