import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.api.main import app
from src.models.search_query import SearchQuery
from src.models.validation_test import ValidationTest


class TestAPIEndpoints:
    """
    Integration tests for the API endpoints.
    """

    def setup_method(self):
        """
        Setup method to create a test client.
        """
        self.client = TestClient(app)

    @patch('src.api.search_router.get_retrieval_service')
    def test_search_endpoint(self, mock_get_retrieval_service):
        """
        Test the search endpoint.
        """
        # Mock the retrieval service
        mock_query_processor = Mock()
        mock_retrieval_service = Mock()
        mock_query_processor.retrieval_service = mock_retrieval_service

        # Mock the process_search method to return some results
        mock_chunk = Mock()
        mock_chunk.content = "Test content"
        mock_chunk.similarity_score = 0.8
        mock_chunk.chunk_id = "test_chunk_id"
        mock_chunk.metadata = {"url": "test_url", "section": "test_section"}
        mock_chunk.position = 1

        mock_query_processor.process_search.return_value = [mock_chunk]
        mock_get_retrieval_service.return_value = mock_query_processor

        # Prepare the request data
        search_data = {
            "query_text": "test query",
            "top_k": 5,
            "similarity_threshold": 0.5
        }

        # Make the request
        response = self.client.post("/api/v1/search", json=search_data)

        # Verify the response
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["similarity_score"] == 0.8
        assert result[0]["chunk_id"] == "test_chunk_id"

    @patch('src.api.validation_router.get_validation_service')
    def test_validate_endpoint(self, mock_get_validation_service):
        """
        Test the validation endpoint.
        """
        # Mock the validation service
        mock_validation_service = Mock()
        mock_get_validation_service.return_value = mock_validation_service

        # Mock the validate_retrieval_quality method to return some results
        from datetime import datetime
        from src.models.validation_test import ValidationResult

        mock_result = ValidationResult(
            test_id="test1",
            query_text="test query",
            expected_results=[{"content": "expected"}],
            success_criteria="test",
            test_category="factual",
            executed_at=datetime.utcnow(),
            result_accuracy=0.8,
            actual_results=[{"content": "actual"}],
            passed=True
        )

        mock_validation_service.validate_retrieval_quality.return_value = [mock_result]
        mock_get_validation_service.return_value = mock_validation_service

        # Prepare the request data
        validation_data = [{
            "query_text": "test query",
            "expected_results": [{"content": "expected"}],
            "success_criteria": "test",
            "test_category": "factual"
        }]

        # Make the request
        response = self.client.post("/api/v1/validate", json=validation_data)

        # Verify the response
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["result_accuracy"] == 0.8
        assert result[0]["passed"] is True

    def test_health_endpoint(self):
        """
        Test the health endpoint.
        """
        # Make the request
        response = self.client.get("/api/v1/health")

        # Verify the response
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert "timestamp" in result
        assert "services" in result
        assert "retrieval" in result["services"]

    @patch('src.api.search_router.get_retrieval_service')
    def test_search_text_endpoint(self, mock_get_retrieval_service):
        """
        Test the search text endpoint.
        """
        # Mock the retrieval service
        mock_query_processor = Mock()
        mock_retrieval_service = Mock()
        mock_query_processor.retrieval_service = mock_retrieval_service

        # Mock the process_search_text method to return some results
        mock_chunk = Mock()
        mock_chunk.content = "Test content"
        mock_chunk.similarity_score = 0.8
        mock_chunk.chunk_id = "test_chunk_id"
        mock_chunk.metadata = {"url": "test_url", "section": "test_section"}
        mock_chunk.position = 1

        mock_query_processor.process_search_text.return_value = [mock_chunk]
        mock_get_retrieval_service.return_value = mock_query_processor

        # Make the request with query parameters
        response = self.client.post(
            "/api/v1/search/text",
            params={
                "query_text": "test query",
                "top_k": 5,
                "similarity_threshold": 0.5
            }
        )

        # Verify the response
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["similarity_score"] == 0.8