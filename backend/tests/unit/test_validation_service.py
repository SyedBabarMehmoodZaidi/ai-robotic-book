import pytest
from unittest.mock import Mock, patch
from src.models.validation_test import ValidationTest
from src.models.retrieved_chunk import RetrievedChunk
from src.validation.validation_service import ValidationService
from src.retrieval.retrieval_service import RetrievalService


class TestValidationService:
    """
    Unit tests for the ValidationService class.
    """

    @patch('src.validation.validation_service.RetrievalService')
    def test_initialization(self, mock_retrieval_service):
        """
        Test that the validation service initializes correctly with a retrieval service.
        """
        # Mock the retrieval service
        mock_retrieval_instance = Mock()
        mock_retrieval_service.return_value = mock_retrieval_instance

        # Initialize the validation service
        validation_service = ValidationService(mock_retrieval_instance)

        # Verify that the retrieval service was set correctly
        assert validation_service.retrieval_service is mock_retrieval_instance

    def test_calculate_accuracy_perfect_match(self):
        """
        Test accuracy calculation with perfect match.
        """
        validation_service = ValidationService(Mock())

        expected_results = [
            {"content": "This is a test content"}
        ]
        actual_results = [
            {"content": "This is a test content", "similarity_score": 0.9, "chunk_id": "1", "metadata": {}, "position": 0}
        ]

        accuracy = validation_service._calculate_accuracy(expected_results, actual_results, "test")

        # Should have perfect accuracy
        assert accuracy == 1.0

    def test_calculate_accuracy_partial_match(self):
        """
        Test accuracy calculation with partial match.
        """
        validation_service = ValidationService(Mock())

        expected_results = [
            {"content": "This is a test content"},
            {"content": "Another expected result"}
        ]
        actual_results = [
            {"content": "This is a test content", "similarity_score": 0.9, "chunk_id": "1", "metadata": {}, "position": 0}
        ]

        accuracy = validation_service._calculate_accuracy(expected_results, actual_results, "test")

        # Should have 50% accuracy (1 out of 2 matched)
        assert accuracy == 0.5

    def test_calculate_accuracy_no_match(self):
        """
        Test accuracy calculation with no match.
        """
        validation_service = ValidationService(Mock())

        expected_results = [
            {"content": "This is expected content"}
        ]
        actual_results = [
            {"content": "This is completely different content", "similarity_score": 0.1, "chunk_id": "1", "metadata": {}, "position": 0}
        ]

        accuracy = validation_service._calculate_accuracy(expected_results, actual_results, "test")

        # Should have 0% accuracy
        assert accuracy == 0.0

    def test_calculate_content_similarity(self):
        """
        Test content similarity calculation.
        """
        validation_service = ValidationService(Mock())

        content1 = "This is a test content for similarity"
        content2 = "This is a test content for similarity"

        similarity = validation_service._calculate_content_similarity(content1, content2)

        # Should have perfect similarity
        assert similarity == 1.0

        content1 = "This is a test"
        content2 = "Completely different content"

        similarity = validation_service._calculate_content_similarity(content1, content2)

        # Should have low similarity
        assert similarity < 0.5

    def test_create_validation_report(self):
        """
        Test validation report creation.
        """
        validation_service = ValidationService(Mock())

        # Create some mock validation results
        from datetime import datetime
        from src.models.validation_test import ValidationResult

        validation_results = [
            ValidationResult(
                test_id="test1",
                query_text="test query 1",
                expected_results=[{"content": "expected"}],
                success_criteria="test",
                test_category="factual",
                executed_at=datetime.utcnow(),
                result_accuracy=0.8,
                actual_results=[{"content": "actual"}],
                passed=True
            ),
            ValidationResult(
                test_id="test2",
                query_text="test query 2",
                expected_results=[{"content": "expected"}],
                success_criteria="test",
                test_category="conceptual",
                executed_at=datetime.utcnow(),
                result_accuracy=0.6,
                actual_results=[{"content": "actual"}],
                passed=False
            )
        ]

        report = validation_service.create_validation_report(validation_results)

        # Verify the report structure
        assert report['total_tests'] == 2
        assert report['passed_tests'] == 1
        assert report['failed_tests'] == 1
        assert report['overall_accuracy'] == 0.7  # (0.8 + 0.6) / 2
        assert 'factual' in report['test_categories']
        assert 'conceptual' in report['test_categories']