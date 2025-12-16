import logging
from typing import List, Dict, Any
from datetime import datetime
from ..models.validation_test import ValidationTest, ValidationResult
from ..models.retrieved_chunk import RetrievedChunk
from ..retrieval.retrieval_service import RetrievalService
from ..exceptions.base import ValidationException


logger = logging.getLogger(__name__)


class ValidationService:
    """
    Service class for validating the quality of the retrieval pipeline.
    """
    def __init__(self, retrieval_service: RetrievalService):
        """
        Initialize the validation service with a retrieval service.

        Args:
            retrieval_service: Instance of RetrievalService to validate
        """
        self.retrieval_service = retrieval_service

    def validate_retrieval_quality(self, test_queries: List[ValidationTest]) -> List[ValidationResult]:
        """
        Validate retrieval quality by executing multiple test queries and measuring relevance.

        Args:
            test_queries: List of ValidationTest objects with expected outcomes

        Returns:
            List of ValidationResult objects with accuracy measurements
        """
        results = []

        for test_query in test_queries:
            try:
                logger.info(f"Validating retrieval quality for query: '{test_query.query_text[:50]}...'")

                # Perform the search using the retrieval service
                actual_results = self.retrieval_service.search(
                    query_text=test_query.query_text,
                    top_k=len(test_query.expected_results) if test_query.expected_results else 5,
                    similarity_threshold=test_query.success_criteria.get('similarity_threshold', 0.0)
                )

                # Compare actual vs expected results
                validation_result = self._compare_results(
                    test_query=test_query,
                    actual_results=actual_results
                )

                results.append(validation_result)

            except Exception as e:
                logger.error(f"Error during validation for query '{test_query.query_text[:50]}...': {str(e)}")
                # Create a failed validation result
                validation_result = ValidationResult(
                    test_id=test_query.test_id,
                    query_text=test_query.query_text,
                    expected_results=test_query.expected_results,
                    success_criteria=test_query.success_criteria,
                    test_category=test_query.test_category,
                    executed_at=datetime.utcnow(),
                    result_accuracy=0.0,
                    actual_results=[],
                    passed=False
                )
                results.append(validation_result)

        return results

    def _compare_results(self, test_query: ValidationTest, actual_results: List[RetrievedChunk]) -> ValidationResult:
        """
        Compare actual retrieval results with expected results to calculate accuracy.

        Args:
            test_query: The validation test with expected results
            actual_results: The actual results from the retrieval service

        Returns:
            ValidationResult object with accuracy measurement
        """
        # Convert actual results to the expected format for comparison
        actual_result_dicts = []
        for chunk in actual_results:
            actual_result_dicts.append({
                'content': chunk.content,
                'similarity_score': chunk.similarity_score,
                'chunk_id': chunk.chunk_id,
                'metadata': chunk.metadata,
                'position': chunk.position
            })

        # Calculate accuracy based on various criteria
        accuracy = self._calculate_accuracy(
            expected_results=test_query.expected_results,
            actual_results=actual_result_dicts,
            success_criteria=test_query.success_criteria
        )

        # Determine if the test passed based on the accuracy threshold
        threshold = test_query.success_criteria.get('accuracy_threshold', 0.8)
        passed = accuracy >= threshold

        # Create and return the validation result
        result = ValidationResult(
            test_id=test_query.test_id,
            query_text=test_query.query_text,
            expected_results=test_query.expected_results,
            success_criteria=test_query.success_criteria,
            test_category=test_query.test_category,
            executed_at=datetime.utcnow(),
            result_accuracy=accuracy,
            actual_results=actual_result_dicts,
            passed=passed
        )

        return result

    def _calculate_accuracy(self, expected_results: List[Dict[str, Any]],
                          actual_results: List[Dict[str, Any]],
                          success_criteria: str) -> float:
        """
        Calculate accuracy based on the success criteria.

        Args:
            expected_results: Expected results for the query
            actual_results: Actual results from the retrieval service
            success_criteria: Criteria for determining success

        Returns:
            Accuracy score between 0 and 1
        """
        if not expected_results and not actual_results:
            return 1.0  # Both empty is considered accurate
        if not expected_results:
            return 0.0  # Expected nothing but got results
        if not actual_results:
            return 0.0  # Expected results but got nothing

        # For now, implement a simple content-based similarity check
        # In a real implementation, this would be more sophisticated
        matches = 0
        total_expected = len(expected_results)

        for expected in expected_results:
            expected_content = expected.get('content', '').lower()
            for actual in actual_results:
                actual_content = actual.get('content', '').lower()

                # Simple similarity check - if content contains expected content or vice versa
                if (expected_content in actual_content or
                    actual_content in expected_content or
                    self._calculate_content_similarity(expected_content, actual_content) > 0.7):
                    matches += 1
                    break  # Each expected result should match at most one actual result

        accuracy = matches / total_expected if total_expected > 0 else 0.0
        return min(accuracy, 1.0)  # Ensure accuracy doesn't exceed 1.0

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate a simple similarity score between two content strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0 and 1
        """
        if not content1 and not content2:
            return 1.0
        if not content1 or not content2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 and not words2:
            return 1.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 1.0

        # Jaccard similarity
        return len(intersection) / len(union)

    def create_validation_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Create a validation report with detailed results.

        Args:
            validation_results: List of validation results

        Returns:
            Dictionary containing validation report metrics
        """
        if not validation_results:
            return {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'overall_accuracy': 0.0,
                'average_accuracy': 0.0,
                'test_categories': {}
            }

        total_tests = len(validation_results)
        passed_tests = sum(1 for result in validation_results if result.passed)
        failed_tests = total_tests - passed_tests
        overall_accuracy = sum(result.result_accuracy for result in validation_results) / total_tests

        # Group by test category
        categories = {}
        for result in validation_results:
            category = result.test_category
            if category not in categories:
                categories[category] = {
                    'total': 0,
                    'passed': 0,
                    'accuracy_sum': 0.0
                }
            categories[category]['total'] += 1
            if result.passed:
                categories[category]['passed'] += 1
            categories[category]['accuracy_sum'] += result.result_accuracy

        # Calculate category metrics
        category_metrics = {}
        for category, data in categories.items():
            category_metrics[category] = {
                'total': data['total'],
                'passed': data['passed'],
                'accuracy': data['accuracy_sum'] / data['total'] if data['total'] > 0 else 0.0
            }

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'overall_accuracy': overall_accuracy,
            'average_accuracy': overall_accuracy,  # Same in this simple implementation
            'test_categories': category_metrics
        }