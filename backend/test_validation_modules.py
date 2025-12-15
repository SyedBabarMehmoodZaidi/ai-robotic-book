"""
Comprehensive test suite for RAG retrieval validation modules.

This test suite covers all validation modules including:
- Query conversion (query_converter.py)
- Similarity search (retrieval_validator.py)
- Result validation (result_validator.py)
- Validation reporting (validation_reporter.py)
- Configuration management (config.py)
"""
import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

# Add backend to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ValidationConfig, validate_query_text, validate_top_k, validate_retrieved_chunks
from query_converter import query_to_vector
from retrieval_validator import perform_similarity_search
from result_validator import validate_retrieved_chunks, validate_quality_against_expected_results
from result_validator import assess_content_relevance, validate_relevance_threshold, validate_metadata
from validation_reporter import generate_validation_report, export_validation_report, print_validation_summary
from test_queries import TestQuerySuite


class TestConfigModule(unittest.TestCase):
    """Test configuration module functionality"""

    def test_config_defaults(self):
        """Test that Config class has expected default values"""
        self.assertIsInstance(Config.COHERE_API_KEY, str)
        self.assertIsNotNone(Config.LOCAL_QDRANT_PATH)
        self.assertEqual(Config.TOP_K_RESULTS, 5)
        self.assertEqual(Config.SIMILARITY_THRESHOLD, 0.5)
        self.assertEqual(Config.QUALITY_THRESHOLD, 0.9)
        self.assertEqual(Config.METADATA_ACCURACY_THRESHOLD, 1.0)
        self.assertEqual(Config.RETRIEVAL_SUCCESS_RATE_THRESHOLD, 0.95)
        self.assertEqual(Config.MIN_CHUNK_LENGTH, 50)
        self.assertEqual(Config.MAX_CHUNK_LENGTH, 2000)
        self.assertEqual(Config.MIN_SENTENCES_FOR_QUALITY, 5)
        self.assertEqual(Config.COHERE_MODEL, 'embed-multilingual-v3.0')
        self.assertEqual(Config.COHERE_INPUT_TYPE, 'search_query')
        self.assertEqual(Config.MAX_QUERY_LENGTH, 1000)
        self.assertEqual(Config.MAX_RETRIES, 3)
        self.assertEqual(Config.RETRY_DELAY, 1.0)

    def test_validation_config_methods(self):
        """Test ValidationConfig methods"""
        weights = ValidationConfig.get_quality_weights()
        self.assertIn('length_weight', weights)
        self.assertIn('keyword_weight', weights)
        self.assertIn('sentence_quality_weight', weights)

        rules = ValidationConfig.get_validation_rules()
        self.assertIn('min_length', rules)
        self.assertIn('max_length', rules)
        self.assertIn('min_sentences', rules)
        self.assertIn('similarity_threshold', rules)

        thresholds = ValidationConfig.get_thresholds()
        self.assertIn('quality', thresholds)
        self.assertIn('metadata_accuracy', thresholds)
        self.assertIn('retrieval_success_rate', thresholds)

    def test_validate_query_text(self):
        """Test query text validation function"""
        # Valid queries
        self.assertTrue(validate_query_text("Valid query"))
        self.assertTrue(validate_query_text("A"))

        # Invalid queries
        self.assertFalse(validate_query_text(""))
        self.assertFalse(validate_query_text("   "))
        self.assertFalse(validate_query_text(None))
        self.assertFalse(validate_query_text(123))
        self.assertFalse(validate_query_text("x" * 1001))  # Too long

    def test_validate_top_k(self):
        """Test top-k validation function"""
        # Valid values
        self.assertTrue(validate_top_k(1))
        self.assertTrue(validate_top_k(5))
        self.assertTrue(validate_top_k(100))

        # Invalid values
        self.assertFalse(validate_top_k(0))
        self.assertFalse(validate_top_k(-1))
        self.assertFalse(validate_top_k(101))
        self.assertFalse(validate_top_k("5"))

    def test_validate_retrieved_chunks(self):
        """Test chunk validation function"""
        valid_chunks = [
            {
                'content': 'Test content',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'},
                'score': 0.8,
            }
        ]
        self.assertTrue(validate_retrieved_chunks(valid_chunks))

        invalid_chunks = [
            {
                'content': 'Test content',
                'score': 0.8,
                # Missing metadata
            }
        ]
        self.assertFalse(validate_retrieved_chunks(invalid_chunks))

        not_a_list = "not a list"
        self.assertFalse(validate_retrieved_chunks(not_a_list))


class TestQueryConverter(unittest.TestCase):
    """Test query conversion functionality"""

    def test_query_to_vector_with_valid_input(self):
        """Test converting valid query text to vector"""
        with patch('query_converter.get_cohere_client') as mock_client:
            # Mock the Cohere embed response
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_client.return_value.embed.return_value = mock_response

            result = query_to_vector("Test query")

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 5)
            mock_client.return_value.embed.assert_called_once()

    def test_query_to_vector_with_empty_input(self):
        """Test converting empty query text to vector"""
        with self.assertRaises(ValueError):
            query_to_vector("")

    def test_query_to_vector_with_long_input(self):
        """Test converting too long query text to vector"""
        long_query = "x" * 1001  # Exceeds MAX_QUERY_LENGTH
        with self.assertRaises(ValueError):
            query_to_vector(long_query)

    def test_query_to_vector_with_invalid_input(self):
        """Test converting invalid query text to vector"""
        with self.assertRaises(ValueError):
            query_to_vector(None)

        with self.assertRaises(ValueError):
            query_to_vector(123)


class TestRetrievalValidator(unittest.TestCase):
    """Test retrieval validation functionality"""

    def test_perform_similarity_search_with_mock_client(self):
        """Test similarity search with mocked Qdrant client"""
        with patch('retrieval_validator.get_qdrant_client') as mock_client:
            # Mock the search response
            mock_point = Mock()
            mock_point.payload = {
                'content': 'Test content',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'}
            }
            mock_point.score = 0.85
            mock_client.return_value.search.return_value = [mock_point]

            query_vector = [0.1, 0.2, 0.3]
            results = perform_similarity_search(query_vector, top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['content'], 'Test content')
            self.assertEqual(results[0]['score'], 0.85)
            mock_client.return_value.search.assert_called_once()


class TestResultValidator(unittest.TestCase):
    """Test result validation functionality"""

    def test_validate_retrieved_chunks_basic(self):
        """Test basic chunk validation"""
        test_chunks = [
            {
                'content': 'Artificial intelligence is a wonderful field.',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'},
                'score': 0.85,
                'id': 'chunk_1'
            },
            {
                'content': 'This is not relevant content at all.',
                'metadata': {'url': 'test.com', 'section': 'other', 'chunk_id': '2'},
                'score': 0.2,
                'id': 'chunk_2'
            }
        ]

        results = validate_retrieved_chunks(test_chunks)

        # Should have individual results plus overall stats
        self.assertEqual(len(results), 3)  # 2 chunks + 1 overall stats

        # Check the overall stats
        overall_stats = results[-1]['overall_stats']
        self.assertIn('total_chunks', overall_stats)
        self.assertIn('relevant_chunks', overall_stats)
        self.assertIn('relevant_percentage', overall_stats)

    def test_validate_retrieved_chunks_with_expected_content(self):
        """Test chunk validation with expected content for relevance assessment"""
        test_chunks = [
            {
                'content': 'Artificial intelligence and machine learning are related fields.',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'},
                'score': 0.85,
                'id': 'chunk_1'
            }
        ]
        expected_content = ["artificial intelligence", "machine learning"]

        results = validate_retrieved_chunks(test_chunks, expected_content=expected_content)

        # The first chunk should have a higher relevance score due to keyword matching
        chunk_result = results[0]
        self.assertGreaterEqual(chunk_result['relevance_score'], 0.0)
        self.assertIsInstance(chunk_result['relevance_score'], float)

    def test_validate_retrieved_chunks_empty_content(self):
        """Test chunk validation with empty content"""
        test_chunks = [
            {
                'content': '',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'},
                'score': 0.85,
                'id': 'chunk_1'
            }
        ]

        results = validate_retrieved_chunks(test_chunks)

        chunk_result = results[0]
        self.assertEqual(chunk_result['relevance_score'], 0.0)
        self.assertEqual(chunk_result['content_relevant'], False)

    def test_validate_retrieved_chunks_missing_metadata(self):
        """Test chunk validation with missing metadata"""
        test_chunks = [
            {
                'content': 'Valid content',
                'score': 0.85,
                'id': 'chunk_1'
            }
        ]

        results = validate_retrieved_chunks(test_chunks)

        chunk_result = results[0]
        self.assertEqual(chunk_result['metadata_valid'], False)
        self.assertIn('missing_metadata', chunk_result['validation_details'])

    def test_assess_content_relevance(self):
        """Test content relevance assessment"""
        content = "Artificial intelligence is a wonderful field that involves creating smart machines."
        expected_keywords = ["artificial intelligence", "smart machines"]

        relevance_score = assess_content_relevance(content, expected_keywords)

        self.assertIsInstance(relevance_score, float)
        self.assertGreaterEqual(relevance_score, 0.0)
        self.assertLessEqual(relevance_score, 1.0)

    def test_assess_content_relevance_no_expected_content(self):
        """Test content relevance assessment without expected content"""
        content = "This is a sample content without specific expected keywords."

        relevance_score = assess_content_relevance(content)

        self.assertIsInstance(relevance_score, float)
        self.assertGreaterEqual(relevance_score, 0.0)
        self.assertLessEqual(relevance_score, 1.0)

    def test_validate_relevance_threshold(self):
        """Test relevance threshold validation"""
        # Create mock validation results with high relevance
        high_relevance_results = [
            {'content_relevant': True, 'chunk_index': 0},
            {'content_relevant': True, 'chunk_index': 1},
            {'overall_stats': {'relevant_percentage': 100.0}}
        ]

        low_relevance_results = [
            {'content_relevant': False, 'chunk_index': 0},
            {'content_relevant': False, 'chunk_index': 1},
            {'overall_stats': {'relevant_percentage': 0.0}}
        ]

        # Test with 90% threshold (default)
        self.assertTrue(validate_relevance_threshold(high_relevance_results))
        self.assertFalse(validate_relevance_threshold(low_relevance_results))

        # Test with custom threshold
        self.assertTrue(validate_relevance_threshold(low_relevance_results, threshold=0.0))

    def test_validate_metadata(self):
        """Test metadata validation"""
        valid_metadata = [
            {'url': 'https://example.com', 'section': 'intro', 'chunk_id': '1'},
            {'url': 'https://example.com/2', 'section': 'chapter1', 'chunk_id': '2'}
        ]

        accuracy = validate_metadata(valid_metadata)

        self.assertEqual(accuracy, 100.0)  # All metadata is valid

    def test_validate_metadata_with_missing_fields(self):
        """Test metadata validation with missing fields"""
        invalid_metadata = [
            {'url': 'https://example.com', 'section': 'intro', 'chunk_id': '1'},  # Valid
            {'url': 'https://example.com/2', 'section': 'chapter1'},  # Missing chunk_id
            {'section': 'chapter2', 'chunk_id': '3'},  # Missing url
        ]

        accuracy = validate_metadata(invalid_metadata)

        # 1 out of 3 is valid = 33.33%
        self.assertEqual(accuracy, 33.33)

    def test_validate_metadata_with_empty_values(self):
        """Test metadata validation with empty values"""
        invalid_metadata = [
            {'url': '', 'section': 'intro', 'chunk_id': '1'},  # Empty url
            {'url': 'https://example.com', 'section': '', 'chunk_id': '2'},  # Empty section
            {'url': 'https://example.com', 'section': 'intro', 'chunk_id': ''},  # Empty chunk_id
        ]

        accuracy = validate_metadata(invalid_metadata)

        # All have empty values = 0% accuracy
        self.assertEqual(accuracy, 0.0)


class TestValidationReporter(unittest.TestCase):
    """Test validation reporting functionality"""

    def test_generate_validation_report_basic(self):
        """Test basic validation report generation"""
        sample_validation_results = [
            {
                'chunk_index': 0,
                'chunk_id': 'chunk_1',
                'content_length': 150,
                'similarity_score': 0.92,
                'metadata_valid': True,
                'content_relevant': True,
                'relevance_score': 0.85,
                'validation_details': {'content_preview': 'Artificial intelligence is...'}
            },
            {
                'chunk_index': 1,
                'chunk_id': 'chunk_2',
                'content_length': 200,
                'similarity_score': 0.88,
                'metadata_valid': True,
                'content_relevant': True,
                'relevance_score': 0.78,
                'validation_details': {'content_preview': 'Machine learning algorithms...'}
            },
            {
                'overall_stats': {
                    'total_chunks': 2,
                    'relevant_chunks': 2,
                    'relevant_percentage': 100.0,
                    'valid_metadata_chunks': 2,
                    'metadata_validity_percentage': 100.0
                }
            }
        ]

        sample_times = [125.5, 98.2, 142.7]

        report = generate_validation_report(
            validation_results=sample_validation_results,
            metadata_accuracy=100.0,
            query_execution_times=sample_times,
            total_queries=3,
            successful_queries=3
        )

        # Check that report has expected attributes
        self.assertTrue(hasattr(report, 'timestamp'))
        self.assertEqual(report.total_queries_executed, 3)
        self.assertEqual(report.successful_retrievals, 3)
        self.assertEqual(report.metadata_accuracy, 100.0)
        self.assertTrue(report.quality_threshold_met)
        self.assertTrue(report.metadata_accuracy_met)

    def test_generate_validation_report_with_failure_cases(self):
        """Test validation report generation with failure cases"""
        sample_validation_results = [
            {
                'chunk_index': 0,
                'chunk_id': 'chunk_1',
                'content_length': 150,
                'similarity_score': 0.92,
                'metadata_valid': True,
                'content_relevant': False,  # Not relevant
                'relevance_score': 0.3,    # Low relevance
                'validation_details': {'content_preview': 'Artificial intelligence is...'}
            },
            {
                'chunk_index': 1,
                'chunk_id': 'chunk_2',
                'content_length': 200,
                'similarity_score': 0.88,
                'metadata_valid': False,   # Invalid metadata
                'content_relevant': True,
                'relevance_score': 0.78,
                'validation_details': {'content_preview': 'Machine learning algorithms...'}
            },
            {
                'overall_stats': {
                    'total_chunks': 2,
                    'relevant_chunks': 1,    # Only 1 relevant
                    'relevant_percentage': 50.0,  # 50% relevance
                    'valid_metadata_chunks': 1,   # Only 1 valid metadata
                    'metadata_validity_percentage': 50.0  # 50% metadata valid
                }
            }
        ]

        report = generate_validation_report(
            validation_results=sample_validation_results,
            metadata_accuracy=50.0,  # 50% metadata accuracy
            total_queries=2,
            successful_queries=1  # Only 1 successful query
        )

        # With 50% relevance, quality threshold (90%) should not be met
        self.assertFalse(report.quality_threshold_met)
        # With 50% metadata accuracy, metadata threshold (100%) should not be met
        self.assertFalse(report.metadata_accuracy_met)
        # With 50% success rate (1/2), success rate threshold (95%) should not be met
        self.assertEqual(report.retrieval_success_rate, 50.0)
        self.assertFalse(report.summary['overall_validation_passed'])

    def test_export_validation_report_json(self):
        """Test JSON export of validation report"""
        sample_validation_results = [
            {
                'chunk_index': 0,
                'chunk_id': 'chunk_1',
                'content_length': 150,
                'similarity_score': 0.92,
                'metadata_valid': True,
                'content_relevant': True,
                'relevance_score': 0.85,
                'validation_details': {'content_preview': 'Artificial intelligence is...'}
            },
            {
                'overall_stats': {
                    'total_chunks': 1,
                    'relevant_chunks': 1,
                    'relevant_percentage': 100.0,
                    'valid_metadata_chunks': 1,
                    'metadata_validity_percentage': 100.0
                }
            }
        ]

        report = generate_validation_report(
            validation_results=sample_validation_results,
            metadata_accuracy=100.0,
            total_queries=1,
            successful_queries=1
        )

        # Export to JSON format
        export_path = export_validation_report(report, format='json', filename='test_report.json')

        self.assertTrue(export_path.endswith('.json'))

        # Clean up
        if os.path.exists(export_path):
            os.remove(export_path)

    def test_export_validation_report_txt(self):
        """Test TXT export of validation report"""
        sample_validation_results = [
            {
                'chunk_index': 0,
                'chunk_id': 'chunk_1',
                'content_length': 150,
                'similarity_score': 0.92,
                'metadata_valid': True,
                'content_relevant': True,
                'relevance_score': 0.85,
                'validation_details': {'content_preview': 'Artificial intelligence is...'}
            },
            {
                'overall_stats': {
                    'total_chunks': 1,
                    'relevant_chunks': 1,
                    'relevant_percentage': 100.0,
                    'valid_metadata_chunks': 1,
                    'metadata_validity_percentage': 100.0
                }
            }
        ]

        report = generate_validation_report(
            validation_results=sample_validation_results,
            metadata_accuracy=100.0,
            total_queries=1,
            successful_queries=1
        )

        # Export to TXT format
        export_path = export_validation_report(report, format='txt', filename='test_report.txt')

        self.assertTrue(export_path.endswith('.txt'))

        # Clean up
        if os.path.exists(export_path):
            os.remove(export_path)


class TestIntegration(unittest.TestCase):
    """Test integration between modules"""

    def test_validate_quality_against_expected_results(self):
        """Test quality validation against expected results"""
        # Create a mock test suite
        mock_test_suite = Mock()
        mock_test_suite.get_expected_keywords_for_query.return_value = ["artificial intelligence", "machine learning"]
        mock_test_suite.evaluate_retrieval_quality.return_value = {
            'relevance_score': 0.95,
            'found_keywords': ["artificial intelligence"],
            'missing_keywords': ["machine learning"],
            'keyword_coverage': 0.5
        }

        test_chunks = [
            {
                'content': 'Artificial intelligence is a wonderful field.',
                'metadata': {'url': 'test.com', 'section': 'intro', 'chunk_id': '1'},
                'score': 0.85,
                'id': 'chunk_1'
            }
        ]

        result = validate_quality_against_expected_results("What is AI?", test_chunks, mock_test_suite)

        self.assertEqual(result['query'], "What is AI?")
        self.assertGreaterEqual(result['relevance_score'], 0.0)
        self.assertLessEqual(result['relevance_score'], 1.0)
        self.assertIsInstance(result['chunk_validation_results'], list)
        self.assertIn('quality_passed', result)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the validation pipeline"""

    @patch('config.get_cohere_client')
    @patch('config.get_qdrant_client')
    def test_complete_validation_cycle(self, mock_qdrant_client, mock_cohere_client):
        """Test a complete validation cycle with mocked services"""
        # Mock Cohere client
        mock_cohere_response = Mock()
        mock_cohere_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_cohere_client.return_value.embed.return_value = mock_cohere_response

        # Mock Qdrant client
        mock_point = Mock()
        mock_point.payload = {
            'content': 'Artificial intelligence is a wonderful field that involves creating smart machines.',
            'metadata': {'url': 'https://example.com/ai', 'section': 'Introduction', 'chunk_id': 'chunk_001'}
        }
        mock_point.score = 0.85
        mock_qdrant_client.return_value.search.return_value = [mock_point]

        # Simulate the complete validation process
        # 1. Convert query to vector
        query_vector = query_to_vector("What is artificial intelligence?")
        self.assertIsInstance(query_vector, list)
        self.assertEqual(len(query_vector), 5)

        # 2. Perform similarity search
        results = perform_similarity_search(query_vector, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn('content', results[0])
        self.assertIn('metadata', results[0])
        self.assertIn('score', results[0])

        # 3. Validate retrieved chunks
        validation_results = validate_retrieved_chunks(results)
        self.assertIsInstance(validation_results, list)
        self.assertGreaterEqual(len(validation_results), 1)

        # 4. Generate validation report
        report = generate_validation_report(
            validation_results=validation_results,
            metadata_accuracy=100.0,  # Assuming perfect metadata for this test
            total_queries=1,
            successful_queries=1
        )

        # 5. Verify report properties
        self.assertIsNotNone(report.timestamp)
        self.assertEqual(report.total_queries_executed, 1)
        self.assertEqual(report.successful_retrievals, 1)
        self.assertGreaterEqual(report.metadata_accuracy, 0.0)


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)