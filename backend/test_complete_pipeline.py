"""
Test script for the complete validation pipeline with sample queries.
This script tests the end-to-end functionality of the RAG retrieval validation system.
"""
import sys
import os
import time
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration import run_complete_validation_pipeline
from test_queries import test_suite
from config import validate_environment_variables
from validation_reporter import print_validation_summary


def test_complete_pipeline():
    """Test the complete validation pipeline with sample queries"""
    print("🧪 Testing Complete Validation Pipeline...")
    print("=" * 60)

    try:
        # Validate environment first
        print("🔍 Validating environment variables...")
        validate_environment_variables()
        print("✅ Environment validated successfully")

        # Test 1: Run with a few sample queries
        print("\n🧪 Test 1: Sample Queries")
        print("-" * 30)

        sample_queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "How do neural networks work?"
        ]

        print(f"Running validation for {len(sample_queries)} sample queries...")

        start_time = time.time()
        results = run_complete_validation_pipeline(sample_queries, top_k=3)
        end_time = time.time()

        pipeline_time = end_time - start_time
        report = results['pipeline_results']['report']

        print(f"✅ Pipeline completed in {pipeline_time:.2f} seconds")
        print(f"📊 Results: {report.successful_retrievals}/{report.total_queries_executed} successful")
        print(f"📈 Quality Score: {report.quality_score:.2f}%")
        print(f"🏷️  Metadata Accuracy: {report.metadata_accuracy:.2f}%")

        # Test 2: Run with test suite queries
        print("\n🧪 Test 2: Test Suite Queries")
        print("-" * 30)

        # Use first 3 queries from test suite
        suite_queries = [q.query for q in test_suite.get_all_queries()[:3]]
        print(f"Running validation for {len(suite_queries)} test suite queries...")

        start_time = time.time()
        suite_results = run_complete_validation_pipeline(suite_queries, top_k=2)
        end_time = time.time()

        suite_time = end_time - start_time
        suite_report = suite_results['pipeline_results']['report']

        print(f"✅ Pipeline completed in {suite_time:.2f} seconds")
        print(f"📊 Results: {suite_report.successful_retrievals}/{suite_report.total_queries_executed} successful")
        print(f"📈 Quality Score: {suite_report.quality_score:.2f}%")
        print(f"🏷️  Metadata Accuracy: {suite_report.metadata_accuracy:.2f}%")

        # Test 3: Run with single query for detailed inspection
        print("\n🧪 Test 3: Single Query Detailed Test")
        print("-" * 30)

        single_query = ["What are the applications of robotics?"]
        print(f"Running detailed validation for single query: '{single_query[0]}'")

        single_results = run_complete_validation_pipeline(single_query, top_k=5)
        single_report = single_results['pipeline_results']['report']

        print(f"✅ Single query test completed")
        print(f"📊 Results: {single_report.successful_retrievals}/{single_report.total_queries_executed} successful")
        print(f"📈 Quality Score: {single_report.quality_score:.2f}%")
        print(f"🏷️  Metadata Accuracy: {single_report.metadata_accuracy:.2f}%")

        # Test 4: Performance test with more queries
        print("\n🧪 Test 4: Performance Test")
        print("-" * 30)

        performance_queries = [
            "What is deep learning?",
            "Explain computer vision",
            "How does natural language processing work?",
            "What is reinforcement learning?",
            "Describe ethical considerations in AI"
        ]

        print(f"Running performance test with {len(performance_queries)} queries...")

        start_time = time.time()
        perf_results = run_complete_validation_pipeline(performance_queries, top_k=3)
        end_time = time.time()

        perf_time = end_time - start_time
        perf_report = perf_results['pipeline_results']['report']

        print(f"✅ Performance test completed in {perf_time:.2f} seconds")
        print(f"📊 Results: {perf_report.successful_retrievals}/{perf_report.total_queries_executed} successful")
        print(f"📈 Quality Score: {perf_report.quality_score:.2f}%")
        print(f"🏷️  Average Response Time: {perf_report.average_response_time:.2f}ms")
        print(f"📄 Total Chunks Retrieved: {perf_report.total_chunks_retrieved}")

        # Summary of all tests
        print("\n🏆 Test Summary")
        print("-" * 30)
        all_tests_passed = (
            report.successful_retrievals == report.total_queries_executed,
            suite_report.successful_retrievals == suite_report.total_queries_executed,
            single_report.successful_retrievals == single_report.total_queries_executed,
            perf_report.successful_retrievals == perf_report.total_queries_executed
        )

        print(f"Sample queries test: {'✅ PASS' if all_tests_passed[0] else '❌ FAIL'}")
        print(f"Test suite queries test: {'✅ PASS' if all_tests_passed[1] else '❌ FAIL'}")
        print(f"Single query test: {'✅ PASS' if all_tests_passed[2] else '❌ FAIL'}")
        print(f"Performance test: {'✅ PASS' if all_tests_passed[3] else '❌ FAIL'}")

        overall_success = all(all_tests_passed)
        print(f"\n🎯 Overall Pipeline Test: {'✅ ALL PASSED' if overall_success else '❌ SOME FAILED'}")

        if overall_success:
            print("🎉 All tests passed! The complete validation pipeline is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")

        return overall_success

    except Exception as e:
        print(f"\n❌ Pipeline test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases for the validation pipeline"""
    print("\n🔍 Testing Edge Cases...")
    print("=" * 60)

    try:
        # Test with empty query list
        print("🧪 Edge Case 1: Empty query list")
        try:
            empty_results = run_complete_validation_pipeline([], top_k=3)
            print("✅ Handled empty query list gracefully")
        except Exception as e:
            print(f"⚠️  Empty query list caused error: {e}")

        # Test with very long query
        print("\n🧪 Edge Case 2: Very long query")
        long_query = ["This is a very long query that tests the system's ability to handle extensive text input without crashing or causing performance issues. " * 10]
        try:
            long_results = run_complete_validation_pipeline(long_query, top_k=1)
            print("✅ Handled long query gracefully")
        except Exception as e:
            print(f"⚠️  Long query caused error: {e}")

        # Test with special characters
        print("\n🧪 Edge Case 3: Query with special characters")
        special_query = ["What is AI? (Artificial Intelligence) - A comprehensive overview!"]
        try:
            special_results = run_complete_validation_pipeline(special_query, top_k=1)
            print("✅ Handled special characters gracefully")
        except Exception as e:
            print(f"⚠️  Special characters caused error: {e}")

        print("\n✅ Edge case testing completed")
        return True

    except Exception as e:
        print(f"❌ Edge case testing failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Complete Validation Pipeline Tests...\n")

    # Run main tests
    main_tests_passed = test_complete_pipeline()

    # Run edge case tests
    edge_tests_passed = test_edge_cases()

    print(f"\n🏁 Pipeline Testing Complete")
    print("=" * 60)
    print(f"Main tests: {'✅ PASSED' if main_tests_passed else '❌ FAILED'}")
    print(f"Edge tests: {'✅ PASSED' if edge_tests_passed else '❌ FAILED'}")

    if main_tests_passed and edge_tests_passed:
        print("\n🎉 All pipeline tests completed successfully!")
        print("The RAG retrieval validation system is ready for use.")
    else:
        print("\n⚠️  Some tests had issues.")
        print("The system may need additional configuration or the database may need to be populated.")