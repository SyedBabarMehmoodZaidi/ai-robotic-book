"""
Integration module for RAG retrieval validation.
This module integrates query conversion, similarity search, and validation functions.
"""
import time
from typing import List, Dict, Tuple, Optional
import logging
from query_converter import query_to_vector
from retrieval_validator import perform_similarity_search, execute_similarity_search_query
from result_validator import (
    validate_retrieved_chunks,
    validate_quality_against_expected_results,
    validate_metadata_comprehensive
)
from validation_reporter import generate_validation_report, print_validation_summary
from test_queries import test_suite
from config import logger


def execute_single_validation_cycle(query: str, top_k: int = 5, expected_content: Optional[List[str]] = None) -> Dict:
    """
    Execute a complete validation cycle for a single query:
    1. Convert query to vector
    2. Perform similarity search
    3. Validate retrieved chunks
    4. Generate validation metrics

    Args:
        query (str): The query to validate
        top_k (int): Number of results to retrieve
        expected_content (List[str], optional): Expected content for validation

    Returns:
        Dict: Complete validation results for the query
    """
    start_time = time.time()

    try:
        # Step 1: Convert query to vector
        query_vector = query_to_vector(query)
        vector_conversion_time = time.time() - start_time

        # Step 2: Perform similarity search
        search_start = time.time()
        retrieved_chunks = perform_similarity_search(query_vector, top_k)
        search_time = time.time() - search_start

        # Step 3: Validate retrieved chunks
        validation_start = time.time()
        chunk_validation_results = validate_retrieved_chunks(retrieved_chunks, expected_content)
        chunk_validation_time = time.time() - validation_start

        # Step 4: Validate metadata
        metadata_validation_start = time.time()
        metadata_results = validate_metadata_comprehensive(retrieved_chunks)
        metadata_accuracy = metadata_results['accuracy_percentage']
        metadata_validation_time = time.time() - metadata_validation_start

        # Step 5: Quality validation against expected results if test suite query
        quality_validation_start = time.time()
        expected_keywords = test_suite.get_expected_keywords_for_query(query)
        if expected_keywords:
            quality_results = validate_quality_against_expected_results(query, retrieved_chunks, test_suite)
            quality_score = quality_results['relevance_score']
        else:
            # Calculate quality without expected content
            quality_score = sum(result.get('relevance_score', 0) for result in chunk_validation_results[:-1]) / len(chunk_validation_results[:-1]) if chunk_validation_results and len(chunk_validation_results) > 1 else 0
            quality_results = None
        quality_validation_time = time.time() - quality_validation_start

        # Calculate total time
        total_time = time.time() - start_time

        # Compile results
        result = {
            'query': query,
            'query_vector_length': len(query_vector) if query_vector else 0,
            'retrieved_chunks_count': len(retrieved_chunks),
            'retrieved_chunks': retrieved_chunks,
            'chunk_validation_results': chunk_validation_results,
            'metadata_validation_results': metadata_results,
            'metadata_accuracy': metadata_accuracy,
            'quality_score': quality_score,
            'quality_results': quality_results,
            'execution_times': {
                'vector_conversion': vector_conversion_time,
                'similarity_search': search_time,
                'chunk_validation': chunk_validation_time,
                'metadata_validation': metadata_validation_time,
                'quality_validation': quality_validation_time,
                'total': total_time
            },
            'success': True,
            'error': None
        }

        logger.info(f"Validation cycle completed for query '{query[:50]}...' in {total_time:.3f}s. "
                    f"Retrieved {len(retrieved_chunks)} chunks.")

        return result

    except Exception as e:
        error_time = time.time() - start_time
        error_result = {
            'query': query,
            'retrieved_chunks_count': 0,
            'retrieved_chunks': [],
            'chunk_validation_results': [],
            'metadata_validation_results': {},
            'metadata_accuracy': 0.0,
            'quality_score': 0.0,
            'quality_results': None,
            'execution_times': {
                'total': error_time
            },
            'success': False,
            'error': str(e)
        }

        logger.error(f"Validation cycle failed for query '{query}': {str(e)}")
        return error_result


def execute_batch_validation(queries: List[str], top_k: int = 5, use_test_suite: bool = True) -> Dict:
    """
    Execute validation for a batch of queries

    Args:
        queries (List[str]): List of queries to validate
        top_k (int): Number of results to retrieve for each query
        use_test_suite (bool): Whether to use test suite expected results for validation

    Returns:
        Dict: Comprehensive validation results for all queries
    """
    logger.info(f"Starting batch validation for {len(queries)} queries")

    all_results = []
    successful_queries = 0
    total_execution_times = []

    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: '{query[:30]}...'")

        expected_content = None
        if use_test_suite:
            expected_content = test_suite.get_expected_keywords_for_query(query)

        result = execute_single_validation_cycle(query, top_k, expected_content)
        all_results.append(result)

        if result['success']:
            successful_queries += 1
            total_execution_times.append(result['execution_times']['total'])

    # Calculate overall metrics
    total_queries = len(queries)
    success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0

    # Aggregate validation results for report generation
    all_chunk_validation_results = []
    all_metadata_accuracies = []

    for result in all_results:
        all_chunk_validation_results.extend(result.get('chunk_validation_results', []))
        all_metadata_accuracies.append(result.get('metadata_accuracy', 0))

    # Calculate average metadata accuracy
    avg_metadata_accuracy = sum(all_metadata_accuracies) / len(all_metadata_accuracies) if all_metadata_accuracies else 0

    batch_results = {
        'batch_metadata': {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': success_rate,
            'average_execution_time': sum(total_execution_times) / len(total_execution_times) if total_execution_times else 0,
            'queries': queries
        },
        'individual_results': all_results,
        'aggregate_chunk_validation': all_chunk_validation_results,
        'average_metadata_accuracy': avg_metadata_accuracy
    }

    logger.info(f"Batch validation completed. Success rate: {success_rate:.2f}%")
    return batch_results


def generate_comprehensive_validation_report(batch_results: Dict) -> Dict:
    """
    Generate a comprehensive validation report from batch results

    Args:
        batch_results (Dict): Results from execute_batch_validation

    Returns:
        Dict: Comprehensive validation report
    """
    batch_meta = batch_results['batch_metadata']
    chunk_validation = batch_results['aggregate_chunk_validation']
    avg_metadata_accuracy = batch_results['average_metadata_accuracy']

    # Calculate query execution times for the report
    query_times = [r['execution_times']['total'] for r in batch_results['individual_results'] if r['success']]

    # Generate the validation report
    report = generate_validation_report(
        validation_results=chunk_validation,
        metadata_accuracy=avg_metadata_accuracy,
        query_execution_times=query_times,
        total_queries=batch_meta['total_queries'],
        successful_queries=batch_meta['successful_queries']
    )

    return {
        'report': report,
        'batch_results': batch_results
    }


def run_complete_validation_pipeline(queries: Optional[List[str]] = None, top_k: int = 5) -> Dict:
    """
    Run the complete validation pipeline from queries to final report

    Args:
        queries (List[str], optional): List of queries to validate (uses test suite if None)
        top_k (int): Number of results to retrieve for each query

    Returns:
        Dict: Complete pipeline results with validation report
    """
    if queries is None:
        # Use test suite queries
        queries = [q.query for q in test_suite.get_all_queries()[:5]]  # Use first 5 queries as sample
        logger.info(f"Using test suite queries: {len(queries)} queries selected")

    logger.info("Starting complete validation pipeline...")

    # Execute batch validation
    batch_results = execute_batch_validation(queries, top_k)

    # Generate comprehensive report
    report_data = generate_comprehensive_validation_report(batch_results)

    # Print summary
    print_validation_summary(report_data['report'])

    logger.info("Complete validation pipeline finished successfully")

    return {
        'pipeline_results': report_data,
        'batch_results': batch_results
    }


if __name__ == "__main__":
    # Example usage
    print("Running example validation pipeline...")

    # Use a few sample queries
    sample_queries = [
        "What is artificial intelligence?",
        "Explain neural networks",
        "How does machine learning work?"
    ]

    # Run the complete pipeline
    results = run_complete_validation_pipeline(sample_queries, top_k=3)

    print(f"\nPipeline completed with {results['pipeline_results']['report'].total_queries_executed} queries processed.")
    print(f"Success rate: {results['pipeline_results']['report'].retrieval_success_rate:.2f}%")