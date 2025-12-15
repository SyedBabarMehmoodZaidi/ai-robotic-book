import re
from typing import List, Dict, Optional
import logging
from config import Config, logger


def validate_retrieved_chunks(retrieved_chunks: List[Dict], expected_content: Optional[List[str]] = None) -> List[Dict]:
    """
    Validates the accuracy and relevance of retrieved chunks

    Args:
        retrieved_chunks (List[Dict]): The chunks retrieved from the similarity search
        expected_content (List[str], optional): Expected content for validation

    Returns:
        List[Dict]: Validation results for each chunk with relevance scores
    """
    validation_results = []

    for i, chunk in enumerate(retrieved_chunks):
        # Initialize validation result for this chunk
        validation_result = {
            'chunk_index': i,
            'chunk_id': chunk.get('id', f'chunk_{i}'),
            'content_length': len(chunk.get('content', '')),
            'similarity_score': chunk.get('score', 0.0),
            'metadata_valid': True,
            'content_relevant': False,
            'relevance_score': 0.0,
            'validation_details': {}
        }

        # Validate content exists and has minimum length
        content = chunk.get('content', '')
        if not content or len(content.strip()) == 0:
            validation_result['validation_details']['content_error'] = 'Chunk has empty content'
            validation_results.append(validation_result)
            continue

        # Validate metadata exists
        metadata = chunk.get('metadata', {})
        if not metadata:
            validation_result['metadata_valid'] = False
            validation_result['validation_details']['metadata_error'] = 'Chunk has no metadata'
        else:
            # Check required metadata fields
            required_fields = ['url', 'section', 'chunk_id']
            missing_fields = [field for field in required_fields if field not in metadata or not metadata.get(field)]
            if missing_fields:
                validation_result['metadata_valid'] = False
                validation_result['validation_details']['missing_metadata'] = missing_fields

        # Assess content relevance
        relevance_score = assess_content_relevance(content, expected_content)
        validation_result['relevance_score'] = relevance_score
        validation_result['content_relevant'] = relevance_score >= Config.SIMILARITY_THRESHOLD

        # Additional validation checks
        validation_result['validation_details']['content_preview'] = content[:100] + "..." if len(content) > 100 else content

        # Check for quality indicators using configuration parameters
        quality_issues = []
        if len(content) < Config.MIN_CHUNK_LENGTH:  # Too short to be meaningful
            quality_issues.append('content_too_short')
        if len(content) > Config.MAX_CHUNK_LENGTH:  # Too long
            quality_issues.append('content_too_long')
        if content.count('\n') > 10:  # Potentially contains too many line breaks
            quality_issues.append('excessive_line_breaks')
        if len(re.findall(r'[A-Z][a-z]+', content)) < Config.MIN_SENTENCES_FOR_QUALITY:  # Likely not proper sentences
            quality_issues.append('poor_sentence_structure')

        if quality_issues:
            validation_result['validation_details']['quality_issues'] = quality_issues

        validation_results.append(validation_result)

    # Calculate overall validation statistics
    total_chunks = len(validation_results)
    if total_chunks > 0:
        relevant_chunks = sum(1 for vr in validation_results if vr['content_relevant'])
        valid_metadata_chunks = sum(1 for vr in validation_results if vr['metadata_valid'])

        overall_stats = {
            'total_chunks': total_chunks,
            'relevant_chunks': relevant_chunks,
            'relevant_percentage': (relevant_chunks / total_chunks) * 100,
            'valid_metadata_chunks': valid_metadata_chunks,
            'metadata_validity_percentage': (valid_metadata_chunks / total_chunks) * 100
        }

        logger.info(f"Validation completed: {relevant_chunks}/{total_chunks} chunks relevant ({overall_stats['relevant_percentage']:.1f}%)")
    else:
        overall_stats = {
            'total_chunks': 0,
            'relevant_chunks': 0,
            'relevant_percentage': 0.0,
            'valid_metadata_chunks': 0,
            'metadata_validity_percentage': 0.0
        }
        logger.info("Validation completed: No chunks to validate")

    # Add overall stats to the results
    validation_results.append({'overall_stats': overall_stats})

    return validation_results


def validate_quality_against_expected_results(query: str, retrieved_chunks: List[Dict], test_suite) -> Dict:
    """
    Implement automated quality validation against expected results from test suite

    Args:
        query (str): The original query
        retrieved_chunks (List[Dict]): The chunks retrieved from the similarity search
        test_suite: The test query suite with expected results

    Returns:
        Dict: Quality validation results
    """
    # Get expected keywords for this query from the test suite
    expected_keywords = test_suite.get_expected_keywords_for_query(query)

    if not expected_keywords:
        # If query not in test suite, return basic validation
        validation_results = validate_retrieved_chunks(retrieved_chunks)
        return {
            'query': query,
            'expected_keywords': [],
            'found_keywords': [],
            'missing_keywords': [],
            'keyword_coverage': 0.0,
            'relevance_score': 0.0,
            'chunk_validation_results': validation_results,
            'quality_passed': False,
            'validation_message': f'Query "{query}" not found in test suite'
        }

    # Combine all content from retrieved chunks
    all_content = " ".join([chunk.get('content', '') for chunk in retrieved_chunks])

    # Evaluate quality against expected results
    evaluation = test_suite.evaluate_retrieval_quality(query, all_content)

    # Run detailed chunk validation
    chunk_validation_results = validate_retrieved_chunks(retrieved_chunks, expected_keywords)

    # Calculate overall quality metrics
    quality_passed = evaluation['relevance_score'] >= 0.9  # Using 90% as threshold per spec requirement

    quality_results = {
        'query': query,
        'expected_keywords': expected_keywords,
        'found_keywords': evaluation['found_keywords'],
        'missing_keywords': evaluation['missing_keywords'],
        'keyword_coverage': evaluation['keyword_coverage'],
        'relevance_score': evaluation['relevance_score'],
        'chunk_validation_results': chunk_validation_results,
        'quality_passed': quality_passed,
        'validation_message': f"Quality check {'passed' if quality_passed else 'failed'} - Coverage: {evaluation['keyword_coverage']:.2f}"
    }

    logger.info(f"Quality validation for '{query}': {quality_results['validation_message']}")

    return quality_results


def assess_content_relevance(content: str, expected_content: Optional[List[str]] = None) -> float:
    """
    Assess the contextual relevance of content based on various factors.

    Args:
        content (str): The content to assess
        expected_content (List[str], optional): Expected content for direct comparison

    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    if not content or len(content.strip()) == 0:
        return 0.0

    from config import ValidationConfig

    relevance_score = 0.0
    score_components = {}

    # Content length factor (normalized)
    # Use a reference length from config or default to 500 chars
    ref_length = 500  # Could be made configurable
    length_score = min(len(content) / ref_length, 1.0)
    score_components['length_score'] = length_score

    # Keyword relevance if expected content is provided
    keyword_score = 0.0
    if expected_content:
        content_lower = content.lower()
        matched_keywords = 0
        total_keywords = 0

        for expected in expected_content:
            keywords = re.findall(r'\b\w+\b', expected.lower())
            for keyword in keywords:
                if len(keyword) > 3 and keyword in content_lower:  # Only consider meaningful words
                    matched_keywords += 1
                total_keywords += 1

        if total_keywords > 0:
            keyword_score = matched_keywords / total_keywords
        else:
            # If no keywords in expected content, use a simple relevance heuristic
            keyword_score = 0.5

    score_components['keyword_score'] = keyword_score

    # Semantic quality factors
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Count sentences with proper structure
    good_sentences = 0
    for sentence in sentences:
        # Check if sentence starts with capital letter and has reasonable length
        if len(sentence) > 10 and sentence[0].isupper():
            good_sentences += 1

    sentence_quality_score = good_sentences / len(sentences) if sentences else 0.0
    score_components['sentence_quality_score'] = sentence_quality_score

    # Get weights from configuration
    weights = ValidationConfig.get_quality_weights()

    # Combine scores with configurable weights
    relevance_score = (
        weights['length_weight'] * length_score +
        weights['keyword_weight'] * keyword_score +
        weights['sentence_quality_weight'] * sentence_quality_score
    )

    return min(relevance_score, 1.0)  # Ensure score doesn't exceed 1.0


def validate_relevance_threshold(validation_results: List[Dict], threshold: float = 0.9) -> bool:
    """
    Check if the validation results meet the required relevance threshold.

    Args:
        validation_results (List[Dict]): Results from validate_retrieved_chunks
        threshold (float): Minimum required relevance percentage (default 0.9 for 90% per spec)

    Returns:
        bool: True if threshold is met, False otherwise
    """
    if not validation_results or len(validation_results) == 0:
        return False

    # Get overall stats from the last item
    overall_stats = validation_results[-1].get('overall_stats', {})
    relevant_percentage = overall_stats.get('relevant_percentage', 0.0)

    return relevant_percentage >= threshold


def validate_metadata(metadata_list: List[Dict]) -> float:
    """
    Validates that metadata (URL, section, chunk ID) is preserved correctly

    Args:
        metadata_list (List[Dict]): List of metadata records to validate

    Returns:
        float: Percentage of metadata records that are valid
    """
    if not metadata_list:
        logger.info("No metadata records to validate")
        return 0.0

    total_records = len(metadata_list)
    valid_records = 0

    for i, metadata in enumerate(metadata_list):
        if not isinstance(metadata, dict):
            logger.warning(f"Metadata record at index {i} is not a dictionary: {type(metadata)}")
            continue

        # Check required metadata fields
        required_fields = ['url', 'section', 'chunk_id']
        is_valid = True

        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Missing required metadata field '{field}' in record {i}")
                is_valid = False
                continue

            value = metadata[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.warning(f"Empty required metadata field '{field}' in record {i}")
                is_valid = False

        # Additional validation for URL format
        if 'url' in metadata and metadata['url']:
            url = str(metadata['url'])
            if not url.startswith(('http://', 'https://')):
                logger.warning(f"URL format may be invalid in record {i}: {url}")

        # Additional validation for chunk_id format
        if 'chunk_id' in metadata and metadata['chunk_id']:
            chunk_id = str(metadata['chunk_id'])
            if not chunk_id.strip():
                logger.warning(f"Invalid chunk_id format in record {i}: {chunk_id}")

        if is_valid:
            valid_records += 1

    metadata_accuracy = (valid_records / total_records) * 100 if total_records > 0 else 0.0
    logger.info(f"Metadata validation: {valid_records}/{total_records} records valid ({metadata_accuracy:.2f}%)")

    return metadata_accuracy


def validate_metadata_comprehensive(retrieved_chunks: List[Dict]) -> Dict:
    """
    Comprehensive metadata validation for retrieved chunks

    Args:
        retrieved_chunks (List[Dict]): The chunks with metadata to validate

    Returns:
        Dict: Detailed metadata validation results
    """
    metadata_records = [chunk.get('metadata', {}) for chunk in retrieved_chunks if 'metadata' in chunk]

    # Calculate basic metadata accuracy
    accuracy_percentage = validate_metadata(metadata_records)

    # Detailed validation
    detailed_results = []
    for i, chunk in enumerate(retrieved_chunks):
        metadata = chunk.get('metadata', {})
        chunk_validation = {
            'chunk_index': i,
            'chunk_id': chunk.get('id', f'chunk_{i}'),
            'metadata_present': bool(metadata),
            'required_fields_present': True,
            'field_validity': {}
        }

        # Check each required field
        required_fields = ['url', 'section', 'chunk_id']
        for field in required_fields:
            field_present = field in metadata and metadata[field]
            chunk_validation['field_validity'][field] = field_present
            if not field_present:
                chunk_validation['required_fields_present'] = False

        detailed_results.append(chunk_validation)

    # Overall statistics
    total_chunks = len(retrieved_chunks)
    chunks_with_metadata = sum(1 for result in detailed_results if result['metadata_present'])
    chunks_with_all_required = sum(1 for result in detailed_results if result['required_fields_present'])

    overall_stats = {
        'total_chunks': total_chunks,
        'chunks_with_metadata': chunks_with_metadata,
        'chunks_with_all_required_fields': chunks_with_all_required,
        'metadata_accuracy_percentage': accuracy_percentage,
        'metadata_completeness_ratio': (chunks_with_all_required / total_chunks * 100) if total_chunks > 0 else 0.0
    }

    return {
        'detailed_results': detailed_results,
        'overall_stats': overall_stats,
        'accuracy_percentage': accuracy_percentage
    }


if __name__ == "__main__":
    # Test the validation functions
    test_chunks = [
        {
            'content': 'Artificial intelligence is a wonderful field that involves creating smart machines.',
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

    expected = ["artificial intelligence", "smart machines"]
    results = validate_retrieved_chunks(test_chunks, expected_content=expected)

    print("Validation Results:")
    for result in results:
        if 'overall_stats' in result:
            print(f"Overall stats: {result['overall_stats']}")
        else:
            print(f"Chunk {result['chunk_index']}: relevance={result['relevance_score']:.3f}, "
                  f"relevant={result['content_relevant']}, metadata_valid={result['metadata_valid']}")

    print("\nMetadata Validation Test:")
    metadata_accuracy = validate_metadata([chunk['metadata'] for chunk in test_chunks])
    print(f"Metadata accuracy: {metadata_accuracy:.2f}%")

    comprehensive_results = validate_metadata_comprehensive(test_chunks)
    print(f"Comprehensive metadata validation: {comprehensive_results['accuracy_percentage']:.2f}%")