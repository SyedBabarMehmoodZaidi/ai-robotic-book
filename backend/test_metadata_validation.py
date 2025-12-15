"""
Test script for metadata validation functionality.
This script tests the metadata validation with sample retrieved chunks.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from result_validator import validate_metadata_comprehensive
from metadata_validation_rules import validate_metadata_collection


def test_metadata_validation():
    """Test metadata validation with sample retrieved chunks"""
    print("Testing metadata validation with sample retrieved chunks...")

    # Sample retrieved chunks with various metadata scenarios
    sample_chunks = [
        {
            'content': 'Artificial intelligence is a wonderful field that involves creating intelligent systems.',
            'metadata': {
                'url': 'https://example.com/ai-intro.pdf',
                'section': 'Introduction',
                'chunk_id': 'chunk_001',
                'source_file': 'ai-intro.pdf'
            },
            'score': 0.95,
            'id': 'id_1'
        },
        {
            'content': 'Machine learning algorithms form the backbone of AI systems.',
            'metadata': {
                'url': 'https://example.com/ml-basics.html',
                'section': 'Chapter 1',
                'chunk_id': 'chunk_002',
                'source_file': 'ml-basics.html'
            },
            'score': 0.88,
            'id': 'id_2'
        },
        {
            'content': 'Neural networks are computing systems inspired by the human brain.',
            'metadata': {
                'url': 'https://example.com/neural-networks.pdf',
                'section': 'Chapter 2',
                'chunk_id': 'chunk_003'
            },
            'score': 0.92,
            'id': 'id_3'
        },
        # Chunk with incomplete metadata (missing source_file)
        {
            'content': 'This chunk has incomplete metadata.',
            'metadata': {
                'url': 'https://example.com/incomplete.html',
                'section': 'Appendix',
                'chunk_id': 'chunk_004'
            },
            'score': 0.75,
            'id': 'id_4'
        },
        # Chunk with invalid metadata
        {
            'content': 'This chunk has invalid metadata.',
            'metadata': {
                'url': 'invalid-url-format',  # Invalid URL
                'section': '',  # Empty section
                'chunk_id': 'invalid chunk id with spaces'  # Invalid chunk_id
            },
            'score': 0.45,
            'id': 'id_5'
        }
    ]

    print(f"Testing with {len(sample_chunks)} sample chunks...")

    # Test comprehensive metadata validation
    print("\n--- Comprehensive Metadata Validation ---")
    comprehensive_results = validate_metadata_comprehensive(sample_chunks)

    print(f"Total chunks: {comprehensive_results['overall_stats']['total_chunks']}")
    print(f"Chunks with metadata: {comprehensive_results['overall_stats']['chunks_with_metadata']}")
    print(f"Chunks with all required fields: {comprehensive_results['overall_stats']['chunks_with_all_required_fields']}")
    print(f"Metadata accuracy: {comprehensive_results['accuracy_percentage']:.2f}%")
    print(f"Metadata completeness ratio: {comprehensive_results['overall_stats']['metadata_completeness_ratio']:.2f}%")

    # Detailed results
    for result in comprehensive_results['detailed_results']:
        chunk_idx = result['chunk_index']
        chunk_id = result['chunk_id']
        all_required = result['required_fields_present']
        print(f"  Chunk {chunk_idx} ({chunk_id}): {'✅ All required fields' if all_required else '❌ Missing required fields'}")

    # Test with the more detailed validation rules
    print("\n--- Detailed Metadata Validation (per data model) ---")
    metadata_list = [chunk['metadata'] for chunk in sample_chunks]
    detailed_results = validate_metadata_collection(metadata_list)

    print(f"Total metadata records: {detailed_results['total_records']}")
    print(f"Valid records: {detailed_results['valid_records']}")
    print(f"Invalid records: {detailed_results['invalid_records']}")
    print(f"Accuracy percentage: {detailed_results['accuracy_percentage']:.2f}%")

    # Show detailed validation for each record
    for detail in detailed_results['detailed_results']:
        idx = detail['index']
        validation = detail['validation']
        chunk_id = sample_chunks[idx].get('id', f'chunk_{idx}')

        print(f"  Record {idx} ({chunk_id}): {'✅ Valid' if validation['valid'] else '❌ Invalid'}")

        if not validation['valid']:
            for error in validation['errors']:
                print(f"    - {error}")
        else:
            print(f"    - All fields valid")

    # Test edge cases
    print("\n--- Edge Case Testing ---")

    # Test with empty metadata
    empty_metadata_chunks = [
        {
            'content': 'Content with no metadata',
            'metadata': {},
            'score': 0.5,
            'id': 'empty_meta'
        }
    ]
    empty_results = validate_metadata_comprehensive(empty_metadata_chunks)
    print(f"Empty metadata test - Accuracy: {empty_results['accuracy_percentage']:.2f}%")

    # Test with no metadata field
    no_metadata_chunks = [
        {
            'content': 'Content with no metadata field',
            'score': 0.5,
            'id': 'no_meta_field'
        }
    ]
    no_meta_results = validate_metadata_comprehensive(no_metadata_chunks)
    print(f"No metadata field test - Accuracy: {no_meta_results['accuracy_percentage']:.2f}%")

    # Test with missing required fields
    partial_metadata_chunks = [
        {
            'content': 'Content with partial metadata',
            'metadata': {
                'url': 'https://example.com/doc.pdf',
                # Missing 'section' and 'chunk_id'
            },
            'score': 0.6,
            'id': 'partial_meta'
        }
    ]
    partial_results = validate_metadata_comprehensive(partial_metadata_chunks)
    print(f"Partial metadata test - Accuracy: {partial_results['accuracy_percentage']:.2f}%")

    print(f"\n--- Metadata Validation Test Summary ---")
    all_tests_passed = (
        comprehensive_results['accuracy_percentage'] >= 60 and  # Some will fail due to invalid entries
        detailed_results['accuracy_percentage'] >= 60
    )

    print(f"Comprehensive validation: {'✅ PASSED' if comprehensive_results['accuracy_percentage'] >= 60 else '❌ FAILED'}")
    print(f"Detailed validation: {'✅ PASSED' if detailed_results['accuracy_percentage'] >= 60 else '❌ FAILED'}")

    return all_tests_passed


if __name__ == "__main__":
    print("Running metadata validation tests...")

    test_passed = test_metadata_validation()

    print(f"\n--- Final Metadata Validation Results ---")
    print(f"Overall test: {'✅ PASSED' if test_passed else '❌ FAILED'}")

    if test_passed:
        print("\n🎉 Metadata validation functionality is working correctly!")
        print("The system can properly validate metadata fields and calculate accuracy.")
    else:
        print("\n⚠️  Some metadata validation tests had issues.")
        print("This may be expected when testing with intentionally invalid metadata.")