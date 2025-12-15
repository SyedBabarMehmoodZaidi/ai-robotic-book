"""
Metadata validation rules based on the data model.
This module defines validation rules for metadata fields based on the data model specification.
"""
import re
from typing import Dict, Any, List
import logging
from urllib.parse import urlparse


def validate_url_format(url: str) -> bool:
    """
    Validate URL format based on data model requirements.

    Args:
        url: The URL string to validate

    Returns:
        bool: True if URL format is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    try:
        result = urlparse(url)
        # Check if it has both scheme and netloc (domain)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_section_format(section: str) -> bool:
    """
    Validate section format based on data model requirements.

    Args:
        section: The section string to validate

    Returns:
        bool: True if section format is valid, False otherwise
    """
    if not section or not isinstance(section, str):
        return False

    # Section should be non-empty and not just whitespace
    return bool(section.strip())


def validate_chunk_id_format(chunk_id: str) -> bool:
    """
    Validate chunk_id format based on data model requirements.

    Args:
        chunk_id: The chunk_id string to validate

    Returns:
        bool: True if chunk_id format is valid, False otherwise
    """
    if not chunk_id or not isinstance(chunk_id, str):
        return False

    # Chunk ID should be non-empty and follow a reasonable format
    # Could be alphanumeric, with hyphens or underscores
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, chunk_id.strip()))


def validate_metadata_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single metadata record based on data model requirements.

    Args:
        metadata: The metadata dictionary to validate

    Returns:
        Dict: Validation results with field-level validation
    """
    if not isinstance(metadata, dict):
        return {
            'valid': False,
            'errors': ['Metadata must be a dictionary'],
            'field_validations': {}
        }

    # Required fields from the data model
    required_fields = ['url', 'section', 'chunk_id']

    field_validations = {}
    errors = []

    # Validate required fields exist
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
            field_validations[field] = {
                'valid': False,
                'error': f"Missing required field: {field}"
            }
        else:
            # Validate field format
            field_value = metadata[field]
            is_valid = False
            error_msg = ""

            if field == 'url':
                is_valid = validate_url_format(field_value)
                error_msg = "Invalid URL format" if not is_valid else ""
            elif field == 'section':
                is_valid = validate_section_format(field_value)
                error_msg = "Invalid section format" if not is_valid else ""
            elif field == 'chunk_id':
                is_valid = validate_chunk_id_format(field_value)
                error_msg = "Invalid chunk_id format" if not is_valid else ""

            field_validations[field] = {
                'valid': is_valid,
                'value': field_value,
                'error': error_msg
            }

            if not is_valid:
                errors.append(f"{field}: {error_msg}")

    # Check for optional fields and validate them if present
    optional_fields = ['source_file', 'page_number', 'created_at']
    for field in optional_fields:
        if field in metadata:
            field_value = metadata[field]
            field_validations[field] = {
                'valid': field_value is not None,
                'value': field_value,
                'error': "" if field_value is not None else f"{field} cannot be None"
            }

    is_record_valid = len(errors) == 0

    return {
        'valid': is_record_valid,
        'errors': errors,
        'field_validations': field_validations
    }


def validate_metadata_collection(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a collection of metadata records based on data model requirements.

    Args:
        metadata_list: List of metadata dictionaries to validate

    Returns:
        Dict: Overall validation results for the collection
    """
    if not isinstance(metadata_list, list):
        return {
            'valid': False,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'accuracy_percentage': 0.0,
            'detailed_results': [],
            'errors': ['Metadata list must be a list of dictionaries']
        }

    detailed_results = []
    valid_count = 0

    for i, metadata in enumerate(metadata_list):
        validation_result = validate_metadata_record(metadata)
        detailed_results.append({
            'index': i,
            'metadata': metadata,
            'validation': validation_result
        })

        if validation_result['valid']:
            valid_count += 1

    total_records = len(metadata_list)
    accuracy_percentage = (valid_count / total_records * 100) if total_records > 0 else 0.0

    return {
        'valid': total_records == 0 or valid_count > 0,
        'total_records': total_records,
        'valid_records': valid_count,
        'invalid_records': total_records - valid_count,
        'accuracy_percentage': accuracy_percentage,
        'detailed_results': detailed_results,
        'errors': [] if valid_count > 0 else ['All metadata records are invalid']
    }


# Validation rules based on the data model
METADATA_VALIDATION_RULES = {
    'url': {
        'required': True,
        'format': 'Valid URL with scheme and domain',
        'validator': validate_url_format,
        'description': 'URL of the source document'
    },
    'section': {
        'required': True,
        'format': 'Non-empty string',
        'validator': validate_section_format,
        'description': 'Section or chapter of the source document'
    },
    'chunk_id': {
        'required': True,
        'format': 'Alphanumeric with optional hyphens/underscores',
        'validator': validate_chunk_id_format,
        'description': 'Unique identifier for the text chunk'
    },
    'source_file': {
        'required': False,
        'format': 'File path string',
        'validator': lambda x: isinstance(x, str) and bool(x.strip()) if x is not None else True,
        'description': 'Original source file name'
    },
    'page_number': {
        'required': False,
        'format': 'Integer or null',
        'validator': lambda x: isinstance(x, (int, type(None))),
        'description': 'Page number in the source document'
    },
    'created_at': {
        'required': False,
        'format': 'ISO datetime string or null',
        'validator': lambda x: True,  # Basic validation, could be more specific
        'description': 'Timestamp when the chunk was created'
    }
}


def get_validation_rules():
    """
    Get the metadata validation rules based on the data model.

    Returns:
        Dict: Validation rules for all metadata fields
    """
    return METADATA_VALIDATION_RULES


if __name__ == "__main__":
    # Test the validation functions
    test_metadata = [
        {
            'url': 'https://example.com/doc.pdf',
            'section': 'Introduction',
            'chunk_id': 'chunk_001',
            'source_file': 'document.pdf'
        },
        {
            'url': 'invalid-url',
            'section': '',
            'chunk_id': 'invalid id with spaces',
        },
        {
            'url': 'https://example.com/page.html',
            'section': 'Chapter 1',
            'chunk_id': 'chunk_002'
        }
    ]

    print("Testing metadata validation...")
    result = validate_metadata_collection(test_metadata)

    print(f"Total records: {result['total_records']}")
    print(f"Valid records: {result['valid_records']}")
    print(f"Accuracy: {result['accuracy_percentage']:.2f}%")

    for detail in result['detailed_results']:
        idx = detail['index']
        validation = detail['validation']
        print(f"\nRecord {idx}: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
        if not validation['valid']:
            for error in validation['errors']:
                print(f"  - {error}")