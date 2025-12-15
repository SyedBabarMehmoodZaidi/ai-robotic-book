# Validation Parameters Documentation

This document provides comprehensive information about all validation parameters available in the RAG retrieval validation system. These parameters can be configured through environment variables to customize the behavior of the validation pipeline.

## Configuration Overview

The validation system uses a centralized configuration approach with the `Config` class in `config.py`. Parameters are loaded from environment variables with sensible defaults, allowing for easy customization without code changes.

## Environment Variables

### API Keys and Connection Settings
- `COHERE_API_KEY` (required): Your Cohere API key for embedding generation
- `QDRANT_URL`: URL for remote Qdrant instance (optional if using local)
- `QDRANT_API_KEY`: API key for remote Qdrant instance
- `LOCAL_QDRANT_PATH`: Path for local Qdrant storage (default: './qdrant_data')

### Database Settings
- `COLLECTION_NAME`: Name of the Qdrant collection to search (default: 'book_embeddings')

### Validation Thresholds and Settings
- `TOP_K_RESULTS`: Number of top results to retrieve in similarity search (default: 5)
- `VALIDATION_TIMEOUT`: Timeout for validation operations in seconds (default: 30)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for a chunk to be considered relevant (default: 0.5)

### Quality Validation Thresholds
- `QUALITY_THRESHOLD`: Minimum quality percentage required for validation to pass (default: 0.9 or 90%)
- `METADATA_ACCURACY_THRESHOLD`: Minimum metadata accuracy percentage required (default: 1.0 or 100%)
- `RETRIEVAL_SUCCESS_RATE_THRESHOLD`: Minimum retrieval success rate percentage (default: 0.95 or 95%)

### Validation Parameters
- `MIN_CHUNK_LENGTH`: Minimum content length for a chunk to be considered valid (default: 50 characters)
- `MAX_CHUNK_LENGTH`: Maximum content length for a chunk to be considered valid (default: 2000 characters)
- `MIN_SENTENCES_FOR_QUALITY`: Minimum number of sentences for quality assessment (default: 5)

### Cohere Settings
- `COHERE_MODEL`: Cohere model to use for embeddings (default: 'embed-multilingual-v3.0')
- `COHERE_INPUT_TYPE`: Input type for Cohere embeddings (default: 'search_query')

### Performance Settings
- `MAX_QUERY_LENGTH`: Maximum allowed query length (default: 1000 characters)
- `MAX_RETRIES`: Maximum number of retries for API calls (default: 3)
- `RETRY_DELAY`: Delay between retries in seconds (default: 1.0)

### Quality Assessment Weights
- `LENGTH_WEIGHT`: Weight for content length in relevance scoring (default: 0.3)
- `KEYWORD_WEIGHT`: Weight for keyword matching in relevance scoring (default: 0.5)
- `SENTENCE_QUALITY_WEIGHT`: Weight for sentence structure in relevance scoring (default: 0.2)

## Configuration Classes

### Config Class
The main `Config` class contains all application settings and provides a centralized location for all configuration parameters.

### ValidationConfig Class
The `ValidationConfig` class provides specific configuration methods for validation parameters:
- `get_quality_weights()`: Returns weights for different quality assessment components
- `get_validation_rules()`: Returns validation rules for content quality
- `get_thresholds()`: Returns all validation thresholds

## Usage Examples

### Setting Environment Variables
```bash
export COHERE_API_KEY="your-api-key-here"
export QUALITY_THRESHOLD="0.85"  # 85% quality threshold
export TOP_K_RESULTS="10"        # Retrieve 10 results instead of 5
export MIN_CHUNK_LENGTH="100"    # Require minimum 100 characters per chunk
```

### Default Values
If environment variables are not set, the system uses these default values:
- Quality threshold: 90% (0.9) - at least 90% of chunks must be relevant
- Metadata accuracy: 100% (1.0) - all metadata must be preserved correctly
- Retrieval success rate: 95% (0.95) - 95% of queries must succeed
- Minimum chunk length: 50 characters
- Maximum chunk length: 2000 characters

## Validation Process

The validation process uses these parameters to assess:
1. **Relevance**: Whether retrieved chunks are contextually relevant to the query
2. **Quality**: Whether content meets minimum quality standards
3. **Metadata Accuracy**: Whether URL, section, and chunk ID are preserved correctly
4. **Success Rate**: Whether the retrieval system performs reliably

## Threshold Requirements

To meet the system requirements:
- Quality validation must achieve at least 90% relevance (QUALITY_THRESHOLD ≥ 0.9)
- Metadata preservation must achieve 100% accuracy (METADATA_ACCURACY_THRESHOLD = 1.0)
- Retrieval success rate must achieve at least 95% (RETRIEVAL_SUCCESS_RATE_THRESHOLD ≥ 0.95)

## Best Practices

1. **Start with defaults**: Begin with default values and adjust based on your specific use case
2. **Monitor performance**: Track validation results to determine if thresholds are appropriate
3. **Balance precision and recall**: Higher similarity thresholds may improve precision but reduce recall
4. **Consider content characteristics**: Adjust MIN/MAX_CHUNK_LENGTH based on your document structure
5. **Test iteratively**: Make small adjustments and test the impact on validation results

## Troubleshooting

If validation is failing:
- Check that `COHERE_API_KEY` is properly set
- Verify Qdrant connection settings
- Review quality thresholds - they may be too strict for your content
- Ensure your embedded content has proper metadata fields (URL, section, chunk_id)