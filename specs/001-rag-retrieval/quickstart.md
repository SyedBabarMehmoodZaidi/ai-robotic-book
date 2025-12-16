# Quickstart Guide: RAG Retrieval Implementation

## Overview
This guide provides the essential steps to set up and run the RAG retrieval pipeline for semantic search in embedded book content using Qdrant and Cohere embeddings.

## Prerequisites
- Python 3.11+
- pip package manager
- Qdrant database access (URL and API key)
- Cohere API key (for query vectorization)
- Existing Cohere embeddings in Qdrant (from Spec-1)

## Setup

### 1. Initialize the Project
```bash
# Create backend directory (if not already done from previous spec)
cd backend
```

### 2. Install Dependencies
```bash
pip install qdrant-client cohere python-dotenv pydantic uvicorn
```

### 3. Environment Configuration
Update your `.env` file with the necessary keys:
```env
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_COLLECTION_NAME=book_embeddings  # or whatever was used in Spec-1
```

## Core Components

### 1. Retrieval Service
Handles semantic search against Qdrant vector database:
- Converts user queries to vectors using Cohere
- Performs similarity search against existing embeddings
- Returns results with metadata and similarity scores

### 2. Validation Service
Validates retrieval quality with test queries:
- Executes multiple test queries with known expected results
- Measures accuracy and relevance of returned results
- Reports on retrieval quality metrics

### 3. Query Processor
Manages the query workflow:
- Validates input parameters
- Handles error conditions
- Formats results for API responses

## Running the Service

### 1. Start the API Server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 2. Perform a Search
```python
from src.retrieval.retrieval_service import RetrievalService

service = RetrievalService()
results = service.search("What are the key concepts in AI?", top_k=5)
print(results)
```

### 3. Validate Retrieval Quality
```python
from src.validation.validation_service import ValidationService

validator = ValidationService()
validation_results = validator.validate_retrieval_quality(test_queries)
print(f"Overall accuracy: {validation_results.overall_accuracy}")
```

## API Endpoints

Once the service is running, you can use these endpoints:

- `POST /api/v1/search` - Perform semantic search in embedded content
- `POST /api/v1/validate` - Validate retrieval quality with test queries
- `GET /api/v1/health` - Check service health status

## Testing

Run the following to verify your setup:
```bash
python -m pytest tests/
```

## Next Steps

1. Implement the retrieval service with Qdrant integration
2. Build the validation functionality for quality assurance
3. Add comprehensive error handling and logging
4. Implement the API endpoints
5. Test with multiple book-related queries and validate results