# RAG Retrieval Service

A semantic search service that retrieves relevant text chunks from embedded book content using Qdrant vector database and Cohere embeddings.

## Overview

The RAG (Retrieval-Augmented Generation) Retrieval Service provides semantic search capabilities over embedded book content. It uses Cohere embeddings for vectorization and Qdrant for efficient similarity search, enabling contextual retrieval rather than simple keyword matching.

## Features

- Semantic search in embedded book content
- Configurable top-k results and similarity thresholds
- Metadata preservation and retrieval
- Quality validation with test queries
- Comprehensive API endpoints
- Health monitoring and error handling

## Prerequisites

- Python 3.11+
- Qdrant vector database access
- Cohere API key
- Existing Cohere embeddings in Qdrant (from Spec-1)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai-robotic-book
```

### 2. Navigate to the backend directory

```bash
cd backend
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
QDRANT_COLLECTION_NAME=book_embeddings  # or whatever was used in Spec-1
DEBUG=false
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.0
VALIDATION_ACCURACY_THRESHOLD=0.8
COHERE_MODEL=embed-english-v3.0
```

## Usage

### 1. Start the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 2. API Endpoints

Once the service is running, you can use these endpoints:

#### Search Endpoint
- `POST /api/v1/search` - Perform semantic search in embedded content

Example request:
```json
{
  "query_text": "What are the key concepts in AI?",
  "top_k": 5,
  "similarity_threshold": 0.5
}
```

#### Validation Endpoint
- `POST /api/v1/validate` - Validate retrieval quality with test queries

Example request:
```json
[
  {
    "query_text": "Explain neural networks",
    "expected_results": [
      {
        "content": "Neural networks are computing systems inspired by the human brain..."
      }
    ],
    "success_criteria": {
      "accuracy_threshold": 0.8,
      "similarity_threshold": 0.5
    },
    "test_category": "conceptual"
  }
]
```

#### Health Check Endpoint
- `GET /api/v1/health` - Check service health status

### 3. Programmatic Usage

You can also use the retrieval service directly in your Python code:

```python
from src.retrieval.retrieval_service import RetrievalService

service = RetrievalService()
results = service.search("What are the key concepts in AI?", top_k=5)
print(results)
```

## Architecture

### Core Components

1. **Retrieval Service** - Handles semantic search against Qdrant vector database:
   - Converts user queries to vectors using Cohere
   - Performs similarity search against existing embeddings
   - Returns results with metadata and similarity scores

2. **Validation Service** - Validates retrieval quality with test queries:
   - Executes multiple test queries with known expected results
   - Measures accuracy and relevance of returned results
   - Reports on retrieval quality metrics

3. **Query Processor** - Manages the query workflow:
   - Validates input parameters
   - Handles error conditions
   - Formats results for API responses

### Data Models

- `RetrievedChunk` - Represents a text chunk with content, similarity score, and metadata
- `SearchQuery` - Represents a search query with parameters like top_k and similarity threshold
- `MetadataPackage` - Contains metadata like URL, section, chunk ID, etc.
- `ValidationTest` - Represents a validation test with expected outcomes

## Configuration

The service supports different environments through the settings module:

- **Development**: Set `DEBUG=true` for detailed logging
- **Production**: Use appropriate Qdrant and Cohere configurations
- **Environment-specific**: Override settings via environment variables

## Testing

Run the tests using pytest:

```bash
python -m pytest tests/
```

## Performance

- Designed for 95% of queries to return results in under 2 seconds
- Targets 85% of results with similarity scores above 0.7
- Maintains 99% error-free pipeline execution

## Security

- API keys stored in environment variables, not in code
- Input validation on all endpoints
- Proper error handling to prevent information disclosure

## Monitoring

- Comprehensive logging throughout the application
- Health check endpoint for monitoring service status
- Error tracking and metrics collection