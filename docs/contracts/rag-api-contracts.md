# RAG API Contracts

This document defines the API contracts for the Retrieval-Augmented Generation (RAG) system that enables querying the book content with AI assistance.

## API Overview

The RAG system provides HTTP/JSON communication between the Docusaurus frontend and the backend RAG service. All endpoints use JSON for request and response bodies.

## Base URL

All API endpoints are relative to: `http://localhost:8000` (default for local development)

## Endpoints

### 1. Query Endpoint

#### `POST /query`

Submit a query to the RAG system with optional selected text context.

**Request:**

```json
{
  "query_text": "string, required - The question or query to ask the AI",
  "selected_text": "string, optional - Text selected by the user on the page (null if none)"
}
```

**Request Headers:**
- `Content-Type: application/json`

**Example Request:**
```json
{
  "query_text": "What are the key components of ROS 2 architecture?",
  "selected_text": "ROS 2 is not an operating system but rather a middleware framework that provides libraries, tools, and conventions for building robot software."
}
```

**Response:**

```json
{
  "response": {
    "response_text": "string - The AI-generated response to the query",
    "source_context": "array of strings - Source documents/chunks used in generating the response",
    "confidence_score": "float - Confidence score between 0.0 and 1.0"
  }
}
```

**Example Response:**
```json
{
  "response": {
    "response_text": "The key components of ROS 2 architecture include Nodes, Topics, Services, Actions, and Parameters. Nodes are processes that perform computation. Topics enable publish-subscribe communication. Services provide request-response communication. Actions are for long-running tasks with feedback. Parameters allow configuration of nodes.",
    "source_context": ["module-1-ros2/architecture.md", "intro.md"],
    "confidence_score": 0.92
  }
}
```

**Status Codes:**
- `200 OK` - Query processed successfully
- `400 Bad Request` - Invalid request format or missing required fields
- `500 Internal Server Error` - Error occurred during query processing

### 2. Health Check Endpoint

#### `GET /health`

Check the health status of the RAG backend service.

**Request:**

No request body required.

**Example Request:**
```
GET /health
```

**Response:**

```json
{
  "status": "string - Health status (e.g., 'healthy', 'degraded', 'unhealthy')",
  "version": "string - API version",
  "timestamp": "string - ISO 8601 timestamp of the health check",
  "details": "object - Additional health check details (optional)"
}
```

**Example Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "details": {
    "model_loaded": true,
    "embedding_service": "connected",
    "vector_store": "connected"
  }
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

## Data Models

### QueryRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query_text | string | Yes | The question or query to ask the AI |
| selected_text | string or null | No | Text selected by the user on the page (null if none) |

**Validation Rules:**
- `query_text` must be between 3 and 2000 characters
- `selected_text` must be between 10 and 5000 characters (if provided)
- `selected_text` can be null or empty string

### QueryResponse

| Field | Type | Description |
|-------|------|-------------|
| response_text | string | The AI-generated response to the query |
| source_context | array of strings | Source documents/chunks used in generating the response |
| confidence_score | float | Confidence score between 0.0 and 1.0 |

### HealthResponse

| Field | Type | Description |
|-------|------|-------------|
| status | string | Health status (e.g., 'healthy', 'degraded', 'unhealthy') |
| version | string | API version |
| timestamp | string | ISO 8601 timestamp of the health check |
| details | object | Additional health check details (optional) |

## Error Handling

All API errors follow the standard HTTP status codes. When an error occurs, the response body will contain an error message:

```json
{
  "error": "string - Description of the error"
}
```

## Configuration Parameters

The following parameters are used by the API client for communication:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| REQUEST_TIMEOUT | 30000ms | Maximum time to wait for a response |
| MAX_RETRIES | 2 | Number of retry attempts on failure |
| RETRY_DELAY | 1000ms | Delay between retry attempts |
| MIN_QUERY_LENGTH | 3 | Minimum length of query text |
| MAX_QUERY_LENGTH | 2000 | Maximum length of query text |
| MIN_SELECTED_TEXT_LENGTH | 10 | Minimum length of selected text |
| MAX_SELECTED_TEXT_LENGTH | 5000 | Maximum length of selected text |

## Security Considerations

- All communication is over HTTP for local development (HTTPS recommended for production)
- No authentication required for local development
- Input validation is performed on both client and server sides
- Query length limits prevent excessive resource consumption

## Versioning

This API follows semantic versioning. Breaking changes will increment the major version number.