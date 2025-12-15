# RAG Data Models

This document defines the data models used in the Retrieval-Augmented Generation (RAG) system for the Physical AI & Humanoid Robotics book.

## Overview

The RAG system uses structured data models to ensure consistent communication between the frontend and backend components. These models define the structure of requests, responses, and internal data representations.

## Request Models

### QueryRequest

Represents a query request from the frontend to the RAG backend.

```json
{
  "query_text": "string",
  "selected_text": "string or null"
}
```

**Fields:**

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| query_text | string | Yes | 3-2000 characters | The question or query to ask the AI |
| selected_text | string or null | No | 10-5000 characters (if provided) | Text selected by the user on the page, providing additional context for the query |

**Validation Rules:**
- `query_text` must be at least 3 characters and no more than 2000 characters
- `selected_text` must be at least 10 characters and no more than 5000 characters if provided
- `selected_text` can be null, undefined, or an empty string

**Example:**
```json
{
  "query_text": "Explain the differences between ROS and ROS 2",
  "selected_text": "ROS 2 is not an operating system but rather a middleware framework that provides libraries, tools, and conventions for building robot software."
}
```

## Response Models

### QueryResponse

Represents the response from the RAG backend to the frontend query request.

```json
{
  "response": {
    "response_text": "string",
    "source_context": "array of strings",
    "confidence_score": "float"
  }
}
```

**Fields:**

| Field | Type | Required | Range | Description |
|-------|------|----------|-------|-------------|
| response_text | string | Yes | - | The AI-generated response to the query |
| source_context | array of strings | Yes | - | Source documents/chunks used in generating the response |
| confidence_score | float | Yes | 0.0 - 1.0 | Confidence score indicating the reliability of the response |

**Validation Rules:**
- `response_text` must not be empty
- `source_context` must be an array of strings (can be empty)
- `confidence_score` must be between 0.0 and 1.0 (inclusive)

**Example:**
```json
{
  "response": {
    "response_text": "ROS 2 addresses several key challenges in robotics development including improved security, real-time performance, and multi-robot support. It provides standardized ways for robot components to exchange information and enables sharing of robot software components across different platforms.",
    "source_context": ["module-1-ros2/index.md", "intro.md"],
    "confidence_score": 0.87
  }
}
```

### HealthResponse

Represents the health status response from the RAG backend.

```json
{
  "status": "string",
  "version": "string",
  "timestamp": "string",
  "details": "object (optional)"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Yes | Health status (e.g., 'healthy', 'degraded', 'unhealthy') |
| version | string | Yes | API version |
| timestamp | string | Yes | ISO 8601 timestamp of the health check |
| details | object | No | Additional health check details |

**Validation Rules:**
- `status` must be one of: 'healthy', 'degraded', 'unhealthy'
- `version` must follow semantic versioning format (e.g., "1.0.0")
- `timestamp` must be in ISO 8601 format
- `details` is optional and can contain any additional health information

**Example:**
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

## Internal Data Models

### DocumentChunk

Represents a chunk of text from the book content used for retrieval.

```json
{
  "id": "string",
  "content": "string",
  "source_path": "string",
  "metadata": "object"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | Yes | Unique identifier for the document chunk |
| content | string | Yes | The text content of the chunk |
| source_path | string | Yes | Path to the original document source |
| metadata | object | No | Additional metadata about the chunk |

**Example:**
```json
{
  "id": "module-1-ros2-architecture-001",
  "content": "ROS 2 is not an operating system but rather a middleware framework that provides libraries, tools, and conventions for building robot software.",
  "source_path": "docs/module-1-ros2/architecture.md",
  "metadata": {
    "section_title": "Architecture Overview",
    "module": "module-1-ros2",
    "position": 125
  }
}
```

### Embedding

Represents an embedding vector for semantic search.

```json
{
  "vector": "array of floats",
  "chunk_id": "string",
  "model": "string"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| vector | array of floats | Yes | The embedding vector (dimension depends on model) |
| chunk_id | string | Yes | Reference to the document chunk that was embedded |
| model | string | Yes | Name of the embedding model used |

**Example:**
```json
{
  "vector": [0.12, -0.45, 0.89, -0.23, 0.67],
  "chunk_id": "module-1-ros2-architecture-001",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

## Configuration Model

### RAGConfig

Represents the configuration parameters for the RAG system.

```json
{
  "BACKEND_API_URL": "string",
  "DEVELOPMENT_MODE": "boolean",
  "REQUEST_TIMEOUT": "integer",
  "MAX_RETRIES": "integer",
  "RETRY_DELAY": "integer",
  "MIN_QUERY_LENGTH": "integer",
  "MAX_QUERY_LENGTH": "integer",
  "MIN_SELECTED_TEXT_LENGTH": "integer",
  "MAX_SELECTED_TEXT_LENGTH": "integer"
}
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| BACKEND_API_URL | string | "http://localhost:8000" | URL of the RAG backend service |
| DEVELOPMENT_MODE | boolean | process.env.NODE_ENV !== 'production' | Flag indicating development mode |
| REQUEST_TIMEOUT | integer | 30000 | Request timeout in milliseconds |
| MAX_RETRIES | integer | 2 | Maximum number of retry attempts |
| RETRY_DELAY | integer | 1000 | Delay between retries in milliseconds |
| MIN_QUERY_LENGTH | integer | 3 | Minimum length of query text |
| MAX_QUERY_LENGTH | integer | 2000 | Maximum length of query text |
| MIN_SELECTED_TEXT_LENGTH | integer | 10 | Minimum length of selected text |
| MAX_SELECTED_TEXT_LENGTH | integer | 5000 | Maximum length of selected text |

## Error Model

### ErrorResponse

Represents an error response from the RAG backend.

```json
{
  "error": "string",
  "error_code": "string (optional)",
  "timestamp": "string"
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| error | string | Yes | Human-readable error message |
| error_code | string | No | Machine-readable error code |
| timestamp | string | Yes | ISO 8601 timestamp of when the error occurred |

**Example:**
```json
{
  "error": "Query text must be at least 3 characters long",
  "error_code": "INVALID_QUERY_LENGTH",
  "timestamp": "2025-01-15T10:30:15Z"
}
```

## Model Relationships

```
QueryRequest
    ↓
RAG Backend
    ↓
DocumentChunk ←→ Embedding
    ↓
QueryResponse
```

The QueryRequest is sent to the RAG backend, which retrieves relevant DocumentChunks using their Embeddings, then generates a QueryResponse with the answer and source context.

## Versioning

Data models follow semantic versioning. Breaking changes to models will increment the major version number. Backward compatibility is maintained when possible through optional fields.