# API Contracts: RAG-Enabled AI Agent

**Feature**: 3-rag-agent
**Created**: 2025-12-15
**Status**: Draft
**Author**: Claude

## Overview

This document defines the API contracts for the RAG-enabled AI agent system based on the functional requirements from the feature specification.

## Base URL

```
https://api.example.com/api/v1
```

## Common Headers

All requests and responses use the following common headers:

**Request Headers:**
- `Content-Type: application/json`
- `Accept: application/json`
- `Authorization: Bearer {api_key}` (optional for initial implementation)

**Response Headers:**
- `Content-Type: application/json`
- `X-Request-ID: {request_id}` (unique identifier for the request)

## Endpoints

### 1. Query Agent Endpoint

**Endpoint**: `POST /query`

**Description**: Submit a query to the RAG agent and receive a response based on book context.

**Request Schema**:
```json
{
  "type": "object",
  "properties": {
    "query_text": {
      "type": "string",
      "description": "The user's question",
      "minLength": 1,
      "maxLength": 2000
    },
    "selected_text": {
      "type": "string",
      "description": "Optional specific text to focus on",
      "maxLength": 5000,
      "nullable": true
    }
  },
  "required": ["query_text"]
}
```

**Request Example**:
```json
{
  "query_text": "What are the main principles of artificial intelligence?",
  "selected_text": "Artificial intelligence is a branch of computer science..."
}
```

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "response_text": {
      "type": "string",
      "description": "The AI-generated response"
    },
    "source_context": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "References to source material used"
    },
    "confidence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence in response accuracy"
    },
    "tokens_used": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of tokens in response"
    },
    "processing_time": {
      "type": "number",
      "minimum": 0,
      "description": "Time taken to generate response in seconds"
    },
    "query_id": {
      "type": "string",
      "description": "Reference to the original query"
    },
    "is_hallucination_detected": {
      "type": "boolean",
      "description": "Flag if hallucination was detected"
    }
  },
  "required": [
    "response_text",
    "source_context",
    "confidence_score",
    "tokens_used",
    "processing_time",
    "query_id",
    "is_hallucination_detected"
  ]
}
```

**Response Example**:
```json
{
  "response_text": "The main principles of artificial intelligence include machine learning, neural networks, and natural language processing...",
  "source_context": [
    "Section 1.2: Introduction to AI Principles",
    "Chapter 3: Machine Learning Fundamentals"
  ],
  "confidence_score": 0.92,
  "tokens_used": 127,
  "processing_time": 1.24,
  "query_id": "query_12345",
  "is_hallucination_detected": false
}
```

**Status Codes**:
- `200 OK`: Query processed successfully
- `400 Bad Request`: Invalid request format or parameters
- `422 Unprocessable Entity`: Query validation failed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error during processing

### 2. Health Check Endpoint

**Endpoint**: `GET /health`

**Description**: Check the health status of the agent and API.

**Request Schema**: No request body required

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["healthy", "degraded", "unhealthy"]
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp"
    },
    "details": {
      "type": "object",
      "properties": {
        "agent_status": {
          "type": "string",
          "description": "Status of the AI agent"
        },
        "retrieval_status": {
          "type": "string",
          "description": "Status of the retrieval pipeline"
        },
        "api_status": {
          "type": "string",
          "description": "Status of the API service"
        }
      }
    }
  },
  "required": ["status", "timestamp"]
}
```

**Response Example**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:30:00Z",
  "details": {
    "agent_status": "ready",
    "retrieval_status": "connected",
    "api_status": "operational"
  }
}
```

**Status Codes**:
- `200 OK`: System is healthy
- `503 Service Unavailable`: System is unhealthy

### 3. Agent Status Endpoint

**Endpoint**: `GET /agent/status`

**Description**: Get detailed status information about the AI agent.

**Request Schema**: No request body required

**Response Schema**:
```json
{
  "type": "object",
  "properties": {
    "agent_id": {
      "type": "string",
      "description": "Unique identifier for the agent"
    },
    "model": {
      "type": "string",
      "description": "AI model being used"
    },
    "status": {
      "type": "string",
      "enum": ["ready", "processing", "error", "unavailable"]
    },
    "last_query_time": {
      "type": "string",
      "format": "date-time",
      "description": "Time of last processed query"
    },
    "queries_processed": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of queries processed"
    },
    "average_response_time": {
      "type": "number",
      "minimum": 0,
      "description": "Average response time in seconds"
    }
  },
  "required": ["agent_id", "model", "status"]
}
```

**Response Example**:
```json
{
  "agent_id": "rag-agent-001",
  "model": "gpt-4-turbo",
  "status": "ready",
  "last_query_time": "2025-12-15T10:25:30Z",
  "queries_processed": 42,
  "average_response_time": 1.45
}
```

**Status Codes**:
- `200 OK`: Status retrieved successfully
- `500 Internal Server Error`: Error retrieving agent status

## Error Response Format

All error responses follow this standard format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object (optional)"
  }
}
```

**Common Error Codes**:
- `INVALID_QUERY`: Query text is invalid or empty
- `RETRIEVAL_FAILED`: Failed to retrieve relevant context
- `AGENT_ERROR`: Error in AI agent processing
- `RATE_LIMIT_EXCEEDED`: Request rate limit exceeded
- `INTERNAL_ERROR`: Internal server error

**Error Examples**:

```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query text must be between 1 and 2000 characters"
  }
}
```

## Authentication

For initial implementation, authentication is optional. In production, the API expects:
- `Authorization: Bearer {api_key}` header
- API key validation through environment configuration

## Rate Limiting

The API implements rate limiting:
- Default: 100 requests per minute per IP address
- Custom limits possible through configuration
- Returns HTTP 429 when limit exceeded

## Versioning

API versioning is implemented through the URL path:
- Current version: `/api/v1/`
- Future versions: `/api/v2/`, etc.
- Backward compatibility maintained for 6 months after new version release