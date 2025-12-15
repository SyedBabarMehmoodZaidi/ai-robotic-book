# API Contracts: RAG Frontend Integration

**Feature**: 1-rag-frontend-integration
**Created**: 2025-12-15
**Status**: Draft
**Author**: Claude

## Overview

This document defines the API contracts for the communication between the frontend book interface and the RAG backend. These contracts specify the endpoints, request/response formats, and error handling patterns.

## Backend API Endpoints

### POST /query

**Purpose**: Submit a user query to the RAG backend for processing.

**Request**:
- **URL**: `/query`
- **Method**: POST
- **Content-Type**: `application/json`
- **Headers**:
  - `Content-Type: application/json`

**Request Body**:
```json
{
  "query_text": "string (required, 3-2000 characters)",
  "selected_text": "string (optional, 10-5000 characters)",
  "metadata": {
    "user_id": "string (optional)",
    "source_page": "string (optional)"
  }
}
```

**Response**:
- **Success Response** (Status: 200 OK):
```json
{
  "response": {
    "response_text": "string (required)",
    "source_context": ["string (required)"],
    "confidence_score": "number (required, 0.0-1.0)",
    "tokens_used": "number (optional)",
    "processing_time": "number (optional)",
    "query_id": "string (optional)",
    "is_hallucination_detected": "boolean (optional)",
    "detailed_source_references": [
      {
        "source": "string (required)",
        "content_preview": "string (required, max 500 chars)",
        "relevance_score": "number (required, 0.0-1.0)",
        "chunk_id": "string (required)"
      }
    ]
  },
  "request_id": "string (required)",
  "status_code": "number (required)",
  "timestamp": "string (required, ISO 8601)",
  "processing_time": "number (required, seconds)"
}
```

- **Error Response** (Status: 400, 422, 500):
```json
{
  "error_code": "string (required)",
  "message": "string (required)",
  "details": "string (optional)",
  "timestamp": "string (required, ISO 8601)",
  "request_id": "string (optional)"
}
```

**Error Codes**:
- `VALIDATION_ERROR` (422): Query text doesn't meet validation requirements
- `RETRIEVAL_ERROR` (500): Error during context retrieval
- `AGENT_ERROR` (500): Error during AI response generation
- `OPENAI_API_ERROR` (500): Error communicating with OpenAI API
- `CONTEXT_TOO_LONG` (400): Selected text exceeds length limits
- `NO_CONTEXT_FOUND` (400): No relevant context found for query
- `HALLUCINATION_DETECTED` (422): Potential hallucination detected in response
- `INTERNAL_ERROR` (500): General internal server error

### GET /health

**Purpose**: Check the health status of the RAG backend.

**Request**:
- **URL**: `/health`
- **Method**: GET

**Response**:
- **Success Response** (Status: 200 OK):
```json
{
  "status": "string (required)",
  "version": "string (required)",
  "service": "string (required)",
  "timestamp": "string (required, ISO 8601)"
}
```

## Frontend API Implementation

### Query Submission Endpoint

**Purpose**: Wrapper function in frontend to submit queries to the backend.

**Request Format**:
```javascript
{
  "url": "string (backend API URL)",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "query_text": "string",
    "selected_text": "string (optional)"
  },
  "timeout": "number (milliseconds)"
}
```

**Response Format**:
```javascript
{
  "success": "boolean",
  "data": "QueryResponse object or null",
  "error": "Error object or null",
  "status": "number (HTTP status code)"
}
```

## Communication Patterns

### Request-Response Flow

1. Frontend captures user query and selected text
2. Frontend validates inputs according to data model rules
3. Frontend sends POST request to `/query` endpoint
4. Backend processes query using RAG system
5. Backend returns response with AI-generated answer
6. Frontend displays response in book interface

### Error Handling

- **Network Errors**: Frontend displays user-friendly message about connection issues
- **Validation Errors**: Frontend shows specific error about query requirements
- **Backend Errors**: Frontend displays general error message with option to retry
- **Timeout**: Frontend shows timeout message and allows retry

## Security Considerations

- All requests must use HTTPS in production
- Input validation must be performed on both frontend and backend
- Response content must be sanitized before display
- Error messages should not expose sensitive system information

## Performance Requirements

- API response time: Under 5 seconds for 90% of requests
- Frontend should show loading indicators during requests
- Timeout threshold: 30 seconds for query requests
- Retry mechanism: 2 retries on network errors with exponential backoff

## Versioning Strategy

- API versioning through URL path (e.g., `/api/v1/query`)
- Backward compatibility maintained for major version changes
- Breaking changes require new major version number