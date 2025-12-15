# Data Model: RAG Frontend Integration

**Feature**: 1-rag-frontend-integration
**Created**: 2025-12-15
**Status**: Draft
**Author**: Claude

## Overview

This document defines the data models for the RAG frontend integration system based on the key entities identified in the feature specification and implementation research.

## Entity: QueryRequest

**Description**: Represents a user's query request from the frontend to the RAG backend, containing the question and optional selected text context.

**Fields**:
- `query_text` (string, required): The user's question (3-2000 characters)
- `selected_text` (string, optional): Text selected by user for context (10-5000 characters)
- `metadata` (object, optional): Additional query metadata including timestamp, source, etc.

**Validation Rules**:
- `query_text` must be 3-2000 characters
- `selected_text` if provided must be 10-5000 characters
- `query_text` must not contain malicious content (XSS prevention)

**State Transitions**:
- `created` → `submitted` → `processing` → `completed` | `failed`

## Entity: QueryResponse

**Description**: Represents the response from the RAG backend to the frontend, containing the AI-generated answer and metadata.

**Fields**:
- `response` (object, required): The AI-generated response object
  - `response_text` (string, required): The AI's answer
  - `source_context` (array of strings, required): References to sources used in the response
  - `confidence_score` (number, required): Confidence level in the response accuracy (0.0-1.0)
  - `tokens_used` (number, optional): Number of tokens in response
  - `processing_time` (number, optional): Time taken to generate response
  - `query_id` (string, optional): Reference to the original query
  - `is_hallucination_detected` (boolean, optional): Flag if hallucination was detected
- `request_id` (string, required): Unique identifier for the request
- `status_code` (number, required): HTTP status code
- `timestamp` (string, required): ISO 8601 timestamp of the response
- `processing_time` (number, required): Total processing time in seconds

**Validation Rules**:
- `response_text` must be properly formatted and not contain unsafe content
- `confidence_score` must be between 0.0 and 1.0
- `status_code` must be valid HTTP status code
- `source_context` must reference actual sources from the knowledge base

**State Transitions**:
- `received` → `validating` → `displaying` | `error`

## Entity: FrontendQueryState

**Description**: Represents the state of a query in the frontend interface, tracking UI elements and user interactions.

**Fields**:
- `query_text` (string, required): The current query text in the input field
- `selected_text` (string, optional): The currently selected text on the page
- `isLoading` (boolean, required): Whether the query is currently being processed
- `response` (QueryResponse, optional): The response from the backend
- `error` (string, optional): Error message if query failed
- `timestamp` (string, required): ISO 8601 timestamp of the state

**Validation Rules**:
- `isLoading` must be consistent with actual request state
- `error` must be cleared when new query is submitted
- `response` must be updated only when valid response is received

**State Transitions**:
- `idle` → `querying` → `response_received` | `error` → `idle`

## Entity: TextSelection

**Description**: Represents the text currently selected by the user on the book page.

**Fields**:
- `text` (string, required): The selected text content
- `startOffset` (number, optional): Start position of the selection
- `endOffset` (number, optional): End position of the selection
- `elementId` (string, optional): ID of the element containing the selection
- `timestamp` (string, required): ISO 8601 timestamp when selection was captured

**Validation Rules**:
- `text` must be 10-5000 characters when present
- `startOffset` and `endOffset` must be valid positions within the text
- `text` must not contain executable scripts or other unsafe content

**State Transitions**:
- `none` → `active` → `cleared`

## Entity: APICommunication

**Description**: Represents the communication state between frontend and backend.

**Fields**:
- `url` (string, required): The backend API endpoint URL
- `method` (string, required): HTTP method (typically "POST")
- `headers` (object, required): HTTP headers for the request
- `timeout` (number, required): Request timeout in milliseconds
- `status` (string, required): Current status ("idle", "loading", "success", "error")
- `lastError` (string, optional): Last error message if any
- `timestamp` (string, required): ISO 8601 timestamp of the state

**Validation Rules**:
- `url` must be a valid backend endpoint
- `method` must be a valid HTTP method
- `timeout` must be a positive number
- `status` must be one of the allowed values

**State Transitions**:
- `idle` → `loading` → `success` | `error` → `idle`

## Relationships

```
FrontendQueryState 1----1 APICommunication (current communication for the query)
FrontendQueryState 1----1 QueryRequest (the request being processed)
FrontendQueryState 1----1 QueryResponse (the response received)
TextSelection 1----1 FrontendQueryState (selected text used in the query)
```

## Data Flow

1. User selects text on the page → TextSelection entity created/updated
2. User enters query → QueryRequest entity created with query and selected text
3. FrontendQueryState tracks the request → APICommunication handles the HTTP request
4. Backend responds → QueryResponse entity created from the response
5. FrontendQueryState updates with response → UI displays the response

## Constraints

- All text fields should be properly sanitized to prevent XSS attacks
- Query and response data should be validated before display
- Communication timeouts should be handled gracefully
- All entities should have proper timestamps for audit purposes
- Selected text should be captured efficiently without impacting page performance