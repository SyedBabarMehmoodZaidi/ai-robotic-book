# Data Model: RAG-Enabled AI Agent

**Feature**: 3-rag-agent
**Created**: 2025-12-15
**Status**: Draft
**Author**: Claude

## Overview

This document defines the data models for the RAG-enabled AI agent system based on the key entities identified in the feature specification.

## Entity: Query

**Description**: Represents a user's request to the AI agent, containing the question and optional parameters.

**Fields**:
- `query_text` (string, required): The main question or query from the user
- `selected_text` (string, optional): Specific text segment the user wants to focus on
- `context_window` (integer, optional): Size of context window (default: system determined)
- `user_id` (string, optional): Identifier for the requesting user (for future expansion)
- `metadata` (object, optional): Additional query metadata including timestamp, source, etc.

**Validation Rules**:
- `query_text` must be 1-2000 characters
- `selected_text` if provided must be 1-5000 characters
- `context_window` if provided must be between 100-4000 tokens

**State Transitions**:
- `received` → `processing` → `completed` | `failed`

## Entity: RetrievedContext

**Description**: Represents the context retrieved from the book content based on the user's query.

**Fields**:
- `content` (string, required): The retrieved book content
- `source` (string, required): Source document/section identifier
- `relevance_score` (number, required): Relevance score from retrieval (0.0-1.0)
- `chunk_id` (string, required): Unique identifier for the content chunk
- `metadata` (object, required): Additional metadata including url, section, etc.
- `similarity_score` (number, required): Similarity score to the original query

**Validation Rules**:
- `content` must be 10-10000 characters
- `relevance_score` must be between 0.0 and 1.0
- `source` must be a valid URL or document identifier
- `chunk_id` must be unique within the system

**State Transitions**:
- `retrieved` → `validated` → `formatted_for_agent`

## Entity: AgentResponse

**Description**: Represents the AI-generated response to the user's query based on the retrieved context.

**Fields**:
- `response_text` (string, required): The AI-generated response
- `source_context` (array of strings, required): References to source material used
- `confidence_score` (number, required): Confidence in response accuracy (0.0-1.0)
- `tokens_used` (integer, required): Number of tokens in response
- `processing_time` (number, required): Time taken to generate response in seconds
- `query_id` (string, required): Reference to the original query
- `is_hallucination_detected` (boolean, required): Flag if hallucination was detected

**Validation Rules**:
- `response_text` must be 10-10000 characters
- `confidence_score` must be between 0.0 and 1.0
- `tokens_used` must be positive
- `processing_time` must be positive
- `source_context` must reference actual retrieved context

**State Transitions**:
- `generating` → `validated` → `completed` | `failed`

## Entity: APIRequest

**Description**: Represents the HTTP request to the FastAPI endpoint containing the query parameters.

**Fields**:
- `query` (Query object, required): The query object as defined above
- `request_id` (string, required): Unique identifier for the request
- `timestamp` (string, required): ISO 8601 timestamp of the request
- `client_ip` (string, optional): IP address of the requesting client
- `api_key_hash` (string, optional): Hash of the API key used (for rate limiting)

**Validation Rules**:
- `request_id` must be unique
- `timestamp` must be current or recent
- `query` must pass all Query validation rules

## Entity: APIResponse

**Description**: Represents the HTTP response from the FastAPI endpoint containing the agent's answer.

**Fields**:
- `response` (AgentResponse object, required): The agent response as defined above
- `request_id` (string, required): Reference to the original request
- `status_code` (integer, required): HTTP status code
- `timestamp` (string, required): ISO 8601 timestamp of the response
- `processing_time` (number, required): Total processing time in seconds

**Validation Rules**:
- `request_id` must match the original request
- `status_code` must be valid HTTP status
- `response` must pass all AgentResponse validation rules

## Entity: AgentSession (Optional for future multi-turn conversations)

**Description**: Represents a session for multi-turn conversations with the agent.

**Fields**:
- `session_id` (string, required): Unique identifier for the session
- `user_id` (string, optional): Identifier for the user
- `created_at` (string, required): ISO 8601 timestamp of session creation
- `last_activity` (string, required): ISO 8601 timestamp of last activity
- `conversation_history` (array of objects, optional): History of conversation turns
- `context_window` (array of RetrievedContext, optional): Maintained context for the session

**Validation Rules**:
- `session_id` must be unique
- `created_at` must be before `last_activity`
- `conversation_history` must be properly formatted

## Relationships

```
Query 1----* RetrievedContext (one query can retrieve multiple contexts)
RetrievedContext *----1 AgentResponse (multiple contexts contribute to one response)
Query 1----1 AgentResponse (one-to-one relationship for each query-response pair)
APIRequest 1----1 Query (one request contains one query)
APIResponse 1----1 AgentResponse (one response contains one agent response)
```

## Data Flow

1. User submits an `APIRequest` containing a `Query`
2. System retrieves multiple `RetrievedContext` objects based on the query
3. Agent generates an `AgentResponse` using the retrieved contexts
4. System returns an `APIResponse` containing the agent response
5. Optionally, session data is maintained in `AgentSession` for multi-turn conversations

## Constraints

- All text fields should be properly sanitized to prevent injection attacks
- Confidence scores should be calculated based on similarity between query and retrieved content
- Source context references must be verifiable against the actual retrieved content
- Processing times should be monitored for performance optimization
- All entities should have proper timestamps for audit purposes