# Research: RAG-Enabled AI Agent Implementation

**Feature**: 3-rag-agent
**Created**: 2025-12-15
**Status**: Complete
**Author**: Claude

## Research Summary

This document addresses the key unknowns and technical decisions required for implementing the RAG-enabled AI agent using OpenAI Agent SDK and FastAPI.

## OpenAI Agent SDK Configuration

### Decision: Use OpenAI Assistant API as the foundation
**Rationale**: The OpenAI Agent SDK primarily refers to the Assistant API, which provides the orchestration capabilities needed for our RAG system. The Assistant API allows creating agents with custom instructions, tools, and retrieval capabilities.

**Technical Details**:
- Use `openai.Assistant` for agent orchestration
- Configure with `gpt-4-turbo` or `gpt-4o` model for best performance
- Set custom instructions to enforce RAG pattern compliance
- Implement custom tools for retrieval integration

**Implementation Approach**:
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

assistant = client.beta.assistants.create(
    name="RAG Book Assistant",
    instructions="You are a helpful assistant that answers questions based only on the provided book context. Do not make up information.",
    model="gpt-4-turbo",
    tools=[{"type": "retrieval"}]  # This enables file-based retrieval
)
```

**Alternatives Considered**:
- LangChain Agents: More complex but more flexible
- Custom agent implementation: More control but more development work
- OpenAI Functions: Good for tool calling but less orchestration

## Retrieval Pipeline Integration

### Decision: Create a custom tool to integrate with existing retrieval pipeline
**Rationale**: The existing retrieval pipeline from Spec-2 provides the required functionality. Creating a custom tool allows tight integration while maintaining the existing pipeline's validation and quality features.

**Technical Details**:
- Implement a custom OpenAI tool that calls the existing retrieval pipeline
- Pass query text and selected text to the retrieval pipeline
- Format retrieved results for agent consumption
- Handle edge cases like no results or empty queries

**Implementation Approach**:
```python
def retrieve_book_context(query: str, selected_text: Optional[str] = None) -> str:
    # Call existing retrieval pipeline from Spec-2
    # Format results appropriately for the agent
    pass

# Register as OpenAI tool
tool_definition = {
    "type": "function",
    "function": {
        "name": "retrieve_book_context",
        "description": "Retrieve relevant book content for answering questions",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question"},
                "selected_text": {"type": "string", "description": "Optional specific text to focus on"}
            },
            "required": ["query"]
        }
    }
}
```

**Alternatives Considered**:
- File-based retrieval: Would require indexing content differently
- Vector database integration: Would duplicate existing functionality
- Direct embedding: Would require more complex implementation

## Rate Limiting and Concurrency Handling

### Decision: Implement basic rate limiting using in-memory storage with option for Redis
**Rationale**: Rate limiting is essential for API stability and cost management. A simple in-memory solution works for single-server deployments, with Redis as an option for production scaling.

**Technical Details**:
- Use FastAPI middleware for rate limiting
- Track requests by IP or API key
- Implement sliding window algorithm
- Return appropriate HTTP 429 status for exceeded limits

**Implementation Approach**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

**Alternatives Considered**:
- No rate limiting: Would risk API abuse and excessive costs
- External service: Would add complexity and dependencies
- Token bucket algorithm: More complex than needed for initial implementation

## Context Enforcement Mechanisms

### Decision: Combine custom instructions with retrieval tool integration
**Rationale**: To ensure responses are grounded in retrieved context, we'll use a multi-layer approach: custom instructions for the agent, mandatory retrieval tool usage, and validation of response sources.

**Technical Details**:
- Custom instructions prevent the agent from making up information
- Mandatory retrieval tool ensures context is always retrieved first
- Response validation to confirm sources match retrieved context
- Selected-text queries will be handled by preprocessing the context

**Implementation Approach**:
```python
# Agent instructions that enforce context-only responses
instructions = """
Answer questions based ONLY on the provided context.
Do not use prior knowledge or make up information.
Always cite the source of your information.
"""

# Tool that retrieves context before answering
def retrieve_and_answer(query: str) -> dict:
    context = retrieve_book_context(query)
    # Use context to answer the query
    return {"context": context, "query": query}
```

**Alternatives Considered**:
- Post-generation validation: Would be less effective than prevention
- Fine-tuning: Would require more data and resources
- External validation service: Would add latency and complexity

## FastAPI Integration Patterns

### Decision: Use async endpoints with proper request/response validation
**Rationale**: FastAPI's async capabilities align well with AI API calls, and Pydantic validation ensures data integrity.

**Technical Details**:
- Async endpoints to handle I/O efficiently
- Pydantic models for request/response validation
- Proper error handling with meaningful messages
- Structured logging for debugging and monitoring

**Implementation Approach**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query_text: str
    selected_text: Optional[str] = None

class AgentResponse(BaseModel):
    response_text: str
    source_context: List[str]
    confidence_score: float

@app.post("/api/v1/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    try:
        # Process query with agent
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Alternatives Considered**:
- Synchronous endpoints: Would block during AI API calls
- Manual validation: Would be error-prone and verbose
- Custom serialization: Would lose Pydantic benefits

## Additional Research Findings

### Error Handling Strategy
- Implement comprehensive error handling for OpenAI API failures
- Handle rate limit errors with appropriate retry logic
- Graceful degradation when retrieval pipeline is unavailable
- Detailed logging for debugging purposes

### Performance Considerations
- Cache frequently accessed content to reduce retrieval time
- Implement connection pooling for OpenAI API calls
- Use streaming responses for large outputs
- Optimize context window usage to minimize token costs

### Security Measures
- Validate all user inputs to prevent injection attacks
- Implement proper authentication if required
- Sanitize outputs to prevent XSS in client applications
- Secure API key storage and access

## Conclusion

All identified unknowns have been researched and resolved. The implementation plan can now proceed with confidence in the technical approaches selected. The chosen solutions balance functionality, maintainability, and performance while adhering to the project's constraints and requirements.