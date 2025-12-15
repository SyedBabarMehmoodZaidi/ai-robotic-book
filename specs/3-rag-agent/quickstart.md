# Quickstart Guide: RAG-Enabled AI Agent

**Feature**: 3-rag-agent
**Created**: 2025-12-15
**Status**: Draft

## Overview

This guide provides a quick start for implementing the RAG-enabled AI agent using OpenAI Agent SDK and FastAPI. Follow these steps to set up the development environment and begin implementation.

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Access to the retrieval pipeline from Spec-2
- UV package manager (or pip)

## Setup

### 1. Clone and Navigate to Project

```bash
cd F:\GS Assignment\ai-robotic-book
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn openai python-dotenv pydantic slowapi
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
RETRIEVAL_ENDPOINT=http://localhost:8000  # Endpoint for Spec-2 retrieval pipeline
AGENT_MODEL=gpt-4-turbo
RATE_LIMIT_REQUESTS=100
CONTEXT_SIZE_LIMIT=4000
```

## Project Structure

Create the following directory structure:

```
backend/
├── rag_agent/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── agent.py         # OpenAI Agent integration
│   ├── models.py        # Pydantic models based on data-model.md
│   ├── retrieval_tool.py # Integration with Spec-2 retrieval pipeline
│   └── config.py        # Configuration and settings
├── tests/
│   ├── test_agent.py
│   ├── test_api.py
│   └── conftest.py
└── requirements.txt
```

## Implementation Steps

### 1. Create Pydantic Models

Create `backend/rag_agent/models.py` based on the data model:

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query_text: str
    selected_text: Optional[str] = None

class RetrievedContext(BaseModel):
    content: str
    source: str
    relevance_score: float
    chunk_id: str
    metadata: dict
    similarity_score: float

class AgentResponse(BaseModel):
    response_text: str
    source_context: List[str]
    confidence_score: float
    tokens_used: int
    processing_time: float
    query_id: str
    is_hallucination_detected: bool

class APIResponse(BaseModel):
    response: AgentResponse
    request_id: str
    status_code: int
    timestamp: str
    processing_time: float
```

### 2. Implement Agent Integration

Create `backend/rag_agent/agent.py`:

```python
import os
from openai import OpenAI
from typing import List, Optional
from .retrieval_tool import retrieve_book_context

class RAGAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("AGENT_MODEL", "gpt-4-turbo")

    async def process_query(self, query_text: str, selected_text: Optional[str] = None):
        # 1. Retrieve context using the Spec-2 pipeline
        retrieved_context = retrieve_book_context(query_text, selected_text)

        # 2. Format context for the agent
        context_str = "\n".join([ctx.content for ctx in retrieved_context])

        # 3. Create a message for the agent with the context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based only on the provided book context. Do not make up information."
            },
            {
                "role": "user",
                "content": f"Context: {context_str}\n\nQuestion: {query_text}"
            }
        ]

        # 4. Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )

        return {
            "response_text": response.choices[0].message.content,
            "source_context": [ctx.source for ctx in retrieved_context],
            "confidence_score": 0.9,  # This would be calculated based on similarity
            "tokens_used": response.usage.total_tokens,
            "is_hallucination_detected": False  # This would be validated
        }
```

### 3. Create FastAPI Application

Create `backend/rag_agent/main.py`:

```python
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import uuid

from .models import QueryRequest, AgentResponse
from .agent import RAGAgent

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="API for RAG-enabled AI agent",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize agent
rag_agent = RAGAgent()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "details": {
            "agent_status": "ready",
            "retrieval_status": "connected",
            "api_status": "operational"
        }
    }

@app.post("/query", response_model=AgentResponse)
@limiter.limit("100/minute")
async def query_agent(request: Request, query: QueryRequest):
    start_time = time.time()

    try:
        # Process the query with the agent
        result = await rag_agent.process_query(
            query.query_text,
            query.selected_text
        )

        processing_time = time.time() - start_time

        # Create response object
        agent_response = AgentResponse(
            response_text=result["response_text"],
            source_context=result["source_context"],
            confidence_score=result["confidence_score"],
            tokens_used=result["tokens_used"],
            processing_time=processing_time,
            query_id=str(uuid.uuid4()),
            is_hallucination_detected=result["is_hallucination_detected"]
        )

        return agent_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Test the Implementation

Create `backend/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from rag_agent.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    # This test requires the retrieval pipeline to be running
    query_data = {
        "query_text": "What is artificial intelligence?"
    }
    response = client.post("/query", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert "response_text" in data
    assert "source_context" in data
```

## Running the Application

### 1. Start the Application

```bash
cd backend
uvicorn rag_agent.main:app --reload --port 8000
```

### 2. Test the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What are the main principles of artificial intelligence?"}'
```

## Next Steps

1. Implement the retrieval tool integration with Spec-2 pipeline
2. Add comprehensive error handling
3. Implement hallucination detection
4. Add monitoring and logging
5. Create more comprehensive tests
6. Implement security measures
7. Optimize performance

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Ensure `OPENAI_API_KEY` is set in environment variables
2. **Retrieval Pipeline Not Found**: Verify the retrieval pipeline from Spec-2 is running
3. **Rate Limiting**: Check the rate limit configuration in environment variables

### Getting Help

- Check the implementation plan in `plan.md`
- Review the data models in `data-model.md`
- Consult the API contracts in `contracts/api-contracts.md`
- Review the research findings in `research.md`