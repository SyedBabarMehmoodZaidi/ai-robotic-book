# Quickstart Guide: RAG-Enabled AI Agent Implementation

## Overview
This guide provides the essential steps to set up and run the RAG-enabled AI agent that uses OpenAI Agent SDK and FastAPI to answer queries based on book content.

## Prerequisites
- Python 3.11+
- pip package manager
- OpenAI API key
- Access to the retrieval pipeline from Spec-2 (RAG retrieval service)
- Running Qdrant vector database with book embeddings

## Setup

### 1. Navigate to the Backend Directory
```bash
cd backend
```

### 2. Install Dependencies
```bash
pip install openai fastapi uvicorn pydantic python-dotenv requests pytest
```

### 3. Environment Configuration
Update your `.env` file with the necessary keys:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_embeddings
DEBUG=false
AGENT_MODEL_NAME=gpt-4-turbo  # or your preferred model
AGENT_TEMPERATURE=0.1  # Low temperature for factual responses
AGENT_MAX_TOKENS=1000
RETRIEVAL_THRESHOLD=0.5
```

## Core Components

### 1. RAG Agent
Orchestrates retrieval and response generation:
- Initializes OpenAI Agent SDK with proper configuration
- Integrates with retrieval pipeline from Spec-2
- Enforces context-only answering to prevent hallucinations
- Handles both general and context-specific queries

### 2. Retrieval Integration Service
Manages communication with the existing retrieval pipeline:
- Calls retrieval service with user queries
- Processes retrieved content chunks
- Provides context to the agent for response generation

### 3. Response Generator
Processes agent responses and ensures quality:
- Validates that responses are grounded in retrieved content
- Formats responses with proper attribution
- Measures and reports confidence scores

### 4. Agent Router
Handles API requests and responses:
- Validates incoming query requests
- Orchestrates the agent processing workflow
- Returns properly formatted responses

## Running the Service

### 1. Start the API Server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 2. Submit a Query
```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the key concepts in neural networks?",
    "query_type": "general"
  }'
```

### 3. Submit a Context-Specific Query
```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Explain the concept mentioned here",
    "context_text": "Neural networks are computing systems inspired by the human brain...",
    "query_type": "context-specific"
  }'
```

## API Endpoints

Once the service is running, you can use these endpoints:

- `POST /api/v1/agent/query` - Submit queries to the RAG agent
- `GET /api/v1/agent/health` - Check service health status

## Testing

Run the following to verify your setup:
```bash
python -m pytest tests/
```

## Next Steps

1. Implement the RAG agent with OpenAI Agent SDK integration
2. Build the retrieval integration service to connect with Spec-2 pipeline
3. Add comprehensive error handling and validation
4. Implement the API endpoints with proper request/response handling
5. Test with various book-related queries to validate grounded responses