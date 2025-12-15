from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import os
from dotenv import load_dotenv

from .models import QueryRequest
from .services.query_service import query_service

# Load environment variables
load_dotenv()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="API for RAG-enabled AI agent using OpenAI Agent SDK",
    version="0.1.0"
)

# Add rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
async def root():
    return {"message": "RAG Agent API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "rag-agent-api"
    }

@app.post("/query")
@limiter.limit("100/minute")  # Use rate limit from config
async def query_endpoint(request: Request, query: QueryRequest):
    """
    Process a query using the RAG agent.

    Args:
        request: The HTTP request object
        query: The query request containing the question and parameters

    Returns:
        APIResponse: The agent's response with source context and metadata
    """
    try:
        # Get client IP for logging
        client_ip = request.client.host if request.client else None

        # Process the query using the query service
        api_response = query_service.process_query_request(query, client_ip)

        # Return the response from the agent
        return api_response

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)