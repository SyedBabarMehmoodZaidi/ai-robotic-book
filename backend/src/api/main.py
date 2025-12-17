from fastapi import FastAPI
from .search_router import search_router
from .validation_router import validation_router
from .health_router import health_router
from .agent_router import agent_router
from ..config.settings import settings
import logging


# Initialize the main FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG Retrieval API - Semantic search in embedded book content using Qdrant and Cohere",
    debug=settings.debug
)


# Include all routers
app.include_router(search_router)
app.include_router(validation_router)
app.include_router(health_router)
app.include_router(agent_router)


# Add middleware for request logging if needed
@app.middleware("http")
async def log_requests(request, call_next):
    if settings.debug:
        logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# Root endpoint for basic info
@app.get("/")
async def root():
    return {
        "message": "RAG Retrieval API",
        "version": settings.app_version,
        "endpoints": [
            "/api/v1/search - Perform semantic search",
            "/api/v1/validate - Validate retrieval quality",
            "/api/v1/health - Check service health"
        ]
    }