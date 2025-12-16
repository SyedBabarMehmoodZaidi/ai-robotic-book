from fastapi import APIRouter
from typing import Dict, Any
import logging
from ..retrieval.retrieval_service import RetrievalService
from ..utils.logging_config import get_logger


logger = get_logger(__name__)
health_router = APIRouter(prefix="/api/v1", tags=["health"])


@health_router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the application.

    Returns:
        Dictionary containing health status information
    """
    try:
        # Test retrieval service health
        retrieval_service = RetrievalService()
        is_retrieval_healthy = retrieval_service.health_check()

        # In a real implementation, you might also check:
        # - Database connections
        # - External API availability
        # - Resource usage

        health_status = {
            "status": "healthy" if is_retrieval_healthy else "unhealthy",
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "services": {
                "retrieval": {
                    "status": "healthy" if is_retrieval_healthy else "unhealthy",
                    "details": "Qdrant and Cohere connectivity"
                }
            },
            "version": "1.0.0"
        }

        logger.info(f"Health check completed: {health_status['status']}")

        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "error": str(e),
            "services": {
                "retrieval": {
                    "status": "unhealthy",
                    "details": f"Error: {str(e)}"
                }
            },
            "version": "1.0.0"
        }