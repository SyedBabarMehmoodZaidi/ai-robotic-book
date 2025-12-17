from fastapi import APIRouter
from typing import Dict, Any
import logging

from ..retrieval.retrieval_service import RetrievalService
from ..agents.rag_agent import RAGAgent
from ..models.agent_configuration import AgentConfiguration
from ..config.agent_settings import agent_settings
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

        # Test RAG agent health
        try:
            config = AgentConfiguration(
                model_name=agent_settings.openai_model_name,
                temperature=agent_settings.openai_temperature,
                max_tokens=agent_settings.openai_max_tokens,
                retrieval_threshold=agent_settings.agent_retrieval_threshold,
                context_window=agent_settings.agent_context_window
            )
            rag_agent = RAGAgent(config)
            is_agent_healthy = rag_agent.health_check()
        except Exception as agent_error:
            logger.error(f"Agent health check failed: {str(agent_error)}")
            is_agent_healthy = False

        # Determine overall health status
        overall_status = "healthy" if is_retrieval_healthy and is_agent_healthy else "unhealthy"

        health_status = {
            "status": overall_status,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "services": {
                "retrieval": {
                    "status": "healthy" if is_retrieval_healthy else "unhealthy",
                    "details": "Qdrant and Cohere connectivity"
                },
                "rag_agent": {
                    "status": "healthy" if is_agent_healthy else "unhealthy",
                    "details": "OpenAI connectivity and agent initialization"
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
                },
                "rag_agent": {
                    "status": "unhealthy",
                    "details": f"Error: {str(e)}"
                }
            },
            "version": "1.0.0"
        }