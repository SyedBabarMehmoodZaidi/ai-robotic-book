from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
import logging
import html
import re
import time

from ..models.agent_query import AgentQuery
from ..models.agent_response import AgentResponse
from ..models.agent_configuration import AgentConfiguration
from ..agents.rag_agent import RAGAgent
from ..config.agent_settings import agent_settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def sanitize_text(text: str) -> str:
    """
    Sanitize text input by removing potentially harmful content.

    Args:
        text: The text to sanitize

    Returns:
        Sanitized text with potentially harmful content removed
    """
    if not text:
        return text

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove potential script tags (case insensitive)
    text = re.sub(r'(?i)<script[^>]*>.*?</script>', '', text)

    # Remove potential javascript: urls
    text = re.sub(r'(?i)javascript:', '', text)

    # Remove potential data: urls
    text = re.sub(r'(?i)data:', '', text)

    return text.strip()


def validate_query_input(agent_query: AgentQuery) -> None:
    """
    Validate and sanitize the query input before processing.

    Args:
        agent_query: The query object to validate

    Raises:
        HTTPException: If validation fails
    """
    # Sanitize query text
    sanitized_query_text = sanitize_text(agent_query.query_text)
    if sanitized_query_text != agent_query.query_text:
        logger.warning(f"Query text was sanitized: original length {len(agent_query.query_text)}, sanitized length {len(sanitized_query_text)}")
        agent_query.query_text = sanitized_query_text

    # Sanitize context text if provided
    if agent_query.context_text:
        sanitized_context_text = sanitize_text(agent_query.context_text)
        if sanitized_context_text != agent_query.context_text:
            logger.warning(f"Context text was sanitized: original length {len(agent_query.context_text)}, sanitized length {len(sanitized_context_text)}")
            agent_query.context_text = sanitized_context_text

    # Additional validation checks
    if not agent_query.query_text or len(agent_query.query_text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty after sanitization"
        )

    # Check for potentially malicious patterns
    malicious_patterns = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'os\.',
        r'subprocess\.',
        r'import\s+os',
        r'import\s+subprocess',
        r'import\s+sys',
    ]

    for pattern in malicious_patterns:
        if re.search(pattern, agent_query.query_text, re.IGNORECASE):
            logger.warning(f"Potentially malicious pattern detected in query: {pattern}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query contains potentially unsafe content"
            )

    # Validate query type
    if agent_query.query_type not in ["general", "context-specific"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query type must be either 'general' or 'context-specific'"
        )

    # Check content length limits
    if len(agent_query.query_text) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text exceeds maximum length of 2000 characters"
        )

    if agent_query.context_text and len(agent_query.context_text) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context text exceeds maximum length of 10000 characters"
        )


def get_rag_agent():
    """
    Dependency function to get RAG agent instance.

    Returns:
        RAGAgent: Instance of the RAG agent
    """
    try:
        config = AgentConfiguration(
            model_name=agent_settings.openai_model_name,
            temperature=agent_settings.openai_temperature,
            max_tokens=agent_settings.openai_max_tokens,
            retrieval_threshold=agent_settings.agent_retrieval_threshold,
            context_window=agent_settings.agent_context_window
        )
        agent = RAGAgent(config)
        return agent
    except Exception as e:
        logger.error(f"Failed to create RAG agent instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable due to agent initialization error"
        )


@router.post("/query",
             response_model=AgentResponse,
             summary="Process an agent query",
             description="Submit a query to the RAG agent for processing using book content as context")
async def query_agent(
    agent_query: AgentQuery,
    agent: RAGAgent = Depends(get_rag_agent)
) -> AgentResponse:
    """
    Process a query using the RAG agent.

    Args:
        agent_query: The query to process with optional context
        agent: The RAG agent instance (injected via dependency)

    Returns:
        AgentResponse: The response generated by the agent

    Raises:
        HTTPException: If query processing fails
    """
    start_time = time.time()

    try:
        # Validate and sanitize the query input
        validate_query_input(agent_query)

        logger.info(f"Processing agent query: '{agent_query.query_text[:50]}...'")
        logger.info(f"Query type: {agent_query.query_type}, User ID: {agent_query.user_id}")

        # Process the query using the RAG agent
        response = agent.process_query(agent_query)

        # Add processing time to metadata if not already present
        processing_time = time.time() - start_time
        if response.metadata is None:
            response.metadata = {}
        response.metadata['processing_time_seconds'] = round(processing_time, 3)
        response.metadata['query_timestamp'] = agent_query.created_at.isoformat()

        logger.info(f"Query processed successfully, response length: {len(response.response_text)}, processing time: {processing_time:.3f}s")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        processing_time = time.time() - start_time
        logger.warning(f"Query failed after {processing_time:.3f}s processing time")
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing agent query after {processing_time:.3f}s: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/health",
            summary="Check agent health",
            description="Verify that the RAG agent is healthy and ready to process queries")
async def agent_health_check(agent: RAGAgent = Depends(get_rag_agent)) -> dict:
    """
    Check the health of the RAG agent.

    Args:
        agent: The RAG agent instance (injected via dependency)

    Returns:
        dict: Health status information
    """
    try:
        is_healthy = agent.health_check()

        if is_healthy:
            return {
                "status": "healthy",
                "message": "RAG agent is operational and ready to process queries",
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG agent is not healthy"
            )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.post("/query/configure",
             summary="Update agent configuration",
             description="Update the configuration settings for the RAG agent")
async def configure_agent(
    config: AgentConfiguration,
    agent: RAGAgent = Depends(get_rag_agent)
) -> dict:
    """
    Update the agent configuration.

    Args:
        config: New configuration settings
        agent: The RAG agent instance (injected via dependency)

    Returns:
        dict: Confirmation of configuration update
    """
    try:
        # Update agent configuration
        agent.config = config

        return {
            "status": "success",
            "message": "Agent configuration updated successfully",
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"Error updating agent configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}"
        )