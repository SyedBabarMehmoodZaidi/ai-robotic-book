from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from ..models.search_query import SearchQuery
from ..models.retrieved_chunk import RetrievedChunk
from ..retrieval.retrieval_service import RetrievalService
from ..retrieval.query_processor import QueryProcessor
from ..utils.logging_config import get_logger


logger = get_logger(__name__)
search_router = APIRouter(prefix="/api/v1", tags=["search"])


def get_retrieval_service() -> QueryProcessor:
    """
    Dependency to get the retrieval service.
    In a real implementation, this would be managed by a DI container.
    """
    retrieval_service = RetrievalService()
    query_processor = QueryProcessor(retrieval_service)
    return query_processor


@search_router.post("/search", response_model=List[RetrievedChunk])
async def search_endpoint(
    search_query: SearchQuery,
    query_processor: QueryProcessor = Depends(get_retrieval_service)
) -> List[RetrievedChunk]:
    """
    Perform semantic search in the embedded book content.

    Args:
        search_query: Search parameters including query text, top_k, and similarity threshold
        query_processor: Query processor service

    Returns:
        List of retrieved chunks ranked by similarity score
    """
    try:
        logger.info(f"Received search request: '{search_query.query_text[:50]}...'")

        results = query_processor.process_search(search_query)

        logger.info(f"Returning {len(results)} results for query: '{search_query.query_text[:50]}...'")

        return results
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )


@search_router.post("/search/text", response_model=List[RetrievedChunk])
async def search_text_endpoint(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    query_processor: QueryProcessor = Depends(get_retrieval_service)
) -> List[RetrievedChunk]:
    """
    Perform semantic search using query text directly.

    Args:
        query_text: The text to search for
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score threshold
        query_processor: Query processor service

    Returns:
        List of retrieved chunks ranked by similarity score
    """
    try:
        logger.info(f"Received text search request: '{query_text[:50]}...'")

        results = query_processor.process_search_text(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        logger.info(f"Returning {len(results)} results for text query: '{query_text[:50]}...'")

        return results
    except Exception as e:
        logger.error(f"Error processing text search request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )