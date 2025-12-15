from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from config import get_qdrant_client, Config, validate_top_k, logger


def perform_similarity_search(query_vector: List[float], top_k: int = 5) -> List[Dict]:
    """
    Performs top-k similarity search in Qdrant using the query vector

    Args:
        query_vector (List[float]): The vector representation of the query
        top_k (int, optional): Number of top results to return (default: 5)

    Returns:
        List[Dict]: List of retrieved chunks with content, metadata, and similarity scores

    Raises:
        Exception: If Qdrant connection fails
    """
    # Validate inputs
    if not isinstance(query_vector, list) or len(query_vector) == 0:
        raise ValueError("query_vector must be a non-empty list of floats")

    if not validate_top_k(top_k):
        raise ValueError(f"top_k must be a positive integer between 1 and 100, got {top_k}")

    try:
        # Get Qdrant client
        qdrant_client = get_qdrant_client()

        # Perform similarity search
        search_results = qdrant_client.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,  # Include the stored content
            with_vectors=False,  # We don't need the vectors back
            score_threshold=Config.SIMILARITY_THRESHOLD  # Filter results below threshold
        )

        # Format results to match expected structure
        retrieved_chunks = []
        for result in search_results:
            chunk = {
                'content': result.payload.get('content', ''),
                'metadata': {
                    'url': result.payload.get('url', ''),
                    'section': result.payload.get('section', ''),
                    'chunk_id': result.payload.get('chunk_id', ''),
                    'source_file': result.payload.get('source_file', ''),
                    # Include any other metadata fields that were stored
                    **{k: v for k, v in result.payload.items()
                       if k not in ['content', 'url', 'section', 'chunk_id', 'source_file']}
                },
                'score': result.score,
                'id': result.id
            }
            retrieved_chunks.append(chunk)

        logger.info(f"Similarity search returned {len(retrieved_chunks)} results")
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Qdrant similarity search failed: {str(e)}")
        # Return empty list if no relevant results found, as specified in function contract
        if "no relevant results" in str(e).lower() or "not found" in str(e).lower():
            logger.info("No relevant results found in similarity search")
            return []
        else:
            # Re-raise for connection or other errors
            raise Exception(f"Qdrant similarity search failed: {str(e)}")


def execute_similarity_search_query(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    High-level function to execute a similarity search from query text to retrieved chunks.
    This combines query conversion and similarity search in one function.

    Args:
        query_text (str): The text query to search for
        top_k (int, optional): Number of top results to return (default: 5)

    Returns:
        List[Dict]: List of retrieved chunks with content, metadata, and similarity scores
    """
    from query_converter import query_to_vector

    # Convert query text to vector
    query_vector = query_to_vector(query_text)

    # Perform similarity search
    results = perform_similarity_search(query_vector, top_k)

    return results


if __name__ == "__main__":
    # Test the function with a sample query
    sample_query = "What is artificial intelligence?"
    try:
        # This would require the query_converter module to be available
        # For testing purposes, we'll just test the structure with a mock vector
        mock_vector = [0.1] * 1024  # Assuming 1024-dimensional embeddings
        results = perform_similarity_search(mock_vector, top_k=3)
        print(f"Retrieved {len(results)} chunks")
        for i, chunk in enumerate(results):
            print(f"Result {i+1}: Score={chunk['score']:.4f}, Content length={len(chunk['content'])}")
    except Exception as e:
        print(f"Error during similarity search: {e}")
        print("This is expected if Qdrant is not running or collection doesn't exist yet.")