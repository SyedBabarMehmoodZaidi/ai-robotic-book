import cohere
from typing import List
import logging
from config import get_cohere_client, Config, validate_query_text, logger


def query_to_vector(query_text: str) -> List[float]:
    """
    Converts a user query text to a vector representation using Cohere

    Args:
        query_text (str): The text query to convert

    Returns:
        List[float]: The vector representation of the query

    Raises:
        ValueError: If the query text is invalid
        Exception: If Cohere API call fails
    """
    from config import Config

    # Validate input using configuration parameter
    if not query_text or not isinstance(query_text, str) or len(query_text.strip()) == 0:
        raise ValueError(f"Invalid query text: must be a non-empty string")

    if len(query_text) > Config.MAX_QUERY_LENGTH:
        raise ValueError(f"Query text too long: maximum {Config.MAX_QUERY_LENGTH} characters allowed")

    try:
        # Get Cohere client
        cohere_client = get_cohere_client()

        # Convert query to vector using Cohere's embed API
        response = cohere_client.embed(
            texts=[query_text],
            model=Config.COHERE_MODEL,
            input_type=Config.COHERE_INPUT_TYPE  # Use configured input type
        )

        # Extract the embedding vector from the response
        if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
            query_vector = response.embeddings[0]  # Take the first (and only) embedding
            logger.info(f"Successfully converted query to vector with dimension {len(query_vector)}")
            return query_vector
        else:
            raise Exception("Cohere API returned empty embeddings")

    except Exception as e:
        logger.error(f"Cohere API call failed: {str(e)}")
        raise Exception(f"Cohere API call failed: {str(e)}")


if __name__ == "__main__":
    # Test the function
    test_query = "What is artificial intelligence?"
    try:
        vector = query_to_vector(test_query)
        print(f"Query: '{test_query}'")
        print(f"Vector length: {len(vector)}")
        print(f"First 5 elements: {vector[:5]}")
    except Exception as e:
        print(f"Error: {e}")