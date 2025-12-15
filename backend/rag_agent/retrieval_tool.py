"""
Integration module for connecting the RAG agent with the existing retrieval pipeline from Spec-2.
This module provides functions to retrieve book context based on user queries.
"""
import sys
import os
from typing import List, Optional
from .models import RetrievedContext
from .config import settings

# Add the backend directory to the path so imports work
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

# Import from the existing retrieval pipeline
from retrieval_validator import execute_similarity_search_query
from query_converter import query_to_vector


def retrieve_book_context(query_text: str, selected_text: Optional[str] = None, top_k: int = 5) -> List[RetrievedContext]:
    """
    Retrieve book context based on the user's query using the existing retrieval pipeline.

    Args:
        query_text (str): The main query text from the user
        selected_text (Optional[str]): Specific text segment to focus on (if provided)
        top_k (int): Number of results to retrieve (default: 5)

    Returns:
        List[RetrievedContext]: List of retrieved context objects with content and metadata
    """
    # If selected_text is provided, we'll create a more targeted search
    search_query = query_text
    if selected_text:
        # Combine the query with selected text for better retrieval
        # This helps the search algorithm find relevant context around the selected text
        search_query = f"{query_text} {selected_text}"

        # If selected text is very long, we might want to summarize or truncate it
        if len(selected_text) > 1000:
            # For very long selected text, we'll use the first 1000 characters
            selected_text = selected_text[:1000]
            search_query = f"{query_text} {selected_text}"

    # Execute similarity search using the existing pipeline
    results = execute_similarity_search_query(search_query, top_k)

    # Convert results to RetrievedContext objects
    retrieved_contexts = []
    for result in results:
        context = RetrievedContext(
            content=result.get('content', ''),
            source=result.get('metadata', {}).get('url', ''),
            relevance_score=result.get('score', 0.0),
            chunk_id=result.get('metadata', {}).get('chunk_id', ''),
            metadata=result.get('metadata', {}),
            similarity_score=result.get('score', 0.0)
        )
        retrieved_contexts.append(context)

    # If selected_text was provided, we might want to prioritize or include it directly
    # as a high-priority context item, but for now we rely on the retrieval algorithm
    # to find relevant content based on the combined query

    return retrieved_contexts


def preprocess_selected_text(selected_text: Optional[str]) -> Optional[str]:
    """
    Preprocess the selected text to ensure it's in a suitable format for retrieval.

    Args:
        selected_text (Optional[str]): The selected text to preprocess

    Returns:
        Optional[str]: The preprocessed selected text, or None if input was None
    """
    if selected_text is None:
        return None

    # Clean up the selected text
    cleaned_text = selected_text.strip()

    # If the text is too long, we might want to truncate or summarize it
    if len(cleaned_text) > 5000:  # Maximum length as per model validation
        cleaned_text = cleaned_text[:5000]

    # Additional preprocessing could include:
    # - Removing excessive whitespace
    # - Normalizing special characters
    # - Removing or replacing problematic characters

    return cleaned_text if cleaned_text else None


def retrieve_context_for_agent(query_text: str, selected_text: Optional[str] = None) -> List[RetrievedContext]:
    """
    Retrieve context specifically formatted for the RAG agent.

    Args:
        query_text (str): The query text from the user
        selected_text (Optional[str]): Specific text segment to focus on (if provided)

    Returns:
        List[RetrievedContext]: List of retrieved context objects
    """
    # Use the main retrieval function with default top_k
    return retrieve_book_context(query_text, selected_text, top_k=5)


def validate_retrieval_quality(retrieved_contexts: List[RetrievedContext]) -> bool:
    """
    Validate the quality of retrieved contexts.

    Args:
        retrieved_contexts (List[RetrievedContext]): List of retrieved contexts to validate

    Returns:
        bool: True if retrieval quality is acceptable, False otherwise
    """
    if not retrieved_contexts:
        return False

    # Check if we have contexts with acceptable relevance scores
    # For now, we'll consider any context with relevance > 0.3 as acceptable
    acceptable_contexts = [ctx for ctx in retrieved_contexts if ctx.relevance_score > 0.3]

    # We should have at least one acceptable context
    return len(acceptable_contexts) > 0