"""
Basic tests for US1 functionality: Query the AI Agent with Book Context
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from ..models import QueryRequest
from ..agent import RAGAgent
from ..retrieval_tool import retrieve_context_for_agent


def test_query_request_validation():
    """Test validation of query requests."""
    # Valid query
    query = QueryRequest(query_text="What is artificial intelligence?")
    assert query.query_text == "What is artificial intelligence?"

    # Test query text validation
    with pytest.raises(ValidationError):
        QueryRequest(query_text="")  # Empty query

    with pytest.raises(ValidationError):
        QueryRequest(query_text="  ")  # Whitespace only

    with pytest.raises(ValidationError):
        QueryRequest(query_text="A")  # Too short when stripped

    # Valid selected text
    query_with_selected = QueryRequest(
        query_text="What is AI?",
        selected_text="Artificial intelligence is a branch of computer science..."
    )
    assert query_with_selected.selected_text is not None

    # Invalid selected text
    with pytest.raises(ValidationError):
        QueryRequest(
            query_text="What is AI?",
            selected_text=""  # Empty selected text
        )


@patch('backend.rag_agent.retrieval_tool.retrieve_book_context')
def test_retrieve_context_for_agent(mock_retrieve):
    """Test context retrieval functionality."""
    # Mock the retrieval result
    mock_context = MagicMock()
    mock_context.content = "Artificial intelligence is a branch of computer science..."
    mock_context.source = "test_source"
    mock_context.relevance_score = 0.85
    mock_context.chunk_id = "chunk_1"
    mock_context.metadata = {}
    mock_context.similarity_score = 0.85

    mock_retrieve.return_value = [mock_context]

    # Test retrieval
    contexts = retrieve_context_for_agent("What is AI?")
    assert len(contexts) == 1
    assert contexts[0].content == "Artificial intelligence is a branch of computer science..."


@patch('backend.rag_agent.clients.openai_client')
def test_rag_agent_initialization(mock_client):
    """Test RAG agent initialization."""
    # Mock the OpenAI client
    mock_openai = MagicMock()
    mock_client.get_client.return_value = mock_openai
    mock_client.get_model.return_value = "gpt-4-turbo"

    agent = RAGAgent()
    assert agent.model == "gpt-4-turbo"


def test_env_vars_loaded():
    """Test that environment variables are accessible."""
    # These should not raise exceptions if .env is properly configured
    assert os.getenv("OPENAI_API_KEY") is not None  # May be empty string but not None if file exists
    assert os.getenv("AGENT_MODEL") is not None


if __name__ == "__main__":
    # Run basic tests
    test_query_request_validation()
    test_env_vars_loaded()
    print("✅ Basic US1 functionality tests passed!")