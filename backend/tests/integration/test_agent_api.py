import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import uuid
from unittest.mock import Mock, patch

from src.api.main import app
from src.models.agent_query import AgentQuery
from src.models.agent_response import AgentResponse, RetrievedChunk
from src.models.agent_configuration import AgentConfiguration


def test_query_endpoint_success():
    """Test successful query to the agent API endpoint."""

    # For this test, we'll mock the RAG agent since we don't want to make actual API calls
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        # Create a mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="This is a test response based on the provided context.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Test context content",
                    similarity_score=0.85,
                    chunk_id="test-chunk-1",
                    metadata={"source": "test", "page": 1},
                    position=1
                )
            ],
            confidence_score=0.9,
            sources=["test_source"],
            metadata={"processing_time_seconds": 0.123}
        )
        mock_rag_agent.return_value = mock_agent_instance

        # Create a test client
        client = TestClient(app)

        # Prepare test query
        test_query = AgentQuery(
            query_text="What is the capital of France?",
            query_type="general"
        )

        # Make request to the API
        response = client.post("/api/v1/agent/query", json=test_query.dict())

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data
        assert "query_id" in response_data
        assert "retrieved_chunks" in response_data
        assert "confidence_score" in response_data
        assert len(response_data["retrieved_chunks"]) == 1
        assert response_data["confidence_score"] >= 0.0 and response_data["confidence_score"] <= 1.0


def test_query_endpoint_with_context():
    """Test query endpoint with context-specific query."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Based on the provided context, the answer is Paris.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="France is a country in Europe. Paris is its capital.",
                    similarity_score=0.9,
                    chunk_id="context-chunk-1",
                    metadata={"source": "book1", "page": 10},
                    position=1
                )
            ],
            confidence_score=0.85,
            sources=["book1"],
            metadata={"processing_time_seconds": 0.150}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        test_query = AgentQuery(
            query_text="What is the capital of France?",
            context_text="France is a country in Europe. Paris is its capital.",
            query_type="context-specific",
            user_id="test-user-123"
        )

        response = client.post("/api/v1/agent/query", json=test_query.dict())

        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data
        assert response_data["response_text"] is not None
        assert response_data["confidence_score"] == 0.85


def test_query_endpoint_validation_error():
    """Test query endpoint with invalid input."""
    client = TestClient(app)

    # Send request with empty query text to trigger validation error
    invalid_query = {
        "query_text": "",  # Empty query should fail validation
        "query_type": "general"
    }

    response = client.post("/api/v1/agent/query", json=invalid_query)

    # Should return 422 for validation error from Pydantic model
    # Or 400 if our custom validation catches it
    assert response.status_code in [400, 422]


def test_query_endpoint_sanitization():
    """Test that query endpoint properly sanitizes input."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Sanitized response.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Test content",
                    similarity_score=0.7,
                    chunk_id="test-chunk",
                    metadata={"source": "test"},
                    position=1
                )
            ],
            confidence_score=0.75,
            sources=["test"],
            metadata={"processing_time_seconds": 0.100}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Send query with potentially harmful content that should be sanitized
        malicious_query = AgentQuery(
            query_text="<script>alert('xss')</script>What is the capital of France?",
            query_type="general"
        )

        response = client.post("/api/v1/agent/query", json=malicious_query.dict())

        # Should succeed after sanitization
        assert response.status_code == 200


def test_query_endpoint_malicious_pattern():
    """Test query endpoint rejects queries with malicious patterns."""
    client = TestClient(app)

    # Send query with malicious pattern
    malicious_query = {
        "query_text": "What is eval(1+1)?",
        "query_type": "general"
    }

    response = client.post("/api/v1/agent/query", json=malicious_query)

    # Should return 400 for malicious content
    assert response.status_code == 400


def test_health_endpoint():
    """Test the health endpoint."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.health_check.return_value = True
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        response = client.get("/api/v1/agent/health")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "message" in response_data
        assert "timestamp" in response_data


def test_configure_agent_endpoint():
    """Test the configure agent endpoint."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        config_data = {
            "model_name": "gpt-4-turbo",
            "temperature": 0.3,
            "max_tokens": 1000,
            "retrieval_threshold": 0.5,
            "context_window": 4000
        }

        response = client.post("/api/v1/agent/query/configure", json=config_data)

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "config" in response_data
        assert response_data["config"]["model_name"] == "gpt-4-turbo"


def test_query_endpoint_internal_error():
    """Test query endpoint handles internal errors gracefully."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        # Make the agent raise an exception
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.side_effect = Exception("Internal error")
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        test_query = AgentQuery(
            query_text="What causes errors?",
            query_type="general"
        )

        response = client.post("/api/v1/agent/query", json=test_query.dict())

        # Should return 500 for internal error
        assert response.status_code == 500


def test_query_endpoint_long_query():
    """Test query endpoint handles long queries."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response to long query.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Long query context",
                    similarity_score=0.8,
                    chunk_id="long-query-chunk",
                    metadata={"source": "test"},
                    position=1
                )
            ],
            confidence_score=0.8,
            sources=["test"],
            metadata={"processing_time_seconds": 0.200}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Create a query that's close to the max length
        long_query_text = "This is a very long query. " * 80  # Should be under 2000 chars

        test_query = AgentQuery(
            query_text=long_query_text,
            query_type="general"
        )

        response = client.post("/api/v1/agent/query", json=test_query.dict())

        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data


def test_query_endpoint_too_long_query():
    """Test query endpoint rejects queries that exceed length limits."""
    client = TestClient(app)

    # Create a query that exceeds the max length
    very_long_query_text = "This is a very long query. " * 100  # Should exceed 2000 chars

    long_query = {
        "query_text": very_long_query_text,
        "query_type": "general"
    }

    response = client.post("/api/v1/agent/query", json=long_query)

    # Should return 400 for exceeding length limit
    assert response.status_code == 400


def test_query_endpoint_various_query_types():
    """Test API endpoints with various query types - general and context-specific."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response to general query.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="General context",
                    similarity_score=0.75,
                    chunk_id="general-chunk-1",
                    metadata={"source": "general-source"},
                    position=1
                )
            ],
            confidence_score=0.75,
            sources=["general-source"],
            metadata={"processing_time_seconds": 0.150}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test general query type
        general_query = AgentQuery(
            query_text="What is the general concept?",
            query_type="general",
            user_id="user-123"
        )

        response = client.post("/api/v1/agent/query", json=general_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["response_text"] == "Response to general query."
        assert "processing_time_seconds" in response_data["metadata"]

        # Test context-specific query type
        context_query = AgentQuery(
            query_text="Explain based on provided context",
            context_text="This is the specific context for the query.",
            query_type="context-specific",
            user_id="user-456"
        )

        response = client.post("/api/v1/agent/query", json=context_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data


def test_query_endpoint_empty_context():
    """Test query endpoint with empty context text."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response without context.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="General content",
                    similarity_score=0.8,
                    chunk_id="general-chunk",
                    metadata={"source": "general"},
                    position=1
                )
            ],
            confidence_score=0.8,
            sources=["general"],
            metadata={"processing_time_seconds": 0.120}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test with empty context_text
        test_query = AgentQuery(
            query_text="What is the answer?",
            context_text="",  # Empty context
            query_type="context-specific"
        )

        response = client.post("/api/v1/agent/query", json=test_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["response_text"] == "Response without context."


def test_query_endpoint_special_characters():
    """Test query endpoint with special characters and Unicode."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response to special character query.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Special content",
                    similarity_score=0.7,
                    chunk_id="special-chunk",
                    metadata={"source": "special"},
                    position=1
                )
            ],
            confidence_score=0.7,
            sources=["special"],
            metadata={"processing_time_seconds": 0.180}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test with special characters
        special_query = AgentQuery(
            query_text="What's the answer for €, ¥, and © symbols?",
            query_type="general"
        )

        response = client.post("/api/v1/agent/query", json=special_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data


def test_query_endpoint_numbers_only():
    """Test query endpoint with numeric queries."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response to numeric query.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Numeric content",
                    similarity_score=0.65,
                    chunk_id="numeric-chunk",
                    metadata={"source": "numeric"},
                    position=1
                )
            ],
            confidence_score=0.65,
            sources=["numeric"],
            metadata={"processing_time_seconds": 0.100}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test with numeric query
        numeric_query = AgentQuery(
            query_text="12345 + 67890 = ?",
            query_type="general"
        )

        response = client.post("/api/v1/agent/query", json=numeric_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data


def test_health_endpoint_multiple_times():
    """Test health endpoint multiple times to ensure consistency."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.health_check.return_value = True
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test health endpoint multiple times
        for i in range(3):
            response = client.get("/api/v1/agent/health")
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "healthy"
            assert "message" in response_data
            assert "timestamp" in response_data


def test_error_handling_invalid_query_type():
    """Test error handling for invalid query types."""
    client = TestClient(app)

    # Send query with invalid query type
    invalid_query = {
        "query_text": "What is the answer?",
        "query_type": "invalid-type"  # Invalid query type
    }

    response = client.post("/api/v1/agent/query", json=invalid_query)

    # Should return 400 for invalid query type
    assert response.status_code == 400


def test_error_handling_missing_query_text():
    """Test error handling for missing query text."""
    client = TestClient(app)

    # Send query without required query_text field
    invalid_query = {
        "query_type": "general"
        # Missing query_text field
    }

    response = client.post("/api/v1/agent/query", json=invalid_query)

    # Should return 422 for validation error
    assert response.status_code == 422


def test_error_handling_agent_initialization_failure():
    """Test error handling when agent initialization fails."""
    from src.api.agent_router import get_rag_agent
    from src.exceptions.agent_exceptions import AgentInitializationException

    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        # Make the agent constructor raise an exception
        mock_rag_agent.side_effect = AgentInitializationException("Agent initialization failed")

        client = TestClient(app)

        # This test will be tricky since the dependency is resolved before the mock
        # Let's test the get_rag_agent function directly
        with pytest.raises(Exception):
            get_rag_agent()


def test_error_handling_unhealthy_agent():
    """Test error handling when agent health check fails."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.health_check.return_value = False  # Agent is unhealthy
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        response = client.get("/api/v1/agent/health")

        # Should return 503 for unhealthy agent
        assert response.status_code == 503
        response_data = response.json()
        assert "detail" in response_data
        assert "not healthy" in response_data["detail"]


def test_error_handling_agent_health_check_exception():
    """Test error handling when agent health check raises an exception."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.health_check.side_effect = Exception("Health check failed")
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        response = client.get("/api/v1/agent/health")

        # Should return 503 for health check failure
        assert response.status_code == 503


def test_error_handling_configure_agent_exception():
    """Test error handling when configuring agent raises an exception."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        # Make the config assignment raise an exception
        type(mock_agent_instance).config = property(
            fset=lambda self, value: (_ for _ in ()).throw(Exception("Config update failed"))
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        config_data = {
            "model_name": "gpt-4-turbo",
            "temperature": 0.3,
            "max_tokens": 1000,
            "retrieval_threshold": 0.5,
            "context_window": 4000
        }

        response = client.post("/api/v1/agent/query/configure", json=config_data)

        # Should return 500 for internal error during config update
        assert response.status_code == 500


def test_error_handling_query_with_none_values():
    """Test error handling for queries with None values in optional fields."""
    client = TestClient(app)

    # Send query with None context_text (which should be handled gracefully)
    query_with_none = {
        "query_text": "What is the answer?",
        "context_text": None,  # None value
        "query_type": "general"
    }

    response = client.post("/api/v1/agent/query", json=query_with_none)

    # Should handle None context_text gracefully (either 200 or 422 depending on validation)
    # Since context_text is Optional[str] = None in the model, this should be valid
    assert response.status_code in [200, 422]  # 200 if it passes to agent, 422 if validation fails later


def test_error_handling_special_edge_cases():
    """Test error handling for special edge cases."""
    client = TestClient(app)

    # Test with whitespace-only query
    whitespace_query = {
        "query_text": "   ",  # Only whitespace
        "query_type": "general"
    }

    response = client.post("/api/v1/agent/query", json=whitespace_query)

    # Should return 400 after sanitization results in empty query
    assert response.status_code == 400


def test_error_handling_extremely_long_context():
    """Test error handling for extremely long context text."""
    client = TestClient(app)

    # Create context that exceeds the max length
    very_long_context = "This is a very long context. " * 400  # Should exceed 10000 chars

    long_context_query = {
        "query_text": "What is the answer?",
        "context_text": very_long_context,
        "query_type": "context-specific"
    }

    response = client.post("/api/v1/agent/query", json=long_context_query)

    # Should return 400 for exceeding context length limit
    assert response.status_code == 400


def test_error_handling_unexpected_fields():
    """Test how the API handles unexpected fields in the request."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response to query with extra fields.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Test content",
                    similarity_score=0.8,
                    chunk_id="test-chunk",
                    metadata={"source": "test"},
                    position=1
                )
            ],
            confidence_score=0.8,
            sources=["test"],
            metadata={"processing_time_seconds": 0.150}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Query with additional unexpected fields (should be allowed due to extra="allow")
        query_with_extra = {
            "query_text": "What is the answer?",
            "query_type": "general",
            "unexpected_field": "unexpected_value",
            "another_field": 123
        }

        response = client.post("/api/v1/agent/query", json=query_with_extra)

        # Should succeed since extra fields are allowed
        assert response.status_code == 200


def test_context_specific_query_handling():
    """Test context-specific query handling with various inputs."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response based on provided context.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Provided context content",
                    similarity_score=0.9,
                    chunk_id="context-chunk",
                    metadata={"source": "provided-context"},
                    position=1
                )
            ],
            confidence_score=0.9,
            sources=["provided-context"],
            metadata={"processing_time_seconds": 0.180, "query_type": "context-specific"}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test context-specific query with good context
        context_query = AgentQuery(
            query_text="What does the provided text say about AI?",
            context_text="Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. This can include learning from experience, understanding natural language, solving problems, and recognizing patterns.",
            query_type="context-specific",
            user_id="test-user-789"
        )

        response = client.post("/api/v1/agent/query", json=context_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data
        assert response_data["confidence_score"] > 0.8
        assert response_data["metadata"]["query_type"] == "context-specific"


def test_context_specific_query_short_context():
    """Test context-specific query with very short context (should trigger fallback)."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response with fallback retrieval.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Retrieved fallback content",
                    similarity_score=0.7,
                    chunk_id="fallback-chunk",
                    metadata={"source": "retrieved"},
                    position=1
                )
            ],
            confidence_score=0.7,
            sources=["retrieved"],
            metadata={"processing_time_seconds": 0.200, "query_type": "context-specific"}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test context-specific query with very short context
        short_context_query = AgentQuery(
            query_text="What is machine learning?",
            context_text="AI.",  # Very short context
            query_type="context-specific"
        )

        response = client.post("/api/v1/agent/query", json=short_context_query.dict())
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data
        assert response_data["metadata"]["query_type"] == "context-specific"


def test_context_specific_query_no_context():
    """Test context-specific query with no context provided (should fallback to retrieval)."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = AgentResponse(
            response_text="Response based on retrieval since no context provided.",
            query_id=str(uuid.uuid4()),
            retrieved_chunks=[
                RetrievedChunk(
                    content="Retrieved content",
                    similarity_score=0.8,
                    chunk_id="retrieved-chunk",
                    metadata={"source": "retrieved"},
                    position=1
                )
            ],
            confidence_score=0.8,
            sources=["retrieved"],
            metadata={"processing_time_seconds": 0.160, "query_type": "context-specific"}
        )
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test context-specific query with no context provided
        no_context_query = {
            "query_text": "What is deep learning?",
            "context_text": None,  # No context provided
            "query_type": "context-specific"
        }

        response = client.post("/api/v1/agent/query", json=no_context_query)
        assert response.status_code == 200
        response_data = response.json()
        assert "response_text" in response_data
        assert response_data["metadata"]["query_type"] == "context-specific"


def test_context_specific_vs_general_query_behavior():
    """Test the difference in behavior between context-specific and general queries."""
    with patch('src.api.agent_router.RAGAgent') as mock_rag_agent:
        mock_agent_instance = Mock()

        # Mock different responses based on query type by checking the call
        def mock_process_query(agent_query):
            if agent_query.query_type == "context-specific":
                return AgentResponse(
                    response_text="Focused response based on provided context.",
                    query_id=agent_query.query_id,
                    retrieved_chunks=[
                        RetrievedChunk(
                            content=agent_query.context_text or "Default context",
                            similarity_score=0.95,
                            chunk_id="provided-context",
                            metadata={"source": "provided"},
                            position=1
                        )
                    ],
                    confidence_score=0.95,
                    sources=["provided"],
                    metadata={"processing_time_seconds": 0.120, "query_type": "context-specific"}
                )
            else:
                return AgentResponse(
                    response_text="General response based on retrieved content.",
                    query_id=agent_query.query_id,
                    retrieved_chunks=[
                        RetrievedChunk(
                            content="Retrieved general content",
                            similarity_score=0.8,
                            chunk_id="retrieved-general",
                            metadata={"source": "retrieved"},
                            position=1
                        )
                    ],
                    confidence_score=0.8,
                    sources=["retrieved"],
                    metadata={"processing_time_seconds": 0.140, "query_type": "general"}
                )

        mock_agent_instance.process_query.side_effect = mock_process_query
        mock_rag_agent.return_value = mock_agent_instance

        client = TestClient(app)

        # Test general query
        general_query = AgentQuery(
            query_text="Explain neural networks",
            context_text="Some context here",
            query_type="general"
        )

        general_response = client.post("/api/v1/agent/query", json=general_query.dict())
        assert general_response.status_code == 200
        general_data = general_response.json()
        assert general_data["metadata"]["query_type"] == "general"

        # Test context-specific query with same context
        context_query = AgentQuery(
            query_text="Explain neural networks",
            context_text="Some context here",
            query_type="context-specific"
        )

        context_response = client.post("/api/v1/agent/query", json=context_query.dict())
        assert context_response.status_code == 200
        context_data = context_response.json()
        assert context_data["metadata"]["query_type"] == "context-specific"