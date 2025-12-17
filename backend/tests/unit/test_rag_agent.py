import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.agent_query import AgentQuery
from src.models.agent_response import AgentResponse
from src.models.agent_configuration import AgentConfiguration
from src.agents.rag_agent import RAGAgent
from src.exceptions.agent_exceptions import (
    AgentInitializationException,
    QueryProcessingException,
    OpenAIServiceException
)


class TestRAGAgent:
    """Unit tests for RAGAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the OpenAI API key
        os.environ["OPENAI_API_KEY"] = "test-key"

        # Create a basic configuration for testing
        self.config = AgentConfiguration(
            model_name="gpt-4-turbo",
            temperature=0.1,
            max_tokens=1000
        )

        # Create a sample agent query for testing
        self.sample_query = AgentQuery(
            query_text="What is the capital of France?",
            context_text="France is a country in Europe. Paris is the capital and largest city of France.",
            query_id="test-query-123",
            created_at=datetime.utcnow(),
            query_type="general"
        )

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_initialization_success(self, mock_openai_client):
        """Test successful initialization of RAGAgent."""
        # Mock the assistant creation
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Assertions
        assert agent.config == self.config
        assert agent.assistant.id == "test-assistant-id"
        mock_openai_client.assert_called_once()
        mock_client.beta.assistants.create.assert_called_once()

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_initialization_missing_api_key(self, mock_openai_client):
        """Test initialization failure when OpenAI API key is missing."""
        # Remove the API key from environment
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(AgentInitializationException) as exc_info:
                RAGAgent(self.config)

            assert "OPENAI_API_KEY environment variable is required" in str(exc_info.value)
        finally:
            # Restore the original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_process_query_success(self, mock_openai_client):
        """Test successful query processing."""
        # Mock the assistant and thread operations
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_thread = Mock()
        mock_thread.id = "test-thread-id"

        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.status = "completed"

        mock_message = Mock()
        mock_message.role = "assistant"
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text.value = "The capital of France is Paris."
        mock_message.content = [mock_content]

        mock_messages_response = Mock()
        mock_messages_response.data = [mock_message]

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant
        mock_client.beta.threads.create.return_value = mock_thread
        mock_client.beta.threads.runs.create.return_value = mock_run
        mock_client.beta.threads.runs.retrieve.return_value = mock_run
        mock_client.beta.threads.messages.list.return_value = mock_messages_response

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Process the query
        response = agent.process_query(self.sample_query)

        # Assertions
        assert isinstance(response, AgentResponse)
        assert response.response_text == "The capital of France is Paris."
        assert response.query_id == self.sample_query.query_id
        assert response.confidence_score > 0.5  # Should be high for a direct answer

        # Verify the correct methods were called
        mock_client.beta.threads.create.assert_called_once()
        mock_client.beta.threads.runs.create.assert_called_once()
        mock_client.beta.threads.messages.list.assert_called_once()

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_process_query_insufficient_context(self, mock_openai_client):
        """Test query processing when response indicates insufficient context."""
        # Mock the assistant and thread operations
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_thread = Mock()
        mock_thread.id = "test-thread-id"

        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.status = "completed"

        mock_message = Mock()
        mock_message.role = "assistant"
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text.value = "I don't have enough information from the provided context to answer this question."
        mock_message.content = [mock_content]

        mock_messages_response = Mock()
        mock_messages_response.data = [mock_message]

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant
        mock_client.beta.threads.create.return_value = mock_thread
        mock_client.beta.threads.runs.create.return_value = mock_run
        mock_client.beta.threads.runs.retrieve.return_value = mock_run
        mock_client.beta.threads.messages.list.return_value = mock_messages_response

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Process the query
        response = agent.process_query(self.sample_query)

        # Assertions
        assert isinstance(response, AgentResponse)
        assert "don't have enough information" in response.response_text.lower()
        assert response.confidence_score == 0.1  # Should be low for insufficient context

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_process_query_with_no_context(self, mock_openai_client):
        """Test query processing when no context is provided."""
        # Create a query without context
        query_without_context = AgentQuery(
            query_text="What is the capital of France?",
            query_id="test-query-no-context",
            created_at=datetime.utcnow(),
            query_type="general"
        )

        # Mock the assistant and thread operations
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_thread = Mock()
        mock_thread.id = "test-thread-id"

        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.status = "completed"

        mock_message = Mock()
        mock_message.role = "assistant"
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text.value = "Based on general knowledge, the capital of France is Paris."
        mock_message.content = [mock_content]

        mock_messages_response = Mock()
        mock_messages_response.data = [mock_message]

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant
        mock_client.beta.threads.create.return_value = mock_thread
        mock_client.beta.threads.runs.create.return_value = mock_run
        mock_client.beta.threads.runs.retrieve.return_value = mock_run
        mock_client.beta.threads.messages.list.return_value = mock_messages_response

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Process the query
        response = agent.process_query(query_without_context)

        # Assertions
        assert isinstance(response, AgentResponse)
        assert "Paris" in response.response_text

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_health_check_success(self, mock_openai_client):
        """Test health check returns True for successful connection."""
        # Mock the assistant
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_thread = Mock()
        mock_thread.id = "test-thread-id"

        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.status = "completed"

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant
        mock_client.beta.threads.create.return_value = mock_thread
        mock_client.beta.threads.runs.create.return_value = mock_run
        mock_client.beta.threads.runs.retrieve.return_value = mock_run

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Perform health check
        result = agent.health_check()

        # Assertion
        assert result is True

    @patch('src.agents.rag_agent.openai.OpenAI')
    def test_health_check_failure(self, mock_openai_client):
        """Test health check returns False for failed connection."""
        # Mock the assistant
        mock_assistant = Mock()
        mock_assistant.id = "test-assistant-id"

        mock_client = Mock()
        mock_client.beta.assistants.create.return_value = mock_assistant
        mock_client.beta.threads.create.side_effect = Exception("Connection failed")

        mock_openai_client.return_value = mock_client

        # Initialize the agent
        agent = RAGAgent(self.config)

        # Perform health check
        result = agent.health_check()

        # Assertion
        assert result is False

    def test_calculate_confidence_score_insufficient_info(self):
        """Test confidence score calculation for insufficient information responses."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test response indicating insufficient information
            response_text = "I don't have enough information from the provided context to answer this question."
            confidence_score = agent._calculate_confidence_score(response_text, self.sample_query)

            assert confidence_score == 0.1

    def test_calculate_confidence_score_with_context(self):
        """Test confidence score calculation when context is provided."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test response with context provided
            response_text = "The capital of France is Paris, as mentioned in the provided context."
            confidence_score = agent._calculate_confidence_score(response_text, self.sample_query)

            # Should be higher than 0.6 due to context being provided
            assert confidence_score > 0.6

    def test_calculate_confidence_score_without_context(self):
        """Test confidence score calculation when no context is provided."""
        # Create a query without context
        query_without_context = AgentQuery(
            query_text="What is the capital of France?",
            query_id="test-query-no-context",
            created_at=datetime.utcnow(),
            query_type="general"
        )

        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test response without context
            response_text = "The capital of France is Paris."
            confidence_score = agent._calculate_confidence_score(response_text, query_without_context)

            # Should be lower since no context was provided
            assert confidence_score <= 0.5  # Less than when context is provided

    def test_context_validation_short_text(self):
        """Test context validation for short text."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test short context text
            short_context = "Hi"
            is_valid = agent._validate_context_text(short_context)

            assert is_valid is False

    def test_context_validation_long_text(self):
        """Test context validation for long text."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test long context text
            long_context = "This is a very long context. " * 500  # More than 10,000 chars
            is_valid = agent._validate_context_text(long_context)

            assert is_valid is False

    def test_context_validation_valid_text(self):
        """Test context validation for valid text."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test valid context text
            valid_context = "This is a valid context that is neither too short nor too long."
            is_valid = agent._validate_context_text(valid_context)

            assert is_valid is True

    def test_context_processing(self):
        """Test context text processing and cleaning."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Test context with extra whitespace
            raw_context = "  This   has   extra   spaces.  \n\t And  newlines.  "
            processed_context = agent._process_context_text(raw_context)

            # Should normalize whitespace
            expected = "This has extra spaces. And newlines."
            assert processed_context == expected

    def test_calculate_context_specific_confidence_score_insufficient_info(self):
        """Test context-specific confidence score calculation for insufficient information responses."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Create a context-specific query
            context_query = AgentQuery(
                query_text="What is the capital of France?",
                context_text="This context is about something else entirely.",
                query_id="test-context-query",
                created_at=datetime.utcnow(),
                query_type="context-specific"
            )

            # Test response indicating insufficient information
            response_text = "I don't have enough information from the provided context to answer this question."
            confidence_score = agent._calculate_context_specific_confidence_score(response_text, context_query, 0)

            assert confidence_score == 0.1

    def test_calculate_context_specific_confidence_score_with_context(self):
        """Test context-specific confidence score for context-specific queries."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Create a context-specific query
            context_query = AgentQuery(
                query_text="What is the capital of France?",
                context_text="France is a country in Europe. Paris is the capital and largest city of France.",
                query_id="test-context-query",
                created_at=datetime.utcnow(),
                query_type="context-specific"
            )

            # Test response with good context
            response_text = "Based on the provided context, the capital of France is Paris."
            confidence_score = agent._calculate_context_specific_confidence_score(response_text, context_query, 2)

            # Should be higher due to context being provided and additional chunks retrieved
            assert confidence_score > 0.8

    def test_calculate_context_specific_confidence_score_general_query(self):
        """Test context-specific confidence score for general queries."""
        # Mock the OpenAI client to avoid actual initialization
        with patch('src.agents.rag_agent.openai.OpenAI') as mock_openai_client:
            mock_assistant = Mock()
            mock_assistant.id = "test-assistant-id"

            mock_client = Mock()
            mock_client.beta.assistants.create.return_value = mock_assistant
            mock_openai_client.return_value = mock_client

            agent = RAGAgent(self.config)

            # Create a general query (no context)
            general_query = AgentQuery(
                query_text="What is the capital of France?",
                query_id="test-general-query",
                created_at=datetime.utcnow(),
                query_type="general"
            )

            # Test response for general query
            response_text = "The capital of France is Paris."
            confidence_score = agent._calculate_context_specific_confidence_score(response_text, general_query, 2)

            # Should use general scoring logic
            assert confidence_score >= 0.5  # Reasonable score with retrieved chunks