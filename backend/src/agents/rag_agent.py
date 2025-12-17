import openai
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from ..models.agent_query import AgentQuery
from ..models.agent_response import AgentResponse, RetrievedChunk
from ..models.agent_configuration import AgentConfiguration
from ..config.agent_settings import agent_settings
from ..services.retrieval_integration import RetrievalIntegration
from ..exceptions.agent_exceptions import (
    AgentInitializationException,
    QueryProcessingException,
    ResponseGenerationException,
    OpenAIServiceException,
    RetrievalServiceException
)


logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent that processes queries using
    book content context and generates grounded responses using OpenAI Assistant API.
    """

    def __init__(self, config: Optional[AgentConfiguration] = None):
        """
        Initialize the RAG Agent with configuration using OpenAI Assistant API.

        Args:
            config: Agent configuration settings (uses defaults if not provided)
        """
        self.config = config or AgentConfiguration()

        # Initialize OpenAI client with API key
        try:
            self.client = openai.OpenAI(api_key=agent_settings.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise AgentInitializationException(
                "Failed to initialize OpenAI client",
                details={"error": str(e)}
            )

        # Initialize retrieval integration service
        try:
            self.retrieval_service = RetrievalIntegration(
                qdrant_url=agent_settings.qdrant_url,
                collection_name=agent_settings.qdrant_collection_name,
                top_k=agent_settings.default_top_k
            )
            logger.info("RAG Agent initialized successfully with retrieval integration")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval service: {str(e)}")
            raise AgentInitializationException(
                "Failed to initialize retrieval service",
                details={"error": str(e)}
            )

        # Create or retrieve the assistant
        try:
            self.assistant = self.client.beta.assistants.create(
                name="RAG Book Assistant",
                description="An AI assistant that answers questions based on book content with retrieval-augmented generation",
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                instructions=(
                    "You are a helpful AI assistant that answers questions based on provided book content. "
                    "Only use the information provided in the context to answer questions. "
                    "Do not make up information or use general knowledge. "
                    "If the provided context doesn't contain information to answer the question, "
                    "say 'I don't have enough information from the provided context to answer this question.'"
                )
            )
            logger.info(f"RAG Agent initialized successfully with assistant ID: {self.assistant.id}")
        except Exception as e:
            logger.error(f"Failed to create assistant: {str(e)}")
            raise AgentInitializationException(
                "Failed to create OpenAI assistant",
                details={"error": str(e)}
            )

    def process_query(self, agent_query: AgentQuery) -> AgentResponse:
        """
        Process an agent query and return a response using OpenAI Assistant API.

        Args:
            agent_query: The query to process

        Returns:
            AgentResponse containing the generated response
        """
        try:
            logger.info(f"Processing query: '{agent_query.query_text[:50]}...', type: {agent_query.query_type}")

            # For context-specific queries, validate and process the provided context
            if agent_query.query_type == "context-specific" and agent_query.context_text:
                # Validate the context text
                if not self._validate_context_text(agent_query.context_text):
                    logger.warning(f"Context text failed validation for query {agent_query.query_id}")
                    # We can still proceed but with a warning
                    pass

                # Process the context text
                processed_context = self._process_context_text(agent_query.context_text)
                # Update the agent_query with processed context
                agent_query.context_text = processed_context

            # Handle context-specific queries differently from general queries
            if agent_query.query_type == "context-specific" and agent_query.context_text:
                # For context-specific queries, prioritize the provided context
                filtered_chunks = self._handle_context_specific_query(agent_query)
            else:
                # For general queries, retrieve content before agent generation (RAG pattern)
                filtered_chunks = self._retrieve_content_before_generation(agent_query)

            logger.info(f"Retrieved {len(filtered_chunks)} relevant chunks after filtering with threshold {agent_settings.agent_retrieval_threshold}")

            # Prepare the messages with proper context formatting for book content
            messages = []

            # Add system message with instructions about using book content and hallucination prevention
            system_instructions = {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that answers questions based on provided book content. "
                    "Only use the information provided in the context to answer questions. "
                    "Do not make up information or use general knowledge. "
                    "If the provided context doesn't contain information to answer the question, "
                    "say 'I don't have enough information from the provided context to answer this question.'\n\n"
                    "IMPORTANT: Always verify that your response is grounded in the provided context. "
                    "If you cannot find the answer in the context, explicitly state that you don't have enough information. "
                    "Do not fabricate, infer, or guess information that is not directly stated in the context. "
                    "Avoid making assumptions or using common knowledge to fill gaps in the provided context."
                )
            }
            messages.append(system_instructions)

            # Add context based on query type and available content
            if agent_query.query_type == "context-specific" and agent_query.context_text:
                # For context-specific queries, prioritize the provided context
                context_message = {
                    "role": "user",
                    "content": f"Book Context:\n{agent_query.context_text}"
                }
                messages.append(context_message)

                # Add a message to indicate the end of context
                context_separator = {
                    "role": "assistant",
                    "content": "I have received the specific book context. I will focus on this information to answer the question."
                }
                messages.append(context_separator)

                # Also include retrieved chunks if any (for additional context)
                if filtered_chunks:
                    additional_context = "\n\n".join([chunk.content for chunk in filtered_chunks])
                    additional_context_message = {
                        "role": "user",
                        "content": f"Additional Context:\n{additional_context}"
                    }
                    messages.append(additional_context_message)
            else:
                # For general queries, use retrieved content as context
                if filtered_chunks:
                    context_text = "\n\n".join([chunk.content for chunk in filtered_chunks])
                    context_message = {
                        "role": "user",
                        "content": f"Book Context:\n{context_text}"
                    }
                    messages.append(context_message)

                    # Add a message to indicate the end of context
                    context_separator = {
                        "role": "assistant",
                        "content": f"I have received {len(filtered_chunks)} book context chunks. I will use this information to answer questions."
                    }
                    messages.append(context_separator)

            # Add the actual user query
            user_query_message = {
                "role": "user",
                "content": f"Question: {agent_query.query_text}"
            }
            messages.append(user_query_message)

            # Create a thread and run the assistant
            thread = self.client.beta.threads.create(
                messages=messages
            )

            # Run the assistant on the thread
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Wait for the run to complete
            import time
            while run.status in ['queued', 'in_progress', 'requires_action']:
                time.sleep(0.5)  # Wait 0.5 seconds before checking again
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            # Get the messages from the thread
            messages_response = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="asc"
            )

            # Extract the assistant's response
            response_text = ""
            for msg in messages_response.data:
                if msg.role == "assistant":
                    for content_block in msg.content:
                        if content_block.type == "text":
                            response_text = content_block.text.value
                            break
                    break

            # Apply additional hallucination prevention checks
            response_text = self._apply_hallucination_prevention(response_text, agent_query)

            # Calculate confidence score based on response characteristics
            confidence_score = self._calculate_context_specific_confidence_score(response_text, agent_query, len(filtered_chunks))

            # Create and return the agent response with retrieved chunks
            agent_response = AgentResponse(
                response_text=response_text,
                query_id=agent_query.query_id,
                retrieved_chunks=filtered_chunks,  # Now populated with actual retrieved chunks
                confidence_score=confidence_score,
                sources=[chunk.metadata.get("source", "unknown") for chunk in filtered_chunks],
                metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "retrieval_threshold": agent_settings.agent_retrieval_threshold,
                    "retrieved_chunk_count": len(filtered_chunks),
                    "query_type": agent_query.query_type
                }
            )

            logger.info(f"Query processed successfully, response length: {len(response_text)}, confidence: {confidence_score}")
            return agent_response

        except RetrievalServiceException as e:
            logger.error(f"Retrieval service error: {str(e)}")
            raise QueryProcessingException(
                "Error occurred during content retrieval",
                details={"error": str(e), "query_id": agent_query.query_id}
            )
        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed")
            raise OpenAIServiceException(
                "OpenAI authentication failed - check your API key",
                details={"error_type": "AuthenticationError"}
            )
        except openai.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            raise OpenAIServiceException(
                "OpenAI rate limit exceeded",
                details={"error_type": "RateLimitError"}
            )
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise OpenAIServiceException(
                f"OpenAI API error: {str(e)}",
                details={"error_type": "APIError", "error_message": str(e)}
            )
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise QueryProcessingException(
                "Error occurred during query processing",
                details={"error": str(e), "query_id": agent_query.query_id}
            )

    def _retrieve_content_before_generation(self, agent_query: AgentQuery):
        """
        Retrieve content before agent generation following the RAG pattern.

        Args:
            agent_query: The query to retrieve content for

        Returns:
            Filtered list of retrieved chunks based on similarity threshold
        """
        # Retrieve relevant content using the retrieval integration service
        retrieved_chunks = self.retrieval_service.retrieve_content(agent_query)

        # Process retrieval results with similarity threshold
        threshold = agent_settings.agent_retrieval_threshold
        filtered_chunks = self.retrieval_service.process_retrieval_results(
            retrieved_chunks,
            threshold=threshold
        )

        return filtered_chunks

    def _calculate_confidence_score(self, response_text: str, agent_query: AgentQuery) -> float:
        """
        Calculate a confidence score for the response based on various factors.

        Args:
            response_text: The response text from the assistant
            agent_query: The original query for context

        Returns:
            A confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Check if response indicates insufficient context (low confidence)
        insufficient_context_indicators = [
            "don't have enough information",
            "not enough information",
            "insufficient information",
            "no information",
            "not mentioned",
            "not stated",
            "not provided",
            "not specified",
            "not found"
        ]

        response_lower = response_text.lower()
        for indicator in insufficient_context_indicators:
            if indicator in response_lower:
                # If the model explicitly states insufficient information, return low confidence
                return 0.1

        # Base score calculation based on context availability
        if agent_query.context_text and len(agent_query.context_text.strip()) > 0:
            # If context was provided, start with higher base score
            score += 0.6
        else:
            # If no context was provided, start with lower base score
            score += 0.2

        # Adjust score based on response length and coherence
        if len(response_text.strip()) > 0:
            score += 0.2  # Base score for having a response

            # Adjust for response quality indicators
            # Longer, more detailed responses might indicate higher confidence
            if len(response_text) > 50:
                score += 0.1
            else:
                # Shorter responses might be less confident
                score -= 0.05

        # Cap the score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        return score

    def _apply_hallucination_prevention(self, response_text: str, agent_query: AgentQuery) -> str:
        """
        Apply additional hallucination prevention checks to the response.

        Args:
            response_text: The response text from the assistant
            agent_query: The original query for context

        Returns:
            The response text, potentially modified to prevent hallucinations
        """
        # Check if the response indicates insufficient context
        insufficient_context_indicators = [
            "don't have enough information",
            "not enough information",
            "insufficient information",
            "no information",
            "not mentioned",
            "not stated",
            "not provided",
            "not specified"
        ]

        # If the response already indicates insufficient context, return as is
        response_lower = response_text.lower()
        for indicator in insufficient_context_indicators:
            if indicator in response_lower:
                return response_text

        # If no context was provided but the model gave a confident answer,
        # it might be hallucinating
        if not agent_query.context_text and len(response_text.strip()) > 20:
            # Check if the response seems to be making claims without context
            # For now, we'll log this for monitoring
            logger.warning(f"Response generated without context: '{response_text[:100]}...'")
            # In a full implementation, we might want to modify the response here
            # For now, we'll return the response as is but with a warning

        return response_text

    def health_check(self) -> bool:
        """
        Check if the agent is healthy by testing OpenAI Assistant connectivity.

        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            # Test OpenAI connectivity by creating a simple thread
            thread = self.client.beta.threads.create(
                messages=[{"role": "user", "content": "Hello"}]
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
                temperature=0.1,
                max_tokens=10
            )

            # Wait briefly for the run to complete
            import time
            start_time = time.time()
            while run.status in ['queued', 'in_progress'] and time.time() - start_time < 10:  # 10 second timeout
                time.sleep(0.5)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def _handle_context_specific_query(self, agent_query: AgentQuery):
        """
        Handle context-specific queries by optionally retrieving additional context
        while prioritizing the provided context. Implements fallback to retrieval
        when the provided context is insufficient.

        Args:
            agent_query: The context-specific query to handle

        Returns:
            List of retrieved chunks (may be empty if retrieval is not needed)
        """
        logger.info(f"Handling context-specific query for: '{agent_query.query_text[:50]}...'")

        # For context-specific queries, we may still want to retrieve additional context
        # to supplement the provided context, but we'll do this selectively
        if agent_query.context_text and len(agent_query.context_text.strip()) > 0:
            # Check if the provided context is sufficient for the query
            # This is a basic check - in a more sophisticated implementation,
            # we might use semantic analysis to determine if the context is relevant
            context_length = len(agent_query.context_text.strip())

            # If context is very short, we should likely retrieve additional context
            if context_length < 50:
                logger.info(f"Context is very short ({context_length} chars), retrieving additional context")
                try:
                    retrieved_chunks = self._retrieve_content_before_generation(agent_query)
                    logger.info(f"Retrieved {len(retrieved_chunks)} chunks as fallback for short context")
                    return retrieved_chunks
                except Exception as e:
                    logger.warning(f"Failed to retrieve additional context as fallback: {str(e)}")
                    return []
            else:
                # Optionally retrieve additional context based on the query text
                # This allows for supplementary information while focusing on the provided context
                try:
                    # Retrieve content related to the query to supplement the provided context
                    additional_chunks = self._retrieve_content_before_generation(agent_query)
                    logger.info(f"Retrieved {len(additional_chunks)} additional chunks for context-specific query")
                    return additional_chunks
                except Exception as e:
                    logger.warning(f"Failed to retrieve additional context for context-specific query: {str(e)}")
                    # Return empty list if retrieval fails, but the query can still proceed with provided context
                    return []
        else:
            # If no context was provided, try to retrieve content based on the query
            logger.info("No context provided, falling back to retrieval")
            try:
                retrieved_chunks = self._retrieve_content_before_generation(agent_query)
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks for context-specific query (no provided context)")
                return retrieved_chunks
            except Exception as e:
                logger.warning(f"Failed to retrieve content for context-specific query: {str(e)}")
                return []

    def _validate_context_text(self, context_text: str) -> bool:
        """
        Validate the context text for context-specific queries.

        Args:
            context_text: The context text to validate

        Returns:
            True if context is valid, False otherwise
        """
        if not context_text:
            return False

        # Check if context text is too short
        if len(context_text.strip()) < 10:
            logger.warning("Context text is too short (< 10 characters)")
            return False

        # Check if context text is too long (more than 10,000 characters as per model limits)
        if len(context_text) > 10000:
            logger.warning("Context text exceeds maximum length (10,000 characters)")
            return False

        # Additional validation could be added here
        # For example, checking for potentially malicious content, etc.
        return True

    def _process_context_text(self, context_text: str) -> str:
        """
        Process and clean the context text for context-specific queries.

        Args:
            context_text: The raw context text

        Returns:
            Processed and cleaned context text
        """
        if not context_text:
            return context_text

        # Remove extra whitespace and normalize
        processed_context = ' '.join(context_text.split())

        # Additional processing could be added here
        # For example, removing special characters, normalizing formatting, etc.
        return processed_context

    def _calculate_context_specific_confidence_score(self, response_text: str, agent_query: AgentQuery, retrieved_chunk_count: int) -> float:
        """
        Calculate a confidence score for context-specific queries based on various factors.

        Args:
            response_text: The response text from the assistant
            agent_query: The original query for context
            retrieved_chunk_count: Number of retrieved chunks used

        Returns:
            A confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Check if response indicates insufficient context (low confidence)
        insufficient_context_indicators = [
            "don't have enough information",
            "not enough information",
            "insufficient information",
            "no information",
            "not mentioned",
            "not stated",
            "not provided",
            "not specified",
            "not found"
        ]

        response_lower = response_text.lower()
        for indicator in insufficient_context_indicators:
            if indicator in response_lower:
                # If the model explicitly states insufficient information, return low confidence
                return 0.1

        # Base score calculation based on query type and context availability
        if agent_query.query_type == "context-specific":
            # For context-specific queries, prioritize the presence of provided context
            if agent_query.context_text and len(agent_query.context_text.strip()) > 0:
                # If context was provided, start with higher base score
                score += 0.7
            else:
                # If no context was provided for a context-specific query, start with lower score
                score += 0.3
        else:
            # For general queries, use standard scoring
            if retrieved_chunk_count > 0:
                score += 0.5
            else:
                score += 0.2

        # Adjust score based on response length and coherence
        if len(response_text.strip()) > 0:
            score += 0.2  # Base score for having a response

            # Adjust for response quality indicators
            # Longer, more detailed responses might indicate higher confidence
            if len(response_text) > 50:
                score += 0.1
            else:
                # Shorter responses might be less confident
                score -= 0.05

        # If we retrieved additional chunks for a context-specific query, that may improve confidence
        if agent_query.query_type == "context-specific" and retrieved_chunk_count > 0:
            score += 0.1  # Bonus for having additional supporting context

        # Cap the score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        return score