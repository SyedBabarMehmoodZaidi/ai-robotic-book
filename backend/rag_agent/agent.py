"""
RAG Agent implementation using OpenAI Agent SDK.
This class orchestrates the retrieval and generation process.
"""
from typing import List, Optional
import time
import re
from .models import QueryRequest, AgentResponse, RetrievedContext
from .clients import openai_client
from .retrieval_tool import retrieve_context_for_agent, validate_retrieval_quality
from .utils import Timer
from .errors import RAGAgentError, RetrievalError, AgentError, HallucinationError


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent that integrates retrieval and generation.
    """
    def __init__(self):
        self.client = openai_client.get_client()
        self.model = openai_client.get_model()

    def process_query(self, query_request: QueryRequest) -> AgentResponse:
        """
        Process a query by retrieving context and generating a response.

        Args:
            query_request (QueryRequest): The query request containing the question and parameters

        Returns:
            AgentResponse: The agent's response with source context and metadata
        """
        timer = Timer()
        timer.start()

        try:
            # Step 1: Retrieve context based on the query
            retrieved_contexts = retrieve_context_for_agent(
                query_request.query_text,
                query_request.selected_text
            )

            # Validate retrieval quality
            if not validate_retrieval_quality(retrieved_contexts):
                raise RetrievalError("Insufficient quality context retrieved for query")

            # Step 1.5: If selected_text is provided, we should prioritize it in the context
            if query_request.selected_text:
                retrieved_contexts = self._prioritize_selected_text_context(
                    retrieved_contexts, query_request.selected_text
                )

            # Step 2: Format the context for the agent
            context_str = self._format_context_for_agent(retrieved_contexts)

            # Step 3: Create messages for the OpenAI API
            messages = self._create_messages_for_agent(
                query_request.query_text,
                context_str,
                query_request.selected_text
            )

            # Step 4: Call the OpenAI API to generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent, fact-based responses
                max_tokens=1000  # Limit response length
            )

            # Step 5: Extract response and calculate metrics
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Perform source verification to ensure response is grounded in context
            source_verification_result = self._verify_sources_in_response(
                response_text, retrieved_contexts
            )

            # Check for potential hallucinations using the dedicated detector
            from .utils.hallucination_detector import hallucination_detector
            is_hallucination_detected, hallucination_details = hallucination_detector.detect_hallucinations(
                response_text, retrieved_contexts
            )

            # Calculate confidence score based on context relevance, source verification, and hallucination detection
            confidence_score = self._calculate_enhanced_confidence_score(
                retrieved_contexts, source_verification_result, is_hallucination_detected
            )

            # Stop timer and calculate processing time
            timer.stop()
            processing_time = timer.elapsed()

            # Create detailed source references
            detailed_references = []
            for ctx in retrieved_contexts:
                if ctx.source:  # Only include contexts that have a source
                    preview = ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
                    reference = {
                        'source': ctx.source,
                        'content_preview': preview,
                        'relevance_score': ctx.relevance_score,
                        'chunk_id': ctx.chunk_id
                    }
                    from .models import SourceReference
                    detailed_references.append(SourceReference(**reference))

            # Create and return the agent response
            agent_response = AgentResponse(
                response_text=response_text,
                source_context=[ctx.source for ctx in retrieved_contexts if ctx.source],
                detailed_source_references=detailed_references,
                confidence_score=confidence_score,
                tokens_used=tokens_used,
                processing_time=processing_time,
                query_id=query_request.query_text[:50] + "..." if len(query_request.query_text) > 50 else query_request.query_text,  # Using a portion of query as ID
                is_hallucination_detected=is_hallucination_detected
            )

            # If hallucination detected, raise an error
            if is_hallucination_detected:
                raise HallucinationError(
                    "Potential hallucination detected in agent response",
                    details=str(hallucination_details)
                )

            return agent_response

        except Exception as e:
            timer.stop()
            processing_time = timer.elapsed()

            # Re-raise known errors with additional context
            if isinstance(e, RAGAgentError):
                raise
            else:
                raise AgentError(f"Error processing query: {str(e)}")

    def _prioritize_selected_text_context(self, retrieved_contexts: List[RetrievedContext], selected_text: str) -> List[RetrievedContext]:
        """
        Prioritize the selected text in the retrieved contexts by potentially adding it directly
        or boosting its relevance score.

        Args:
            retrieved_contexts (List[RetrievedContext]): List of retrieved contexts
            selected_text (str): The selected text to prioritize

        Returns:
            List[RetrievedContext]: Updated list of contexts with selected text prioritized
        """
        # If the selected text is already in the retrieved contexts, we could boost its score
        # For now, we'll add the selected text as a high-priority context if it's not already well represented

        # Check if any of the retrieved contexts contain the selected text or are highly relevant to it
        contains_selected_text = any(
            selected_text.lower() in ctx.content.lower() for ctx in retrieved_contexts
        )

        if not contains_selected_text:
            # Create a special context entry for the selected text with high relevance
            # This is a placeholder - in practice, we'd have a specific source for the selected text
            selected_context = RetrievedContext(
                content=selected_text,
                source="user_selected_text",
                relevance_score=1.0,  # Highest possible relevance since user specifically selected it
                chunk_id="selected_text_chunk",
                metadata={"type": "selected_text", "original_query": "user_selection"},
                similarity_score=1.0
            )

            # Add the selected text context to the beginning of the list to prioritize it
            retrieved_contexts = [selected_context] + retrieved_contexts

        return retrieved_contexts

    def _format_context_for_agent(self, retrieved_contexts: List[RetrievedContext]) -> str:
        """
        Format the retrieved contexts into a string suitable for the agent.

        Args:
            retrieved_contexts (List[RetrievedContext]): List of retrieved contexts

        Returns:
            str: Formatted context string
        """
        if not retrieved_contexts:
            return "No relevant context found."

        formatted_contexts = []
        for i, ctx in enumerate(retrieved_contexts, 1):
            formatted_context = f"Context {i} (Source: {ctx.source}, Relevance: {ctx.relevance_score:.2f}):\n{ctx.content}\n"
            formatted_contexts.append(formatted_context)

        return "\n".join(formatted_contexts)

    def _create_messages_for_agent(self, query_text: str, context_str: str, selected_text: Optional[str] = None) -> List[dict]:
        """
        Create the message structure for the OpenAI API.

        Args:
            query_text (str): The original query
            context_str (str): Formatted context string
            selected_text (Optional[str]): Selected text to focus on (if provided)

        Returns:
            List[dict]: List of messages for the OpenAI API
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based only on the provided book context. "
                "Do not make up information. If the context doesn't contain enough information to answer the question, "
                "say so clearly. Always cite your sources from the provided context."
            )
        }

        if selected_text:
            # If selected text is provided, emphasize it in the prompt
            user_message = {
                "role": "user",
                "content": (
                    f"Context: {context_str}\n\n"
                    f"Specific text to focus on: {selected_text}\n\n"
                    f"Question: {query_text}\n\n"
                    f"Please provide a detailed answer based on the context provided, "
                    f"with special attention to the specific text segment provided. Cite your sources."
                )
            }
        else:
            # Standard query without selected text
            user_message = {
                "role": "user",
                "content": f"Context: {context_str}\n\nQuestion: {query_text}\n\nPlease provide a detailed answer based on the context provided, citing your sources."
            }

        return [system_message, user_message]

    def _calculate_enhanced_confidence_score(
        self,
        retrieved_contexts: List[RetrievedContext],
        source_verification_result: Dict,
        is_hallucination_detected: bool
    ) -> float:
        """
        Calculate an enhanced confidence score based on context quality, source verification, and hallucination detection.

        Args:
            retrieved_contexts (List[RetrievedContext]): List of retrieved contexts
            source_verification_result (Dict): Result from source verification
            is_hallucination_detected (bool): Whether hallucination was detected

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not retrieved_contexts:
            return 0.0

        # Start with the base confidence from context relevance
        base_confidence = self._calculate_confidence_score(retrieved_contexts)

        # Adjust based on source verification
        verification_boost = 0.0
        if source_verification_result.get('is_verified', False):
            verification_boost = source_verification_result.get('confidence', 0.0) * 0.3  # Max 30% boost

        # Adjust based on hallucination detection (significant penalty if hallucination detected)
        hallucination_penalty = -0.5 if is_hallucination_detected else 0.0

        # Calculate final confidence with adjustments
        adjusted_confidence = base_confidence + verification_boost + hallucination_penalty

        # Ensure confidence stays within [0, 1] bounds
        final_confidence = max(0.0, min(1.0, adjusted_confidence))

        return final_confidence

    def _calculate_confidence_score(self, retrieved_contexts: List[RetrievedContext]) -> float:
        """
        Calculate a basic confidence score based on the quality of retrieved contexts.

        Args:
            retrieved_contexts (List[RetrievedContext]): List of retrieved contexts

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not retrieved_contexts:
            return 0.0

        # Calculate average relevance score
        avg_relevance = sum(ctx.relevance_score for ctx in retrieved_contexts) / len(retrieved_contexts)

        # Calculate score based on number of contexts and their relevance
        num_contexts = len(retrieved_contexts)
        if num_contexts >= 3:
            # High confidence if we have multiple relevant contexts
            return min(1.0, avg_relevance * 1.2)
        elif num_contexts >= 1:
            # Medium confidence if we have at least one relevant context
            return avg_relevance
        else:
            # Low confidence if no contexts retrieved
            return 0.0

    def _verify_sources_in_response(self, response_text: str, retrieved_contexts: List[RetrievedContext]) -> Dict:
        """
        Verify that the response properly cites or is grounded in the provided sources.

        Args:
            response_text (str): The agent's response
            retrieved_contexts (List[RetrievedContext]): The contexts provided to the agent

        Returns:
            Dict: Verification results including confidence and details
        """
        verification_result = {
            'is_verified': False,
            'confidence': 0.0,
            'details': {
                'citations_found': [],
                'contexts_referenced': [],
                'content_overlap': 0.0
            }
        }

        if not retrieved_contexts:
            return verification_result

        # Check if response text contains references to context sources
        response_lower = response_text.lower()
        contexts_lower = [ctx.content.lower() for ctx in retrieved_contexts]

        # Look for overlap between response and contexts
        total_overlap = 0
        contexts_referenced = 0

        for i, ctx in enumerate(contexts_lower):
            # Calculate content overlap using simple word matching
            response_words = set(response_lower.split())
            context_words = set(ctx.split())
            overlap = len(response_words & context_words)

            if overlap > 0:
                total_overlap += overlap
                contexts_referenced += 1

                # Look for potential citations in the response
                for source in [retrieved_contexts[i].source]:
                    if source.lower() in response_lower:
                        verification_result['details']['citations_found'].append(source)

        # Calculate overall verification confidence
        if contexts_referenced > 0:
            verification_result['is_verified'] = True
            verification_result['confidence'] = min(1.0, total_overlap / max(len(response_lower.split()), 1))
            verification_result['details']['contexts_referenced'] = contexts_referenced
            verification_result['details']['content_overlap'] = verification_result['confidence']

        return verification_result

    def _detect_hallucinations(self, response_text: str, context_str: str) -> bool:
        """
        Simple hallucination detection by checking if response content is supported by context.

        Args:
            response_text (str): The agent's response
            context_str (str): The context provided to the agent

        Returns:
            bool: True if hallucination is detected, False otherwise
        """
        # This is a simplified hallucination detection
        # In a real implementation, this would be more sophisticated
        # For now, we'll check if the response contains key terms from the context
        response_lower = response_text.lower()
        context_lower = context_str.lower()

        # If context is empty but response is not, it's likely a hallucination
        if not context_str.strip() and response_text.strip():
            return True

        # This is a very basic check - a more sophisticated implementation would use
        # semantic similarity, fact-checking, etc.
        # For now, we'll return False as this is a complex problem that requires more advanced techniques
        return False