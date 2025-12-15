"""
Validation service for the RAG agent.
This service handles various validation tasks including selected text validation and response validation.
"""
from typing import Optional, List
from ..models import QueryRequest, AgentResponse, RetrievedContext
from ..errors import ValidationError, HallucinationError


class ValidationService:
    """
    Service class to handle various validation tasks for the RAG agent.
    """

    @staticmethod
    def validate_selected_text(selected_text: Optional[str]) -> bool:
        """
        Validate the selected text based on business rules.

        Args:
            selected_text (Optional[str]): The selected text to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        if selected_text is None:
            return True  # None is valid

        # Check if the selected text is empty or just whitespace after stripping
        stripped_text = selected_text.strip()
        if not stripped_text:
            raise ValidationError(
                "Selected text cannot be empty or contain only whitespace",
                details="selected_text was empty after stripping whitespace"
            )

        # Check minimum length (at least 10 characters as per original validation)
        if len(stripped_text) < 10:
            raise ValidationError(
                f"Selected text is too short: {len(stripped_text)} characters. Minimum is 10.",
                details=f"selected_text length was {len(stripped_text)} characters"
            )

        # Check maximum length (as per model validation)
        if len(selected_text) > 5000:
            raise ValidationError(
                f"Selected text is too long: {len(selected_text)} characters. Maximum is 5000.",
                details=f"selected_text length was {len(selected_text)} characters"
            )

        # Check for any problematic patterns (optional - can be extended)
        # For example, check for excessive repetition or other quality issues
        if ValidationService._has_excessive_repetition(selected_text):
            raise ValidationError(
                "Selected text contains excessive repetition and may not be suitable for processing",
                details="selected_text contains excessive repetition"
            )

        return True

    @staticmethod
    def validate_query_request(query_request: QueryRequest) -> bool:
        """
        Validate a complete query request.

        Args:
            query_request (QueryRequest): The query request to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        # The Pydantic model validation already handles basic validation
        # Here we can add additional business logic validation if needed

        # Validate selected text if present
        if query_request.selected_text is not None:
            ValidationService.validate_selected_text(query_request.selected_text)

        # Additional validations can be added here as needed
        # For example: check if query and selected_text are related, etc.

        return True

    @staticmethod
    def validate_agent_response(
        response: AgentResponse,
        query_request: QueryRequest,
        retrieved_contexts: List[RetrievedContext]
    ) -> bool:
        """
        Validate an agent response for grounding and accuracy.

        Args:
            response (AgentResponse): The agent's response to validate
            query_request (QueryRequest): The original query request
            retrieved_contexts (List[RetrievedContext]): The contexts used to generate the response

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check if the response has source context references
        if not response.source_context:
            raise ValidationError(
                "Agent response does not include source context references",
                details="response.source_context is empty"
            )

        # Check if the response text is grounded in the provided context
        is_response_grounded = ValidationService._is_response_grounded_in_context(
            response.response_text,
            retrieved_contexts
        )

        if not is_response_grounded:
            raise ValidationError(
                "Agent response is not properly grounded in the provided context",
                details="response text does not reference content from retrieved contexts"
            )

        # Check confidence score is within valid range
        if not (0.0 <= response.confidence_score <= 1.0):
            raise ValidationError(
                f"Confidence score {response.confidence_score} is out of valid range [0.0, 1.0]",
                details="confidence_score must be between 0.0 and 1.0"
            )

        # Check if hallucination was detected
        if response.is_hallucination_detected:
            raise HallucinationError(
                "Hallucination detected in agent response",
                details="response flagged as containing hallucination"
            )

        return True

    @staticmethod
    def _is_response_grounded_in_context(response_text: str, contexts: List[RetrievedContext]) -> bool:
        """
        Check if the response text is grounded in the provided contexts.

        Args:
            response_text (str): The agent's response text
            contexts (List[RetrievedContext]): The contexts used to generate the response

        Returns:
            bool: True if response is grounded in contexts, False otherwise
        """
        if not contexts:
            return False

        # Simple heuristic: check if key phrases from contexts appear in the response
        response_lower = response_text.lower()

        # Look for significant overlap between context content and response
        for context in contexts:
            context_lower = context.content.lower()

            # Check for content overlap (this is a simplified approach)
            # In a more sophisticated implementation, we might use semantic similarity
            if len(context_lower) > 20:  # Only check substantial context chunks
                # Look for at least some overlap between context and response
                common_words = set(response_lower.split()) & set(context_lower.split())
                if len(common_words) > 0:
                    return True

        # Alternative: Check if response cites sources that match the contexts
        for source in response_text.split():
            # Look for source references in the response that match context sources
            if any(source.lower() in ctx.source.lower() for ctx in contexts):
                return True

        # For a more robust implementation, we could use:
        # - Semantic similarity between response and contexts
        # - Named entity recognition to check if entities match
        # - Citation detection to ensure sources are referenced
        return False

    @staticmethod
    def _has_excessive_repetition(text: str, threshold: float = 0.3) -> bool:
        """
        Check if the text contains excessive repetition.

        Args:
            text (str): The text to check
            threshold (float): Threshold for what constitutes excessive repetition (default 30%)

        Returns:
            bool: True if excessive repetition is detected, False otherwise
        """
        if len(text) < 100:  # Skip short texts
            return False

        # Simple heuristic: check if any 10-character substring appears more than threshold times
        min_substring_len = 10
        if len(text) < min_substring_len * 2:
            return False

        # Check for repeated substrings
        for i in range(len(text) - min_substring_len):
            substring = text[i:i + min_substring_len]
            count = text.count(substring)
            if count > 1:
                # Calculate the ratio of repeated content
                repeated_content_ratio = (count * len(substring)) / len(text)
                if repeated_content_ratio > threshold:
                    return True

        return False


# Create a singleton instance of the validation service
validation_service = ValidationService()