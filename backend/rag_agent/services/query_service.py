"""
Query processing service for the RAG agent.
This service orchestrates the query processing workflow.
"""
from typing import Optional
import time
from ..models import QueryRequest, AgentResponse, APIRequest, APIResponse
from ..agent import RAGAgent
from ..utils import generate_request_id, get_current_timestamp
from ..errors import RAGAgentError, ValidationError
from .validation_service import validation_service


class QueryService:
    """
    Service class to handle query processing workflows.
    """
    def __init__(self):
        self.agent = RAGAgent()

    def process_query_request(self, query_request: QueryRequest, client_ip: Optional[str] = None) -> APIResponse:
        """
        Process a query request through the full workflow.

        Args:
            query_request (QueryRequest): The incoming query request
            client_ip (Optional[str]): Client IP address for logging

        Returns:
            APIResponse: The complete API response
        """
        # Generate request ID and start timing
        request_id = generate_request_id()
        start_time = time.time()

        # Create API request wrapper
        api_request = APIRequest(
            query=query_request,
            request_id=request_id,
            client_ip=client_ip
        )

        try:
            # Perform additional validation using the validation service
            validation_service.validate_query_request(query_request)

            # Process the query with the RAG agent
            agent_response = self.agent.process_query(query_request)

            # Calculate total processing time
            total_processing_time = time.time() - start_time

            # Create API response
            api_response = APIResponse(
                response=agent_response,
                request_id=request_id,
                status_code=200,
                processing_time=total_processing_time
            )

            return api_response

        except ValidationError as e:
            # Handle validation errors specifically
            total_processing_time = time.time() - start_time
            error_response = self._create_error_response(
                request_id, 422, str(e), e.error_code.value, total_processing_time
            )
            return error_response

        except RAGAgentError as e:
            # Handle RAG agent specific errors
            total_processing_time = time.time() - start_time
            error_response = self._create_error_response(
                request_id, 500, str(e), e.error_code.value, total_processing_time
            )
            return error_response

        except Exception as e:
            # Handle unexpected errors
            total_processing_time = time.time() - start_time
            error_response = self._create_error_response(
                request_id, 500, f"Internal server error: {str(e)}", "INTERNAL_ERROR", total_processing_time
            )
            return error_response

    def _create_error_response(self, request_id: str, status_code: int, message: str,
                              error_code: str, processing_time: float) -> 'APIResponse':
        """
        Create an error response when exceptions occur.

        Args:
            request_id (str): Request ID
            status_code (int): HTTP status code
            message (str): Error message
            error_code (str): Error code
            processing_time (float): Processing time

        Returns:
            APIResponse: Error response
        """
        # For error responses, we'll create a minimal AgentResponse with error information
        from ..models import AgentResponse

        error_agent_response = AgentResponse(
            response_text=f"Error: {message}",
            source_context=[],
            confidence_score=0.0,
            tokens_used=0,
            processing_time=processing_time,
            query_id=request_id,
            is_hallucination_detected=False
        )

        error_api_response = APIResponse(
            response=error_agent_response,
            request_id=request_id,
            status_code=status_code,
            processing_time=processing_time
        )

        return error_api_response


# Create a singleton instance of the query service
query_service = QueryService()