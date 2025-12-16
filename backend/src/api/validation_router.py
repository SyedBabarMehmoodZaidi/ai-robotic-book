from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from ..models.validation_test import ValidationTest, ValidationResult
from ..validation.validation_service import ValidationService
from ..retrieval.retrieval_service import RetrievalService
from ..utils.logging_config import get_logger


logger = get_logger(__name__)
validation_router = APIRouter(prefix="/api/v1", tags=["validation"])


def get_validation_service() -> ValidationService:
    """
    Dependency to get the validation service.
    In a real implementation, this would be managed by a DI container.
    """
    retrieval_service = RetrievalService()
    validation_service = ValidationService(retrieval_service)
    return validation_service


@validation_router.post("/validate", response_model=List[ValidationResult])
async def validate_endpoint(
    test_queries: List[ValidationTest],
    validation_service: ValidationService = Depends(get_validation_service)
) -> List[ValidationResult]:
    """
    Validate retrieval quality using multiple test queries.

    Args:
        test_queries: List of validation tests with expected outcomes
        validation_service: Validation service

    Returns:
        List of validation results with accuracy measurements
    """
    try:
        logger.info(f"Received validation request with {len(test_queries)} test queries")

        results = validation_service.validate_retrieval_quality(test_queries)

        logger.info(f"Completed validation for {len(test_queries)} queries, returning {len(results)} results")

        return results
    except Exception as e:
        logger.error(f"Error processing validation request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing validation request: {str(e)}"
        )


@validation_router.post("/validate/report", response_model=Dict[str, Any])
async def validation_report_endpoint(
    test_queries: List[ValidationTest],
    validation_service: ValidationService = Depends(get_validation_service)
) -> Dict[str, Any]:
    """
    Generate a validation report with detailed metrics.

    Args:
        test_queries: List of validation tests with expected outcomes
        validation_service: Validation service

    Returns:
        Dictionary containing validation report metrics
    """
    try:
        logger.info(f"Received validation report request with {len(test_queries)} test queries")

        # First validate the queries
        validation_results = validation_service.validate_retrieval_quality(test_queries)

        # Then create the report
        report = validation_service.create_validation_report(validation_results)

        logger.info(f"Generated validation report for {len(test_queries)} queries")

        return report
    except Exception as e:
        logger.error(f"Error processing validation report request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing validation report request: {str(e)}"
        )