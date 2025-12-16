from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


class ValidationResult(BaseModel):
    """
    Model representing the result of a single validation test.

    Attributes:
        test_id: Unique identifier for the test
        query_text: The query text used for validation
        expected_results: Expected results for the query
        success_criteria: Criteria for determining if test passed
        test_category: Category of test (factual, conceptual, contextual)
        executed_at: Timestamp when the test was executed
        result_accuracy: Accuracy score of the results (0-1)
        actual_results: Actual results returned by the system
        passed: Whether the test passed or failed
    """
    test_id: str
    query_text: str
    expected_results: List[Dict[str, Any]]
    success_criteria: str
    test_category: str
    executed_at: datetime
    result_accuracy: float
    actual_results: List[Dict[str, Any]]
    passed: bool


class ValidationTest(BaseModel):
    """
    Model representing a validation test for the retrieval system.

    Attributes:
        test_id: Unique identifier for the test (auto-generated)
        query_text: The text of the validation query
        expected_results: Expected results for the query
        success_criteria: Criteria for determining if test passed
        test_category: Category of test (factual, conceptual, contextual)
        executed_at: Timestamp when the test was executed
        result_accuracy: Accuracy score of the results (0-1)
    """
    test_id: str = None
    query_text: str
    expected_results: List[Dict[str, Any]]
    success_criteria: str
    test_category: str
    executed_at: datetime = None
    result_accuracy: Optional[float] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.test_id is None:
            self.test_id = str(uuid.uuid4())
        if self.executed_at is None:
            self.executed_at = datetime.utcnow()

    class Config:
        # Allow extra fields in case additional validation parameters are added
        extra = "allow"