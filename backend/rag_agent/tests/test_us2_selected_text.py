"""
Tests for US2 functionality: Select Specific Text for Questioning
"""
import pytest
from pydantic import ValidationError

from ..models import QueryRequest
from ..services.validation_service import validation_service


def test_selected_text_validation():
    """Test validation of selected text."""
    # Valid selected text
    assert validation_service.validate_selected_text("This is a valid selected text segment.") is True

    # Test with None (should be valid)
    assert validation_service.validate_selected_text(None) is True

    # Test short selected text (should fail)
    with pytest.raises(ValidationError) as exc_info:
        validation_service.validate_selected_text("Short")
    assert "too short" in str(exc_info.value).lower()

    # Test empty selected text after stripping (should fail)
    with pytest.raises(ValidationError) as exc_info:
        validation_service.validate_selected_text("   ")
    assert "cannot be empty" in str(exc_info.value).lower()

    # Test selected text that's too long (should fail)
    long_text = "A" * 5001  # More than 5000 characters
    with pytest.raises(ValidationError) as exc_info:
        validation_service.validate_selected_text(long_text)
    assert "too long" in str(exc_info.value).lower()


def test_query_with_selected_text():
    """Test query request with selected text."""
    # Valid query with selected text
    query = QueryRequest(
        query_text="What is the main concept?",
        selected_text="The main concept is that artificial intelligence involves machines that can perform tasks that typically require human intelligence."
    )
    assert query.selected_text is not None

    # Test validation service with query containing selected text
    assert validation_service.validate_query_request(query) is True


def test_query_validation_with_selected_text():
    """Test that queries with invalid selected text are rejected."""
    # Try to create a query with invalid selected text (too short)
    with pytest.raises(ValidationError):
        QueryRequest(
            query_text="What is AI?",
            selected_text="Hi"  # Too short
        )

    # Try to create a query with invalid selected text (empty after strip)
    with pytest.raises(ValidationError):
        QueryRequest(
            query_text="What is AI?",
            selected_text="   "  # Just whitespace
        )


def test_selected_text_preprocessing():
    """Test preprocessing of selected text."""
    from ..retrieval_tool import preprocess_selected_text

    # Test normal text preprocessing
    result = preprocess_selected_text("  This is a test.  ")
    assert result == "This is a test."

    # Test with None
    result = preprocess_selected_text(None)
    assert result is None

    # Test with empty string after strip
    result = preprocess_selected_text("   ")
    assert result is None

    # Test with very long text (should be truncated)
    long_text = "A" * 5001
    result = preprocess_selected_text(long_text)
    assert result is not None
    assert len(result) <= 5000


def test_query_request_with_edge_cases():
    """Test query requests with edge cases for selected text."""
    # Valid case with long but acceptable selected text
    long_acceptable_text = "A" * 5000  # Exactly at the limit
    query = QueryRequest(
        query_text="What does this long text say?",
        selected_text=long_acceptable_text
    )
    assert len(query.selected_text) == 5000

    # Test validation through the validation service
    assert validation_service.validate_query_request(query) is True


if __name__ == "__main__":
    # Run basic tests
    test_selected_text_validation()
    test_query_with_selected_text()
    test_query_validation_with_selected_text()
    test_selected_text_preprocessing()
    test_query_request_with_edge_cases()
    print("✅ US2 functionality tests passed!")