from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uuid


class SearchQuery(BaseModel):
    """
    Model representing a search query for the retrieval system.

    Attributes:
        query_text: The text of the search query
        top_k: Number of top results to return (default 5)
        similarity_threshold: Minimum similarity score threshold (default 0.0)
        query_id: Unique identifier for the query (auto-generated)
        created_at: Timestamp when the query was created (auto-generated)
    """
    query_text: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    query_id: str = None
    created_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.query_id is None:
            self.query_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    class Config:
        # Allow extra fields in case additional parameters are added
        extra = "allow"