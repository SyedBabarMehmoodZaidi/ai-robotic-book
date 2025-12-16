from pydantic import BaseModel
from typing import Dict, Any, Optional


class RetrievedChunk(BaseModel):
    """
    Model representing a text chunk retrieved from the vector database.

    Attributes:
        content: The actual text content of the chunk
        similarity_score: The similarity score between query and this chunk (0-1)
        chunk_id: Unique identifier for this chunk
        metadata: Additional metadata associated with the chunk
        position: Position of the chunk in the original document
    """
    content: str
    similarity_score: float
    chunk_id: str
    metadata: Dict[str, Any]
    position: int

    class Config:
        # Allow extra fields in case additional metadata is added
        extra = "allow"