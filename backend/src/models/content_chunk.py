"""
Model representing a segment of extracted book content suitable for embedding generation.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ContentChunk:
    """
    A segment of extracted book content suitable for embedding generation,
    including the text content, source document identifier, position within document, and metadata.
    """
    content: str
    source_document_id: str
    position: int
    chunk_id: Optional[str] = None
    word_count: Optional[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate the content chunk after initialization."""
        if not self.content:
            raise ValueError("Content cannot be empty")

        if self.position < 0:
            raise ValueError("Position must be a positive integer")

        if self.word_count is None:
            self.word_count = len(self.content.split())

        if self.created_at is None:
            self.created_at = datetime.now()

        if self.chunk_id is None:
            # Generate a unique chunk ID based on source and position
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            self.chunk_id = f"{self.source_document_id[:10]}_{self.position}_{unique_id}"