"""
Model representing information about the source document.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class ProcessingStatus(Enum):
    """Enumeration of possible processing statuses."""
    PENDING = "pending"
    EXTRACTED = "extracted"
    EMBEDDED = "embedded"
    STORED = "stored"


@dataclass
class DocumentMetadata:
    """
    Information about the source document including URL, title, creation date, and processing status.
    """
    document_id: str
    url: str
    title: str
    source_type: str = "docusaurus-page"
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate the document metadata after initialization."""
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")

        if not self.url:
            raise ValueError("URL cannot be empty")

        # Basic URL validation
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError("URL must be a valid, publicly accessible URL")

        if not self.title:
            raise ValueError("Title cannot be empty")

        if self.created_at is None:
            self.created_at = datetime.now()

        if self.updated_at is None:
            self.updated_at = self.created_at