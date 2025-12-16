from pydantic import BaseModel
from typing import Optional


class MetadataPackage(BaseModel):
    """
    Model representing metadata associated with a retrieved text chunk.

    Attributes:
        url: URL where the original content can be found
        section: Section or chapter name in the document
        chunk_id: Unique identifier for this chunk
        document_id: Identifier for the source document
        document_title: Title of the source document
        source_type: Type of source (e.g., 'book', 'article', 'webpage')
    """
    url: Optional[str] = None
    section: Optional[str] = None
    chunk_id: str
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    source_type: Optional[str] = "book"  # Default to book as per the project context

    class Config:
        # Allow extra fields in case additional metadata is added
        extra = "allow"