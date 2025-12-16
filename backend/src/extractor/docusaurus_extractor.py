"""
Module for extracting content from Docusaurus pages.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import time
from ..models.content_chunk import ContentChunk
from ..models.document_metadata import DocumentMetadata, ProcessingStatus
from ..exceptions import ContentExtractionError
from ..logging_config import logger


class DocusaurusExtractor:
    """
    Extractor for Docusaurus pages that retrieves clean text content
    while preserving important elements like code blocks and tables.
    """

    def __init__(self, timeout: int = 30, retry_attempts: int = 3):
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.session = requests.Session()
        # Set a user agent to avoid being blocked by some sites
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def extract_from_url(self, url: str, include_code_blocks: bool = True, include_headers: bool = True) -> DocumentMetadata:
        """
        Extract content from a single Docusaurus URL.

        Args:
            url: The URL to extract content from
            include_code_blocks: Whether to include code blocks in extraction
            include_headers: Whether to include headers in extraction

        Returns:
            DocumentMetadata containing the extracted content and metadata
        """
        # Validate URL accessibility
        if not self._is_url_accessible(url):
            raise ContentExtractionError(f"URL is not publicly accessible: {url}")

        # Fetch the page content
        response = self._fetch_with_retry(url)

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract clean text content
        clean_content = self._extract_clean_content(
            soup,
            include_code_blocks=include_code_blocks,
            include_headers=include_headers
        )

        # Create document metadata
        document_id = self._generate_document_id(url)
        title = self._extract_title(soup)

        document_metadata = DocumentMetadata(
            document_id=document_id,
            url=url,
            title=title,
            source_type="docusaurus-page",
            processing_status=ProcessingStatus.EXTRACTED
        )

        logger.info(f"Successfully extracted content from {url}")
        return document_metadata

    def extract_content_from_url(self, url: str, include_code_blocks: bool = True, include_headers: bool = True) -> tuple:
        """
        Extract content from a single Docusaurus URL.

        Args:
            url: The URL to extract content from
            include_code_blocks: Whether to include code blocks in extraction
            include_headers: Whether to include headers in extraction

        Returns:
            Tuple of (DocumentMetadata, clean_content) containing the extracted content and metadata
        """
        # Validate URL accessibility
        if not self._is_url_accessible(url):
            raise ContentExtractionError(f"URL is not publicly accessible: {url}")

        # Fetch the page content
        response = self._fetch_with_retry(url)

        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract clean text content
        clean_content = self._extract_clean_content(
            soup,
            include_code_blocks=include_code_blocks,
            include_headers=include_headers
        )

        # Create document metadata
        document_id = self._generate_document_id(url)
        title = self._extract_title(soup)

        document_metadata = DocumentMetadata(
            document_id=document_id,
            url=url,
            title=title,
            source_type="docusaurus-page",
            processing_status=ProcessingStatus.EXTRACTED
        )

        logger.info(f"Successfully extracted content from {url}")
        return document_metadata, clean_content

    def extract_from_urls(self, urls: List[str], options: Optional[Dict] = None) -> List[DocumentMetadata]:
        """
        Extract content from multiple Docusaurus URLs.

        Args:
            urls: List of URLs to extract content from
            options: Optional dictionary with extraction options

        Returns:
            List of DocumentMetadata for each successfully extracted URL
        """
        if options is None:
            options = {"include_code_blocks": True, "include_headers": True}

        results = []
        failed_urls = []

        for url in urls:
            try:
                doc_meta, _ = self.extract_content_from_url(
                    url,
                    include_code_blocks=options.get("include_code_blocks", True),
                    include_headers=options.get("include_headers", True)
                )
                results.append(doc_meta)
            except Exception as e:
                logger.error(f"Failed to extract content from {url}: {str(e)}")
                failed_urls.append(url)

        logger.info(f"Extraction completed: {len(results)} successful, {len(failed_urls)} failed")
        return results

    def _is_url_accessible(self, url: str) -> bool:
        """Check if the URL is publicly accessible."""
        try:
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            return response.status_code == 200
        except:
            return False

    def _fetch_with_retry(self, url: str) -> requests.Response:
        """Fetch URL content with retry logic."""
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise ContentExtractionError(f"Failed to fetch {url} after {self.retry_attempts} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise ContentExtractionError(f"Failed to fetch {url} after {self.retry_attempts} attempts")

    def _extract_clean_content(self, soup: BeautifulSoup, include_code_blocks: bool = True, include_headers: bool = True) -> str:
        """
        Extract clean text content from the parsed HTML, removing navigation elements.
        """
        # Remove navigation and UI elements that are common in Docusaurus sites
        for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Remove elements with common Docusaurus class names for navigation
        for element in soup.find_all(class_=['navbar', 'menu', 'sidebar', 'toc', 'theme-doc-sidebar-menu']):
            element.decompose()

        # Remove edit links and other UI elements
        for element in soup.find_all(class_=['edit-this-page', 'theme-edit-this-page', 'pagination-nav']):
            element.decompose()

        # Extract text content
        content_parts = []

        # Process different content types
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
            if include_headers or not element.name.startswith('h'):
                text = element.get_text(strip=True)
                if text:
                    content_parts.append(text)

        # Optionally include code blocks
        if include_code_blocks:
            for code_element in soup.find_all(['code', 'pre']):
                code_text = code_element.get_text(strip=True)
                if code_text:
                    content_parts.append(f"```\n{code_text}\n```")

        # Join all content parts with proper spacing
        clean_content = '\n\n'.join(content_parts)
        return clean_content

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title from the soup object."""
        # Try to find the title in various common locations
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)

        # Look for h1 elements
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)

        # Look for meta title
        meta_title = soup.find('meta', attrs={'property': 'og:title'})
        if meta_title:
            return meta_title.get('content', '').strip()

        # Fallback to URL path
        return soup.title.string if soup.title else 'Untitled Document'

    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID based on the URL."""
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"doc_{url_hash}"