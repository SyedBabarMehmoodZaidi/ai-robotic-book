# Function Contracts: Embedding Pipeline

## get_all_urls(base_url, max_depth=2)
**Description**: Crawls a Docusaurus site and returns all valid URLs up to specified depth

**Input**:
- `base_url` (string): The base URL of the Docusaurus site to crawl
- `max_depth` (int, optional): Maximum depth to crawl (default: 2)

**Output**:
- `urls` (list[string]): List of all discovered URLs

**Errors**:
- Raises exception if base_url is invalid or inaccessible

## extract_text_from_url(url)
**Description**: Extracts clean text content from a single URL

**Input**:
- `url` (string): The URL to extract text from

**Output**:
- `content` (dict): Contains 'text', 'title', and 'url' fields

**Errors**:
- Returns error status if URL is inaccessible

## chunk_text(text, chunk_size=1000, chunk_overlap=100)
**Description**: Splits text into smaller chunks that fit within token limits

**Input**:
- `text` (string): The text to chunk
- `chunk_size` (int, optional): Maximum size of each chunk (default: 1000)
- `chunk_overlap` (int, optional): Overlap between chunks (default: 100)

**Output**:
- `chunks` (list[string]): List of text chunks

## embed(text_chunks)
**Description**: Generates embeddings for text chunks using Cohere API

**Input**:
- `text_chunks` (list[string]): List of text chunks to embed

**Output**:
- `embeddings` (list[list[float]]): List of embedding vectors

**Errors**:
- Raises exception if Cohere API call fails

## save_chunk_to_qdrant(text_chunk, embedding, metadata)
**Description**: Stores a text chunk and its embedding in Qdrant

**Input**:
- `text_chunk` (string): The original text chunk
- `embedding` (list[float]): The embedding vector
- `metadata` (dict): Metadata including source URL, title, etc.

**Output**:
- `success` (bool): Whether the operation was successful
- `id` (string): The ID assigned in Qdrant

**Errors**:
- Returns False if storage fails