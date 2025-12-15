# Embedding Pipeline for Docusaurus Content

This project implements an embedding pipeline that extracts text from deployed Docusaurus URLs, generates embeddings using Cohere, and stores them in Qdrant for RAG-based retrieval.

## Prerequisites

- Python 3.11 or higher
- UV package manager
- Cohere API key
- Qdrant instance (local or cloud)

## Setup

1. **Install UV package manager** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   uv pip install -e .
   # Or if you want to install in development mode:
   uv pip install -e ".[dev]"  # if you have dev dependencies
   ```

3. **Set up environment variables**:
   Copy the `.env` file and add your API keys:
   ```bash
   cp .env .env.local
   # Edit .env.local and add your Cohere and Qdrant API keys
   ```

4. **Run the pipeline**:
   ```bash
   cd backend
   python main.py
   ```

## Configuration

The pipeline can be configured using environment variables in the `.env` file:

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: Qdrant instance URL (optional, for remote instances)
- `QDRANT_API_KEY`: Qdrant API key (optional, for remote instances)
- `LOCAL_QDRANT_PATH`: Path for local Qdrant storage (default: ./qdrant_data)
- `CHUNK_SIZE`: Size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 characters)
- `MAX_TOKENS`: Maximum tokens for Cohere API (default: 3000)
- `CRAWL_DEPTH`: How deep to crawl the site (default: 2 levels)

## Usage

The pipeline is implemented in a single `main.py` file with the following main functions:

- `get_all_urls(base_url, max_depth=2)`: Crawls a Docusaurus site and returns all valid URLs
- `extract_text_from_url(url)`: Extracts clean text content from a single URL
- `chunk_text(text, chunk_size=1000, chunk_overlap=100)`: Splits text into smaller chunks
- `embed(text_chunks)`: Generates embeddings for text chunks using Cohere API
- `save_chunk_to_qdrant(text_chunk, embedding, metadata)`: Stores text and embeddings in Qdrant
- `main()`: Orchestrates the entire pipeline

The pipeline will process the target Docusaurus site (https://ai-robotic-book.vercel.app/) by default.

## Architecture

The implementation follows a single-file architecture in `main.py` containing all required functionality:
- URL crawling and validation
- Text extraction and cleaning using BeautifulSoup
- Content chunking to handle token limits
- Cohere embedding generation
- Qdrant vector storage with metadata
- Error handling and logging
- Progress tracking

## Components

### Data Models
- `DocumentChunk`: Represents a segment of text content with metadata
- `EmbeddingVector`: Numeric representation of text semantics
- `VectorRecord`: Complete entity stored in Qdrant with vector and metadata
- `CrawlResult`: Result of crawling operations

### Main Pipeline Flow
1. URL crawling with `get_all_urls`
2. Text extraction with `extract_text_from_url`
3. Content chunking with `chunk_text`
4. Embedding generation with `embed`
5. Vector storage with `save_chunk_to_qdrant`
6. Execution orchestration in `main`