# Quickstart Guide: RAG Qdrant Implementation

## Overview
This guide provides the essential steps to set up and run the RAG Qdrant pipeline for processing Docusaurus book content and storing embeddings.

## Prerequisites
- Python 3.11+
- pip package manager
- Cohere API key
- Qdrant cloud account (free tier)

## Setup

### 1. Initialize the Project
```bash
# Create backend directory
mkdir backend
cd backend

# Initialize Python project
uv init
```

### 2. Install Dependencies
```bash
pip install cohere-client qdrant-client beautifulsoup4 requests python-dotenv
```

### 3. Environment Configuration
Create a `.env` file with your API keys:
```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Core Components

### 1. Content Extractor
Handles extraction of clean text from Docusaurus pages:
- Uses BeautifulSoup4 for HTML parsing
- Removes navigation and UI elements
- Preserves code blocks and headers as configured

### 2. Content Chunker
Splits extracted content into appropriate-sized chunks:
- Configurable chunk size (default 300 words)
- Overlap between chunks for context preservation
- Maintains semantic boundaries

### 3. Embedding Generator
Creates vector embeddings using Cohere:
- Uses Cohere's multilingual embedding model
- Processes content chunks in batches
- Handles API rate limits and errors

### 4. Vector Storage
Stores embeddings in Qdrant:
- Creates Qdrant collection with proper schema
- Stores embeddings with metadata
- Handles batch storage for efficiency

## Running the Pipeline

### 1. Extract Content
```python
from extractor import DocusaurusExtractor

extractor = DocusaurusExtractor()
urls = ["https://your-book-site.com/page1", "https://your-book-site.com/page2"]
documents = extractor.extract_from_urls(urls)
```

### 2. Chunk Content
```python
from chunker import ContentChunker

chunker = ContentChunker(chunk_size=300, overlap=50)
chunks = chunker.chunk_documents(documents)
```

### 3. Generate Embeddings
```python
from embedder import CohereEmbedder

embedder = CohereEmbedder(model="cohere-embed-multilingual-v3.0")
embeddings = embedder.generate_embeddings(chunks)
```

### 4. Store in Qdrant
```python
from storage import QdrantStorage

storage = QdrantStorage(collection_name="book_embeddings")
storage.store_embeddings(embeddings)
```

## API Endpoints

Once the service is running, you can use these endpoints:

- `POST /api/v1/extract` - Extract content from Docusaurus URLs
- `POST /api/v1/chunk` - Chunk extracted content
- `POST /api/v1/embeddings` - Generate embeddings for chunks
- `POST /api/v1/storage` - Store embeddings in Qdrant
- `POST /api/v1/search` - Perform semantic search
- `GET /api/v1/status/{job_id}` - Check job status

## Testing

Run the following to verify your setup:
```bash
python -m pytest tests/
```

## Next Steps

1. Implement the content extraction service
2. Build the embedding generation pipeline
3. Set up Qdrant vector storage
4. Create the search functionality
5. Add monitoring and error handling