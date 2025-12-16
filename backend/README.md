# RAG Qdrant Pipeline

This backend service implements a RAG (Retrieval Augmented Generation) pipeline that extracts content from Docusaurus pages, generates embeddings using Cohere, and stores them in Qdrant for semantic search.

## Features

- Content extraction from Docusaurus pages
- Scalable content chunking strategy
- Embedding generation with Cohere API
- Vector storage in Qdrant database
- Semantic search capabilities
- REST API endpoints for the entire pipeline

## Setup

1. Clone the repository
2. Navigate to the `backend` directory
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and fill in your API keys

## Usage

Run the application with:
```
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /api/v1/extract` - Extract content from Docusaurus URLs
- `POST /api/v1/chunk` - Chunk extracted content
- `POST /api/v1/embeddings` - Generate embeddings for chunks
- `POST /api/v1/storage` - Store embeddings in Qdrant
- `POST /api/v1/search` - Perform semantic search
- `GET /api/v1/status/{job_id}` - Check job status

## Configuration

The application uses environment variables for configuration:

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: Your Qdrant cluster URL
- `QDRANT_API_KEY`: Your Qdrant API key