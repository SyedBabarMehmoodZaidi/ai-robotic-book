# Quickstart: Embedding Pipeline

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

2. **Create project directory and initialize**:
   ```bash
   mkdir backend
   cd backend
   uv init
   ```

3. **Install required dependencies**:
   ```bash
   uv add cohere qdrant-client beautifulsoup4 requests python-dotenv
   ```

4. **Set up environment variables**:
   Create a `.env` file in the backend directory:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_url_here  # Optional, for remote Qdrant
   QDRANT_API_KEY=your_qdrant_api_key_here  # Optional, for remote Qdrant
   LOCAL_QDRANT_PATH=./qdrant_data  # Optional, for local storage
   ```

## Usage

1. **Run the embedding pipeline**:
   ```bash
   python main.py
   ```

2. **The pipeline will execute the following steps**:
   - Crawl all URLs from the target Docusaurus site (https://ai-robotic-book.vercel.app/)
   - Extract clean text content from each page
   - Chunk the content to fit within token limits
   - Generate embeddings using Cohere
   - Create the "rag_embedding" collection in Qdrant if it doesn't exist
   - Store embeddings with metadata in Qdrant

## Configuration

The pipeline can be configured by modifying parameters in the main.py file:
- `CHUNK_SIZE`: Size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 characters)
- `MAX_TOKENS`: Maximum tokens for Cohere API (default: 3000)
- `CRAWL_DEPTH`: How deep to crawl the site (default: 2 levels)

## Verification

After running the pipeline:
1. Check the Qdrant collection "rag_embedding" has been created
2. Verify that vectors have been stored with appropriate metadata
3. Test retrieval by performing a similarity search with a sample query