# Quickstart: RAG Retrieval Validation

## Prerequisites

- Python 3.11 or higher
- UV package manager
- Access to the same Cohere API key used for embedding
- Access to the Qdrant instance containing the embedded content
- Valid API keys for both services

## Setup

1. **Install UV package manager** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Navigate to backend directory and initialize**:
   ```bash
   cd backend
   uv init
   ```

3. **Install required dependencies**:
   ```bash
   uv add qdrant-client cohere python-dotenv pytest
   ```

4. **Set up environment variables**:
   Create a `.env` file in the backend directory:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_url_here  # For remote instances
   QDRANT_API_KEY=your_qdrant_api_key_here  # For remote instances
   LOCAL_QDRANT_PATH=./qdrant_data  # For local storage
   ```

## Usage

1. **Run the validation tool**:
   ```bash
   python retrieval_validator.py
   ```

2. **The validation process will execute**:
   - Convert user queries to vectors using Cohere
   - Perform top-k similarity search against Qdrant database
   - Validate retrieved chunks and metadata accuracy
   - Generate comprehensive validation reports

## Configuration

The validation can be configured by modifying parameters in config.py or via environment variables:

- `TOP_K_RESULTS`: Number of results to retrieve per query (default: 5)
- `VALIDATION_TIMEOUT`: Maximum time to wait for each query (default: 30 seconds)
- `TEST_QUERIES`: List of test queries to validate against
- `SIMILARITY_THRESHOLD`: Minimum similarity score for relevance (default: 0.5)

## Validation Process

The validation tool will:
1. Execute a series of test queries against the RAG pipeline
2. Verify that metadata (URL, section, chunk ID) is preserved correctly
3. Assess the contextual relevance of retrieved content
4. Generate a detailed validation report with quality metrics
5. Highlight any issues with retrieval accuracy or metadata preservation

## Output

The tool will generate a validation report containing:
- Success rate for similarity searches
- Accuracy of retrieved content relevance
- Metadata preservation rate
- Average response times
- Detailed results for each test query