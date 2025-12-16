# Data Model: RAG Retrieval Implementation

## Retrieved Chunk Entity

**Description**: A text segment returned by the semantic search, containing the actual content, similarity score, and associated metadata

**Fields**:
- `content`: The actual text content of the retrieved chunk
- `similarity_score`: A numerical value representing the semantic relevance to the search query
- `chunk_id`: Identifier for the source chunk in the original embedding database
- `metadata`: Object containing source document information
  - `url`: Source URL of the original document
  - `section`: Section or heading in the original document
  - `document_id`: Identifier for the source document
- `position`: Sequential position of the chunk within the source document

**Validation Rules**:
- Content must not be empty
- Similarity score must be between 0 and 1
- Chunk_id must be a valid identifier
- Metadata must contain at least the URL

## Search Query Entity

**Description**: The input text that is used to find semantically similar content in the embedded book data

**Fields**:
- `query_text`: The input text for the semantic search
- `top_k`: Number of results to retrieve (default: 5)
- `similarity_threshold`: Minimum similarity score for inclusion (default: 0.5)
- `query_id`: Unique identifier for the query (auto-generated)
- `created_at`: Timestamp of query creation

**Validation Rules**:
- Query text must not be empty
- Top_k must be a positive integer (1-100 range)
- Similarity threshold must be between 0 and 1
- Query text should have reasonable length (10-500 characters)

## Metadata Package Entity

**Description**: Information associated with each retrieved chunk including source URL, document section, and chunk identifier

**Fields**:
- `url`: Source URL of the original document
- `section`: Section or heading in the original document
- `chunk_id`: Identifier for the specific chunk
- `document_id`: Identifier for the source document
- `document_title`: Title of the source document
- `source_type`: Type of source document (e.g., "docusaurus-page")

**Validation Rules**:
- URL must be a valid, accessible URL
- Chunk_id must be a valid identifier
- Document_id must be a valid identifier
- URL cannot be empty

## Similarity Score Entity

**Description**: A numerical value representing the semantic relevance of a retrieved chunk to the search query

**Fields**:
- `score`: The similarity score value (0.0 to 1.0)
- `algorithm`: The algorithm used to calculate the similarity (e.g., "cosine-similarity")
- `query_embedding_id`: Identifier for the query embedding used
- `result_embedding_id`: Identifier for the result embedding compared

**Validation Rules**:
- Score must be between 0.0 and 1.0
- Algorithm must be from approved list
- Both embedding IDs must be valid

## Query Response Entity

**Description**: Structured result containing multiple retrieved chunks with their metadata and similarity scores

**Fields**:
- `query_id`: Identifier for the original query
- `results`: Array of RetrievedChunk objects
- `total_results`: Total number of results returned
- `query_time_ms`: Time taken to execute the query in milliseconds
- `search_params`: Parameters used for the search
  - `top_k`: Number of results requested
  - `similarity_threshold`: Minimum similarity threshold used
- `status`: Status of the query execution (e.g., "success", "partial", "no_results")

**Validation Rules**:
- Results array must contain valid RetrievedChunk objects
- Total_results must match the actual count in results array
- Query_time_ms must be a non-negative number
- Status must be from predefined list

## Validation Test Entity

**Description**: Test case for validating retrieval quality

**Fields**:
- `test_id`: Unique identifier for the test case
- `query_text`: The query text used for testing
- `expected_results`: Array of expected chunk IDs that should be returned
- `success_criteria`: Criteria for determining test success
  - `min_accuracy`: Minimum accuracy percentage required
  - `min_similarity_score`: Minimum similarity score for relevance
- `test_category`: Category of test (e.g., "factual", "conceptual", "contextual")
- `executed_at`: Timestamp when test was executed
- `result_accuracy`: Actual accuracy achieved in the test

**Validation Rules**:
- Test_id must be unique
- Query_text must not be empty
- Expected_results must contain valid chunk IDs
- Success criteria values must be within valid ranges