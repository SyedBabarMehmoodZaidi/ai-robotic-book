# Data Model: RAG Retrieval Validation

## Entities

### RetrievalQuery
**Description**: Input text query used to search for semantically similar content in the vector database

**Fields**:
- `query_text` (string): The original text query from the user
- `query_vector` (list[float]): The vector representation of the query text
- `query_id` (string): Unique identifier for the query
- `created_at` (datetime): Timestamp when the query was created
- `expected_results` (list[string]): Optional expected results for validation

**Validation Rules**:
- Query text must not be empty
- Query vector must match the dimension of the target vector space
- Query ID must be unique within the validation session

### RetrievedChunk
**Description**: Text segment returned by the similarity search with associated metadata and relevance score

**Fields**:
- `chunk_id` (string): Unique identifier for the chunk in the vector database
- `content` (string): The actual text content of the chunk
- `similarity_score` (float): Relevance score returned by the similarity search
- `metadata` (dict): Original metadata including source URL, section, and chunk ID
- `retrieved_at` (datetime): Timestamp when the chunk was retrieved

**Validation Rules**:
- Content must not be empty
- Similarity score must be between 0 and 1
- Metadata must contain required fields (URL, section, chunk ID)

### ValidationReport
**Description**: Comprehensive report containing retrieval quality metrics, accuracy assessments, and test results

**Fields**:
- `report_id` (string): Unique identifier for the validation report
- `created_at` (datetime): Timestamp when the report was generated
- `total_queries` (int): Total number of queries executed
- `successful_queries` (int): Number of queries that returned results
- `accuracy_rate` (float): Percentage of queries that returned contextually relevant results
- `metadata_accuracy` (float): Percentage of results with correct metadata
- `average_response_time` (float): Average time taken per query in seconds
- `detailed_results` (list[dict]): Detailed results for each query
- `quality_metrics` (dict): Additional quality metrics and statistics

**Validation Rules**:
- Accuracy rate must be between 0 and 100
- Metadata accuracy must be between 0 and 100
- Total queries must be greater than 0
- All required metrics must be present

### MetadataRecord
**Description**: Information associated with each retrieved chunk including source URL, document section, and chunk identifier

**Fields**:
- `source_url` (string): Original URL where the content was found
- `section` (string): Document section or title associated with the content
- `chunk_id` (string): Original chunk identifier from the embedding process
- `document_id` (string): Identifier for the original document
- `chunk_index` (int): Index of this chunk within the original document
- `total_chunks` (int): Total number of chunks in the original document

**Validation Rules**:
- Source URL must be a valid URL format
- Chunk ID must match the identifier in the vector database
- Chunk index must be non-negative and less than total chunks

### ValidationResult
**Description**: Result of validating a single retrieved chunk for relevance and accuracy

**Fields**:
- `query_id` (string): Reference to the original query
- `chunk_id` (string): Reference to the retrieved chunk
- `is_relevant` (bool): Whether the chunk is contextually relevant to the query
- `relevance_score` (float): Subjective relevance score (0-1)
- `validation_notes` (string): Additional notes about the validation
- `validator` (string): Identifier of the validator (human or automated)

**Validation Rules**:
- Relevance score must be between 0 and 1
- Is_relevant must be consistent with relevance score (if score > 0.5, is_relevant should be true)