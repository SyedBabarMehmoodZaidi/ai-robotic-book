# Function Contracts: RAG Retrieval Validation

## query_to_vector(query_text)
**Description**: Converts a user query text to a vector representation using Cohere

**Input**:
- `query_text` (string): The text query to convert

**Output**:
- `query_vector` (list[float]): The vector representation of the query

**Errors**:
- Raises exception if Cohere API call fails

## perform_similarity_search(query_vector, top_k=5)
**Description**: Performs top-k similarity search in Qdrant using the query vector

**Input**:
- `query_vector` (list[float]): The vector representation of the query
- `top_k` (int, optional): Number of top results to return (default: 5)

**Output**:
- `retrieved_chunks` (list[dict]): List of retrieved chunks with content, metadata, and similarity scores

**Errors**:
- Returns empty list if no relevant results found
- Raises exception if Qdrant connection fails

## validate_retrieved_chunks(retrieved_chunks, expected_content=None)
**Description**: Validates the accuracy and relevance of retrieved chunks

**Input**:
- `retrieved_chunks` (list[dict]): The chunks retrieved from the similarity search
- `expected_content` (list[string], optional): Expected content for validation

**Output**:
- `validation_results` (list[dict]): Validation results for each chunk with relevance scores

## validate_metadata(metadata_list)
**Description**: Validates that metadata (URL, section, chunk ID) is preserved correctly

**Input**:
- `metadata_list` (list[dict]): List of metadata records to validate

**Output**:
- `metadata_accuracy` (float): Percentage of metadata records that are valid

## generate_validation_report(validation_results, metadata_accuracy)
**Description**: Generates a comprehensive validation report with quality metrics

**Input**:
- `validation_results` (list[dict]): Results from chunk validation
- `metadata_accuracy` (float): Metadata accuracy percentage

**Output**:
- `validation_report` (dict): Comprehensive validation report with metrics and statistics