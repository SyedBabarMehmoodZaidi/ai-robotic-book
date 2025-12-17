# Data Model: Frontend Integration with RAG Backend

## Entities

### Frontend Query Request
**Description**: The data structure for sending queries from frontend to backend
**Fields**:
- query_text: string (required) - The user's question
- context_text: string (optional) - Selected text context from book interface
- query_type: "general" | "context-specific" (required) - Type of query
- user_id: string (optional) - Identifier for the user making the query
- query_id: string (optional, auto-generated) - Unique identifier for the query

**Validation Rules**:
- query_text must be 1-2000 characters
- context_text can be up to 10000 characters if provided
- query_type must be either "general" or "context-specific"

### Backend Response
**Description**: The data structure returned by the backend RAG system
**Fields**:
- response_text: string (required) - The AI-generated answer
- query_id: string (required) - Reference to the original query
- retrieved_chunks: array of RetrievedChunk (required) - Content chunks used for response
- confidence_score: number (0-1) (required) - Confidence level of the response
- response_id: string (required) - Unique identifier for the response
- created_at: string (ISO date) (required) - Timestamp when response was created
- sources: array of string (optional) - List of source references
- metadata: object (optional) - Additional response metadata

### RetrievedChunk
**Description**: Model representing a content chunk retrieved from the RAG pipeline
**Fields**:
- content: string (required) - The actual content text
- similarity_score: number (0-1) (required) - Similarity score from retrieval
- chunk_id: string (required) - Unique identifier for the chunk
- metadata: object (required) - Additional metadata about the source
- position: number (required) - Position in the original document

### Frontend State
**Description**: Internal state structure for the frontend application
**Fields**:
- currentQuery: string - The current query being processed
- selectedText: string - Text currently selected in the book interface
- isProcessing: boolean - Whether a query is currently being processed
- response: BackendResponse (optional) - The latest response from the backend
- error: string (optional) - Any error message to display
- backendStatus: "online" | "offline" | "checking" - Status of backend connection

## API Endpoints

### POST /api/v1/agent/query
**Description**: Submit a query to the RAG agent for processing
**Request Body**: Frontend Query Request model
**Response**: Backend Response model
**Authentication**: Optional (may require API key in headers)
**Error Responses**:
- 400: Bad request (invalid query format)
- 401: Unauthorized (missing/invalid API key)
- 500: Internal server error (backend processing error)
- 503: Service unavailable (backend temporarily down)

### GET /api/v1/agent/health
**Description**: Check the health status of the RAG agent
**Response**: Health status object
**Authentication**: None required
**Success Response**:
```json
{
  "status": "healthy",
  "message": "RAG agent is operational and ready to process queries",
  "timestamp": "2025-12-17T10:30:00.000Z"
}
```

## State Transitions

### Query Processing Flow
1. User enters query or selects text → Frontend State: { currentQuery: "...", selectedText: "..." }
2. User submits query → Frontend State: { isProcessing: true }
3. Query sent to backend → Frontend State: { isProcessing: true }
4. Response received → Frontend State: { isProcessing: false, response: {...} }
5. Error occurs → Frontend State: { isProcessing: false, error: "..." }

### Backend Connection Status
1. Initial check → Backend Status: "checking"
2. Connection successful → Backend Status: "online"
3. Connection failed → Backend Status: "offline"
4. Re-check initiated → Backend Status: "checking"