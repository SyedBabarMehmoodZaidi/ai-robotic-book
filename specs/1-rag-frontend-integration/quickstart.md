# Quickstart Guide: RAG Frontend Integration

**Feature**: 1-rag-frontend-integration
**Created**: 2025-12-15
**Status**: Draft

## Overview

This guide provides a quick start for implementing the RAG backend integration with the frontend book interface. Follow these steps to set up the development environment and begin implementation.

## Prerequisites

- Node.js 16+ (for Docusaurus)
- Python 3.11+ (for RAG backend from Spec-3)
- Docusaurus CLI installed (`npm install -g @docusaurus/cli`)
- RAG backend from Spec-3 running locally
- Git for version control

## Setup

### 1. Clone and Navigate to Project

```bash
cd F:\GS Assignment\ai-robotic-book
```

### 2. Verify RAG Backend is Running

Make sure the RAG backend from Spec-3 is running locally:

```bash
# Navigate to the backend directory
cd backend/rag_agent

# Start the backend (assuming you have the environment set up)
uvicorn main:app --reload --port 8000
```

### 3. Install Docusaurus Dependencies

```bash
# Navigate to your Docusaurus project root
cd F:\GS Assignment\ai-robotic-book

# Install dependencies
npm install
```

## Project Structure

Add the following components to integrate the RAG functionality:

```
src/
├── components/
│   ├── QueryInterface/
│   │   ├── QueryInterface.js
│   │   ├── QueryInterface.module.css
│   │   └── index.js
│   └── SelectedTextCapture/
│       ├── SelectedTextCapture.js
│       └── index.js
static/
└── js/
    └── api-client.js
```

## Implementation Steps

### 1. Create API Client

Create `static/js/api-client.js` for backend communication:

```javascript
// API client for communicating with RAG backend
class RAGApiClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async query(queryText, selectedText = null) {
    const response = await fetch(`${this.baseURL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query_text: queryText,
        selected_text: selectedText
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }
}

// Export singleton instance
window.RAGApiClient = new RAGApiClient();
```

### 2. Create Query Interface Component

Create `src/components/QueryInterface/QueryInterface.js`:

```javascript
import React, { useState } from 'react';
import styles from './QueryInterface.module.css';

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedText, setSelectedText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      // Use the global API client created in api-client.js
      const result = await window.RAGApiClient.query(query, selectedText);
      setResponse(result.response);
    } catch (err) {
      setError('Failed to get response from AI. Please try again.');
      console.error('Query error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.queryContainer}>
      <form onSubmit={handleSubmit} className={styles.queryForm}>
        <div className={styles.inputGroup}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about this book..."
            className={styles.queryInput}
            rows="3"
          />
          {selectedText && (
            <div className={styles.selectedTextPreview}>
              Selected text: "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className={styles.submitButton}
        >
          {isLoading ? 'Asking AI...' : 'Ask AI'}
        </button>
      </form>

      {isLoading && (
        <div className={styles.loadingIndicator}>
          AI is thinking...
        </div>
      )}

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      {response && (
        <div className={styles.responseContainer}>
          <h4>AI Response</h4>
          <div className={styles.responseText}>
            {response.response_text}
          </div>

          {response.source_context && response.source_context.length > 0 && (
            <div className={styles.sourceContext}>
              <strong>Sources:</strong> {response.source_context.join(', ')}
            </div>
          )}

          {response.confidence_score && (
            <div className={styles.confidence}>
              Confidence: {(response.confidence_score * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInterface;
```

### 3. Create CSS Styles

Create `src/components/QueryInterface/QueryInterface.module.css`:

```css
.queryContainer {
  margin: 2rem 0;
  padding: 1.5rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background-color: #fafafa;
}

.queryForm {
  margin-bottom: 1rem;
}

.inputGroup {
  margin-bottom: 1rem;
}

.queryInput {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  resize: vertical;
}

.queryInput:focus {
  outline: none;
  border-color: #3578e5;
  box-shadow: 0 0 0 2px rgba(53, 120, 229, 0.2);
}

.selectedTextPreview {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: #e3f2fd;
  border-radius: 4px;
  font-size: 0.9rem;
  color: #1a237e;
}

.submitButton {
  background-color: #3578e5;
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

.submitButton:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.submitButton:not(:disabled):hover {
  background-color: #1d5fb6;
}

.loadingIndicator {
  padding: 1rem;
  text-align: center;
  color: #666;
}

.error {
  padding: 1rem;
  background-color: #ffebee;
  color: #c62828;
  border-radius: 4px;
  margin-top: 1rem;
}

.responseContainer {
  margin-top: 1rem;
  padding: 1rem;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.responseText {
  margin: 1rem 0;
  line-height: 1.6;
}

.sourceContext {
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: #666;
}

.confidence {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #4caf50;
  font-weight: bold;
}
```

### 4. Create Selected Text Capture Functionality

Create `src/components/SelectedTextCapture/SelectedTextCapture.js`:

```javascript
import { useEffect } from 'react';

const SelectedTextCapture = ({ onTextSelected }) => {
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();

      // Only trigger if there's actually selected text
      if (selectedText.length > 0) {
        onTextSelected(selectedText);
      }
    };

    // Listen for selection changes
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    // Cleanup event listeners
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, [onTextSelected]);

  return null; // This component doesn't render anything
};

export default SelectedTextCapture;
```

### 5. Integrate Components into Docusaurus Pages

To add the query interface to a Docusaurus page, use it in your MDX files:

```mdx
import QueryInterface from '@site/src/components/QueryInterface';
import SelectedTextCapture from '@site/src/components/SelectedTextCapture';

<SelectedTextCapture onTextSelected={(text) => {
  // This would typically update a parent state
  // For now, we'll handle this in the QueryInterface component
}} />

<QueryInterface />

## Your Book Content Here

This is where your regular book content goes. Users can select text and ask questions about it using the interface above.
```

## Running the Application

### 1. Start the RAG Backend

```bash
cd backend/rag_agent
uvicorn main:app --reload --port 8000
```

### 2. Start the Docusaurus Frontend

```bash
cd F:\GS Assignment\ai-robotic-book
npm run start
```

### 3. Test the Integration

1. Open your browser to `http://localhost:3000`
2. Navigate to a book page with the query interface
3. Select some text on the page
4. Enter a question in the query interface
5. Click "Ask AI" and verify you receive a response

## Testing the Integration

### Manual Testing Steps

1. **Connectivity Test**: Verify the frontend can reach the backend
   - Use the health check endpoint: `http://localhost:8000/health`

2. **Query Functionality Test**:
   - Submit a simple query like "What is this book about?"
   - Verify response appears in the interface

3. **Selected Text Integration Test**:
   - Select text on the page
   - Verify selected text indicator appears
   - Submit a query about the selected text
   - Verify response addresses the selected content

4. **Error Handling Test**:
   - Stop the backend temporarily
   - Submit a query and verify error message appears
   - Restart backend and verify functionality resumes

## Next Steps

1. Implement the query interface component in your Docusaurus pages
2. Add the API client script to your site
3. Test end-to-end functionality with the RAG backend
4. Customize the UI to match your book's design
5. Add additional features like query history or advanced response formatting

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the RAG backend allows requests from `http://localhost:3000`
2. **Connection Failures**: Verify the backend is running on the correct port
3. **Selected Text Not Captured**: Check that event listeners are properly attached
4. **Response Formatting Issues**: Verify the response format matches the API contract