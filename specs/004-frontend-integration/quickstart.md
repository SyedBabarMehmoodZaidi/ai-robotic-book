# Quickstart Guide: Frontend Integration with RAG Backend

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.11+ with uv package manager
- Running RAG backend from Spec-3 (FastAPI server)

## Setup Instructions

### 1. Clone and Navigate to Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Start the RAG Backend (if not already running)
```bash
cd backend
uv venv  # Create virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
python -m main  # or uv run main
```

Verify the backend is running by accessing: `http://localhost:8000/api/v1/agent/health`

### 3. Set up the Frontend
```bash
cd frontend
npm install  # or yarn install
```

### 4. Configure Environment Variables
Create a `.env` file in the frontend directory:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_API_KEY=your-openai-api-key  # if required
```

### 5. Run the Frontend
```bash
npm start  # or yarn start
```

## Key Components

### Query Service (`frontend/src/services/queryService.js`)
Handles communication with the RAG backend:

```javascript
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export const queryService = {
  // Verify backend availability
  async checkHealth() {
    const response = await axios.get(`${BACKEND_URL}/api/v1/agent/health`);
    return response.data;
  },

  // Submit a query to the RAG agent
  async submitQuery(queryData) {
    const response = await axios.post(`${BACKEND_URL}/api/v1/agent/query`, queryData);
    return response.data;
  }
};
```

### Text Selection Utility
Capture selected text in the book interface:

```javascript
export const textSelectionUtils = {
  getSelectedText() {
    const selection = window.getSelection();
    return selection.toString().trim();
  },

  getSelectedRange() {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
      return selection.getRangeAt(0);
    }
    return null;
  }
};
```

## API Usage Examples

### Submit a General Query
```javascript
import { queryService } from './services/queryService';

const response = await queryService.submitQuery({
  query_text: "What is the main concept discussed in this book?",
  query_type: "general"
});
```

### Submit a Context-Specific Query
```javascript
import { queryService, textSelectionUtils } from './utils';

const selectedText = textSelectionUtils.getSelectedText();
const response = await queryService.submitQuery({
  query_text: "Explain this concept in more detail",
  context_text: selectedText,
  query_type: "context-specific"
});
```

## Error Handling

The frontend includes error handling for common scenarios:

- **Backend Unavailable**: Shows user-friendly message and suggests retrying
- **Invalid Queries**: Validates input before sending to backend
- **Timeouts**: Implements reasonable timeout limits with feedback
- **Rate Limits**: Handles API rate limiting gracefully

## Development Tips

1. **Testing Locally**: Use the health check endpoint to verify backend connectivity
2. **Debugging**: Enable detailed logging in development mode
3. **Environment Configuration**: Use different backend URLs for development/staging/production
4. **CORS Issues**: Ensure the backend has appropriate CORS configuration for frontend origins

## Troubleshooting

### Backend Connection Issues
- Verify the backend server is running on the configured port
- Check CORS configuration on the backend
- Confirm network connectivity between frontend and backend

### Text Selection Not Working
- Ensure the book interface allows text selection
- Check for JavaScript errors in the browser console
- Verify the text selection API is supported in the target browsers

### Query Processing Errors
- Check that query text meets length requirements (1-2000 characters)
- Verify context text length (max 10000 characters)
- Ensure query_type is either "general" or "context-specific"