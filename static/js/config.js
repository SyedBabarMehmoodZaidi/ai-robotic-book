// Configuration for RAG frontend integration
const RAGConfig = {
  // Backend API URL - defaults to local development
  BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://localhost:8000',

  // Development mode flag
  DEVELOPMENT_MODE: process.env.NODE_ENV !== 'production',

  // Request timeout in milliseconds
  REQUEST_TIMEOUT: 30000, // 30 seconds

  // Retry settings
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000, // 1 second

  // Validation settings
  MIN_QUERY_LENGTH: 3,
  MAX_QUERY_LENGTH: 2000,
  MIN_SELECTED_TEXT_LENGTH: 10,
  MAX_SELECTED_TEXT_LENGTH: 5000
};

// Make config available globally
window.RAGConfig = RAGConfig;

export default RAGConfig;