import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// Create an axios instance with default configuration
const apiClient = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor to add any needed headers
apiClient.interceptors.request.use(
  (config) => {
    // Add any auth tokens or other headers here if needed
    const apiKey = process.env.REACT_APP_API_KEY;
    if (apiKey) {
      config.headers['Authorization'] = `Bearer ${apiKey}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling responses and errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request was made but no response received
      console.error('Network Error: No response received', error.request);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export const queryService = {
  // Verify backend availability
  async checkHealth() {
    try {
      const response = await apiClient.get('/api/v1/agent/health');
      return response.data;
    } catch (error) {
      throw formatError(error);
    }
  },

  // Submit a query to the RAG agent
  async submitQuery(queryData) {
    try {
      const response = await apiClient.post('/api/v1/agent/query', queryData);
      return response.data;
    } catch (error) {
      throw formatError(error);
    }
  },

  // Update agent configuration
  async updateConfiguration(configData) {
    try {
      const response = await apiClient.post('/api/v1/agent/query/configure', configData);
      return response.data;
    } catch (error) {
      throw formatError(error);
    }
  }
};

// Helper function to format errors consistently
function formatError(error) {
  if (error.response) {
    // Server responded with error status
    return new Error(
      error.response.data.detail ||
      `API Error: ${error.response.status} - ${error.response.statusText}`
    );
  } else if (error.request) {
    // Network error - no response received
    return new Error('Network Error: Unable to connect to the server. Please check your connection and try again.');
  } else {
    // Other error
    return new Error(error.message || 'An unexpected error occurred');
  }
}