// API client for communicating with RAG backend
class RAGApiClient {
  constructor(baseURL = null) {
    // Use config if available, otherwise default
    this.baseURL = baseURL || (typeof window.RAGConfig !== 'undefined' ? window.RAGConfig.BACKEND_API_URL : 'http://localhost:8000');
    this.config = typeof window.RAGConfig !== 'undefined' ? window.RAGConfig : {
      REQUEST_TIMEOUT: 30000,
      MAX_RETRIES: 2,
      RETRY_DELAY: 1000
    };
  }

  /**
   * Helper function to create a timeout promise
   */
  timeoutPromise(timeoutMs) {
    return new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Request timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });
  }

  /**
   * Helper function to delay execution
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Main request function with timeout and error handling
   */
  async makeRequest(url, options = {}) {
    const timeoutMs = this.config.REQUEST_TIMEOUT || 30000;

    // Create promises for fetch and timeout
    const fetchPromise = fetch(url, options);
    const timeoutPromiseObj = this.timeoutPromise(timeoutMs);

    try {
      // Race between fetch and timeout
      const response = await Promise.race([fetchPromise, timeoutPromiseObj]);

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      return response;
    } catch (error) {
      // If it's a timeout error, throw it directly
      if (error.message.includes('timed out')) {
        throw error;
      }
      // Re-throw other errors as well
      throw error;
    }
  }

  /**
   * Function to execute request with retries
   */
  async makeRequestWithRetry(url, options = {}, maxRetries = this.config.MAX_RETRIES || 2) {
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          // Wait before retrying (exponential backoff could be implemented here)
          await this.delay(this.config.RETRY_DELAY || 1000);
        }

        const response = await this.makeRequest(url, options);
        return response; // Success, return response
      } catch (error) {
        lastError = error;

        // If this was the last attempt, throw the error
        if (attempt === maxRetries) {
          break;
        }

        console.warn(`Request failed (attempt ${attempt + 1}/${maxRetries + 1}):`, error.message);
      }
    }

    // Throw the last error after all retries are exhausted
    throw lastError;
  }

  /**
   * Query the RAG backend with the provided query and selected text
   */
  async query(queryText, selectedText = null) {
    // Validate input lengths based on config
    const minQueryLength = this.config.MIN_QUERY_LENGTH || 3;
    const maxQueryLength = this.config.MAX_QUERY_LENGTH || 2000;
    const minSelectedTextLength = this.config.MIN_SELECTED_TEXT_LENGTH || 10;
    const maxSelectedTextLength = this.config.MAX_SELECTED_TEXT_LENGTH || 5000;

    if (!queryText || queryText.trim().length < minQueryLength) {
      throw new Error(`Query must be at least ${minQueryLength} characters long`);
    }

    if (queryText.length > maxQueryLength) {
      throw new Error(`Query must not exceed ${maxQueryLength} characters`);
    }

    if (selectedText && selectedText.length > 0) {
      if (selectedText.length < minSelectedTextLength) {
        throw new Error(`Selected text must be at least ${minSelectedTextLength} characters long`);
      }

      if (selectedText.length > maxSelectedTextLength) {
        throw new Error(`Selected text must not exceed ${maxSelectedTextLength} characters`);
      }
    }

    const url = `${this.baseURL}/chat`;
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query_text: queryText,
        selected_text: selectedText
      })
    };

    try {
      const response = await this.makeRequestWithRetry(url, options);
      return response.json();
    } catch (error) {
      // Enhanced error reporting
      console.error('Query request failed:', error);
      throw new Error(`Query request failed: ${error.message}`);
    }
  }

  /**
   * Health check to verify backend connectivity
   */
  async healthCheck() {
    const url = `${this.baseURL}/health`;
    const options = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    };

    try {
      const response = await this.makeRequestWithRetry(url, options);
      return response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  /**
   * Test connectivity to the backend
   */
  async testConnection() {
    try {
      const health = await this.healthCheck();
      return {
        connected: true,
        health: health
      };
    } catch (error) {
      return {
        connected: false,
        error: error.message
      };
    }
  }
}

// Export singleton instance
window.RAGApiClient = new RAGApiClient();