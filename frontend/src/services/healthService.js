import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export const healthService = {
  /**
   * Check the health status of the RAG agent backend
   * @returns {Promise<Object>} Health status response
   */
  async checkHealth() {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/v1/agent/health`);
      return response.data;
    } catch (error) {
      // If there's an error, we still want to return a structured response
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.detail || 'Health check failed');
      } else {
        throw new Error('Unable to connect to backend service');
      }
    }
  },

  /**
   * Continuously check health until the service is ready or timeout is reached
   * @param {number} maxAttempts - Maximum number of attempts (default: 30)
   * @param {number} intervalMs - Interval between attempts in milliseconds (default: 1000)
   * @returns {Promise<boolean>} True if service becomes healthy, false if timeout
   */
  async waitForService(maxAttempts = 30, intervalMs = 1000) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const health = await this.checkHealth();
        if (health.status === 'healthy') {
          return true;
        }
      } catch (error) {
        // Service not available yet, continue waiting
      }

      // Wait for the specified interval before next attempt
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
    return false; // Timeout reached
  }
};