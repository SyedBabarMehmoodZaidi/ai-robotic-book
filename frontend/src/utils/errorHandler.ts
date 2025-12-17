// Centralized error handling utility for the application

export interface ApplicationError {
  type: 'network' | 'validation' | 'server' | 'unknown';
  message: string;
  code?: string;
  details?: any;
}

export class ErrorHandler {
  /**
   * Format an error for consistent application-wide handling
   */
  static formatError(error: any): ApplicationError {
    if (error.type === 'ApplicationError') {
      return error;
    }

    // Network errors
    if (error.message && error.message.includes('Network Error')) {
      return {
        type: 'network',
        message: 'Unable to connect to the server. Please check your internet connection and try again.',
        code: 'NETWORK_ERROR'
      };
    }

    // Axios errors with response
    if (error.response) {
      const status = error.response.status;
      const detail = error.response.data?.detail || error.response.data?.message;

      if (status >= 500) {
        return {
          type: 'server',
          message: detail || 'Server error occurred. Please try again later.',
          code: 'SERVER_ERROR',
          details: error.response.data
        };
      } else if (status >= 400) {
        return {
          type: 'server',
          message: detail || `Request failed with status ${status}`,
          code: `HTTP_${status}`,
          details: error.response.data
        };
      }
    }

    // Validation errors
    if (error.message && error.message.includes('Validation')) {
      return {
        type: 'validation',
        message: error.message,
        code: 'VALIDATION_ERROR'
      };
    }

    // General errors
    return {
      type: 'unknown',
      message: error.message || 'An unexpected error occurred',
      code: 'UNKNOWN_ERROR',
      details: error
    };
  }

  /**
   * Log error with additional context
   */
  static logError(error: ApplicationError, context?: string): void {
    console.group(`%cError: ${error.message}`, 'color: #ff0000; font-weight: bold;');
    console.log('Type:', error.type);
    console.log('Code:', error.code);
    if (context) {
      console.log('Context:', context);
    }
    if (error.details) {
      console.log('Details:', error.details);
    }
    console.trace();
    console.groupEnd();
  }

  /**
   * Display user-friendly error message
   */
  static getUserFriendlyMessage(error: ApplicationError): string {
    switch (error.type) {
      case 'network':
        return error.message;
      case 'server':
        if (error.code === 'HTTP_429') {
          return 'Too many requests. Please wait a moment and try again.';
        }
        return 'The service is temporarily unavailable. Please try again later.';
      case 'validation':
        return `Invalid input: ${error.message}`;
      default:
        return 'An unexpected error occurred. Please try again.';
    }
  }

  /**
   * Handle error in a component-friendly way
   */
  static async handle(error: any, context?: string): Promise<ApplicationError> {
    const formattedError = this.formatError(error);
    this.logError(formattedError, context);
    return formattedError;
  }
}

export default ErrorHandler;