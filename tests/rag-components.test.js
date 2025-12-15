// Comprehensive test suite for RAG components
import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import QueryInterface from '../src/components/QueryInterface/QueryInterface';
import SelectedTextCapture from '../src/components/SelectedTextCapture/SelectedTextCapture';
import RAGValidation from '../src/components/RAGValidation/RAGValidation';
import RAGApiClient from '../static/js/api-client.js';

// Mock the global RAGApiClient
global.window.RAGApiClient = {
  query: jest.fn(),
  healthCheck: jest.fn(),
  testConnection: jest.fn(),
};

global.window.RAGConfig = {
  REQUEST_TIMEOUT: 30000,
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000,
  MIN_QUERY_LENGTH: 3,
  MAX_QUERY_LENGTH: 2000,
  MIN_SELECTED_TEXT_LENGTH: 10,
  MAX_SELECTED_TEXT_LENGTH: 5000
};

// Mock fetch API
global.fetch = jest.fn();

describe('RAG Components Test Suite', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset fetch mock
    fetch.mockClear();
  });

  describe('QueryInterface Component', () => {
    test('renders correctly with initial state', () => {
      render(<QueryInterface />);

      expect(screen.getByPlaceholderText('Ask a question about this book...')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Ask AI/i })).toBeInTheDocument();
      expect(screen.getByRole('button')).toBeDisabled(); // Initially disabled due to empty query
    });

    test('enables submit button when query is entered', () => {
      render(<QueryInterface />);

      const queryInput = screen.getByPlaceholderText('Ask a question about this book...');
      fireEvent.change(queryInput, { target: { value: 'What is ROS 2?' } });

      const submitButton = screen.getByRole('button', { name: /Ask AI/i });
      expect(submitButton).not.toBeDisabled();
    });

    test('disables submit button when query is cleared', () => {
      render(<QueryInterface />);

      const queryInput = screen.getByPlaceholderText('Ask a question about this book...');
      fireEvent.change(queryInput, { target: { value: 'What is ROS 2?' } });

      let submitButton = screen.getByRole('button', { name: /Ask AI/i });
      expect(submitButton).not.toBeDisabled();

      fireEvent.change(queryInput, { target: { value: '' } });
      submitButton = screen.getByRole('button', { name: /Ask AI/i });
      expect(submitButton).toBeDisabled();
    });

    test('submits query and shows loading state', async () => {
      // Mock successful API response
      window.RAGApiClient.query.mockResolvedValue({
        response: {
          response_text: 'ROS 2 is a middleware framework...',
          source_context: ['module-1-ros2/index.md'],
          confidence_score: 0.92
        }
      });

      render(<QueryInterface />);

      const queryInput = screen.getByPlaceholderText('Ask a question about this book...');
      fireEvent.change(queryInput, { target: { value: 'What is ROS 2?' } });

      const submitButton = screen.getByRole('button', { name: /Ask AI/i });
      fireEvent.click(submitButton);

      // Check loading state
      expect(screen.getByText(/Asking AI\.\.\./i)).toBeInTheDocument();
      expect(screen.getByText(/AI is thinking\.\.\./i)).toBeInTheDocument();

      // Wait for response
      await waitFor(() => {
        expect(screen.getByText(/ROS 2 is a middleware framework\.\.\./i)).toBeInTheDocument();
      });
    });

    test('handles API error gracefully', async () => {
      // Mock API error
      window.RAGApiClient.query.mockRejectedValue(new Error('API Error'));

      render(<QueryInterface />);

      const queryInput = screen.getByPlaceholderText('Ask a question about this book...');
      fireEvent.change(queryInput, { target: { value: 'What is ROS 2?' } });

      const submitButton = screen.getByRole('button', { name: /Ask AI/i });
      fireEvent.click(submitButton);

      // Wait for error message
      await waitFor(() => {
        expect(screen.getByText(/Failed to get response from AI\. Please try again\./i)).toBeInTheDocument();
      });
    });

    test('displays response with source context and confidence', async () => {
      const mockResponse = {
        response: {
          response_text: 'Test response from AI',
          source_context: ['test-source.md', 'another-source.md'],
          confidence_score: 0.85
        }
      };

      window.RAGApiClient.query.mockResolvedValue(mockResponse);

      render(<QueryInterface />);

      const queryInput = screen.getByPlaceholderText('Ask a question about this book...');
      fireEvent.change(queryInput, { target: { value: 'Test query' } });

      const submitButton = screen.getByRole('button', { name: /Ask AI/i });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/Test response from AI/i)).toBeInTheDocument();
      });

      expect(screen.getByText(/Sources:/i)).toBeInTheDocument();
      expect(screen.getByText(/test-source\.md, another-source\.md/i)).toBeInTheDocument();
      expect(screen.getByText(/Confidence: 85\.0%/i)).toBeInTheDocument();
    });
  });

  describe('SelectedTextCapture Component', () => {
    let originalGetSelection;

    beforeEach(() => {
      originalGetSelection = window.getSelection;

      // Mock window.getSelection
      window.getSelection = jest.fn(() => ({
        toString: jest.fn(),
      }));
    });

    afterEach(() => {
      window.getSelection = originalGetSelection;
    });

    test('adds and removes event listeners', () => {
      const addEventListenerSpy = jest.spyOn(document, 'addEventListener');
      const removeEventListenerSpy = jest.spyOn(document, 'removeEventListener');

      const { unmount } = render(<SelectedTextCapture onTextSelected={jest.fn()} />);

      expect(addEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));
      expect(addEventListenerSpy).toHaveBeenCalledWith('keyup', expect.any(Function));

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('keyup', expect.any(Function));
    });

    test('calls onTextSelected when valid text is selected', () => {
      const mockOnTextSelected = jest.fn();
      const selectionText = 'This is a valid selection that meets the minimum length requirement';

      window.getSelection = jest.fn(() => ({
        toString: jest.fn(() => selectionText),
      }));

      render(<SelectedTextCapture onTextSelected={mockOnTextSelected} />);

      // Simulate a mouseup event
      fireEvent.mouseUp(document);

      expect(mockOnTextSelected).toHaveBeenCalledWith(selectionText);
    });

    test('does not call onTextSelected for short text', () => {
      const mockOnTextSelected = jest.fn();
      const shortSelection = 'Hi'; // Less than minimum length

      window.getSelection = jest.fn(() => ({
        toString: jest.fn(() => shortSelection),
      }));

      render(<SelectedTextCapture onTextSelected={mockOnTextSelected} />);

      // Simulate a mouseup event
      fireEvent.mouseUp(document);

      expect(mockOnTextSelected).not.toHaveBeenCalled();
    });

    test('does not call onTextSelected for empty text', () => {
      const mockOnTextSelected = jest.fn();
      const emptySelection = '';

      window.getSelection = jest.fn(() => ({
        toString: jest.fn(() => emptySelection),
      }));

      render(<SelectedTextCapture onTextSelected={mockOnTextSelected} />);

      // Simulate a mouseup event
      fireEvent.mouseUp(document);

      expect(mockOnTextSelected).not.toHaveBeenCalled();
    });
  });

  describe('RAGApiClient', () => {
    let apiClient;

    beforeEach(() => {
      apiClient = new RAGApiClient('http://test-api:8000');
    });

    test('constructor sets baseURL and config correctly', () => {
      expect(apiClient.baseURL).toBe('http://test-api:8000');
      expect(apiClient.config).toBeDefined();
    });

    test('query method validates input length', async () => {
      await expect(apiClient.query('')).rejects.toThrow('Query must be at least 3 characters long');
      await expect(apiClient.query('ab')).rejects.toThrow('Query must be at least 3 characters long');
      await expect(apiClient.query('a'.repeat(2001))).rejects.toThrow('Query must not exceed 2000 characters');
    });

    test('query method validates selected text length', async () => {
      await expect(apiClient.query('valid query', 'sh')).rejects.toThrow('Selected text must be at least 10 characters long');
      await expect(apiClient.query('valid query', 'a'.repeat(5001))).rejects.toThrow('Selected text must not exceed 5000 characters');
    });

    test('query method makes correct API call', async () => {
      const mockResponse = {
        response: {
          response_text: 'Test response',
          source_context: ['test.md'],
          confidence_score: 0.9
        }
      };

      fetch.mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await apiClient.query('Test query');

      expect(fetch).toHaveBeenCalledWith(
        'http://test-api:8000/query',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query_text: 'Test query',
            selected_text: null
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('healthCheck method makes correct API call', async () => {
      const mockHealthResponse = { status: 'healthy', version: '1.0.0' };

      fetch.mockResolvedValue({
        ok: true,
        json: async () => mockHealthResponse,
      });

      const result = await apiClient.healthCheck();

      expect(fetch).toHaveBeenCalledWith('http://test-api:8000/health',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        })
      );

      expect(result).toEqual(mockHealthResponse);
    });

    test('handles network errors gracefully', async () => {
      fetch.mockRejectedValue(new Error('Network error'));

      await expect(apiClient.query('Test query')).rejects.toThrow('Query request failed: Network error');
    });

    test('handles HTTP error responses', async () => {
      fetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      await expect(apiClient.query('Test query')).rejects.toThrow('API request failed: 500 Internal Server Error');
    });
  });

  describe('RAGValidation Component', () => {
    test('renders with initial state', () => {
      render(<RAGValidation />);

      expect(screen.getByText(/RAG System Validation/i)).toBeInTheDocument();
      expect(screen.getByText(/Total Tests/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Run All Validations/i })).toBeInTheDocument();
    });

    test('initial stats show 0 tests', () => {
      render(<RAGValidation />);

      expect(screen.getByText('0')).toBeInTheDocument(); // Total tests
      expect(screen.getByText('0%')).toBeInTheDocument(); // Success rate
    });

    test('run all validations button exists', () => {
      render(<RAGValidation />);

      const runButton = screen.getByRole('button', { name: /Run All Validations/i });
      expect(runButton).toBeInTheDocument();
      expect(runButton).not.toBeDisabled();
    });
  });

  describe('Integration Tests', () => {
    test('QueryInterface properly integrates with SelectedTextCapture', () => {
      render(
        <div>
          <SelectedTextCapture onTextSelected={jest.fn()} />
          <QueryInterface />
        </div>
      );

      // Both components should render without errors
      expect(screen.getByPlaceholderText('Ask a question about this book...')).toBeInTheDocument();
    });
  });
});

// Additional helper functions for testing
const createMockResponse = (data) => ({
  ok: true,
  json: async () => data,
});

const createMockErrorResponse = (status = 500, statusText = 'Internal Server Error') => ({
  ok: false,
  status,
  statusText,
});

// Export for potential use in other test files
export {
  createMockResponse,
  createMockErrorResponse
};