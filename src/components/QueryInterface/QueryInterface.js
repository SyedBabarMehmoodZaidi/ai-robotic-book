import React, { useState } from 'react';
import SelectedTextCapture from '../SelectedTextCapture/SelectedTextCapture';
import styles from './QueryInterface.module.css';

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedText, setSelectedText] = useState('');

  const handleTextSelected = (text) => {
    setSelectedText(text);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      // Use the global API client created in api-client.js
      const result = await window.RAGApiClient.query(query, selectedText);
      setResponse(result);
    } catch (err) {
      setError('Failed to get response from AI. Please try again.');
      console.error('Query error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.queryContainer}>
      {/* SelectedTextCapture component to capture selected text from the page */}
      <SelectedTextCapture onTextSelected={handleTextSelected} />

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