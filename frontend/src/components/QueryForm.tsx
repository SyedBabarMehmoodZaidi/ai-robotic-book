import React, { useState } from 'react';
import { AgentQuery } from '../types/queryTypes';

interface QueryFormProps {
  onSubmit: (queryData: AgentQuery) => void;
  isLoading: boolean;
}

const QueryForm: React.FC<QueryFormProps> = ({ onSubmit, isLoading }) => {
  const [queryText, setQueryText] = useState('');
  const [queryType, setQueryType] = useState<'general' | 'context-specific'>('general');
  const [contextText, setContextText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!queryText.trim()) {
      alert('Please enter a query');
      return;
    }

    const queryData: AgentQuery = {
      query_text: queryText,
      query_type: queryType,
      ...(contextText && { context_text: contextText }),
      user_id: 'user-123', // In a real app, this would come from auth
      query_id: `query-${Date.now()}`, // Generate a unique ID
      created_at: new Date().toISOString()
    };

    onSubmit(queryData);
  };

  return (
    <form onSubmit={handleSubmit} className="query-form space-y-4">
      <div className="form-group">
        <label htmlFor="queryType" className="block text-sm font-medium text-gray-700 mb-1">
          Query Type
        </label>
        <select
          id="queryType"
          value={queryType}
          onChange={(e) => setQueryType(e.target.value as 'general' | 'context-specific')}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="general">General Query</option>
          <option value="context-specific">Context-Specific Query</option>
        </select>
        <p className="mt-1 text-sm text-gray-500">
          {queryType === 'general'
            ? 'Ask a general question about the book content'
            : 'Ask a question with specific text context'}
        </p>
      </div>

      {queryType === 'context-specific' && (
        <div className="form-group">
          <label htmlFor="contextText" className="block text-sm font-medium text-gray-700 mb-1">
            Context Text (Optional)
          </label>
          <textarea
            id="contextText"
            value={contextText}
            onChange={(e) => setContextText(e.target.value)}
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Paste or type the text context here..."
          />
          <p className="mt-1 text-sm text-gray-500">
            Provide specific text that your question relates to
          </p>
        </div>
      )}

      <div className="form-group">
        <label htmlFor="queryText" className="block text-sm font-medium text-gray-700 mb-1">
          Your Query
        </label>
        <textarea
          id="queryText"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          placeholder="Ask your question about the book content..."
          disabled={isLoading}
        />
        <div className="mt-1 text-sm text-gray-500 flex justify-between">
          <span>{queryText.length}/2000 characters</span>
          <span>Required</span>
        </div>
      </div>

      <div className="form-actions">
        <button
          type="submit"
          disabled={isLoading || !queryText.trim()}
          className={`w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-base font-medium text-white ${
            isLoading || !queryText.trim()
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
          }`}
        >
          {isLoading ? 'Processing...' : 'Submit Query'}
        </button>
      </div>
    </form>
  );
};

export default QueryForm;