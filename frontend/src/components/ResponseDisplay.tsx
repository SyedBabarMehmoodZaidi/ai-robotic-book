import React from 'react';
import { AgentResponse, RetrievedChunk } from '../types/queryTypes';

interface ResponseDisplayProps {
  response: AgentResponse | null;
  isLoading: boolean;
  error: string | null;
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({ response, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="response-display p-6 bg-gray-50 rounded-lg">
        <div className="flex flex-col items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-lg text-gray-600">Processing your query...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a moment</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="response-display p-6 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="response-display p-6 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <p className="text-center text-gray-500">Your response will appear here</p>
      </div>
    );
  }

  return (
    <div className="response-display bg-white rounded-lg shadow">
      <div className="p-6">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Response</h2>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
              Confidence: {(response.confidence_score * 100).toFixed(1)}%
            </span>
          </div>

          <div className="prose prose-blue max-w-none">
            <div className="text-gray-800 whitespace-pre-wrap">{response.response_text}</div>
          </div>
        </div>

        {response.retrieved_chunks && response.retrieved_chunks.length > 0 && (
          <div className="chunks-section mt-8">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Retrieved Context</h3>
            <div className="space-y-4">
              {response.retrieved_chunks.map((chunk: RetrievedChunk, index: number) => (
                <div key={chunk.chunk_id || index} className="chunk-item bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-gray-700">Source {index + 1}</span>
                    <span className="text-xs font-medium bg-gray-200 text-gray-800 px-2 py-1 rounded">
                      Score: {(chunk.similarity_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-gray-700 whitespace-pre-wrap">{chunk.content}</p>
                  {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                    <div className="mt-2 text-xs text-gray-500">
                      {Object.entries(chunk.metadata).map(([key, value]) => (
                        <span key={key} className="mr-3">
                          {key}: {String(value)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {response.sources && response.sources.length > 0 && (
          <div className="sources-section mt-6 pt-6 border-t border-gray-200">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Sources</h4>
            <div className="flex flex-wrap gap-2">
              {response.sources.map((source, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {source}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResponseDisplay;