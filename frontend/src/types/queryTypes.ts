// TypeScript interfaces for query data based on the RAG agent API contract

export interface AgentQuery {
  query_text: string;
  context_text?: string;
  query_id?: string;
  created_at?: string;
  user_id?: string;
  query_type?: 'general' | 'context-specific';
  [key: string]: any; // Allow additional properties
}

export interface RetrievedChunk {
  content: string;
  similarity_score: number;
  chunk_id: string;
  metadata: {
    [key: string]: any;
  };
  position: number;
}

export interface AgentResponse {
  response_text: string;
  query_id: string;
  retrieved_chunks: RetrievedChunk[];
  confidence_score: number;
  response_id?: string;
  created_at?: string;
  sources?: string[];
  metadata?: {
    [key: string]: any;
  };
}

export interface AgentConfiguration {
  model_name?: string;
  temperature?: number;
  max_tokens?: number;
  retrieval_threshold?: number;
  context_window?: number;
  [key: string]: any; // Allow additional properties
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  message: string;
  timestamp: string;
  services?: {
    retrieval?: {
      status: 'healthy' | 'unhealthy';
      details?: string;
    };
    rag_agent?: {
      status: 'healthy' | 'unhealthy';
      details?: string;
    };
  };
}

export interface ErrorResponse {
  detail: string;
}

export interface ConfigurationResponse {
  status: 'success' | 'error';
  message: string;
  config: AgentConfiguration;
}