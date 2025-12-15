// Simple script to verify RAG backend connectivity
async function verifyBackendConnection() {
  try {
    const apiClient = new window.RAGApiClient || new (await import('./api-client.js')).RAGApiClient;
    const health = await apiClient.healthCheck();
    console.log('Backend connection successful:', health);
    return true;
  } catch (error) {
    console.error('Backend connection failed:', error);
    return false;
  }
}

// Run verification
verifyBackendConnection().then(success => {
  if (success) {
    console.log('✓ RAG backend from Spec-3 is accessible locally');
  } else {
    console.log('⚠ RAG backend may not be running. Please start the backend from Spec-3: cd backend/rag_agent && uvicorn main:app --reload --port 8000');
  }
});

export { verifyBackendConnection };