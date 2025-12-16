"""
Main entry point for the RAG Qdrant pipeline application.
"""
import uvicorn
from src.api.main import app


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    run_server()