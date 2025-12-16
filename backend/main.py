"""
Main entry point for the RAG Retrieval Service.
"""
import uvicorn
from src.api.main import app
from src.config.settings import settings


def run_server():
    """Run the FastAPI server."""
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"API available at: http://localhost:8000")
    print(f"Docs available at: http://localhost:8000/docs")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        debug=settings.debug
    )


if __name__ == "__main__":
    run_server()