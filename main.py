"""
MnemoGraph Server Entry Point

Run with: python main.py
Or with uvicorn: uvicorn main:app --reload
"""

import os

import uvicorn

if __name__ == "__main__":
    # Use reload only in development
    is_dev = os.getenv("ENVIRONMENT", "development") == "development"

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev,  # Only reload in development
        log_level="info",
    )
