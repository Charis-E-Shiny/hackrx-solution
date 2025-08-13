#!/usr/bin/env python3
"""
Start script for the HackRx LLM Query-Retrieval System
This ensures proper ASGI server startup
"""

import uvicorn
import os

if __name__ == "__main__":
    # Use environment variables for configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    
    # Start the FastAPI application with uvicorn
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )