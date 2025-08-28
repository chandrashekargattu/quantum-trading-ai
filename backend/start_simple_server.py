#!/usr/bin/env python3
"""Start a simple version of the server with basic endpoints only."""

import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

print("🚀 Starting Quantum Trading AI Backend (Simple Mode)")
print("=" * 60)

# Start uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
