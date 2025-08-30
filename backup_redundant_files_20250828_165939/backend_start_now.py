#!/usr/bin/env python3
"""
Simple server startup script for Quantum Trading AI Backend
"""
import os
import sys
import uvicorn

# Ensure we're in the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)

# Add backend to Python path
sys.path.insert(0, backend_dir)

print("🚀 Starting Quantum Trading AI Backend Server")
print("=" * 50)
print(f"📁 Working Directory: {backend_dir}")
print(f"🐍 Python Version: {sys.version.split()[0]}")
print("=" * 50)
print("")
print("🌐 Server will be available at:")
print("   - API: http://localhost:8000")
print("   - Docs: http://localhost:8000/docs")
print("")
print("Press Ctrl+C to stop the server")
print("=" * 50)

# Import and run the app
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
