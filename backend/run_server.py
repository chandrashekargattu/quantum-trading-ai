#!/usr/bin/env python3
"""Simple script to run the Quantum Trading AI backend server."""

import uvicorn
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        print("🚀 Starting Quantum Trading AI Backend Server...")
        print("=" * 50)
        
        # Import the app
        from app.main import app
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
