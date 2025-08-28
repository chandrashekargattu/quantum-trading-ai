#!/usr/bin/env python3
"""
Start the Quantum Trading AI Backend Server
"""
import os
import sys
import subprocess

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

print("🚀 Starting Quantum Trading AI Backend Server")
print("=" * 50)
print(f"📁 Working Directory: {backend_dir}")
print(f"🐍 Python Version: {sys.version.split()[0]}")
print("=" * 50)

# Start the server
try:
    # Run uvicorn directly
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ])
except KeyboardInterrupt:
    print("\n⏹️  Server stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
