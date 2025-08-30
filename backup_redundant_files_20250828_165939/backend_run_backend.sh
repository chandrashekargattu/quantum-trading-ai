#!/bin/bash

echo "🚀 Starting Quantum Trading AI Backend"
echo "===================================="

# Kill any existing process on port 8000
echo "🔧 Cleaning up port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null

# Ensure we're in the right directory
cd "$(dirname "$0")"
echo "📁 Working from: $(pwd)"

# Start the server
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API Docs will be at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
