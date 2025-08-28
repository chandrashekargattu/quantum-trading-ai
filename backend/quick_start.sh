#!/bin/bash

echo "🚀 Quick Start - Quantum Trading AI Backend"
echo "=========================================="

# Ensure we're in the backend directory
cd "$(dirname "$0")"

# Export Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "📁 Working directory: ${PWD}"
echo "🐍 Python path includes: ${PWD}"

# Quick numpy fix
echo ""
echo "🔧 Fixing numpy compatibility..."
pip uninstall numpy -y > /dev/null 2>&1
pip install 'numpy<2.0,>=1.23.5' --quiet

# Start the server
echo ""
echo "🚀 Starting server..."
echo "Backend URL: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
