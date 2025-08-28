#!/bin/bash

echo "🚀 Starting Quantum Trading AI Backend Server"
echo "============================================"
echo ""
echo "📦 Installing any missing dependencies..."
pip install -q fastapi uvicorn sqlalchemy pydantic pydantic-settings httpx \
    python-jose[cryptography] passlib[bcrypt] python-multipart \
    aiosqlite email-validator aiohttp redis celery \
    yfinance pandas numpy alpha-vantage ta websockets python-socketio

echo ""
echo "🔧 Starting server..."
echo ""

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
