#!/bin/bash

# Quantum Trading AI - Server Startup Script

echo "🚀 Starting Quantum Trading AI Servers..."
echo "========================================"

# Start backend server
echo "📡 Starting Backend Server (FastAPI)..."
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend server
echo "🌐 Starting Frontend Server (Next.js)..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to start
sleep 5

echo ""
echo "✅ Servers Started Successfully!"
echo "================================"
echo "📊 Backend API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🌐 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to handle shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup INT

# Keep script running
while true; do
    sleep 1
done
