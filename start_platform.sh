#!/bin/bash

echo "🚀 Starting Quantum Trading AI Platform"
echo "======================================"
echo ""

# Function to check if a port is in use
check_port() {
    lsof -i :$1 >/dev/null 2>&1
}

# Kill any existing processes
echo "🔧 Cleaning up existing processes..."
if check_port 3000; then
    echo "   Killing process on port 3000..."
    lsof -ti :3000 | xargs kill -9 2>/dev/null
fi
if check_port 8000; then
    echo "   Killing process on port 8000..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null
fi

# Start Frontend
echo ""
echo "🎨 Starting Frontend..."
cd frontend
# Ensure we're using the right Node version
source ~/.nvm/nvm.sh
nvm use 18 >/dev/null 2>&1
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend starting (PID: $FRONTEND_PID)..."

# Start Backend
echo ""
echo "🔧 Starting Backend..."
cd ../backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend starting (PID: $BACKEND_PID)..."

# Wait and check status
echo ""
echo "⏳ Waiting for servers to start..."
sleep 10

echo ""
echo "📊 Status Check:"
echo "==============="

# Check Frontend
if curl -s -o /dev/null http://localhost:3000; then
    echo "✅ Frontend: http://localhost:3000"
else
    echo "❌ Frontend: Not responding"
    echo "   Check frontend.log for errors"
fi

# Check Backend
if curl -s -o /dev/null http://localhost:8000; then
    echo "✅ Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
else
    echo "❌ Backend: Not responding"
    echo "   Check backend.log for errors"
fi

echo ""
echo "📝 Logs are available at:"
echo "   - frontend/frontend.log"
echo "   - backend/backend.log"
echo ""
echo "Press Ctrl+C to stop both servers"

# Keep script running
wait
