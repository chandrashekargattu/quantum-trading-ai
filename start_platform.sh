#!/bin/bash

echo "🚀 Starting Quantum Trading AI Platform"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
sleep 2

# Start Frontend
echo ""
echo "🎨 Starting Frontend..."
cd frontend
# Ensure we're using the right Node version
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use 18.20.0 2>/dev/null || nvm use 18 2>/dev/null || echo "   Using system Node.js"
fi
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend starting (PID: $FRONTEND_PID)..."

# Start Backend
echo ""
echo "🔧 Starting Backend..."
cd ../backend

# Ensure bsedata is installed
if ! pip show bsedata >/dev/null 2>&1; then
    echo "   Installing bsedata dependency..."
    pip install bsedata >/dev/null 2>&1
fi

# Set Python path properly
export PYTHONPATH="$(pwd)"
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
    echo -e "${GREEN}✅ Frontend: http://localhost:3000${NC}"
else
    echo -e "${RED}❌ Frontend: Not responding${NC}"
    echo "   Check frontend/frontend.log for errors"
    echo "   Last 5 lines of frontend.log:"
    tail -5 frontend/frontend.log | sed 's/^/     /'
fi

# Check Backend
if curl -s -o /dev/null http://localhost:8000/docs; then
    echo -e "${GREEN}✅ Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}   API Docs: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}❌ Backend: Not responding${NC}"
    echo "   Check backend/backend.log for errors"
    echo "   Last 5 lines of backend.log:"
    tail -5 backend/backend.log | sed 's/^/     /'
fi

echo ""
echo "📝 Logs are available at:"
echo "   - frontend/frontend.log"
echo "   - backend/backend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"

# Handle cleanup on exit
trap "echo -e '\n🛑 Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
