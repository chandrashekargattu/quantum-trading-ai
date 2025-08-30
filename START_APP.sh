#!/bin/bash

# Quantum Trading AI - One-Click Start Script
# This script handles all setup and starts both frontend and backend

echo "🚀 Quantum Trading AI - Starting Application"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to kill processes on a port
kill_port() {
    local port=$1
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Killing process on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
kill_port 3000
kill_port 8000
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

# Setup Node.js environment
echo -e "\n${BLUE}📦 Setting up Node.js environment...${NC}"
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    
    # Use Node 18.20.0 or install it if not available
    if ! nvm ls 18.20.0 >/dev/null 2>&1; then
        echo "Installing Node.js 18.20.0..."
        nvm install 18.20.0
    fi
    nvm use 18.20.0
    echo -e "${GREEN}✓ Using Node.js $(node --version)${NC}"
else
    echo -e "${YELLOW}⚠️  NVM not found. Make sure you have Node.js 18.17.0+ installed${NC}"
fi

# Start Frontend
echo -e "\n${BLUE}🎨 Starting Frontend...${NC}"
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Clear Next.js cache for fresh start
rm -rf .next

# Start frontend in background
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
cd ..

# Start Backend
echo -e "\n${BLUE}🔧 Starting Backend...${NC}"
cd backend

# Set Python environment
if command_exists pyenv; then
    pyenv local 3.10.13 2>/dev/null || pyenv local 3.10 2>/dev/null || echo "Using system Python"
fi

# Install missing dependencies if needed
pip install bsedata >/dev/null 2>&1

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Start backend in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd ..

# Wait for services to start
echo -e "\n⏳ Waiting for services to start..."
sleep 8

# Check status
echo -e "\n${BLUE}📊 Status Check:${NC}"
echo "==============="

# Check Frontend
if curl -s -o /dev/null http://localhost:3000; then
    echo -e "${GREEN}✅ Frontend: http://localhost:3000${NC}"
    # Try to load the main page content
    if curl -s http://localhost:3000 | grep -q "AI-Powered Options Trading"; then
        echo -e "${GREEN}   ✓ Main page loaded successfully${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Main page may not be loading correctly${NC}"
        echo "   Check frontend.log for details"
    fi
else
    echo -e "${RED}❌ Frontend: Not responding${NC}"
    echo "   Last 5 lines of frontend.log:"
    tail -5 frontend.log 2>/dev/null | sed 's/^/     /' || echo "     No log file found"
fi

# Check Backend
if curl -s http://localhost:8000/docs | grep -q "FastAPI"; then
    echo -e "${GREEN}✅ Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}   API Docs: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}❌ Backend: Not responding${NC}"
    echo "   Last 5 lines of backend.log:"
    tail -5 backend.log 2>/dev/null | sed 's/^/     /' || echo "     No log file found"
fi

# Save PIDs for easy shutdown
echo "$FRONTEND_PID" > .frontend.pid
echo "$BACKEND_PID" > .backend.pid

echo -e "\n${BLUE}📝 Quick Commands:${NC}"
echo "=================="
echo "• View frontend logs: tail -f frontend.log"
echo "• View backend logs:  tail -f backend.log"
echo "• Stop all servers:   ./STOP_APP.sh"
echo "• Restart servers:    ./START_APP.sh"

echo -e "\n${GREEN}🎉 Application is ready!${NC}"
echo -e "Open your browser and go to: ${BLUE}http://localhost:3000${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop watching (servers will continue running)${NC}"

# Keep script running to show logs
trap "echo -e '\n${YELLOW}Exiting... (servers still running in background)${NC}'" INT TERM
tail -f frontend.log backend.log 2>/dev/null

