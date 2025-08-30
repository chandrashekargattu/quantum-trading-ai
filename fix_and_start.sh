#!/bin/bash

echo "🔧 Quantum Trading AI - Complete Setup and Start"
echo "============================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kill existing processes
echo -e "\n${YELLOW}🧹 Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 2

# Backend setup
echo -e "\n${YELLOW}📦 Setting up backend dependencies...${NC}"
cd backend

# Install all dependencies
echo "Installing backend dependencies..."
pip install openai textblob transformers huggingface-hub tokenizers==0.21.0 regex \
    bsedata kiteconnect nsetools torch torchvision opencv-python \
    rasterio geopandas shapely vaderSentiment arch copulas \
    sortedcontainers numba pymongo --upgrade --quiet

# Start backend
echo -e "\n${YELLOW}🚀 Starting backend server...${NC}"
export PYTHONPATH="$(pwd)"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend starting (PID: $BACKEND_PID)..."

# Frontend setup
echo -e "\n${YELLOW}🎨 Starting frontend server...${NC}"
cd ../frontend

# Ensure correct Node version
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use 18.20.0 2>/dev/null || nvm use 18 2>/dev/null || echo "Using system Node.js"
fi

# Start frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend starting (PID: $FRONTEND_PID)..."

# Wait for services
echo -e "\n${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 20

# Check status
echo -e "\n${YELLOW}📊 Status Check:${NC}"
echo "==============="

# Check Frontend
if curl -s -o /dev/null http://localhost:3000; then
    echo -e "${GREEN}✅ Frontend: http://localhost:3000${NC}"
else
    echo -e "${RED}❌ Frontend: Not responding${NC}"
    echo "   Last 10 lines of frontend.log:"
    tail -10 frontend/frontend.log | sed 's/^/     /'
fi

# Check Backend
if curl -s -o /dev/null http://localhost:8000/docs; then
    echo -e "${GREEN}✅ Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}   API Docs: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}❌ Backend: Not responding${NC}"
    echo "   Last 10 lines of backend.log:"
    tail -10 backend/backend.log | sed 's/^/     /'
fi

echo -e "\n${GREEN}✨ Setup complete!${NC}"
echo -e "\n📍 Access Points:"
echo -e "   Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "   Backend:   ${GREEN}http://localhost:8000${NC}"
echo -e "   API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Handle cleanup
trap "echo -e '\n🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep running
wait

