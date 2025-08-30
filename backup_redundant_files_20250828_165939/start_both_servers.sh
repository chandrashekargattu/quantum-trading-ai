#!/bin/bash
# Start both backend and frontend servers

echo "🚀 Starting Quantum Trading AI Servers..."

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 2

# Start Backend
echo -e "\n📡 Starting Backend Server..."
cd backend
python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 5

# Use correct Node version and start Frontend
echo -e "\n🎨 Starting Frontend Server..."
source ~/.nvm/nvm.sh
nvm use 18.20.0
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for servers to start
echo -e "\n⏳ Waiting for servers to start..."
sleep 10

# Check status
echo -e "\n✅ Server Status:"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

echo -e "\n📍 Access Points:"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"

echo -e "\n🛑 To stop servers, run:"
echo "pkill -f uvicorn"
echo "pkill -f 'next dev'"
