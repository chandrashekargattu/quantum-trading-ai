#!/bin/bash

# Quantum Trading AI - Recovery Mode Startup Script
# This script helps you start your journey to recover losses and build wealth

echo "🚀 Welcome to Quantum Trading AI - Recovery Mode!"
echo "================================================"
echo ""
echo "📊 Starting your personal trading recovery system..."
echo ""

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ Detected macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Detected Linux"
else
    echo "⚠️  Warning: This script is optimized for macOS/Linux"
fi

# Check prerequisites
echo ""
echo "🔍 Checking prerequisites..."

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo "✓ Node.js installed: $NODE_VERSION"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python installed: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Redis
if command -v redis-cli &> /dev/null; then
    echo "✓ Redis installed"
else
    echo "⚠️  Redis not found. Installing via Docker..."
fi

echo ""
echo "📦 Installing dependencies..."

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install --silent
cd ..

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt --quiet
cd ..

echo ""
echo "🗄️  Setting up databases..."

# Start Redis if not running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    if command -v docker &> /dev/null; then
        docker run -d --name quantum-redis -p 6379:6379 redis:alpine > /dev/null 2>&1
        echo "✓ Redis started in Docker"
    else
        echo "⚠️  Please start Redis manually"
    fi
fi

# Initialize database
echo "Initializing database..."
cd backend
python -c "
import asyncio
from app.db.database import init_db

async def setup():
    await init_db()
    print('✓ Database initialized')

asyncio.run(setup())
" 2>/dev/null || echo "⚠️  Database already initialized"
cd ..

echo ""
echo "🚀 Starting services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "Starting backend API..."
cd backend
python main.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 5

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📱 Access Points:"
echo "   Frontend:  http://localhost:3000"
echo "   Recovery:  http://localhost:3000/zerodha-recovery"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📚 Next Steps:"
echo "   1. Open http://localhost:3000/zerodha-recovery"
echo "   2. Connect your Zerodha account"
echo "   3. Start with paper trading"
echo "   4. Read ZERODHA_RECOVERY_GUIDE.md"
echo ""
echo "💡 Tips:"
echo "   - Start with paper trading for at least 2 weeks"
echo "   - Complete the risk management course first"
echo "   - Follow the 30-day challenge in the guide"
echo "   - Join our Discord community for support"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Show live logs
echo "📋 Showing live logs (Ctrl+C to stop)..."
echo "=====================================\n"

# Wait and show logs
tail -f backend/backend.log frontend/frontend.log 2>/dev/null

# Keep script running
wait
