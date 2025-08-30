#!/bin/bash

# Stop script for Quantum Trading AI

echo "🛑 Stopping Quantum Trading AI..."

# Read PIDs if available
if [ -f .frontend.pid ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo "✓ Stopped frontend (PID: $FRONTEND_PID)"
    fi
    rm .frontend.pid
fi

if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo "✓ Stopped backend (PID: $BACKEND_PID)"
    fi
    rm .backend.pid
fi

# Also kill by port just to be sure
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "✅ All services stopped"

