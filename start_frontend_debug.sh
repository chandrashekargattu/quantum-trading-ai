#!/bin/bash

echo "Starting Frontend Debug..."
echo "Current directory: $(pwd)"
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"

cd frontend 2>/dev/null || echo "Already in frontend directory"

echo ""
echo "Starting Next.js development server..."
echo ""

# Start the server and capture output
npx next dev --port 3000 2>&1 | tee frontend-debug.log &
PID=$!

echo "Frontend process started with PID: $PID"
echo ""
echo "Waiting for server to start..."
sleep 5

# Check if server is running
if curl -s -o /dev/null http://localhost:3000; then
    echo "✅ Frontend is running at http://localhost:3000"
else
    echo "❌ Frontend failed to start"
    echo ""
    echo "Last 20 lines of debug log:"
    tail -20 frontend-debug.log
fi

echo ""
echo "To stop the server, run: kill $PID"

