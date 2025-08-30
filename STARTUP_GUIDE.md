# 🚀 Quantum Trading AI - Startup Guide

## Current Status

The application has been successfully cleaned up and reorganized. Here's the current state:

### ✅ Frontend - Working
- Running on: http://localhost:3000
- Node.js version: 18.20.0 (via nvm)
- Status: **Fully operational**

### ⚠️ Backend - Partial (Dependency Issues)
- Port: 8000
- Issue: TensorFlow/ML library conflicts
- Solution: ML features temporarily disabled

## Quick Start

### Option 1: Use the Startup Script
```bash
./start_platform.sh
```

### Option 2: Manual Start

#### Frontend
```bash
cd frontend
nvm use 18.20.0
npm run dev
```

#### Backend (Minimal Mode)
```bash
cd backend
export PYTHONPATH="$(pwd)"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## What's Working

### Frontend ✅
- Dashboard
- Real-time charts
- Portfolio management
- Market data visualization
- Authentication UI

### Backend (Basic Features) ✅
- Authentication & Authorization
- User management
- Portfolio CRUD operations
- Basic market data
- Alerts system
- Trading strategies (basic)

### Temporarily Disabled ⚠️
- Quantum computing features
- Deep reinforcement learning
- Advanced ML predictions
- Zerodha integration
- High-frequency trading engine

## Fixing ML Dependencies

To re-enable ML features, you'll need to:

1. **Fix numpy compatibility**:
   ```bash
   pip install numpy==1.24.3
   ```

2. **Install TensorFlow properly**:
   ```bash
   pip uninstall tensorflow tensorflow-macos
   pip install tensorflow-macos==2.13.0
   ```

3. **Re-enable imports** in `backend/app/api/v1/api.py`:
   - Uncomment quantum, deep_rl, zerodha imports
   - Uncomment their router registrations

4. **Re-enable ModelManager** in `backend/app/main.py`:
   - Uncomment ModelManager import
   - Uncomment model initialization in lifespan

## Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Troubleshooting

### Frontend Issues
- Ensure Node.js 18.17+ is active: `nvm use 18.20.0`
- Clear cache: `rm -rf .next node_modules && npm install`

### Backend Issues
- Check logs: `tail -f backend/backend.log`
- Verify Python path: `echo $PYTHONPATH`
- Install missing deps: `pip install -r requirements.txt`

### Port Conflicts
```bash
# Kill processes on ports
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

## Clean Architecture

The project has been cleaned up:
- Removed 21+ redundant files
- Consolidated startup scripts
- Organized documentation
- Created clear separation of concerns

## Next Steps

1. **For Development**: Use the application with basic features
2. **For Production**: Fix ML dependencies and re-enable advanced features
3. **For Zerodha Integration**: Configure API keys in `.env` file

