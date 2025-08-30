# 🚀 Quantum Trading AI - Complete Setup Guide

This guide will help you get both the frontend and backend running successfully.

## ✅ Current Status

- **Frontend**: Running successfully on http://localhost:3000
- **Backend**: Needs to be started with proper configuration

## 🔧 Quick Start Commands

### Option 1: Run Everything from Project Root (Recommended)

Open **TWO TERMINAL WINDOWS**:

#### Terminal 1 - Backend:
```bash
cd /Users/chandrashekargattu/quantum-trading-ai/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 - Frontend:
```bash
cd /Users/chandrashekargattu/quantum-trading-ai/frontend
npm run dev
```

### Option 2: Use the Quick Start Script

```bash
cd /Users/chandrashekargattu/quantum-trading-ai/backend
./quick_start.sh
```

## 🛠️ Troubleshooting

### If Backend Fails to Start:

1. **NumPy Compatibility Issue**:
```bash
cd backend
pip uninstall numpy -y
pip install 'numpy<2.0,>=1.23.5'
```

2. **Module Not Found Error**:
Make sure you're in the `backend` directory when running the server:
```bash
cd /Users/chandrashekargattu/quantum-trading-ai/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Missing Dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

### If Frontend Has Issues:

1. **Node Version**:
```bash
nvm use 18.20.8
```

2. **Reinstall Dependencies**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 📋 Complete Dependency List

The backend requires these key packages:
- fastapi
- uvicorn
- sqlalchemy
- pydantic
- python-jose[cryptography]
- passlib[bcrypt]
- numpy<2.0
- pandas
- yfinance
- And many more...

## 🎯 Verification

Once both servers are running, you should be able to:

1. Access the frontend at: http://localhost:3000
2. Access the backend API at: http://localhost:8000
3. View API documentation at: http://localhost:8000/docs

## 💡 Tips

1. Always run the backend from the `backend` directory
2. Keep both terminal windows open to see logs
3. The backend may take 10-20 seconds to fully start
4. If port 8000 is busy, kill the process: `lsof -ti :8000 | xargs kill -9`

## 🚨 Current Known Issues

1. Some advanced features (HFT, Alternative Data) are temporarily disabled due to complex dependencies
2. Quantum computing features are using mock implementations
3. Some ML models may need additional setup

## ✨ Next Steps

Once both servers are running:
1. Create a user account through the frontend
2. Explore the trading features
3. Check the API documentation for available endpoints
