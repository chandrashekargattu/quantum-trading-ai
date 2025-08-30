# 🚀 Start Your Quantum Trading AI Platform

## ✅ Frontend Status
Your frontend is now working perfectly! The dark theme issue has been fixed and you now have:
- Light mode as default
- Theme toggle button in the top-right corner
- Beautiful UI ready for trading

## 🔧 Backend Setup

The backend needs to be started from the correct directory. Here's how:

### Option 1: Simple Command (Recommended)
Open a new terminal and run:
```bash
cd /Users/chandrashekargattu/quantum-trading-ai/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Using the Script
```bash
cd /Users/chandrashekargattu/quantum-trading-ai/backend
./start_backend.sh
```

### Common Issues & Solutions:

**Issue: "ModuleNotFoundError: No module named 'app'"**
- Solution: Make sure you're in the `backend` directory, not the project root

**Issue: "Address already in use"**
- Solution: Kill any existing process on port 8000:
```bash
lsof -ti :8000 | xargs kill -9
```

**Issue: Missing dependencies**
- Solution: Install them from the backend directory:
```bash
cd backend
pip install -r requirements.txt
```

## 📍 Access Points

Once both servers are running:
- **Frontend**: http://localhost:3000 ✅ (Already running!)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 💰 Start Trading

1. **Create an Account**: Register on the frontend
2. **Paper Trade First**: Test strategies without real money
3. **Follow the Guides**:
   - `TRADING_GUIDE.md` - Complete platform overview
   - `DAILY_TRADING_PLAYBOOK.md` - Step-by-step daily routine

## 🎯 Quick Trading Tips

1. **Start Small**: Target 0.5-1% daily returns
2. **Use AI Signals**: Follow >80% confidence predictions
3. **Risk Management**: Never risk more than 2% per trade
4. **Take Profits**: Close winners at 50% of max profit

The platform is designed to help you make consistent daily profits through:
- AI-powered price predictions
- Automated options strategies
- Real-time risk management
- Portfolio optimization

**Happy Trading! 📈**
