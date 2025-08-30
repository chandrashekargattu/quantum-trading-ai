# 🚀 Quick Start Guide - Quantum Trading AI

## Getting Started in 3 Minutes

### Step 1: Start the Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
Backend will be available at: http://localhost:8000

### Step 2: Start the Frontend (in a new terminal)
```bash
cd frontend
npm install  # First time only
npm run dev
```
Frontend will be available at: http://localhost:3000

### Step 3: Access the Application
Open your browser and go to: **http://localhost:3000**

## 🎯 First Steps

### 1. Create Your Account
- Click "Sign Up" on the homepage
- Enter your email and create a password
- You'll start with $100,000 in paper trading money

### 2. Explore Features
- **Dashboard**: View your portfolio and P&L
- **Market Data**: Real-time stock and options data
- **Trading**: Place trades (paper trading by default)
- **Strategies**: Use AI-powered trading strategies
- **Backtesting**: Test strategies on historical data
- **Risk Analysis**: Monitor your portfolio risk

### 3. For Zerodha Users (Loss Recovery)
If you have a Zerodha account and want to recover losses:

1. Go to **Zerodha Recovery** section
2. Click "Connect Zerodha Account"
3. Log in with your Zerodha credentials
4. The AI will analyze your loss patterns
5. Follow the personalized recovery plan

## 🔥 Key Features to Try

### 1. AI Trading Signals
- Go to "Trading Signals"
- Select a stock (e.g., RELIANCE, TCS)
- View AI predictions and recommendations

### 2. Options Strategy Builder
- Navigate to "Options Trading"
- Choose a strategy (Bull Call Spread, Iron Condor, etc.)
- See max profit/loss and breakeven points

### 3. Backtesting
- Go to "Backtesting"
- Select a strategy
- Choose date range and initial capital
- Run backtest and analyze results

### 4. Risk Management
- View "Risk Dashboard"
- Check your portfolio Greeks
- Monitor Value at Risk (VaR)
- Set stop losses automatically

## 🇮🇳 Indian Market Features

- **NSE/BSE Data**: Real-time Indian market data
- **NIFTY/BANKNIFTY**: Options strategies
- **F&O Analysis**: Futures and Options analytics
- **Indian Sentiment**: MoneyControl, ET news analysis

## ⚡ Quick Commands

### Start Everything at Once
```bash
# From project root
./start_recovery.sh
```

### Run Tests
```bash
cd backend
python simple_test_runner.py
```

### Check API Documentation
Visit: http://localhost:8000/docs

## 🆘 Troubleshooting

### If Frontend Shows Black Screen
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### If Backend Won't Start
```bash
cd backend
pip install -r requirements.txt
```

### If You See Import Errors
```bash
cd backend
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 📱 Mobile Access
The application is responsive and works on mobile devices. Just access the same URL from your phone/tablet.

## 🎓 Learning Resources

1. **Video Tutorials**: Coming soon
2. **API Docs**: http://localhost:8000/docs
3. **Recovery Guide**: See ZERODHA_RECOVERY_GUIDE.md

## 💡 Pro Tips

1. **Start with Paper Trading**: Test strategies without real money
2. **Use Stop Losses**: Always set stop losses on trades
3. **Monitor Risk**: Keep position sizes under 5% of portfolio
4. **Follow AI Signals**: The AI learns from your patterns
5. **Daily Review**: Check the recovery dashboard daily

## 🚨 Important Notes

- **Paper Trading First**: Always test in paper mode before real trading
- **Risk Management**: Never risk more than 2% per trade
- **Market Hours**: Indian markets: 9:15 AM - 3:30 PM IST
- **Options Expiry**: Indian options expire on Thursday

## Need Help?

- Check the logs in `backend/logs/` for errors
- Review test results with `python simple_test_runner.py`
- See integration guide: INTEGRATION_GUIDE.md

---

**Ready to start trading smarter? Let's go! 🚀**
