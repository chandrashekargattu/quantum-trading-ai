# 🔧 Complete Integration Guide - Quantum Trading AI

This guide will walk you through integrating all components of your loss recovery system.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Zerodha Account Setup](#zerodha-account-setup)
4. [Database Configuration](#database-configuration)
5. [Starting the Application](#starting-the-application)
6. [Using the Recovery Dashboard](#using-the-recovery-dashboard)
7. [Configuring Trading Strategies](#configuring-trading-strategies)
8. [Monitoring & Alerts](#monitoring--alerts)
9. [Troubleshooting](#troubleshooting)

## 🔑 Prerequisites

### Required Software
- Node.js 18+ and npm
- Python 3.11+
- Redis server
- Git

### Optional (Recommended)
- Docker & Docker Compose
- PostgreSQL (for production)
- Grafana & Prometheus (for monitoring)

## 🌍 Environment Setup

### Step 1: Create Environment Files

Create `.env` file in the root directory:

```bash
# Create .env file
cat > .env << 'EOF'
# Application Settings
NODE_ENV=development
DEBUG=true

# API URLs
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Database
DATABASE_URL=sqlite+aiosqlite:///./quantum_trading.db

# Redis
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Zerodha API Credentials (GET THESE FROM ZERODHA CONSOLE)
ZERODHA_API_KEY=your-zerodha-api-key
ZERODHA_API_SECRET=your-zerodha-api-secret
ZERODHA_USER_ID=your-zerodha-user-id

# Risk Management Settings
MAX_DAILY_LOSS_PERCENT=2.0
MAX_POSITION_SIZE_PERCENT=5.0
COOLING_PERIOD_MINUTES=30
MAX_DAILY_TRADES=5

# Paper Trading
ENABLE_PAPER_TRADING=true
PAPER_TRADING_CAPITAL=500000

# Feature Flags
ENABLE_ARBITRAGE_BOT=true
ENABLE_SOCIAL_SENTIMENT=true
EOF
```

Create `.env.local` in the frontend directory:

```bash
cd frontend
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF
cd ..
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt
pip install kiteconnect  # Zerodha SDK

# Install additional Indian market packages
pip install nsepy yfinance

cd ..

# Install frontend dependencies
cd frontend
npm install
cd ..
```

## 🏦 Zerodha Account Setup

### Step 1: Get API Credentials

1. Log in to [Zerodha Console](https://console.zerodha.com/)
2. Create a new app under "My Apps"
3. Note down your:
   - API Key
   - API Secret
4. Add redirect URL: `http://localhost:3000/zerodha/callback`

### Step 2: Update Environment Variables

Edit your `.env` file and add:

```bash
ZERODHA_API_KEY=your-actual-api-key
ZERODHA_API_SECRET=your-actual-api-secret
```

### Step 3: Generate Access Token

The first time you connect, you'll need to:

1. Visit: `https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY&v=3`
2. Log in with your Zerodha credentials
3. You'll be redirected to your app with a `request_token`
4. The app will automatically exchange this for an access token

## 🗄️ Database Configuration

### Development (SQLite)

```bash
# Initialize database
cd backend
python -c "
import asyncio
from app.db.database import init_db

asyncio.run(init_db())
print('Database initialized!')
"
```

### Production (PostgreSQL)

1. Install PostgreSQL with TimescaleDB extension
2. Create database:

```sql
CREATE DATABASE quantum_trading_db;
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

3. Update `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost/quantum_trading_db
```

## 🚀 Starting the Application

### Option 1: Use the Start Script (Recommended)

```bash
# Make script executable
chmod +x start_recovery.sh

# Start everything
./start_recovery.sh
```

### Option 2: Manual Start

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 3: Start Frontend
cd frontend
npm run dev

# Terminal 4: Start Celery Worker (for background tasks)
cd backend
celery -A app.celery_app worker --loglevel=info
```

### Option 3: Docker Compose

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d
```

## 📊 Using the Recovery Dashboard

### 1. Initial Connection

1. Open http://localhost:3000/zerodha-recovery
2. Click "Connect Zerodha Account"
3. Enter your API credentials
4. Complete OAuth flow

### 2. Portfolio Analysis

Once connected, the dashboard will:
- Analyze your complete trading history
- Identify loss patterns
- Generate recovery strategies
- Show current positions with AI recommendations

### 3. Paper Trading Mode

**IMPORTANT**: Start with paper trading!

```javascript
// The system enforces paper trading for new users
// You must complete 20 successful paper trades before live trading
```

### 4. Activate Recovery Strategies

1. **Conservative Mode** (Recommended to start):
   - NIFTY credit spreads
   - 3-5% monthly target
   - Maximum 2% risk

2. **Systematic Recovery**:
   - AI-powered signals
   - Multiple strategies
   - 5-8% monthly target

3. **Growth Mode** (After recovery):
   - Aggressive strategies
   - Higher position sizes
   - 10%+ monthly target

## ⚙️ Configuring Trading Strategies

### 1. Edit Strategy Parameters

Create `config/strategies.json`:

```json
{
  "credit_spreads": {
    "enabled": true,
    "instruments": ["NIFTY", "BANKNIFTY"],
    "max_positions": 2,
    "strike_distance": 200,
    "stop_loss_points": 50,
    "target_points": 100,
    "entry_time": "10:30",
    "exit_time": "15:00"
  },
  "momentum_trading": {
    "enabled": true,
    "scan_interval": 300,
    "min_volume": 1000000,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "position_size_percent": 5
  },
  "arbitrage": {
    "enabled": true,
    "min_spread_percent": 0.15,
    "max_execution_time": 1000,
    "instruments": ["RELIANCE", "TCS", "INFY", "HDFC"]
  }
}
```

### 2. Risk Management Rules

Create `config/risk_rules.json`:

```json
{
  "global": {
    "max_daily_loss_percent": 2.0,
    "max_open_positions": 3,
    "force_square_off_time": "15:15"
  },
  "per_trade": {
    "max_risk_percent": 1.0,
    "mandatory_stop_loss": true,
    "trailing_stop_percent": 1.0
  },
  "behavioral": {
    "cooling_period_after_loss": 1800,
    "max_trades_after_loss": 1,
    "profit_booking_percent": 80
  }
}
```

## 🔔 Monitoring & Alerts

### 1. Configure Telegram Alerts

1. Create a bot using @BotFather on Telegram
2. Get your chat ID
3. Update `.env`:

```bash
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

### 2. Email Alerts

Update `.env` with SMTP settings:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
```

### 3. Real-time Monitoring

Access monitoring dashboards:
- Trading Dashboard: http://localhost:3000/dashboard
- API Metrics: http://localhost:8000/metrics
- WebSocket Status: http://localhost:8000/ws/status

## 🛠️ Integration Examples

### Example 1: Fetch Portfolio Analysis

```python
import requests

# Get auth token
response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    'email': 'your-email@example.com',
    'password': 'your-password'
})
token = response.json()['access_token']

# Get portfolio analysis
headers = {'Authorization': f'Bearer {token}'}
portfolio = requests.get(
    'http://localhost:8000/api/v1/zerodha/portfolio/analysis',
    headers=headers
).json()

print(f"Total P&L: {portfolio['total_pnl']}")
print(f"Recovery Strategies: {portfolio['recovery_strategies']}")
```

### Example 2: Execute Smart Order

```python
# Place a smart order with AI optimization
order_data = {
    'symbol': 'RELIANCE',
    'quantity': 10,
    'order_type': 'LIMIT',
    'transaction_type': 'BUY'
}

response = requests.post(
    'http://localhost:8000/api/v1/zerodha/order/smart',
    json=order_data,
    headers=headers
)

if response.json()['success']:
    print(f"Order placed: {response.json()['order_id']}")
```

### Example 3: Start Recovery Strategy

```python
# Activate AI trading signals
strategy_data = {
    'strategy_id': 'ai_ml_signals',
    'capital_allocation': 100000,
    'paper_trading': True  # Start with paper trading
}

response = requests.post(
    'http://localhost:8000/api/v1/zerodha/strategy/activate',
    json=strategy_data,
    headers=headers
)

print(f"Strategy activated: {response.json()}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # Start Redis
   redis-server
   ```

2. **Zerodha API Error**
   - Ensure API key and secret are correct
   - Check if your IP is whitelisted in Zerodha console
   - Verify redirect URL matches

3. **Database Connection Error**
   ```bash
   # Reset database
   cd backend
   rm quantum_trading.db
   python -c "from app.db.database import init_db; import asyncio; asyncio.run(init_db())"
   ```

4. **WebSocket Connection Issues**
   - Check if backend is running on port 8000
   - Verify CORS settings in backend
   - Clear browser cache

### Debug Mode

Enable detailed logging:

```bash
# Backend
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug

# Frontend
export NEXT_PUBLIC_DEBUG=true
npm run dev
```

## 📱 Mobile App Integration (Future)

The API is designed to support mobile apps:

1. Use JWT authentication
2. WebSocket for real-time updates
3. REST API for all operations
4. Push notifications via Firebase

## 🔐 Security Checklist

- [ ] Change default JWT secret key
- [ ] Use HTTPS in production
- [ ] Enable rate limiting
- [ ] Set up API key rotation
- [ ] Configure firewall rules
- [ ] Enable 2FA for Zerodha
- [ ] Use environment-specific configs
- [ ] Regular security audits

## 🚦 Health Checks

Monitor system health:

```bash
# API Health
curl http://localhost:8000/health

# Database Health
curl http://localhost:8000/api/v1/health/db

# Redis Health
curl http://localhost:8000/api/v1/health/redis

# WebSocket Health
curl http://localhost:8000/api/v1/health/websocket
```

## 📈 Performance Optimization

1. **Enable caching**:
   - Redis for market data
   - Local storage for user preferences

2. **Database optimization**:
   - Create indexes on frequently queried fields
   - Use connection pooling
   - Enable query caching

3. **Frontend optimization**:
   - Enable Next.js production build
   - Use CDN for static assets
   - Implement lazy loading

## 🎯 Next Steps

1. **Complete Setup**:
   - [ ] Set up environment variables
   - [ ] Connect Zerodha account
   - [ ] Initialize database
   - [ ] Start services

2. **Begin Recovery**:
   - [ ] Complete risk assessment
   - [ ] Start paper trading
   - [ ] Review education materials
   - [ ] Join community

3. **Go Live**:
   - [ ] Complete 20 paper trades
   - [ ] Achieve 60% win rate
   - [ ] Start with small capital
   - [ ] Scale gradually

## 📞 Support

- Documentation: `/docs`
- API Reference: http://localhost:8000/docs
- Community Discord: [Join Here]
- Email Support: support@quantumtrading.ai

---

Remember: **Start with paper trading!** The system enforces this for your protection.

Good luck with your trading journey! 🚀
