# 🚀 Quick Start Guide - Quantum Trading AI

This guide will help you get the Quantum Trading AI platform up and running quickly.

## Prerequisites

- Docker & Docker Compose installed
- Node.js 20+ (for local development)
- Python 3.11+ (for local development)
- Git

## 🐳 Running with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd quantum-trading-ai
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 💻 Local Development Setup

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

5. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

## 🔑 Default Credentials

For demo purposes, you can create a new account through the registration page:
- Navigate to http://localhost:3000/auth/register
- Create a new account with your email

## 📊 Sample Data

To populate the database with sample data:

```bash
cd backend
python scripts/populate_sample_data.py
```

## 🛠️ Common Issues

### Port Already in Use
If you get port conflicts, you can change the ports in `docker-compose.yml` or stop conflicting services:
```bash
# Find process using port 3000
lsof -ti:3000 | xargs kill -9

# Find process using port 8000
lsof -ti:8000 | xargs kill -9
```

### Database Connection Issues
Make sure PostgreSQL is running:
```bash
docker-compose up -d postgres
```

### Missing API Keys
The application will work without external API keys but with limited functionality. To get full features:
- Get Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
- Get IEX Cloud API key: https://iexcloud.io/console/tokens

## 📚 Next Steps

1. **Explore the Dashboard**: Login and check out the main dashboard
2. **Add Stocks to Watchlist**: Search and add stocks to monitor
3. **View Options Chains**: Click on any stock to see its options
4. **Create Trading Strategies**: Use the strategy builder
5. **Run Backtests**: Test your strategies on historical data

## 🤝 Support

- Check the [full documentation](./docs/README.md)
- Report issues on GitHub
- Join our Discord community

Happy Trading! 🚀
