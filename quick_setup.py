#!/usr/bin/env python3
"""
Quick Setup Script for Quantum Trading AI
This script automates the initial setup and integration
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path
import getpass
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def check_prerequisites():
    """Check if all required software is installed."""
    print_header("Checking Prerequisites")
    
    requirements = {
        'python3': 'Python 3.11+',
        'node': 'Node.js 18+',
        'npm': 'npm',
        'redis-cli': 'Redis',
        'git': 'Git'
    }
    
    all_good = True
    
    for cmd, name in requirements.items():
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            print_success(f"{name} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_error(f"{name} is not installed")
            all_good = False
    
    return all_good


def create_env_file():
    """Create .env file with user inputs."""
    print_header("Environment Configuration")
    
    env_path = Path('.env')
    if env_path.exists():
        response = input(f"{Colors.YELLOW}.env file already exists. Overwrite? (y/N): {Colors.END}")
        if response.lower() != 'y':
            print_info("Keeping existing .env file")
            return
    
    print_info("Let's set up your environment variables...")
    
    # Collect user inputs
    config = {
        'zerodha_api_key': input("\nZerodha API Key (from console.zerodha.com): "),
        'zerodha_api_secret': getpass.getpass("Zerodha API Secret: "),
        'zerodha_user_id': input("Zerodha User ID: "),
        'email': input("\nYour email for alerts: "),
        'telegram_bot_token': input("Telegram Bot Token (optional, press Enter to skip): "),
        'openai_api_key': input("OpenAI API Key for GPT-4 (optional, press Enter to skip): "),
    }
    
    # Generate secure JWT secret
    import secrets
    jwt_secret = secrets.token_urlsafe(32)
    
    # Create .env content
    env_content = f"""# Quantum Trading AI Configuration
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
JWT_SECRET_KEY={jwt_secret}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Zerodha API Credentials
ZERODHA_API_KEY={config['zerodha_api_key']}
ZERODHA_API_SECRET={config['zerodha_api_secret']}
ZERODHA_USER_ID={config['zerodha_user_id']}

# Alerts
USER_EMAIL={config['email']}
TELEGRAM_BOT_TOKEN={config['telegram_bot_token'] or 'your-telegram-bot-token'}
TELEGRAM_CHAT_ID=your-telegram-chat-id

# AI Integration
OPENAI_API_KEY={config['openai_api_key'] or 'your-openai-api-key'}

# Risk Management
MAX_DAILY_LOSS_PERCENT=2.0
MAX_POSITION_SIZE_PERCENT=5.0
COOLING_PERIOD_MINUTES=30
MAX_DAILY_TRADES=5

# Paper Trading (MANDATORY for first 2 weeks)
ENABLE_PAPER_TRADING=true
PAPER_TRADING_CAPITAL=500000

# Feature Flags
ENABLE_ARBITRAGE_BOT=true
ENABLE_SOCIAL_SENTIMENT=true
ENABLE_LOSS_RECOVERY=true

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Create frontend .env.local
    frontend_env = f"""NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
"""
    
    with open('frontend/.env.local', 'w') as f:
        f.write(frontend_env)
    
    print_success("Environment files created successfully!")


def install_dependencies():
    """Install all required dependencies."""
    print_header("Installing Dependencies")
    
    # Install Python dependencies
    print_info("Installing Python packages...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'
        ], check=True)
        
        # Install additional packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'kiteconnect', 'nsepy'
        ], check=True)
        
        print_success("Python dependencies installed")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Python dependencies: {e}")
        return False
    
    # Install frontend dependencies
    print_info("Installing Node.js packages...")
    try:
        subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        print_success("Frontend dependencies installed")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install frontend dependencies: {e}")
        return False
    
    return True


async def initialize_database():
    """Initialize the database."""
    print_header("Database Setup")
    
    try:
        # Add backend to Python path
        sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
        
        from app.db.database import init_db
        
        print_info("Initializing database...")
        await init_db()
        print_success("Database initialized successfully!")
        
        # Create default user
        from app.models.user import User
        from app.core.security import get_password_hash
        from app.db.database import get_db
        from sqlalchemy import select
        
        async for db in get_db():
            # Check if user exists
            result = await db.execute(
                select(User).where(User.email == "demo@quantumtrading.ai")
            )
            existing_user = result.scalar_one_or_none()
            
            if not existing_user:
                demo_user = User(
                    email="demo@quantumtrading.ai",
                    username="demo_trader",
                    hashed_password=get_password_hash("demo123"),
                    is_active=True,
                    is_paper_trading=True
                )
                db.add(demo_user)
                await db.commit()
                print_success("Created demo user: demo@quantumtrading.ai / demo123")
            else:
                print_info("Demo user already exists")
            
            break
            
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        return False
    
    return True


def create_strategy_configs():
    """Create default strategy configuration files."""
    print_header("Creating Strategy Configurations")
    
    # Create config directory
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # Default strategies configuration
    strategies = {
        "credit_spreads": {
            "enabled": True,
            "instruments": ["NIFTY", "BANKNIFTY"],
            "max_positions": 2,
            "strike_distance": 200,
            "stop_loss_points": 50,
            "target_points": 100,
            "entry_time": "10:30",
            "exit_time": "15:00"
        },
        "momentum_trading": {
            "enabled": True,
            "scan_interval": 300,
            "min_volume": 1000000,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "position_size_percent": 5
        },
        "arbitrage": {
            "enabled": True,
            "min_spread_percent": 0.15,
            "max_execution_time": 1000,
            "instruments": ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
        },
        "ai_signals": {
            "enabled": True,
            "confidence_threshold": 0.75,
            "max_concurrent_signals": 3,
            "risk_per_signal": 1.0
        }
    }
    
    # Risk management rules
    risk_rules = {
        "global": {
            "max_daily_loss_percent": 2.0,
            "max_open_positions": 3,
            "force_square_off_time": "15:15",
            "panic_button": True
        },
        "per_trade": {
            "max_risk_percent": 1.0,
            "mandatory_stop_loss": True,
            "trailing_stop_percent": 1.0,
            "partial_profit_booking": {
                "enabled": True,
                "levels": [
                    {"profit_percent": 50, "book_percent": 30},
                    {"profit_percent": 75, "book_percent": 30},
                    {"profit_percent": 100, "book_percent": 40}
                ]
            }
        },
        "behavioral": {
            "cooling_period_after_loss": 1800,
            "max_trades_after_loss": 1,
            "revenge_trading_prevention": True,
            "overtrading_limit": 5
        },
        "paper_trading": {
            "mandatory_days": 14,
            "min_trades_required": 20,
            "min_win_rate_required": 0.5,
            "graduate_conditions": {
                "profitable_days": 7,
                "max_drawdown_percent": 10
            }
        }
    }
    
    # Write configuration files
    with open(config_dir / 'strategies.json', 'w') as f:
        json.dump(strategies, f, indent=2)
    
    with open(config_dir / 'risk_rules.json', 'w') as f:
        json.dump(risk_rules, f, indent=2)
    
    print_success("Strategy configurations created")


def check_services():
    """Check if required services are running."""
    print_header("Checking Services")
    
    # Check Redis
    try:
        subprocess.run(['redis-cli', 'ping'], capture_output=True, check=True)
        print_success("Redis is running")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("Redis is not running. Starting Redis...")
        try:
            # Try to start Redis
            subprocess.Popen(['redis-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(2)
            print_success("Redis started")
        except:
            print_error("Failed to start Redis. Please start it manually.")
            return False
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! 🎉")
    
    print(f"""
{Colors.GREEN}Your Quantum Trading AI system is ready!{Colors.END}

{Colors.BOLD}Next Steps:{Colors.END}

1. {Colors.BLUE}Start the application:{Colors.END}
   ./start_recovery.sh
   
   Or manually:
   - Terminal 1: cd backend && uvicorn app.main:app --reload
   - Terminal 2: cd frontend && npm run dev

2. {Colors.BLUE}Access the recovery dashboard:{Colors.END}
   http://localhost:3000/zerodha-recovery

3. {Colors.BLUE}Log in with demo account:{Colors.END}
   Email: demo@quantumtrading.ai
   Password: demo123

4. {Colors.BLUE}Connect your Zerodha account:{Colors.END}
   - Click "Connect Zerodha Account"
   - Complete OAuth flow
   - Start with paper trading

5. {Colors.BLUE}Read the guides:{Colors.END}
   - ZERODHA_RECOVERY_GUIDE.md - Your personal recovery roadmap
   - INTEGRATION_GUIDE.md - Technical integration details

{Colors.YELLOW}⚠️  IMPORTANT REMINDERS:{Colors.END}
- You MUST complete 2 weeks of paper trading first
- Maximum 2% daily loss limit is enforced
- Start with conservative strategies
- Complete risk management course before live trading

{Colors.GREEN}Good luck with your trading journey! 🚀{Colors.END}
""")


def main():
    """Main setup function."""
    print_header("Quantum Trading AI - Quick Setup")
    
    # Check prerequisites
    if not check_prerequisites():
        print_error("\nPlease install missing prerequisites and run again.")
        sys.exit(1)
    
    # Create environment files
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print_error("\nDependency installation failed. Please check errors above.")
        sys.exit(1)
    
    # Initialize database
    if not asyncio.run(initialize_database()):
        print_error("\nDatabase initialization failed.")
        sys.exit(1)
    
    # Create configurations
    create_strategy_configs()
    
    # Check services
    if not check_services():
        print_warning("\nSome services need to be started manually.")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        sys.exit(1)
