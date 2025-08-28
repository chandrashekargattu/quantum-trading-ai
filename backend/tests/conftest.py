"""Test configuration and fixtures."""
import os
import pytest
from typing import AsyncGenerator, Generator
from datetime import datetime, timedelta
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.main import app
from app.db.database import Base, get_db
from app.models.user import User
from app.core.security import get_password_hash, create_access_token
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_quantum_trading.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
    poolclass=NullPool
)

# Create test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_database():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Cleanup
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """Get test database session."""
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Get test client."""
    
    async def override_get_db():
        yield db
    
    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(db: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="testuser@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
        is_active=True,
        is_superuser=False,
        created_at=datetime.utcnow()
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.fixture
async def test_superuser(db: AsyncSession) -> User:
    """Create a test superuser."""
    user = User(
        email="admin@example.com",
        username="admin",
        hashed_password=get_password_hash("adminpassword123"),
        full_name="Admin User",
        is_active=True,
        is_superuser=True,
        created_at=datetime.utcnow()
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Get authentication headers for test user."""
    access_token = create_access_token(
        subject=str(test_user.id),
        expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def superuser_auth_headers(test_superuser: User) -> dict:
    """Get authentication headers for test superuser."""
    access_token = create_access_token(
        subject=str(test_superuser.id),
        expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
async def test_portfolio(db: AsyncSession, test_user: User):
    """Create a test portfolio."""
    from app.models.portfolio import Portfolio
    
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Test Portfolio",
        description="Test portfolio for unit tests",
        initial_balance=10000.0,
        current_balance=10000.0,
        is_active=True,
        created_at=datetime.utcnow()
    )
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return portfolio


@pytest.fixture
async def test_strategy(db: AsyncSession, test_user: User):
    """Create a test strategy."""
    from app.models.strategy import Strategy
    
    strategy = Strategy(
        user_id=test_user.id,
        name="Test Strategy",
        description="Test strategy for unit tests",
        strategy_type="technical",
        parameters={"sma_short": 20, "sma_long": 50},
        is_active=True,
        created_at=datetime.utcnow()
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    return strategy


@pytest.fixture
async def test_backtest_result(db: AsyncSession, test_user: User, test_strategy):
    """Create a test backtest result."""
    from app.models.backtest import BacktestResult
    
    backtest = BacktestResult(
        user_id=test_user.id,
        strategy_id=test_strategy.id,
        symbol="AAPL",
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
        initial_capital=10000.0,
        final_capital=11000.0,
        status="completed",
        metrics={
            "total_return": 10.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -5.0,
            "win_rate": 60.0
        },
        created_at=datetime.utcnow()
    )
    db.add(backtest)
    await db.commit()
    await db.refresh(backtest)
    return backtest


@pytest.fixture
async def test_stock(db: AsyncSession):
    """Create a test stock."""
    from app.models.stock import Stock
    
    stock = Stock(
        symbol="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=2900000000000,
        created_at=datetime.utcnow()
    )
    db.add(stock)
    await db.commit()
    await db.refresh(stock)
    return stock


@pytest.fixture
async def test_option(db: AsyncSession, test_stock):
    """Create a test option."""
    from app.models.option import Option
    from datetime import date
    
    option = Option(
        symbol="AAPL230120C00150000",
        underlying_symbol=test_stock.symbol,
        option_type="CALL",
        strike_price=150.0,
        expiration_date=date(2024, 1, 20),
        contract_size=100,
        currency="USD",
        created_at=datetime.utcnow()
    )
    db.add(option)
    await db.commit()
    await db.refresh(option)
    return option


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 75000000,
        "high": 151.50,
        "low": 149.00,
        "open": 149.75,
        "close": 150.25,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_option_chain():
    """Mock option chain data for testing."""
    return {
        "symbol": "AAPL",
        "calls": [
            {
                "strike": 145.0,
                "bid": 5.50,
                "ask": 5.60,
                "volume": 1500,
                "openInterest": 5000,
                "impliedVolatility": 0.25
            },
            {
                "strike": 150.0,
                "bid": 2.20,
                "ask": 2.30,
                "volume": 3000,
                "openInterest": 10000,
                "impliedVolatility": 0.28
            }
        ],
        "puts": [
            {
                "strike": 145.0,
                "bid": 1.10,
                "ask": 1.20,
                "volume": 800,
                "openInterest": 3000,
                "impliedVolatility": 0.24
            },
            {
                "strike": 150.0,
                "bid": 3.80,
                "ask": 3.90,
                "volume": 2000,
                "openInterest": 8000,
                "impliedVolatility": 0.27
            }
        ]
    }


# Clean up any existing test database before tests
@pytest.fixture(scope="function", autouse=True)
async def cleanup_db():
    """Clean up database between tests."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Clean up after each test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)