"""
Simplified pytest configuration for tests.
"""

import os
import sys
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_ALGORITHM'] = 'HS256'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

from app.db.database import Base
from app.main import app


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database engine."""
    from app.core.config import settings
    
    # Use test database
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        future=True
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def db(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
