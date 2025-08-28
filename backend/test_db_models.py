"""Test database models can be created successfully."""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.db.database import Base

async def test_create_tables():
    """Test creating all database tables."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///./test_models.db",
        echo=True
    )
    
    async with engine.begin() as conn:
        # Drop all tables first
        await conn.run_sync(Base.metadata.drop_all)
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ All database tables created successfully!")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_create_tables())
