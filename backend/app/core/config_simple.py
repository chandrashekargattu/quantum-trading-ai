"""Simplified configuration to avoid import issues."""

import os
from typing import Optional

# Disable problematic imports temporarily
os.environ["DISABLE_ML_IMPORTS"] = "1"

class Settings:
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Quantum Trading AI"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./quantum_trading.db")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # External APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    IEX_CLOUD_API_KEY: Optional[str] = os.getenv("IEX_CLOUD_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001"]

settings = Settings()

