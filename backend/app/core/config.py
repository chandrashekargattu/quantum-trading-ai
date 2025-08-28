"""Configuration settings for the Quantum Trading AI backend."""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, Field


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings for environment variable support."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Quantum Trading AI"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="BACKEND_CORS_ORIGINS"
    )
    
    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/quantum_trading",
        env="DATABASE_URL"
    )
    
    # Redis Settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security Settings
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Keys for Market Data
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    IEX_CLOUD_API_KEY: Optional[str] = Field(default=None, env="IEX_CLOUD_API_KEY")
    POLYGON_API_KEY: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
    
    # ML Model Settings
    MODEL_PATH: str = Field(default="./models", env="MODEL_PATH")
    MODEL_UPDATE_INTERVAL: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")  # seconds
    
    # WebSocket Settings
    WS_MESSAGE_QUEUE: str = Field(default="redis://localhost:6379/1", env="WS_MESSAGE_QUEUE")
    
    # Celery Settings
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Email Settings (for alerts)
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # Trading Settings
    PAPER_TRADING_ENABLED: bool = Field(default=True, env="PAPER_TRADING_ENABLED")
    MAX_POSITION_SIZE: float = Field(default=10000.0, env="MAX_POSITION_SIZE")
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
