"""User model for authentication and user management."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class User(Base):
    """User model for storing user account information."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # Trading account info
    account_type = Column(String, default="paper")  # paper, live
    account_balance = Column(Float, default=100000.0)  # Starting balance for paper trading
    buying_power = Column(Float, default=100000.0)
    
    # Subscription info
    subscription_tier = Column(String, default="free")  # free, basic, pro, enterprise
    subscription_expires = Column(DateTime)
    
    # API access
    api_key = Column(String, unique=True)
    api_key_created = Column(DateTime)
    
    # Preferences
    preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("app.models.trade.Trade", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    strategies = relationship("app.models.alert.Strategy", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"


class UserSession(Base):
    """User session model for tracking active sessions."""
    
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    access_token = Column(String, unique=True, nullable=False)
    refresh_token = Column(String, unique=True)
    
    # Session info
    ip_address = Column(String)
    user_agent = Column(String)
    device_type = Column(String)  # web, mobile, api
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Status
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, device={self.device_type})>"
