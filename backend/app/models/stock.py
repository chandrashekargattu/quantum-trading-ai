"""Stock model for storing stock market data."""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class Stock(Base):
    """Stock model for storing stock information and real-time data."""
    
    __tablename__ = "stocks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    exchange = Column(String)  # NYSE, NASDAQ, etc.
    
    # Current market data
    current_price = Column(Float)
    previous_close = Column(Float)
    open_price = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    volume = Column(Integer)
    avg_volume = Column(Integer)
    
    # Market metrics
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    
    # 52-week data
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    
    # Change metrics
    change_amount = Column(Float)
    change_percent = Column(Float)
    
    # Technical indicators (latest values)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Volatility metrics
    implied_volatility = Column(Float)
    historical_volatility = Column(Float)
    
    # Sector and industry
    sector = Column(String)
    industry = Column(String)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_optionable = Column(Boolean, default=False)
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    options = relationship("app.models.option.Option", back_populates="stock", cascade="all, delete-orphan")
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_stock_symbol_active', 'symbol', 'is_active'),
        Index('idx_stock_sector', 'sector'),
    )
    
    def __repr__(self):
        return f"<Stock(symbol={self.symbol}, price={self.current_price})>"


class PriceHistory(Base):
    """Historical price data for stocks."""
    
    __tablename__ = "price_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    
    # OHLCV data
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    
    # Adjusted values
    adjusted_close = Column(Float)
    
    # Timeframe
    timeframe = Column(String)  # 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M
    
    # Relationships
    stock = relationship("Stock", back_populates="price_history")
    
    # Indexes
    __table_args__ = (
        Index('idx_price_history_stock_timestamp', 'stock_id', 'timestamp'),
        Index('idx_price_history_timeframe', 'timeframe'),
    )
    
    def __repr__(self):
        return f"<PriceHistory(stock_id={self.stock_id}, timestamp={self.timestamp}, close={self.close})>"


class MarketIndicator(Base):
    """Market-wide indicators and indices."""
    
    __tablename__ = "market_indicators"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, unique=True, nullable=False)  # SPY, VIX, DXY, etc.
    name = Column(String)
    
    # Current values
    value = Column(Float)
    change_amount = Column(Float)
    change_percent = Column(Float)
    
    # Additional data
    data = Column(JSON)  # Flexible field for indicator-specific data
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketIndicator(symbol={self.symbol}, value={self.value})>"
