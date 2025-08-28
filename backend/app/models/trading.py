"""Trading models for orders and trades."""

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class TimeInForce(str, enum.Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date


class Order(Base):
    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    
    # Price fields
    limit_price = Column(Float)
    stop_price = Column(Float)
    average_fill_price = Column(Float)
    
    # Time fields
    time_in_force = Column(Enum(TimeInForce), default=TimeInForce.DAY)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Extended order features
    extended_hours = Column(Boolean, default=False)
    hidden = Column(Boolean, default=False)
    display_quantity = Column(Float)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="orders")





class Strategy(Base):
    __tablename__ = "trading_strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    strategy_type = Column(String, nullable=False)
    
    # Strategy parameters (JSON)
    parameters = Column(String)  # JSON string
    
    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    trades = relationship("app.models.trade.Trade", back_populates="strategy")
    backtest_results = relationship("BacktestResult", back_populates="strategy")
