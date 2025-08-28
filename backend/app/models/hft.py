"""High-Frequency Trading (HFT) models."""

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum, Boolean, BigInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class HFTOrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    PEG = "peg"
    HIDDEN = "hidden"
    ICEBERG = "iceberg"
    MIDPOINT = "midpoint"


class HFTStrategyType(str, enum.Enum):
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARB = "statistical_arb"
    LATENCY_ARB = "latency_arb"


class VenueType(str, enum.Enum):
    LIT = "lit"
    DARK = "dark"
    ECN = "ecn"
    ATS = "ats"


class HFTOrder(Base):
    __tablename__ = "hft_orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("hft_strategies.id"))
    
    # Order details
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # buy/sell
    quantity = Column(Integer, nullable=False)
    order_type = Column(Enum(HFTOrderType), nullable=False)
    
    # Pricing
    price = Column(Float)
    limit_price = Column(Float)
    peg_offset = Column(Float)
    
    # Execution details
    venue = Column(Enum(VenueType))
    venue_order_id = Column(String)
    latency_ns = Column(BigInteger)  # Nanoseconds
    
    # Status
    status = Column(String, default="pending")
    filled_quantity = Column(Integer, default=0)
    average_fill_price = Column(Float)
    
    # Hidden/Iceberg orders
    display_quantity = Column(Integer)
    
    # Timestamps (microsecond precision)
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)
    acked_at = Column(DateTime)
    filled_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio")
    strategy = relationship("HFTStrategy")


class OrderBook(Base):
    __tablename__ = "order_books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, nullable=False, index=True)
    venue = Column(String, nullable=False)
    
    # Order book data (stored as JSON for flexibility)
    bids = Column(JSON)  # [{price, size, orders}, ...]
    asks = Column(JSON)  # [{price, size, orders}, ...]
    
    # Calculated fields
    bid_price = Column(Float)
    ask_price = Column(Float)
    spread = Column(Float)
    mid_price = Column(Float)
    imbalance = Column(Float)
    
    # Depth metrics
    bid_depth = Column(Integer)
    ask_depth = Column(Integer)
    total_bid_volume = Column(BigInteger)
    total_ask_volume = Column(BigInteger)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    exchange_timestamp = Column(DateTime)


class MarketMaker(Base):
    __tablename__ = "market_makers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("hft_strategies.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    
    # Market making parameters
    target_spread_bps = Column(Float)  # Basis points
    min_spread_bps = Column(Float)
    max_spread_bps = Column(Float)
    
    # Position management
    max_position = Column(Integer)
    current_position = Column(Integer, default=0)
    inventory_target = Column(Integer, default=0)
    
    # Risk limits
    max_order_size = Column(Integer)
    position_limit = Column(Integer)
    loss_limit = Column(Float)
    
    # Performance tracking
    filled_orders = Column(Integer, default=0)
    cancelled_orders = Column(Integer, default=0)
    total_volume = Column(BigInteger, default=0)
    realized_pnl = Column(Float, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    strategy = relationship("HFTStrategy")


class HFTStrategy(Base):
    __tablename__ = "hft_strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    name = Column(String, nullable=False)
    strategy_type = Column(Enum(HFTStrategyType), nullable=False)
    
    # Strategy parameters (JSON)
    parameters = Column(JSON)
    
    # Risk management
    max_orders_per_second = Column(Integer, default=100)
    max_position_value = Column(Float)
    max_daily_loss = Column(Float)
    current_daily_pnl = Column(Float, default=0)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)
    sharpe_ratio = Column(Float)
    
    # Circuit breaker
    circuit_breaker_triggered = Column(Boolean, default=False)
    circuit_breaker_threshold = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio")
    orders = relationship("HFTOrder", back_populates="strategy")
    market_makers = relationship("MarketMaker", back_populates="strategy")


class TickData(Base):
    __tablename__ = "tick_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    
    # Tick details
    price = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)
    side = Column(String)  # bid/ask/trade
    
    # Venue information
    venue = Column(String)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange_timestamp = Column(DateTime)
    
    # Flags
    is_odd_lot = Column(Boolean, default=False)
    is_intermarket_sweep = Column(Boolean, default=False)
