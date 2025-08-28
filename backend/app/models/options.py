"""Options trading models."""

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum, Boolean, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class OptionType(str, enum.Enum):
    CALL = "call"
    PUT = "put"


class OptionStyle(str, enum.Enum):
    AMERICAN = "american"
    EUROPEAN = "european"


class OptionAction(str, enum.Enum):
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"
    EXERCISE = "exercise"
    ASSIGN = "assign"
    EXPIRE = "expire"


class Option(Base):
    __tablename__ = "option_contracts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, nullable=False, index=True)
    underlying_symbol = Column(String, nullable=False, index=True)
    option_type = Column(Enum(OptionType), nullable=False)
    strike_price = Column(Float, nullable=False)
    expiration_date = Column(Date, nullable=False)
    style = Column(Enum(OptionStyle), default=OptionStyle.AMERICAN)
    
    # Market data
    bid = Column(Float)
    ask = Column(Float)
    last_price = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = relationship("app.models.trade.Trade", back_populates="option")


class OptionOrder(Base):
    __tablename__ = "option_orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    option_id = Column(UUID(as_uuid=True), ForeignKey("option_contracts.id"), nullable=False)
    
    # Order details
    quantity = Column(Integer, nullable=False)  # Number of contracts
    action = Column(Enum(OptionAction), nullable=False)
    order_type = Column(String, nullable=False)  # market, limit, etc.
    limit_price = Column(Float)
    
    # Status
    status = Column(String, default="pending")
    filled_quantity = Column(Integer, default=0)
    average_fill_price = Column(Float)
    commission = Column(Float, default=0)
    
    # Position tracking
    position_id = Column(UUID(as_uuid=True))
    is_closing = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio")
    option = relationship("app.models.options.Option")


class OptionPosition(Base):
    __tablename__ = "option_positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    option_id = Column(UUID(as_uuid=True), ForeignKey("option_contracts.id"), nullable=False)
    
    # Position details
    quantity = Column(Integer, nullable=False)  # Positive for long, negative for short
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    
    # P&L tracking
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0)
    
    # Risk metrics
    delta_exposure = Column(Float)
    gamma_exposure = Column(Float)
    theta_exposure = Column(Float)
    vega_exposure = Column(Float)
    
    # Status
    is_open = Column(Boolean, default=True)
    is_covered = Column(Boolean, default=False)  # For covered calls/puts
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio")
    option = relationship("app.models.options.Option")


class OptionStrategy(Base):
    __tablename__ = "option_strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    strategy_name = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)  # spread, straddle, etc.
    
    # Strategy details
    max_profit = Column(Float)
    max_loss = Column(Float)
    breakeven_points = Column(String)  # JSON array
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio")
    legs = relationship("OptionStrategyLeg", back_populates="strategy")


class OptionStrategyLeg(Base):
    __tablename__ = "option_strategy_legs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("option_strategies.id"), nullable=False)
    option_id = Column(UUID(as_uuid=True), ForeignKey("option_contracts.id"), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey("option_positions.id"))
    
    # Leg details
    action = Column(String, nullable=False)  # buy or sell
    quantity = Column(Integer, nullable=False)
    leg_type = Column(String)  # long_call, short_put, etc.
    
    # Relationships
    strategy = relationship("OptionStrategy", back_populates="legs")
    option = relationship("app.models.options.Option")
    position = relationship("OptionPosition")
