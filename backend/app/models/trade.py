"""Trade model for storing trading transactions."""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class Trade(Base):
    """Trade model for tracking all trading activities."""
    
    __tablename__ = "trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"))
    
    # Trade identification
    trade_id = Column(String, unique=True, nullable=False)  # External trade ID
    order_id = Column(String, unique=True)  # Order ID from broker
    
    # Asset information
    symbol = Column(String, index=True, nullable=False)
    asset_type = Column(String, nullable=False)  # stock, option, etf
    option_id = Column(UUID(as_uuid=True), ForeignKey("option_contracts.id"))
    
    # Trade details
    side = Column(String, nullable=False)  # buy, sell
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    
    # Fees and costs
    commission = Column(Float, default=0)
    fees = Column(Float, default=0)
    
    # Order type and execution
    order_type = Column(String)  # market, limit, stop, stop_limit
    time_in_force = Column(String)  # day, gtc, ioc, fok
    
    # Status
    status = Column(String, nullable=False)  # pending, filled, partial, cancelled, rejected
    fill_price = Column(Float)
    filled_quantity = Column(Integer, default=0)
    
    # Strategy information
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"))
    strategy_name = Column(String)
    signal_reason = Column(String)  # Why the trade was made
    
    # Risk metrics at time of trade
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_reward_ratio = Column(Float)
    
    # Paper trading flag
    is_paper = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional metadata
    trade_metadata = Column(JSON)  # Store any additional trade-specific data
    
    # Relationships
    user = relationship("User", back_populates="trades")
    portfolio = relationship("Portfolio", back_populates="trades")
    option = relationship("app.models.options.Option", back_populates="trades")
    strategy = relationship("app.models.trading.Strategy", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('idx_trade_user_created', 'user_id', 'created_at'),
        Index('idx_trade_symbol_status', 'symbol', 'status'),
        Index('idx_trade_portfolio', 'portfolio_id'),
    )
    
    def __repr__(self):
        return f"<Trade(id={self.trade_id}, symbol={self.symbol}, side={self.side}, quantity={self.quantity})>"




class OrderBook(Base):
    """Order book for tracking pending orders."""
    
    __tablename__ = "order_book"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Order details
    order_id = Column(String, unique=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    asset_type = Column(String, nullable=False)
    
    # Order specifications
    side = Column(String, nullable=False)  # buy, sell
    quantity = Column(Integer, nullable=False)
    order_type = Column(String, nullable=False)  # market, limit, stop, stop_limit
    limit_price = Column(Float)
    stop_price = Column(Float)
    
    # Time in force
    time_in_force = Column(String, default="day")  # day, gtc, ioc, fok
    
    # Status
    status = Column(String, nullable=False)  # pending, partial, filled, cancelled, rejected
    filled_quantity = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_order_user_status', 'user_id', 'status'),
        Index('idx_order_symbol', 'symbol'),
    )
    
    def __repr__(self):
        return f"<OrderBook(order_id={self.order_id}, symbol={self.symbol}, status={self.status})>"
