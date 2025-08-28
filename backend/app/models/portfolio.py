"""Portfolio model for managing user portfolios."""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class Portfolio(Base):
    """Portfolio model for tracking user investment portfolios."""
    
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Portfolio identification
    name = Column(String, nullable=False)
    description = Column(String)
    portfolio_type = Column(String, default="trading")  # trading, investment, paper
    
    # Account values
    total_value = Column(Float, default=0)
    cash_balance = Column(Float, default=0)
    buying_power = Column(Float, default=0)
    
    # Performance metrics
    total_return = Column(Float, default=0)
    total_return_percent = Column(Float, default=0)
    daily_return = Column(Float, default=0)
    daily_return_percent = Column(Float, default=0)
    
    # Risk metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    volatility = Column(Float)
    
    # Portfolio Greeks (for options)
    portfolio_delta = Column(Float, default=0)
    portfolio_gamma = Column(Float, default=0)
    portfolio_theta = Column(Float, default=0)
    portfolio_vega = Column(Float, default=0)
    
    # Realized P&L
    realized_pnl_day = Column(Float, default=0)
    realized_pnl_week = Column(Float, default=0)
    realized_pnl_month = Column(Float, default=0)
    realized_pnl_year = Column(Float, default=0)
    realized_pnl_all = Column(Float, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # Settings
    settings = Column(JSON, default=dict)  # Risk limits, alerts, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("app.models.position.Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("app.models.trade.Trade", back_populates="portfolio")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")
    orders = relationship("app.models.trading.Order", back_populates="portfolio")
    performance_history = relationship("PortfolioPerformance", back_populates="portfolio", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_portfolio_user', 'user_id'),
        Index('idx_portfolio_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Portfolio(name={self.name}, value={self.total_value})>"


class PortfolioPerformance(Base):
    """Historical performance tracking for portfolios."""
    
    __tablename__ = "portfolio_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Snapshot date
    date = Column(DateTime, nullable=False)
    
    # Values
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float)
    
    # Daily metrics
    daily_return = Column(Float)
    daily_return_percent = Column(Float)
    
    # Cumulative metrics
    cumulative_return = Column(Float)
    cumulative_return_percent = Column(Float)
    
    # Risk metrics snapshot
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    volatility = Column(Float)
    
    # Position count
    position_count = Column(Integer)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_history")
    
    # Indexes
    __table_args__ = (
        Index('idx_portfolio_performance_date', 'portfolio_id', 'date'),
    )
    
    def __repr__(self):
        return f"<PortfolioPerformance(portfolio_id={self.portfolio_id}, date={self.date}, value={self.total_value})>"


class Watchlist(Base):
    """User watchlists for tracking symbols."""
    
    __tablename__ = "watchlists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Watchlist details
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Symbols
    symbols = Column(JSON, default=list)  # List of symbol dictionaries
    
    # Settings
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="watchlists")
    
    def __repr__(self):
        return f"<Watchlist(name={self.name}, symbols={len(self.symbols)})>"


class RiskMetrics(Base):
    """Real-time risk metrics for portfolios."""
    
    __tablename__ = "risk_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Value at Risk
    var_95 = Column(Float)  # 95% confidence
    var_99 = Column(Float)  # 99% confidence
    cvar_95 = Column(Float)  # Conditional VaR
    
    # Exposure metrics
    gross_exposure = Column(Float)
    net_exposure = Column(Float)
    long_exposure = Column(Float)
    short_exposure = Column(Float)
    
    # Concentration risk
    largest_position_weight = Column(Float)
    top_5_concentration = Column(Float)
    sector_concentration = Column(JSON)  # Dict of sector weights
    
    # Options specific
    max_loss = Column(Float)
    margin_requirement = Column(Float)
    
    # Timestamp
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RiskMetrics(portfolio_id={self.portfolio_id}, var_95={self.var_95})>"
