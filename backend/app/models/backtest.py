"""Backtesting models."""

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class BacktestStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)
    
    # Backtest configuration
    name = Column(String, nullable=False)
    description = Column(Text)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # Transaction costs
    commission = Column(Float, default=0)
    slippage = Column(Float, default=0)
    
    # Status
    status = Column(Enum(BacktestStatus), default=BacktestStatus.PENDING)
    progress = Column(Float, default=0)  # 0-100%
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User")
    strategy = relationship("app.models.trading.Strategy")
    results = relationship("BacktestResult", back_populates="backtest", uselist=False)
    trades = relationship("BacktestTrade", back_populates="backtest")


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(UUID(as_uuid=True), ForeignKey("backtests.id"), nullable=False, unique=True)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)
    
    # Performance metrics
    total_return = Column(Float)
    annualized_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Risk metrics
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # Days
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional VaR 95%
    
    # Trading metrics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    expectancy = Column(Float)
    
    # Additional metrics
    beta = Column(Float)
    alpha = Column(Float)
    information_ratio = Column(Float)
    
    # Time series data (stored as JSON)
    equity_curve = Column(JSON)  # [{date, value}, ...]
    drawdown_curve = Column(JSON)
    returns_series = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    backtest = relationship("Backtest", back_populates="results")
    strategy = relationship("app.models.trading.Strategy", back_populates="backtest_results")


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(UUID(as_uuid=True), ForeignKey("backtests.id"), nullable=False)
    
    # Trade details
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    
    # Dates
    entry_date = Column(DateTime, nullable=False)
    exit_date = Column(DateTime)
    
    # P&L
    gross_pnl = Column(Float)
    commission = Column(Float)
    slippage = Column(Float)
    net_pnl = Column(Float)
    
    # Trade metadata
    entry_reason = Column(String)
    exit_reason = Column(String)
    holding_period = Column(Integer)  # Days
    
    # Relationships
    backtest = relationship("Backtest", back_populates="trades")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Optimization configuration
    name = Column(String, nullable=False)
    parameters_to_optimize = Column(JSON)  # {param_name: {min, max, step}}
    optimization_metric = Column(String, nullable=False)  # sharpe_ratio, total_return, etc.
    method = Column(String, nullable=False)  # grid_search, genetic, bayesian
    
    # Date range
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Status
    status = Column(Enum(BacktestStatus), default=BacktestStatus.PENDING)
    progress = Column(Float, default=0)
    total_iterations = Column(Integer)
    completed_iterations = Column(Integer, default=0)
    
    # Results
    optimal_parameters = Column(JSON)
    optimal_metric_value = Column(Float)
    
    # Performance surface (for visualization)
    performance_surface = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    strategy = relationship("app.models.trading.Strategy")
    user = relationship("User")
    results = relationship("OptimizationResult", back_populates="optimization_run")


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    optimization_run_id = Column(UUID(as_uuid=True), ForeignKey("optimization_runs.id"), nullable=False)
    
    # Parameters tested
    parameters = Column(JSON, nullable=False)
    
    # Performance metrics
    metric_value = Column(Float, nullable=False)  # The optimization metric
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    # Iteration info
    iteration_number = Column(Integer)
    
    # Relationships
    optimization_run = relationship("OptimizationRun", back_populates="results")


class WalkForwardAnalysis(Base):
    __tablename__ = "walk_forward_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Configuration
    name = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    in_sample_periods = Column(Integer, nullable=False)  # Days
    out_sample_periods = Column(Integer, nullable=False)  # Days
    optimization_metric = Column(String, nullable=False)
    
    # Status
    status = Column(Enum(BacktestStatus), default=BacktestStatus.PENDING)
    current_period = Column(Integer, default=0)
    total_periods = Column(Integer)
    
    # Aggregate results
    avg_in_sample_performance = Column(Float)
    avg_out_sample_performance = Column(Float)
    performance_degradation = Column(Float)  # Difference between IS and OOS
    
    # Period results (stored as JSON)
    period_results = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    strategy = relationship("app.models.trading.Strategy")
    user = relationship("User")
