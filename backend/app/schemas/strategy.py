"""Strategy-related schemas."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from uuid import UUID


class StrategyCreate(BaseModel):
    """Strategy creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    strategy_type: str = Field(..., pattern="^(trend_following|mean_reversion|arbitrage|options_spread)$")
    config: Dict[str, Any]
    max_position_size: Optional[float] = Field(None, gt=0, le=1)
    stop_loss_percent: Optional[float] = Field(None, gt=0, le=100)
    take_profit_percent: Optional[float] = Field(None, gt=0)
    is_public: bool = False


class StrategyUpdate(BaseModel):
    """Strategy update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    max_position_size: Optional[float] = Field(None, gt=0, le=1)
    stop_loss_percent: Optional[float] = Field(None, gt=0, le=100)
    take_profit_percent: Optional[float] = Field(None, gt=0)
    is_public: Optional[bool] = None


class StrategyResponse(BaseModel):
    """Strategy response schema."""
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str] = None
    strategy_type: str
    config: Dict[str, Any]
    backtest_results: Optional[Dict[str, Any]] = None
    is_active: bool
    is_public: bool
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0
    win_rate: Optional[float] = None
    avg_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_position_size: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    last_executed: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class BacktestRequest(BaseModel):
    """Backtest request schema."""
    strategy_id: UUID
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(100000.0, gt=0)
    symbols: List[str] = Field(..., min_items=1)
    commission: float = Field(0.001, ge=0)  # 0.1% default commission


class BacktestResult(BaseModel):
    """Backtest result schema."""
    strategy_id: UUID
    period: Dict[str, str]  # start and end dates
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    monthly_returns: Dict[str, float]
    completed_at: datetime
