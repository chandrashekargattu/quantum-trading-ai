"""Portfolio-related schemas."""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from uuid import UUID


class PortfolioCreate(BaseModel):
    """Portfolio creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    portfolio_type: Optional[str] = Field("trading", pattern="^(trading|investment|paper)$")
    initial_cash: Optional[float] = Field(100000.0, gt=0)


class PortfolioUpdate(BaseModel):
    """Portfolio update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    settings: Optional[Dict] = None


class PortfolioResponse(BaseModel):
    """Portfolio response schema."""
    id: UUID
    name: str
    description: Optional[str] = None
    portfolio_type: str
    total_value: float
    cash_balance: float
    buying_power: float
    total_return: float
    total_return_percent: float
    daily_return: Optional[float] = None
    daily_return_percent: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    portfolio_delta: Optional[float] = None
    portfolio_gamma: Optional[float] = None
    portfolio_theta: Optional[float] = None
    portfolio_vega: Optional[float] = None
    is_active: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioPerformanceResponse(BaseModel):
    """Portfolio performance response schema."""
    date: datetime
    total_value: float
    cash_balance: Optional[float] = None
    daily_return: Optional[float] = None
    daily_return_percent: Optional[float] = None
    cumulative_return: Optional[float] = None
    cumulative_return_percent: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    position_count: Optional[int] = None
    
    class Config:
        from_attributes = True


class RiskMetricsResponse(BaseModel):
    """Risk metrics response schema."""
    portfolio_id: UUID
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None
    long_exposure: Optional[float] = None
    short_exposure: Optional[float] = None
    largest_position_weight: Optional[float] = None
    top_5_concentration: Optional[float] = None
    sector_concentration: Optional[Dict[str, float]] = None
    max_loss: Optional[float] = None
    margin_requirement: Optional[float] = None
    calculated_at: datetime
    
    class Config:
        from_attributes = True


class WatchlistCreate(BaseModel):
    """Watchlist creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)
    is_public: bool = False


class WatchlistResponse(BaseModel):
    """Watchlist response schema."""
    id: UUID
    name: str
    description: Optional[str] = None
    symbols: List[Dict] = Field(default_factory=list)
    is_public: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    """Portfolio summary for dashboard."""
    portfolio_id: UUID
    name: str
    total_value: float
    daily_change: float
    daily_change_percent: float
    total_return: float
    total_return_percent: float
    position_count: int
    cash_balance: float
