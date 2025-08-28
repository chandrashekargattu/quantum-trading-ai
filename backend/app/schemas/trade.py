"""Trade-related schemas."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class OrderRequest(BaseModel):
    """Order placement request."""
    symbol: str
    asset_type: str = Field(..., pattern="^(stock|option|etf)$")
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., gt=0)
    order_type: str = Field(..., pattern="^(market|limit|stop|stop_limit)$")
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: Optional[str] = Field("day", pattern="^(day|gtc|ioc|fok)$")
    option_id: Optional[UUID] = None


class OrderResponse(BaseModel):
    """Order placement response."""
    order_id: str
    status: str
    message: Optional[str] = None


class TradeResponse(BaseModel):
    """Trade response schema."""
    id: UUID
    trade_id: str
    order_id: Optional[str] = None
    symbol: str
    asset_type: str
    side: str
    quantity: int
    price: float
    total_amount: float
    commission: float = 0
    fees: float = 0
    status: str
    order_type: Optional[str] = None
    fill_price: Optional[float] = None
    filled_quantity: int = 0
    is_paper: bool
    created_at: datetime
    executed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PositionResponse(BaseModel):
    """Position response schema."""
    id: UUID
    symbol: str
    asset_type: str
    quantity: int
    avg_cost: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_percent: Optional[float] = None
    realized_pnl: Optional[float] = None
    is_open: bool
    opened_at: datetime
    
    class Config:
        from_attributes = True


class OrderBookResponse(BaseModel):
    """Order book response schema."""
    id: UUID
    order_id: str
    symbol: str
    asset_type: str
    side: str
    quantity: int
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str
    status: str
    filled_quantity: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TradeStatistics(BaseModel):
    """Trade statistics schema."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    avg_holding_period: Optional[float] = None


class ExecutionReport(BaseModel):
    """Trade execution report."""
    order_id: str
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    status: str
    timestamp: datetime
    message: Optional[str] = None
