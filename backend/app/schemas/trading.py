"""Trading-related schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from enum import Enum


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class TradingSignalBase(BaseModel):
    """Base trading signal schema."""
    symbol: str
    signal_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[int] = None


class TradingSignalCreate(TradingSignalBase):
    """Trading signal creation schema."""
    pass


class TradingSignalResponse(TradingSignalBase):
    """Trading signal response schema."""
    id: UUID
    created_at: datetime
    is_active: bool = True
    
    model_config = ConfigDict(from_attributes=True)


class MarketOrderBase(BaseModel):
    """Base market order schema."""
    symbol: str
    side: OrderSide
    quantity: int = Field(..., gt=0)
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class MarketOrderCreate(MarketOrderBase):
    """Market order creation schema."""
    pass


class MarketOrderResponse(MarketOrderBase):
    """Market order response schema."""
    id: UUID
    status: OrderStatus
    filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    portfolio_id: UUID
    
    model_config = ConfigDict(from_attributes=True)


class BacktestRequest(BaseModel):
    """Backtest request schema."""
    strategy_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(..., gt=0)
    parameters: Dict[str, Any] = {}


class BacktestResponse(BaseModel):
    """Backtest response schema."""
    strategy_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    profit_factor: float
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    
    model_config = ConfigDict(from_attributes=True)


class TradingBotStatus(BaseModel):
    """Trading bot status schema."""
    bot_id: str
    is_active: bool
    strategy: str
    current_positions: int
    daily_pnl: float
    total_pnl: float
    last_trade_at: Optional[datetime] = None
    errors: List[str] = []


class RiskMetrics(BaseModel):
    """Risk metrics schema."""
    portfolio_id: UUID
    value_at_risk: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    beta: float
    alpha: float
    correlation_to_market: float
    position_concentration: Dict[str, float]
    sector_exposure: Dict[str, float]
    
    model_config = ConfigDict(from_attributes=True)


class PortfolioPerformance(BaseModel):
    """Portfolio performance schema."""
    portfolio_id: UUID
    period: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    daily_returns: List[float] = []
    
    model_config = ConfigDict(from_attributes=True)


class AlertCreate(BaseModel):
    """Alert creation schema."""
    alert_type: str
    condition: str
    symbol: Optional[str] = None
    threshold_value: Optional[float] = None
    message: str
    is_active: bool = True


class AlertResponse(AlertCreate):
    """Alert response schema."""
    id: UUID
    user_id: UUID
    created_at: datetime
    triggered_at: Optional[datetime] = None
    triggered_count: int = 0
    
    model_config = ConfigDict(from_attributes=True)


# Additional schemas for compatibility
class OrderBase(BaseModel):
    """Base order schema."""
    symbol: str
    side: OrderSide
    quantity: int = Field(..., gt=0)
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY


class OrderCreate(OrderBase):
    """Order creation schema."""
    portfolio_id: UUID


class OrderUpdate(BaseModel):
    """Order update schema."""
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    quantity: Optional[int] = Field(None, gt=0)


class OrderResponse(OrderBase):
    """Order response schema."""
    id: UUID
    portfolio_id: UUID
    status: OrderStatus
    filled_quantity: int = 0
    average_fill_price: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TradeBase(BaseModel):
    """Base trade schema."""
    order_id: UUID
    symbol: str
    side: OrderSide
    quantity: int
    price: float


class TradeResponse(TradeBase):
    """Trade response schema."""
    id: UUID
    executed_at: datetime
    commission: float = 0.0
    
    model_config = ConfigDict(from_attributes=True)


class OrderBookLevel(BaseModel):
    """Order book level schema."""
    price: float
    quantity: int
    orders: int


class OrderBookResponse(BaseModel):
    """Order book response schema."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    last_updated: datetime
