"""Option-related schemas."""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class OptionBase(BaseModel):
    """Base option schema."""
    symbol: str
    underlying_symbol: str
    strike: float
    expiration: date
    option_type: str  # call or put


class OptionResponse(OptionBase):
    """Option response schema."""
    id: UUID
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    mark: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    theoretical_value: Optional[float] = None
    time_value: Optional[float] = None
    intrinsic_value: Optional[float] = None
    moneyness: Optional[str] = None
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    last_updated: datetime
    
    class Config:
        from_attributes = True


class OptionChainResponse(BaseModel):
    """Option chain response schema."""
    underlying_symbol: str
    calls: List[OptionResponse]
    puts: List[OptionResponse]
    expirations: List[str]
    strikes: List[float]


class OptionGreeksResponse(BaseModel):
    """Option Greeks calculation response."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class OptionQuoteRequest(BaseModel):
    """Option quote request schema."""
    symbol: str
    
    
class OptionOrderRequest(BaseModel):
    """Option order request schema."""
    option_id: UUID
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., gt=0)
    order_type: str = Field(..., pattern="^(market|limit|stop|stop_limit)$")
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = Field(default="day", pattern="^(day|gtc|ioc|fok)$")


class OptionStrategyLeg(BaseModel):
    """Option strategy leg."""
    action: str = Field(..., pattern="^(buy|sell)$")
    option_id: UUID
    quantity: int = Field(..., gt=0)
    
    
class OptionStrategyRequest(BaseModel):
    """Option strategy request."""
    strategy_type: str
    legs: List[OptionStrategyLeg]
    
    
class OptionScreenerRequest(BaseModel):
    """Option screener request."""
    min_volume: Optional[int] = None
    min_open_interest: Optional[int] = None
    min_implied_volatility: Optional[float] = None
    max_implied_volatility: Optional[float] = None
    option_type: Optional[str] = Field(None, pattern="^(call|put)$")
    moneyness: Optional[str] = Field(None, pattern="^(ITM|ATM|OTM)$")
    days_to_expiration_min: Optional[int] = None
    days_to_expiration_max: Optional[int] = None
