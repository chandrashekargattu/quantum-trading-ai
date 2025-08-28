"""Stock-related schemas."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class StockBase(BaseModel):
    """Base stock schema."""
    symbol: str
    name: str
    exchange: Optional[str] = None
    current_price: Optional[float] = None
    is_optionable: bool = False


class StockResponse(StockBase):
    """Stock response schema."""
    id: UUID
    previous_close: Optional[float] = None
    open_price: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    implied_volatility: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    last_updated: datetime
    
    class Config:
        from_attributes = True


class StockSearchResponse(BaseModel):
    """Stock search result schema."""
    symbol: str
    name: str
    exchange: Optional[str] = None
    current_price: Optional[float] = None
    change_percent: Optional[float] = None
    is_optionable: bool
    
    class Config:
        from_attributes = True


class PriceHistoryResponse(BaseModel):
    """Price history response schema."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None
    
    class Config:
        from_attributes = True


class MarketIndicatorResponse(BaseModel):
    """Market indicator response schema."""
    symbol: str
    name: Optional[str] = None
    value: float
    change_amount: Optional[float] = None
    change_percent: Optional[float] = None
    last_updated: datetime
    
    class Config:
        from_attributes = True


class QuoteResponse(BaseModel):
    """Real-time quote response."""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    timestamp: datetime
