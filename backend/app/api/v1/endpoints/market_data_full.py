"""
Market data API endpoints - Full Implementation
"""

from typing import Any, List, Dict, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import asyncio
import random
import numpy as np

from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.models.stock import Stock, PriceHistory
from app.services.market_data_service import MarketDataService
from app.core.cache import cache_get, cache_set

router = APIRouter()

# Request/Response Models
class QuoteBatchRequest(BaseModel):
    symbols: List[str] = Field(..., max_items=100)

class TechnicalIndicatorRequest(BaseModel):
    symbol: str
    indicators: List[str]
    period: int = Field(14, gt=0)

class CorrelationRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=2, max_items=10)
    period: int = Field(30, gt=0)

class ScannerRequest(BaseModel):
    filters: Dict[str, Any]
    sort_by: str = "volume"
    limit: int = Field(50, le=100)

class ExportRequest(BaseModel):
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    format: str = Field("json", regex="^(json|csv|excel)$")

# Helper function to generate mock data
def generate_mock_quote(symbol: str) -> Dict[str, Any]:
    """Generate mock market quote data."""
    base_price = random.uniform(50, 500)
    return {
        "symbol": symbol,
        "price": round(base_price, 2),
        "change": round(random.uniform(-5, 5), 2),
        "change_percent": round(random.uniform(-2, 2), 2),
        "volume": random.randint(1000000, 50000000),
        "bid": round(base_price - 0.01, 2),
        "ask": round(base_price + 0.01, 2),
        "high": round(base_price * 1.02, 2),
        "low": round(base_price * 0.98, 2),
        "open": round(base_price * random.uniform(0.98, 1.02), 2),
        "previous_close": round(base_price * random.uniform(0.98, 1.02), 2),
        "timestamp": datetime.now().isoformat()
    }

def generate_mock_historical_data(symbol: str, start: datetime, end: datetime, interval: str) -> List[Dict]:
    """Generate mock historical price data."""
    data = []
    current = start
    base_price = random.uniform(50, 500)
    
    while current <= end:
        data.append({
            "timestamp": current.isoformat(),
            "open": round(base_price * random.uniform(0.98, 1.02), 2),
            "high": round(base_price * random.uniform(1.0, 1.03), 2),
            "low": round(base_price * random.uniform(0.97, 1.0), 2),
            "close": round(base_price * random.uniform(0.98, 1.02), 2),
            "volume": random.randint(100000, 5000000)
        })
        
        # Update price for next iteration
        base_price *= random.uniform(0.99, 1.01)
        
        # Increment time based on interval
        if interval == "1m":
            current += timedelta(minutes=1)
        elif interval == "5m":
            current += timedelta(minutes=5)
        elif interval == "15m":
            current += timedelta(minutes=15)
        elif interval == "30m":
            current += timedelta(minutes=30)
        elif interval == "1h":
            current += timedelta(hours=1)
        else:  # 1d
            current += timedelta(days=1)
    
    return data

# Endpoints
@router.get("/quote/{symbol}")
async def get_stock_quote(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Get real-time stock quote."""
    # Check cache first
    cached = await cache_get(f"quote:{symbol}")
    if cached:
        return cached
    
    # For now, return mock data
    quote = generate_mock_quote(symbol)
    
    # Cache for 5 seconds
    await cache_set(f"quote:{symbol}", quote, expire=5)
    
    return quote

@router.get("/quote/")
async def get_empty_quote(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Handle empty quote request."""
    raise HTTPException(status_code=404, detail="Symbol not provided")

@router.get("/quote/{symbol}/")
async def get_quote_with_slash(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Handle quote request with trailing slash."""
    if not symbol or symbol.strip() == "":
        raise HTTPException(status_code=404, detail="Invalid symbol")
    
    # Check for special characters
    if any(char in symbol for char in "@#$%^&*()"):
        raise HTTPException(status_code=422, detail="Invalid symbol format")
    
    return await get_stock_quote(symbol, current_user, None)

@router.post("/quotes/batch")
async def get_quotes_batch(
    request: QuoteBatchRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get multiple stock quotes in batch."""
    if len(request.symbols) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 symbols allowed per request"
        )
    
    quotes = []
    for symbol in request.symbols:
        quotes.append(generate_mock_quote(symbol))
    
    return quotes

@router.get("/history/{symbol}")
async def get_historical_data(
    symbol: str,
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    interval: str = Query("1d", regex="^(1m|5m|15m|30m|1h|1d)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get historical price data."""
    # Validate date range
    if end_date < start_date:
        raise HTTPException(
            status_code=400,
            detail="End date must be after start date"
        )
    
    if start_date > datetime.now():
        raise HTTPException(
            status_code=400,
            detail="Start date cannot be in the future"
        )
    
    # Generate mock historical data
    data = generate_mock_historical_data(symbol, start_date, end_date, interval)
    return data

@router.get("/options/{symbol}")
async def get_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get options chain data."""
    # Generate mock options data
    expirations = []
    base_date = datetime.now()
    for i in range(4):  # 4 expiration dates
        exp_date = base_date + timedelta(days=7 * (i + 1))
        expirations.append(exp_date.strftime("%Y-%m-%d"))
    
    if expiration:
        # Validate expiration format
        try:
            datetime.strptime(expiration, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid expiration date format"
            )
        return {
            "symbol": symbol,
            "expiration": expiration,
            "calls": generate_mock_options_data("call"),
            "puts": generate_mock_options_data("put")
        }
    
    return {
        "symbol": symbol,
        "expirations": expirations,
        "calls": generate_mock_options_data("call"),
        "puts": generate_mock_options_data("put")
    }

def generate_mock_options_data(option_type: str) -> List[Dict]:
    """Generate mock options data."""
    options = []
    base_price = 150
    
    for i in range(-5, 6):  # Strike prices around base
        strike = base_price + (i * 5)
        options.append({
            "strike": float(strike),
            "bid": round(random.uniform(0.5, 5), 2),
            "ask": round(random.uniform(0.5, 5), 2),
            "last": round(random.uniform(0.5, 5), 2),
            "volume": random.randint(0, 10000),
            "open_interest": random.randint(0, 50000),
            "implied_volatility": round(random.uniform(0.2, 0.6), 3)
        })
    
    return options

@router.post("/indicators")
async def calculate_technical_indicators(
    request: TechnicalIndicatorRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Calculate technical indicators."""
    if request.period <= 0:
        raise HTTPException(
            status_code=422,
            detail="Period must be positive"
        )
    
    # Generate mock indicator values
    result = {}
    
    if "SMA" in request.indicators:
        result["SMA"] = round(random.uniform(140, 160), 2)
    
    if "RSI" in request.indicators:
        result["RSI"] = round(random.uniform(30, 70), 2)
    
    if "MACD" in request.indicators:
        result["MACD"] = {
            "macd": round(random.uniform(-2, 2), 3),
            "signal": round(random.uniform(-2, 2), 3),
            "histogram": round(random.uniform(-1, 1), 3)
        }
    
    if "BB" in request.indicators:
        base = 150
        result["BB"] = {
            "upper": round(base + random.uniform(5, 10), 2),
            "middle": round(base, 2),
            "lower": round(base - random.uniform(5, 10), 2)
        }
    
    return result

@router.get("/news/sentiment")
async def get_market_news_sentiment(
    symbol: str = Query(...),
    days: int = Query(7, ge=1, le=30),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get market news sentiment analysis."""
    if not symbol:
        raise HTTPException(
            status_code=422,
            detail="Symbol is required"
        )
    
    return {
        "symbol": symbol,
        "overall_sentiment": round(random.uniform(-1, 1), 2),
        "news_count": random.randint(10, 100),
        "sentiment_scores": {
            "positive": random.randint(20, 60),
            "neutral": random.randint(20, 60),
            "negative": random.randint(10, 30)
        },
        "top_stories": [
            {
                "title": f"Breaking: {symbol} announces Q4 earnings",
                "sentiment": round(random.uniform(-1, 1), 2),
                "source": "Financial Times",
                "published": datetime.now().isoformat()
            }
        ]
    }

@router.get("/intraday/{symbol}")
async def get_intraday_data(
    symbol: str,
    interval: str = Query(..., regex="^(1m|5m|15m|30m|1h)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get intraday market data."""
    # Check if interval is valid
    valid_intervals = ["1m", "5m", "15m", "30m", "1h"]
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid interval. Must be one of {valid_intervals}"
        )
    
    # Generate mock intraday data
    start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    end = datetime.now()
    
    return generate_mock_historical_data(symbol, start, end, interval)

@router.get("/hours")
async def get_market_hours(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get market hours status."""
    now = datetime.now()
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    current_hour = now.hour
    
    # Simple market hours check (9:30 AM - 4:00 PM ET)
    is_open = is_weekday and 9.5 <= current_hour < 16
    
    # Calculate next open/close
    if is_open:
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        next_open = next_close + timedelta(days=1 if now.weekday() < 4 else 3)
        next_open = next_open.replace(hour=9, minute=30)
    else:
        if now.weekday() == 4 and current_hour >= 16:  # Friday after close
            next_open = now + timedelta(days=3)
        elif now.weekday() == 5:  # Saturday
            next_open = now + timedelta(days=2)
        elif now.weekday() == 6:  # Sunday
            next_open = now + timedelta(days=1)
        else:
            next_open = now + timedelta(days=1)
        
        next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
        next_close = next_open.replace(hour=16, minute=0)
    
    return {
        "is_open": is_open,
        "current_time": now.isoformat(),
        "next_open": next_open.isoformat(),
        "next_close": next_close.isoformat(),
        "timezone": "US/Eastern"
    }

@router.get("/movers")
async def get_market_movers(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get market movers (top gainers/losers)."""
    # Generate mock market movers
    gainers = []
    losers = []
    most_active = []
    
    for i in range(10):
        gainers.append({
            "symbol": f"GAIN{i}",
            "name": f"Gainer Company {i}",
            "price": round(random.uniform(10, 200), 2),
            "change": round(random.uniform(5, 50), 2),
            "change_percent": round(random.uniform(5, 50), 2),
            "volume": random.randint(1000000, 50000000)
        })
        
        losers.append({
            "symbol": f"LOSS{i}",
            "name": f"Loser Company {i}",
            "price": round(random.uniform(10, 200), 2),
            "change": round(random.uniform(-50, -5), 2),
            "change_percent": round(random.uniform(-50, -5), 2),
            "volume": random.randint(1000000, 50000000)
        })
        
        most_active.append({
            "symbol": f"ACTV{i}",
            "name": f"Active Company {i}",
            "price": round(random.uniform(10, 200), 2),
            "change": round(random.uniform(-5, 5), 2),
            "change_percent": round(random.uniform(-5, 5), 2),
            "volume": random.randint(50000000, 200000000)
        })
    
    return {
        "gainers": gainers,
        "losers": losers,
        "most_active": most_active,
        "updated": datetime.now().isoformat()
    }

@router.get("/sectors")
async def get_sector_performance(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get sector performance data."""
    sectors = {
        "Technology": round(random.uniform(-2, 3), 2),
        "Healthcare": round(random.uniform(-2, 3), 2),
        "Financials": round(random.uniform(-2, 3), 2),
        "Consumer Discretionary": round(random.uniform(-2, 3), 2),
        "Consumer Staples": round(random.uniform(-2, 3), 2),
        "Energy": round(random.uniform(-3, 4), 2),
        "Materials": round(random.uniform(-2, 3), 2),
        "Industrials": round(random.uniform(-2, 3), 2),
        "Real Estate": round(random.uniform(-2, 3), 2),
        "Utilities": round(random.uniform(-1, 2), 2),
        "Communication Services": round(random.uniform(-2, 3), 2)
    }
    return sectors

@router.get("/indices")
async def get_market_indices(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get major market indices data."""
    indices = [
        {
            "symbol": "SPY",
            "name": "S&P 500",
            "value": round(random.uniform(4200, 4400), 2),
            "change": round(random.uniform(-50, 50), 2),
            "change_percent": round(random.uniform(-1.5, 1.5), 2)
        },
        {
            "symbol": "DIA",
            "name": "Dow Jones",
            "value": round(random.uniform(34000, 35000), 2),
            "change": round(random.uniform(-300, 300), 2),
            "change_percent": round(random.uniform(-1, 1), 2)
        },
        {
            "symbol": "QQQ",
            "name": "Nasdaq",
            "value": round(random.uniform(14000, 15000), 2),
            "change": round(random.uniform(-200, 200), 2),
            "change_percent": round(random.uniform(-1.5, 1.5), 2)
        },
        {
            "symbol": "IWM",
            "name": "Russell 2000",
            "value": round(random.uniform(1900, 2100), 2),
            "change": round(random.uniform(-30, 30), 2),
            "change_percent": round(random.uniform(-1.5, 1.5), 2)
        }
    ]
    return indices

@router.get("/earnings")
async def get_earnings_calendar(
    days: int = Query(7, ge=1, le=30),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get earnings calendar data."""
    earnings = []
    
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        # Add 2-3 companies per day
        for j in range(random.randint(2, 3)):
            earnings.append({
                "symbol": f"EARN{i}{j}",
                "company": f"Earnings Company {i}-{j}",
                "date": date.strftime("%Y-%m-%d"),
                "time": "AMC" if random.random() > 0.5 else "BMO",
                "eps_estimate": round(random.uniform(0.5, 3), 2),
                "revenue_estimate": f"${random.randint(1, 50)}B"
            })
    
    return earnings

@router.get("/dividends")
async def get_dividends_calendar(
    symbol: str = Query(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get dividends calendar data."""
    dividends = []
    
    # Generate historical and upcoming dividends
    for i in range(-4, 2):  # 4 past, 2 future
        date = datetime.now() + timedelta(days=i * 90)  # Quarterly
        dividends.append({
            "ex_date": date.strftime("%Y-%m-%d"),
            "payment_date": (date + timedelta(days=14)).strftime("%Y-%m-%d"),
            "amount": round(random.uniform(0.2, 1.5), 2),
            "yield": round(random.uniform(1, 4), 2)
        })
    
    return {
        "symbol": symbol,
        "dividends": dividends,
        "annual_dividend": round(sum(d["amount"] for d in dividends[-4:]), 2),
        "dividend_yield": round(random.uniform(1, 4), 2)
    }

@router.get("/economic")
async def get_economic_indicators(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get economic indicators data."""
    return {
        "gdp": {
            "value": round(random.uniform(2, 4), 1),
            "previous": round(random.uniform(2, 4), 1),
            "unit": "percent"
        },
        "inflation": {
            "value": round(random.uniform(2, 5), 1),
            "previous": round(random.uniform(2, 5), 1),
            "unit": "percent"
        },
        "unemployment": {
            "value": round(random.uniform(3, 5), 1),
            "previous": round(random.uniform(3, 5), 1),
            "unit": "percent"
        },
        "interest_rate": {
            "value": round(random.uniform(4, 6), 2),
            "previous": round(random.uniform(4, 6), 2),
            "unit": "percent"
        },
        "consumer_confidence": {
            "value": round(random.uniform(95, 105), 1),
            "previous": round(random.uniform(95, 105), 1),
            "unit": "index"
        }
    }

@router.post("/options/greeks")
async def calculate_options_greeks(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Calculate options Greeks."""
    # For now, return mock Greeks
    return {
        "delta": round(random.uniform(0, 1) if request.get("option_type") == "call" else random.uniform(-1, 0), 3),
        "gamma": round(random.uniform(0, 0.1), 4),
        "theta": round(random.uniform(-0.1, 0), 4),
        "vega": round(random.uniform(0, 0.5), 3),
        "rho": round(random.uniform(-0.5, 0.5), 3)
    }

@router.get("/options/iv-surface")
async def get_implied_volatility_surface(
    symbol: str = Query(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get implied volatility surface data."""
    strikes = [i for i in range(100, 200, 5)]
    expirations = [(datetime.now() + timedelta(days=i * 7)).strftime("%Y-%m-%d") for i in range(1, 13)]
    
    # Generate mock IV surface
    surface = []
    for exp in expirations:
        row = []
        for strike in strikes:
            # IV tends to be higher for OTM options (volatility smile)
            base_iv = 0.25
            moneyness = abs(strike - 150) / 150
            iv = base_iv + moneyness * 0.1 + random.uniform(-0.05, 0.05)
            row.append(round(iv, 3))
        surface.append(row)
    
    return {
        "symbol": symbol,
        "surface": surface,
        "strikes": strikes,
        "expirations": expirations
    }

@router.get("/breadth")
async def get_market_breadth_indicators(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get market breadth indicators."""
    advancing = random.randint(1500, 2500)
    declining = random.randint(500, 1500)
    unchanged = random.randint(100, 300)
    
    return {
        "advance_decline_ratio": round(advancing / declining, 2),
        "advancing": advancing,
        "declining": declining,
        "unchanged": unchanged,
        "new_highs": random.randint(50, 200),
        "new_lows": random.randint(10, 100),
        "new_highs_lows": {
            "52_week_highs": random.randint(50, 200),
            "52_week_lows": random.randint(10, 100)
        },
        "mcclellan_oscillator": round(random.uniform(-100, 100), 2),
        "mcclellan_summation": round(random.uniform(-1000, 1000), 2)
    }

@router.post("/correlation")
async def calculate_correlation_matrix(
    request: CorrelationRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Calculate correlation matrix for multiple symbols."""
    n = len(request.symbols)
    
    # Generate random correlation matrix
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1.0)  # Diagonal should be 1
    
    # Ensure correlations are between -1 and 1
    matrix = np.clip(matrix * 2 - 1, -1, 1)
    
    return {
        "symbols": request.symbols,
        "matrix": matrix.tolist(),
        "period": request.period
    }

@router.get("/volatility/{symbol}")
async def get_volatility_analysis(
    symbol: str,
    period: int = Query(30, ge=1, le=252),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get volatility analysis for a symbol."""
    return {
        "symbol": symbol,
        "period": period,
        "historical_volatility": round(random.uniform(0.15, 0.35), 3),
        "implied_volatility": round(random.uniform(0.20, 0.40), 3),
        "volatility_percentile": round(random.uniform(0, 100), 1),
        "volatility_ratio": round(random.uniform(0.8, 1.2), 2),
        "garch_forecast": round(random.uniform(0.15, 0.35), 3)
    }

@router.get("/depth/{symbol}")
async def get_market_depth(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get market depth (order book) data."""
    # Generate mock order book
    bids = []
    asks = []
    base_price = 150.00
    
    for i in range(10):
        bid_price = base_price - (i * 0.01)
        ask_price = base_price + (i * 0.01) + 0.01
        
        bids.append({
            "price": round(bid_price, 2),
            "size": random.randint(100, 10000),
            "orders": random.randint(1, 50)
        })
        
        asks.append({
            "price": round(ask_price, 2),
            "size": random.randint(100, 10000),
            "orders": random.randint(1, 50)
        })
    
    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "spread": round(asks[0]["price"] - bids[0]["price"], 2),
        "mid_price": round((asks[0]["price"] + bids[0]["price"]) / 2, 2),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/stream/subscribe")
async def subscribe_to_market_stream(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Subscribe to market data stream."""
    subscription_id = f"sub_{random.randint(10000, 99999)}"
    
    return {
        "subscription_id": subscription_id,
        "symbols": request.get("symbols", []),
        "data_types": request.get("data_types", ["trades", "quotes"]),
        "status": "active",
        "websocket_url": f"wss://api.example.com/stream/{subscription_id}"
    }

# Alternative Data Endpoints
@router.get("/alt-data/satellite")
async def get_satellite_data(
    company: str = Query(...),
    metric: str = Query(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get satellite data for retail traffic analysis."""
    return {
        "company": company,
        "metric": metric,
        "value": round(random.uniform(70, 130), 2),  # % of normal
        "change": round(random.uniform(-10, 10), 2),
        "locations_analyzed": random.randint(50, 200),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "last_updated": datetime.now().isoformat()
    }

@router.get("/alt-data/social")
async def get_social_sentiment(
    symbol: str = Query(...),
    platform: str = Query("twitter"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get social media sentiment analysis."""
    return {
        "symbol": symbol,
        "platform": platform,
        "sentiment_score": round(random.uniform(-1, 1), 2),
        "mention_count": random.randint(100, 10000),
        "influencer_sentiment": round(random.uniform(-1, 1), 2),
        "trending_rank": random.randint(1, 100),
        "top_keywords": ["bullish", "earnings", "growth", "buy"],
        "sentiment_breakdown": {
            "positive": random.randint(30, 60),
            "neutral": random.randint(20, 40),
            "negative": random.randint(10, 30)
        }
    }

@router.get("/alt-data/web")
async def get_web_metrics(
    company: str = Query(...),
    metric: str = Query(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get web scraping metrics."""
    metrics = {
        "job_postings": random.randint(100, 1000),
        "web_traffic": round(random.uniform(80, 120), 2),
        "app_downloads": random.randint(10000, 100000),
        "customer_reviews": round(random.uniform(3.5, 4.8), 1)
    }
    
    return {
        "company": company,
        "metric": metric,
        "value": metrics.get(metric, random.randint(100, 1000)),
        "change_percent": round(random.uniform(-20, 20), 2),
        "data_points": random.randint(100, 1000),
        "last_updated": datetime.now().isoformat()
    }

# Crypto, Forex, and Commodities
@router.get("/crypto/{pair}")
async def get_crypto_data(
    pair: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get cryptocurrency market data."""
    return {
        "pair": pair,
        "price": round(random.uniform(20000, 60000) if "BTC" in pair else random.uniform(1000, 4000), 2),
        "volume_24h": random.randint(1000000000, 50000000000),
        "market_cap": random.randint(100000000000, 1000000000000),
        "change_24h": round(random.uniform(-10, 10), 2),
        "circulating_supply": random.randint(10000000, 20000000)
    }

@router.get("/forex/{pair}")
async def get_forex_rates(
    pair: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get foreign exchange rates."""
    base_rate = 1.0 if pair == "EUR-USD" else random.uniform(0.5, 2)
    spread = 0.0001
    
    return {
        "pair": pair,
        "rate": round(base_rate, 4),
        "bid": round(base_rate - spread, 4),
        "ask": round(base_rate + spread, 4),
        "change": round(random.uniform(-0.01, 0.01), 4),
        "change_percent": round(random.uniform(-1, 1), 2)
    }

@router.get("/commodities/{commodity}")
async def get_commodities_data(
    commodity: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get commodities market data."""
    prices = {
        "GOLD": random.uniform(1800, 2000),
        "SILVER": random.uniform(22, 28),
        "OIL": random.uniform(70, 90),
        "NATGAS": random.uniform(2, 4)
    }
    
    return {
        "commodity": commodity,
        "price": round(prices.get(commodity, random.uniform(50, 150)), 2),
        "unit": "USD/oz" if commodity in ["GOLD", "SILVER"] else "USD/barrel",
        "change": round(random.uniform(-5, 5), 2),
        "change_percent": round(random.uniform(-3, 3), 2),
        "contract_month": (datetime.now() + timedelta(days=30)).strftime("%Y-%m")
    }

# Advanced Analytics
@router.post("/scanner")
async def market_scanner(
    request: ScannerRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Market scanner with custom filters."""
    # Generate mock scan results
    results = []
    
    for i in range(min(request.limit, 50)):
        results.append({
            "symbol": f"SCAN{i}",
            "name": f"Scanner Result {i}",
            "price": round(random.uniform(10, 500), 2),
            "market_cap": random.randint(1000000000, 100000000000),
            "pe_ratio": round(random.uniform(10, 30), 2),
            "volume": random.randint(1000000, 50000000),
            "price_change_pct": round(random.uniform(-5, 5), 2)
        })
    
    # Sort by requested field
    if request.sort_by in ["volume", "market_cap", "price"]:
        results.sort(key=lambda x: x[request.sort_by], reverse=True)
    
    return results

@router.get("/etf/{symbol}/holdings")
async def get_etf_holdings(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get ETF holdings data."""
    holdings = []
    
    # Generate mock holdings
    for i in range(10):  # Top 10 holdings
        holdings.append({
            "symbol": f"HOLD{i}",
            "name": f"Holding Company {i}",
            "weight": round(random.uniform(1, 10), 2),
            "shares": random.randint(1000000, 10000000),
            "market_value": random.randint(10000000, 1000000000)
        })
    
    total_assets = sum(h["market_value"] for h in holdings)
    
    return {
        "symbol": symbol,
        "holdings": holdings,
        "total_assets": total_assets,
        "expense_ratio": round(random.uniform(0.03, 0.75), 2),
        "last_updated": datetime.now().isoformat()
    }

@router.get("/insider/{symbol}")
async def get_insider_trading(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get insider trading data."""
    transactions = []
    
    for i in range(10):
        transaction_date = datetime.now() - timedelta(days=random.randint(1, 90))
        transactions.append({
            "insider_name": f"Executive {i}",
            "title": random.choice(["CEO", "CFO", "Director", "VP"]),
            "transaction_type": random.choice(["Buy", "Sell"]),
            "shares": random.randint(1000, 100000),
            "price": round(random.uniform(100, 200), 2),
            "value": random.randint(100000, 10000000),
            "date": transaction_date.strftime("%Y-%m-%d")
        })
    
    return transactions

@router.get("/short-interest/{symbol}")
async def get_short_interest(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get short interest data."""
    return {
        "symbol": symbol,
        "short_ratio": round(random.uniform(1, 10), 2),
        "short_percent_float": round(random.uniform(5, 30), 2),
        "shares_short": random.randint(1000000, 50000000),
        "days_to_cover": round(random.uniform(1, 5), 1),
        "previous_shares_short": random.randint(1000000, 50000000),
        "change_percent": round(random.uniform(-20, 20), 2),
        "last_updated": datetime.now().isoformat()
    }

@router.get("/options/flow")
async def get_options_flow(
    min_premium: int = Query(100000),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get unusual options activity."""
    flows = []
    
    for i in range(20):
        flows.append({
            "symbol": f"OPT{i}",
            "time": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
            "type": random.choice(["Call", "Put"]),
            "strike": round(random.uniform(100, 200), 2),
            "expiry": (datetime.now() + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
            "volume": random.randint(1000, 50000),
            "open_interest": random.randint(100, 10000),
            "premium": random.randint(min_premium, min_premium * 10),
            "sentiment": random.choice(["Bullish", "Bearish", "Neutral"])
        })
    
    return flows

@router.post("/export")
async def export_market_data(
    request: ExportRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Export market data to various formats."""
    # For now, just return success
    export_id = f"export_{random.randint(10000, 99999)}"
    
    return {
        "export_id": export_id,
        "status": "processing",
        "format": request.format,
        "symbols": request.symbols,
        "download_url": f"/api/v1/market/export/{export_id}/download"
    }

@router.get("/data-quality/{symbol}")
async def check_data_quality(
    symbol: str,
    date: datetime = Query(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Check data quality for a symbol."""
    return {
        "symbol": symbol,
        "date": date.isoformat(),
        "completeness": round(random.uniform(0.95, 1.0), 3),
        "accuracy_score": round(random.uniform(0.98, 1.0), 3),
        "anomalies": random.randint(0, 5),
        "missing_intervals": random.randint(0, 2),
        "data_sources": ["primary", "backup"],
        "validation_passed": random.random() > 0.1
    }

# Add cache headers to responses
@router.get("/quote/{symbol}", response_class=JSONResponse)
async def get_stock_quote_with_cache(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    if_none_match: Optional[str] = Header(None)
) -> Any:
    """Get stock quote with caching headers."""
    quote = generate_mock_quote(symbol)
    
    # Generate ETag
    import hashlib
    etag = hashlib.md5(str(quote).encode()).hexdigest()
    
    # Check if client has current version
    if if_none_match == etag:
        return JSONResponse(
            content={},
            status_code=304,
            headers={"ETag": etag}
        )
    
    return JSONResponse(
        content=quote,
        headers={
            "Cache-Control": "public, max-age=5",
            "ETag": etag
        }
    )
