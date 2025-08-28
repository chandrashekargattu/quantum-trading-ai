"""Stock market data endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.stock import Stock, PriceHistory
from app.core.security import get_current_active_user
from app.schemas.stock import StockResponse, StockSearchResponse, PriceHistoryResponse
from app.services.market_data import MarketDataService

router = APIRouter()


@router.get("/search", response_model=List[StockSearchResponse])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Search stocks by symbol or name."""
    search_term = f"%{q.upper()}%"
    
    result = await db.execute(
        select(Stock).where(
            and_(
                Stock.is_active == True,
                or_(
                    Stock.symbol.ilike(search_term),
                    Stock.name.ilike(search_term)
                )
            )
        ).limit(limit)
    )
    
    stocks = result.scalars().all()
    return stocks


@router.get("/movers", response_model=dict)
async def get_market_movers(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get top gainers, losers, and most active stocks."""
    # Top gainers
    gainers_result = await db.execute(
        select(Stock).where(
            and_(
                Stock.is_active == True,
                Stock.change_percent > 0
            )
        ).order_by(Stock.change_percent.desc()).limit(10)
    )
    gainers = gainers_result.scalars().all()
    
    # Top losers
    losers_result = await db.execute(
        select(Stock).where(
            and_(
                Stock.is_active == True,
                Stock.change_percent < 0
            )
        ).order_by(Stock.change_percent.asc()).limit(10)
    )
    losers = losers_result.scalars().all()
    
    # Most active
    active_result = await db.execute(
        select(Stock).where(
            Stock.is_active == True
        ).order_by(Stock.volume.desc()).limit(10)
    )
    most_active = active_result.scalars().all()
    
    return {
        "gainers": gainers,
        "losers": losers,
        "most_active": most_active
    }


@router.get("/{symbol}", response_model=StockResponse)
async def get_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get stock details by symbol."""
    result = await db.execute(
        select(Stock).where(
            and_(
                Stock.symbol == symbol.upper(),
                Stock.is_active == True
            )
        )
    )
    stock = result.scalar_one_or_none()
    
    if not stock:
        # Try to fetch from external API
        market_service = MarketDataService()
        stock_data = await market_service.fetch_stock_data(symbol)
        
        if not stock_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock {symbol} not found"
            )
        
        # Create new stock record
        stock = Stock(**stock_data)
        db.add(stock)
        await db.commit()
        await db.refresh(stock)
    
    # Update if data is stale (older than 5 minutes)
    elif stock.last_updated < datetime.utcnow() - timedelta(minutes=5):
        market_service = MarketDataService()
        updated_data = await market_service.fetch_stock_data(symbol)
        
        if updated_data:
            for key, value in updated_data.items():
                setattr(stock, key, value)
            stock.last_updated = datetime.utcnow()
            await db.commit()
    
    return stock


@router.get("/{symbol}/history", response_model=List[PriceHistoryResponse])
async def get_price_history(
    symbol: str,
    interval: str = Query("1d", regex="^(1m|5m|15m|30m|1h|1d|1w|1M)$"),
    period: str = Query("1mo", regex="^(1d|5d|1mo|3mo|6mo|1y|2y|5y)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get historical price data for a stock."""
    # First check if stock exists
    stock_result = await db.execute(
        select(Stock).where(Stock.symbol == symbol.upper())
    )
    stock = stock_result.scalar_one_or_none()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found"
        )
    
    # Calculate date range based on period
    end_date = datetime.utcnow()
    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
    }
    start_date = end_date - period_map[period]
    
    # Query historical data
    result = await db.execute(
        select(PriceHistory).where(
            and_(
                PriceHistory.stock_id == stock.id,
                PriceHistory.timeframe == interval,
                PriceHistory.timestamp >= start_date,
                PriceHistory.timestamp <= end_date
            )
        ).order_by(PriceHistory.timestamp.asc())
    )
    
    history = result.scalars().all()
    
    # If no data, fetch from external API
    if not history:
        market_service = MarketDataService()
        history_data = await market_service.fetch_price_history(
            symbol, interval, period
        )
        
        if history_data:
            # Save to database
            for data in history_data:
                price_history = PriceHistory(
                    stock_id=stock.id,
                    timeframe=interval,
                    **data
                )
                db.add(price_history)
            
            await db.commit()
            
            # Re-query
            result = await db.execute(
                select(PriceHistory).where(
                    and_(
                        PriceHistory.stock_id == stock.id,
                        PriceHistory.timeframe == interval,
                        PriceHistory.timestamp >= start_date
                    )
                ).order_by(PriceHistory.timestamp.asc())
            )
            history = result.scalars().all()
    
    return history


@router.post("/{symbol}/refresh")
async def refresh_stock_data(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Force refresh stock data from external API."""
    market_service = MarketDataService()
    stock_data = await market_service.fetch_stock_data(symbol)
    
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found"
        )
    
    # Update or create stock
    result = await db.execute(
        select(Stock).where(Stock.symbol == symbol.upper())
    )
    stock = result.scalar_one_or_none()
    
    if stock:
        for key, value in stock_data.items():
            setattr(stock, key, value)
        stock.last_updated = datetime.utcnow()
    else:
        stock = Stock(**stock_data)
        db.add(stock)
    
    await db.commit()
    await db.refresh(stock)
    
    return stock
