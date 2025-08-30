"""Stock market data endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.stock import Stock, PriceHistory
from app.core.security import get_current_active_user
from app.schemas.stock import StockResponse, StockSearchResponse, PriceHistoryResponse, StockCreate
from app.services.market_data import MarketDataService
from app.services.stock_search_service import StockSearchService

router = APIRouter()


@router.get("/search", response_model=List[StockSearchResponse])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Intelligent stock search by symbol or name with fuzzy matching."""
    search_term = f"%{q}%"
    
    # First, search in database with case-insensitive partial matching
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
    
    # If no or few stocks found, use intelligent search
    if len(stocks) < 3 and len(q) >= 1:
        search_service = StockSearchService()
        
        # Get intelligent search results
        search_results = await search_service.intelligent_search(q, limit)
        
        # Add new stocks to database and merge with existing results
        for stock_data in search_results:
            # Check if stock already exists in our results
            existing = next((s for s in stocks if s.symbol == stock_data['symbol']), None)
            
            if not existing:
                # Check if stock exists in database
                db_result = await db.execute(
                    select(Stock).where(Stock.symbol == stock_data['symbol'])
                )
                existing_db_stock = db_result.scalar_one_or_none()
                
                if existing_db_stock:
                    # Update existing stock with latest data
                    existing_db_stock.current_price = stock_data['current_price']
                    existing_db_stock.change_percent = stock_data['change_percent']
                    existing_db_stock.volume = stock_data['volume']
                    existing_db_stock.last_updated = datetime.utcnow()
                    stocks.append(existing_db_stock)
                else:
                    # Create new stock in database
                    new_stock = Stock(
                        symbol=stock_data['symbol'],
                        name=stock_data['name'],
                        exchange=stock_data['exchange'],
                        current_price=stock_data['current_price'],
                        previous_close=stock_data['previous_close'],
                        market_cap=stock_data['market_cap'],
                        pe_ratio=stock_data['pe_ratio'],
                        volume=stock_data['volume'],
                        week_52_high=stock_data['week_52_high'],
                        week_52_low=stock_data['week_52_low'],
                        change_amount=stock_data['change_amount'],
                        change_percent=stock_data['change_percent'],
                        is_active=True,
                        is_optionable=stock_data.get('is_optionable', False)
                    )
                    db.add(new_stock)
                    stocks.append(new_stock)
        
        # Commit all changes
        await db.commit()
        
        # Refresh all new stocks
        for stock in stocks:
            if stock.id is None:
                await db.refresh(stock)
    
    # Sort by relevance (exact matches first, then partial matches)
    stocks.sort(key=lambda s: (
        not s.symbol.upper().startswith(q.upper()),
        not q.upper() in s.symbol.upper(),
        s.symbol
    ))
    
    return stocks[:limit]


@router.post("/add", response_model=StockResponse)
async def add_stock(
    stock_data: StockCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Add a new stock to track."""
    # Check if stock already exists
    result = await db.execute(
        select(Stock).where(Stock.symbol == stock_data.symbol.upper())
    )
    existing_stock = result.scalar_one_or_none()
    
    if existing_stock:
        if not existing_stock.is_active:
            existing_stock.is_active = True
            await db.commit()
            await db.refresh(existing_stock)
            return existing_stock
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stock already exists"
        )
    
    # Fetch real-time data
    market_service = MarketDataService()
    symbol = stock_data.symbol.upper()
    
    # Add appropriate suffix if not present
    if not any(symbol.endswith(suffix) for suffix in ['.NS', '.BO', '.NSE', '.BSE']):
        # Default to NSE unless BSE is specified
        if stock_data.exchange and stock_data.exchange.upper() in ['BSE', 'BOM']:
            symbol = f"{symbol}.BO"
        else:
            symbol = f"{symbol}.NS"
    
    real_time_data = await market_service.fetch_stock_data(symbol)
    
    if not real_time_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unable to fetch data for symbol {symbol}"
        )
    
    # Create new stock
    new_stock = Stock(
        symbol=symbol,
        name=real_time_data.get('name', stock_data.name or symbol),
        exchange=real_time_data.get('exchange', stock_data.exchange or 'NSE'),
        current_price=real_time_data.get('current_price', 0),
        previous_close=real_time_data.get('previous_close', 0),
        open_price=real_time_data.get('open_price', 0),
        day_high=real_time_data.get('day_high', 0),
        day_low=real_time_data.get('day_low', 0),
        volume=real_time_data.get('volume', 0),
        market_cap=real_time_data.get('market_cap', 0),
        pe_ratio=real_time_data.get('pe_ratio'),
        week_52_high=real_time_data.get('week_52_high', 0),
        week_52_low=real_time_data.get('week_52_low', 0),
        change_amount=real_time_data.get('change_amount', 0),
        change_percent=real_time_data.get('change_percent', 0),
        is_active=True
    )
    
    db.add(new_stock)
    await db.commit()
    await db.refresh(new_stock)
    
    return new_stock


@router.get("/movers", response_model=dict)
async def get_market_movers(
    category: str = Query('all', description="Category: all, gainers, losers, active"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get market movers - gainers, losers, most active stocks dynamically."""
    search_service = StockSearchService()
    
    if category == 'all':
        # Get all categories
        gainers = await search_service.get_popular_stocks('gainers')
        losers = await search_service.get_popular_stocks('losers')
        active = await search_service.get_popular_stocks('active')
        
        return {
            "gainers": gainers[:5],
            "losers": losers[:5],
            "most_active": active[:5]
        }
    else:
        # Get specific category
        stocks = await search_service.get_popular_stocks(category)
        return {
            category: stocks
        }


@router.get("/categories", response_model=dict)
async def get_stock_categories(
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get popular stock categories for browsing."""
    search_service = StockSearchService()
    
    categories = {
        'nifty50': await search_service.get_popular_stocks('nifty50'),
        'banking': await search_service.get_popular_stocks('banking'),
        'it': await search_service.get_popular_stocks('it'),
        'pharma': await search_service.get_popular_stocks('pharma'),
        'auto': await search_service.get_popular_stocks('auto'),
        'fmcg': await search_service.get_popular_stocks('fmcg')
    }
    
    # Limit results for each category
    for cat in categories:
        categories[cat] = categories[cat][:10]
    
    return categories


@router.get("/watchlist", response_model=List[StockResponse])
async def get_watchlist(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user's watchlist stocks."""
    # For now, return all active stocks
    # TODO: Implement user-specific watchlist
    result = await db.execute(
        select(Stock).where(Stock.is_active == True).limit(20)
    )
    
    stocks = result.scalars().all()
    return stocks


@router.get("/{symbol}", response_model=StockResponse)
async def get_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get stock details by symbol."""
    result = await db.execute(
        select(Stock).where(Stock.symbol == symbol.upper())
    )
    
    stock = result.scalar_one_or_none()
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found"
        )
    
    return stock


@router.get("/{symbol}/history", response_model=List[PriceHistoryResponse])
async def get_stock_history(
    symbol: str,
    period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y|5Y)$"),
    interval: str = Query("1d", regex="^(1m|5m|15m|30m|1h|1d|1w|1M)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get stock price history."""
    # Calculate date range based on period
    end_date = datetime.utcnow()
    period_map = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "5Y": timedelta(days=1825)
    }
    start_date = end_date - period_map[period]
    
    # Get price history from database
    result = await db.execute(
        select(PriceHistory).where(
            and_(
                PriceHistory.symbol == symbol.upper(),
                PriceHistory.timestamp >= start_date
            )
        ).order_by(PriceHistory.timestamp.asc())
    )
    
    history = result.scalars().all()
    
    # If no history, fetch from market data service
    if not history:
        market_service = MarketDataService()
        historical_data = await market_service.fetch_price_history(
            symbol, interval, period.lower()
        )
        
        if historical_data:
            # Store in database for future use
            for data_point in historical_data:
                price_history = PriceHistory(
                    symbol=symbol.upper(),
                    timestamp=data_point['timestamp'],
                    open=data_point['open'],
                    high=data_point['high'],
                    low=data_point['low'],
                    close=data_point['close'],
                    volume=data_point['volume'],
                    adjusted_close=data_point.get('adjusted_close', data_point['close'])
                )
                db.add(price_history)
            
            await db.commit()
            
            # Re-fetch from database
            result = await db.execute(
                select(PriceHistory).where(
                    and_(
                        PriceHistory.symbol == symbol.upper(),
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
    """Refresh stock data from market."""
    # Get stock from database
    result = await db.execute(
        select(Stock).where(Stock.symbol == symbol.upper())
    )
    stock = result.scalar_one_or_none()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found"
        )
    
    # Fetch latest data
    market_service = MarketDataService()
    stock_data = await market_service.fetch_stock_data(symbol)
    
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch stock data"
        )
    
    # Update stock data
    stock.current_price = stock_data.get('current_price', stock.current_price)
    stock.previous_close = stock_data.get('previous_close', stock.previous_close)
    stock.open_price = stock_data.get('open_price', stock.open_price)
    stock.day_high = stock_data.get('day_high', stock.day_high)
    stock.day_low = stock_data.get('day_low', stock.day_low)
    stock.volume = stock_data.get('volume', stock.volume)
    stock.change_amount = stock_data.get('change_amount', stock.change_amount)
    stock.change_percent = stock_data.get('change_percent', stock.change_percent)
    stock.last_updated = datetime.utcnow()
    
    await db.commit()
    await db.refresh(stock)
    
    return {"message": "Stock data refreshed successfully", "stock": stock}


@router.delete("/{symbol}")
async def remove_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Remove a stock from user's watchlist (soft delete)."""
    # Get stock from database
    result = await db.execute(
        select(Stock).where(Stock.symbol == symbol.upper())
    )
    stock = result.scalar_one_or_none()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found"
        )
    
    # Soft delete - just mark as inactive
    stock.is_active = False
    stock.last_updated = datetime.utcnow()
    
    await db.commit()
    
    return {"message": f"Stock {symbol} removed from watchlist"}