"""Market data endpoints for real-time quotes and indicators."""

from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import json
import logging

from app.db.database import get_db
from app.models.stock import MarketIndicator
from app.core.security import get_current_active_user
from app.services.market_data import MarketDataService
from app.services.websocket import manager, broadcast_price_update

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get real-time quote for a symbol."""
    market_service = MarketDataService()
    
    try:
        # Fetch real-time quote
        stock_data = await market_service.fetch_stock_data(symbol)
        
        if not stock_data:
            raise HTTPException(
                status_code=404,
                detail=f"Quote not available for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "price": stock_data['current_price'],
            "bid": stock_data.get('bid'),
            "ask": stock_data.get('ask'),
            "bid_size": stock_data.get('bid_size'),
            "ask_size": stock_data.get('ask_size'),
            "volume": stock_data['volume'],
            "timestamp": stock_data.get('last_updated', datetime.utcnow().isoformat())
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching quote: {str(e)}"
        )


@router.post("/quotes/batch")
async def get_batch_quotes(
    symbols: List[str],
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get batch quotes for multiple symbols."""
    if len(symbols) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 symbols allowed per request"
        )
    
    market_service = MarketDataService()
    quotes = {}
    
    for symbol in symbols:
        try:
            stock_data = await market_service.fetch_stock_data(symbol)
            if stock_data:
                quotes[symbol.upper()] = {
                    "price": stock_data['current_price'],
                    "change": stock_data['change_amount'],
                    "change_percent": stock_data['change_percent'],
                    "volume": stock_data['volume']
                }
        except Exception:
            quotes[symbol.upper()] = None
    
    return quotes


@router.get("/indicators")
async def get_market_indicators(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get major market indicators."""
    market_service = MarketDataService()
    
    try:
        indicators = await market_service.fetch_market_indicators()
        
        # Save to database
        for indicator_data in indicators:
            # Update or create indicator
            result = await db.execute(
                select(MarketIndicator).where(
                    MarketIndicator.symbol == indicator_data['symbol']
                )
            )
            indicator = result.scalar_one_or_none()
            
            if indicator:
                for key, value in indicator_data.items():
                    setattr(indicator, key, value)
                indicator.last_updated = datetime.utcnow()
            else:
                indicator = MarketIndicator(**indicator_data)
                db.add(indicator)
        
        await db.commit()
        
        return indicators
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching market indicators: {str(e)}"
        )


@router.get("/history/{symbol}")
async def get_historical_data(
    symbol: str,
    interval: str = "1d",
    period: str = "1mo",
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get historical price data with technical indicators."""
    market_service = MarketDataService()
    
    try:
        history = await market_service.fetch_price_history(symbol, interval, period)
        
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"Historical data not available for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "period": period,
            "data": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching historical data: {str(e)}"
        )


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time market data streaming."""
    await manager.connect(websocket, "market_data")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                for symbol in symbols:
                    # Subscribe to symbol updates
                    await manager.connect(websocket, f"symbol_{symbol}")
                    
                await websocket.send_json({
                    "type": "subscribed",
                    "symbols": symbols
                })
            
            elif message.get("action") == "unsubscribe":
                symbols = message.get("symbols", [])
                for symbol in symbols:
                    # Unsubscribe from symbol updates
                    manager.disconnect(websocket, f"symbol_{symbol}")
                
                await websocket.send_json({
                    "type": "unsubscribed",
                    "symbols": symbols
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "market_data")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, "market_data")


@router.post("/stream/price-update")
async def stream_price_update(
    symbol: str,
    price: float,
    change: float,
    change_percent: float,
    volume: int,
    current_user = Depends(get_current_active_user)
) -> Any:
    """Internal endpoint to broadcast price updates."""
    # This would typically be called by a background task
    await broadcast_price_update(symbol, {
        "price": price,
        "change": change,
        "change_percent": change_percent,
        "volume": volume,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return {"status": "broadcasted"}
