"""High-frequency trading endpoints."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.db.database import get_db
from app.core.security import get_current_active_user
from app.models import User
from app.hft.hft_engine import HFTEngine, HFTOrder

router = APIRouter()

# Store active HFT engines
hft_engines = {}


@router.post("/initialize-hft")
async def initialize_hft_engine(
    symbols: List[str],
    venues: List[str],
    risk_limits: Dict[str, float],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Initialize a high-frequency trading engine.
    
    - **symbols**: List of symbols to trade
    - **venues**: List of trading venues
    - **risk_limits**: Risk parameters (max_position, max_loss, etc.)
    """
    try:
        engine_id = f"hft_{current_user.id}_{len(hft_engines)}"
        
        # Initialize HFT engine
        engine = HFTEngine(
            symbols=symbols,
            venues=venues,
            risk_limits=risk_limits
        )
        
        hft_engines[engine_id] = engine
        
        # Start engine
        asyncio.create_task(engine.start())
        
        return {
            "engine_id": engine_id,
            "status": "running",
            "symbols": symbols,
            "venues": venues,
            "risk_limits": risk_limits
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize HFT engine: {str(e)}"
        )


@router.post("/send-hft-order")
async def send_hft_order(
    engine_id: str,
    symbol: str,
    side: str,
    order_type: str,
    quantity: float,
    price: Optional[float] = None,
    time_in_force: str = "IOC",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Send an order through the HFT engine.
    
    - **engine_id**: HFT engine identifier
    - **symbol**: Trading symbol
    - **side**: buy or sell
    - **order_type**: limit, market, pegged
    - **quantity**: Order quantity
    - **price**: Limit price (required for limit orders)
    - **time_in_force**: IOC, FOK, GTC
    """
    if engine_id not in hft_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HFT engine not found"
        )
    
    try:
        engine = hft_engines[engine_id]
        
        # Create order
        order = HFTOrder(
            order_id=f"API_{current_user.id}_{asyncio.get_event_loop().time()}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=Decimal(str(price)) if price else None,
            quantity=Decimal(str(quantity)),
            time_in_force=time_in_force
        )
        
        # Send order
        await engine._send_order(order)
        
        return {
            "order_id": order.order_id,
            "status": "sent",
            "timestamp": order.timestamp
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send order: {str(e)}"
        )


@router.get("/hft-metrics/{engine_id}")
async def get_hft_metrics(
    engine_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get real-time metrics from HFT engine."""
    if engine_id not in hft_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HFT engine not found"
        )
    
    engine = hft_engines[engine_id]
    
    return {
        "orders_sent": engine.metrics["orders_sent"],
        "orders_filled": engine.metrics["orders_filled"],
        "orders_cancelled": engine.metrics["orders_cancelled"],
        "total_volume": float(engine.metrics["total_volume"]),
        "execution_latency_us": engine.executor.get_average_latency() * 1e6,  # Convert to microseconds
        "position_tracker": {k: float(v) for k, v in engine.position_tracker.items()},
        "pnl_tracker": {k: float(v) for k, v in engine.pnl_tracker.items()},
        "is_running": engine.is_running
    }


@router.get("/order-book/{symbol}")
async def get_order_book(
    symbol: str,
    engine_id: str,
    levels: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get current order book state."""
    if engine_id not in hft_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HFT engine not found"
        )
    
    engine = hft_engines[engine_id]
    
    if symbol not in engine.order_books:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symbol not found"
        )
    
    order_book = engine.order_books[symbol]
    depth = order_book.get_market_depth(levels)
    
    return {
        "symbol": symbol,
        "bids": [
            {
                "price": float(level.price),
                "quantity": float(level.quantity),
                "order_count": level.order_count
            }
            for level in depth["bids"]
        ],
        "asks": [
            {
                "price": float(level.price),
                "quantity": float(level.quantity),
                "order_count": level.order_count
            }
            for level in depth["asks"]
        ],
        "timestamp": order_book.last_update_time
    }


@router.websocket("/hft-stream/{engine_id}")
async def hft_websocket_stream(
    websocket: WebSocket,
    engine_id: str
):
    """WebSocket stream for real-time HFT data."""
    if engine_id not in hft_engines:
        await websocket.close(code=4004, reason="Engine not found")
        return
    
    await websocket.accept()
    engine = hft_engines[engine_id]
    
    try:
        while True:
            # Send metrics every 100ms
            metrics = {
                "type": "metrics",
                "data": {
                    "orders_sent": engine.metrics["orders_sent"],
                    "orders_filled": engine.metrics["orders_filled"],
                    "total_volume": float(engine.metrics["total_volume"]),
                    "positions": {k: float(v) for k, v in engine.position_tracker.items()},
                    "pnl": {k: float(v) for k, v in engine.pnl_tracker.items()}
                }
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))


@router.post("/stop-hft/{engine_id}")
async def stop_hft_engine(
    engine_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, str]:
    """Stop an HFT engine."""
    if engine_id not in hft_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HFT engine not found"
        )
    
    engine = hft_engines[engine_id]
    await engine.stop()
    
    del hft_engines[engine_id]
    
    return {"message": "HFT engine stopped successfully"}
