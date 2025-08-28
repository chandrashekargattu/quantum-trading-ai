"""Trading operations endpoints."""

from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_
from datetime import datetime, timedelta
import uuid

from app.db.database import get_db
from app.models.trading import Order, OrderStatus, OrderType, OrderSide
from app.models.trade import Trade
from app.models.portfolio import Portfolio
from app.models.position import Position, Transaction
from app.models.stock import Stock
from app.core.security import get_current_active_user
from app.schemas.trading import (
    OrderCreate,
    OrderResponse,
    OrderUpdate,
    TradeResponse,
    OrderBookResponse
)
from app.services.trading_engine import TradingEngine
from app.services.market_data import MarketDataService
from app.api.v1.websocket import manager

router = APIRouter()


@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order_data: OrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create a new order."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == order_data.portfolio_id,
                Portfolio.user_id == current_user.id,
                Portfolio.is_active == True
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get current market price
    market_service = MarketDataService()
    stock_data = await market_service.fetch_stock_data(order_data.symbol)
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to get market data for {order_data.symbol}"
        )
    
    current_price = stock_data['current_price']
    
    # Validate order based on type
    if order_data.order_type == OrderType.MARKET:
        # Market orders execute at current price
        order_price = current_price
    elif order_data.order_type == OrderType.LIMIT:
        if not order_data.limit_price:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit price required for limit orders"
            )
        order_price = order_data.limit_price
    elif order_data.order_type == OrderType.STOP:
        if not order_data.stop_price:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Stop price required for stop orders"
            )
        order_price = current_price  # Will execute at market when triggered
    else:  # STOP_LIMIT
        if not order_data.stop_price or not order_data.limit_price:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both stop and limit prices required for stop-limit orders"
            )
        order_price = order_data.limit_price
    
    # Calculate order value
    order_value = order_data.quantity * order_price
    commission = calculate_commission(order_value)
    total_cost = order_value + commission if order_data.side == OrderSide.BUY else 0
    
    # Check buying power for buy orders
    if order_data.side == OrderSide.BUY and total_cost > portfolio.buying_power:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient buying power. Required: ${total_cost:.2f}, Available: ${portfolio.buying_power:.2f}"
        )
    
    # Check position for sell orders
    if order_data.side == OrderSide.SELL:
        position_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio.id,
                    Position.symbol == order_data.symbol,
                    Position.is_open == True
                )
            )
        )
        position = position_result.scalar_one_or_none()
        if not position or position.quantity < order_data.quantity:
            available = position.quantity if position else 0
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient shares. Required: {order_data.quantity}, Available: {available}"
            )
    
    # Create order
    order = Order(
        portfolio_id=portfolio.id,
        symbol=order_data.symbol.upper(),
        quantity=order_data.quantity,
        side=order_data.side,
        order_type=order_data.order_type,
        limit_price=order_data.limit_price,
        stop_price=order_data.stop_price,
        time_in_force=order_data.time_in_force or "DAY",
        status=OrderStatus.PENDING,
        extended_hours=order_data.extended_hours or False
    )
    
    db.add(order)
    await db.commit()
    await db.refresh(order)
    
    # Send order to trading engine for execution
    background_tasks.add_task(
        execute_order,
        order_id=str(order.id),
        db=db,
        user_id=str(current_user.id)
    )
    
    # Send WebSocket update
    await manager.send_order_update(
        user_id=str(current_user.id),
        order_update={
            "order_id": str(order.id),
            "status": order.status.value,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "side": order.side.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return order


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    portfolio_id: Optional[str] = None,
    status: Optional[OrderStatus] = None,
    symbol: Optional[str] = None,
    side: Optional[OrderSide] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user orders with filtering."""
    # Build query
    query = select(Order).join(Portfolio).where(
        Portfolio.user_id == current_user.id
    )
    
    if portfolio_id:
        query = query.where(Order.portfolio_id == portfolio_id)
    
    if status:
        query = query.where(Order.status == status)
    
    if symbol:
        query = query.where(Order.symbol == symbol.upper())
    
    if side:
        query = query.where(Order.side == side)
    
    if start_date:
        query = query.where(Order.created_at >= start_date)
    
    if end_date:
        query = query.where(Order.created_at <= end_date)
    
    # Execute query with pagination
    query = query.order_by(Order.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    orders = result.scalars().all()
    
    return orders


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific order details."""
    result = await db.execute(
        select(Order).join(Portfolio).where(
            and_(
                Order.id == order_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    order = result.scalar_one_or_none()
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    return order


@router.patch("/orders/{order_id}", response_model=OrderResponse)
async def update_order(
    order_id: str,
    order_update: OrderUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Update an order (only pending orders can be modified)."""
    result = await db.execute(
        select(Order).join(Portfolio).where(
            and_(
                Order.id == order_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    order = result.scalar_one_or_none()
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    if order.status != OrderStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot modify order with status: {order.status.value}"
        )
    
    # Update allowed fields
    if order_update.quantity is not None:
        order.quantity = order_update.quantity
    
    if order_update.limit_price is not None:
        if order.order_type not in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot set limit price on non-limit order"
            )
        order.limit_price = order_update.limit_price
    
    if order_update.stop_price is not None:
        if order.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot set stop price on non-stop order"
            )
        order.stop_price = order_update.stop_price
    
    order.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(order)
    
    return order


@router.post("/orders/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Cancel a pending order."""
    result = await db.execute(
        select(Order).join(Portfolio).where(
            and_(
                Order.id == order_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    order = result.scalar_one_or_none()
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found"
        )
    
    if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order with status: {order.status.value}"
        )
    
    # Cancel order
    order.status = OrderStatus.CANCELLED
    order.cancelled_at = datetime.utcnow()
    order.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(order)
    
    # Send WebSocket update
    await manager.send_order_update(
        user_id=str(current_user.id),
        order_update={
            "order_id": str(order.id),
            "status": "cancelled",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return order


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    portfolio_id: Optional[str] = None,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user trades."""
    query = select(Trade).join(Portfolio).where(
        Portfolio.user_id == current_user.id
    )
    
    if portfolio_id:
        query = query.where(Trade.portfolio_id == portfolio_id)
    
    if symbol:
        query = query.where(Trade.symbol == symbol.upper())
    
    if start_date:
        query = query.where(Trade.executed_at >= start_date)
    
    if end_date:
        query = query.where(Trade.executed_at <= end_date)
    
    query = query.order_by(Trade.executed_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    trades = result.scalars().all()
    
    return trades


@router.get("/market-depth/{symbol}", response_model=OrderBookResponse)
async def get_market_depth(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get market depth / order book for a symbol."""
    market_service = MarketDataService()
    
    try:
        order_book = await market_service.fetch_order_book(symbol)
        
        if not order_book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order book not available for {symbol}"
            )
        
        return OrderBookResponse(
            symbol=symbol.upper(),
            bids=order_book.get('bids', []),
            asks=order_book.get('asks', []),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching order book: {str(e)}"
        )


def calculate_commission(order_value: float) -> float:
    """Calculate order commission."""
    # Simple commission model - can be made more sophisticated
    base_commission = 0.0  # Free trading
    per_share_commission = 0.0
    
    return base_commission


async def execute_order(order_id: str, db: AsyncSession, user_id: str):
    """Execute an order using the trading engine."""
    async with db.begin():
        # Get order
        result = await db.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if not order or order.status != OrderStatus.PENDING:
            return
        
        # Initialize trading engine
        engine = TradingEngine()
        
        try:
            # Execute order
            trades = await engine.execute_order(order, db)
            
            if trades:
                # Update order status
                filled_quantity = sum(trade.quantity for trade in trades)
                
                if filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = filled_quantity
                    order.filled_at = datetime.utcnow()
                else:
                    order.status = OrderStatus.PARTIAL
                    order.filled_quantity = filled_quantity
                
                # Calculate average fill price
                total_value = sum(trade.quantity * trade.price for trade in trades)
                order.avg_fill_price = total_value / filled_quantity
                
                # Send WebSocket update
                await manager.send_order_update(
                    user_id=user_id,
                    order_update={
                        "order_id": str(order.id),
                        "status": order.status.value,
                        "filled_quantity": filled_quantity,
                        "avg_fill_price": order.avg_fill_price,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Send execution alert if price improvement
                if order.order_type == OrderType.LIMIT and order.side == OrderSide.BUY:
                    if order.avg_fill_price < order.limit_price:
                        improvement = order.limit_price - order.avg_fill_price
                        await manager.send_execution_alert(
                            user_id=user_id,
                            alert={
                                "type": "execution",
                                "order_id": str(order.id),
                                "message": "Order filled at better price",
                                "improvement": improvement,
                                "saved_amount": improvement * filled_quantity
                            }
                        )
                
            else:
                # Order couldn't be executed immediately
                if order.order_type == OrderType.MARKET:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "No market liquidity"
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = str(e)
        
        order.updated_at = datetime.utcnow()
        await db.commit()
