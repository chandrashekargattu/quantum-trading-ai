"""Trading endpoints for order execution and management."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from datetime import datetime
import uuid

from app.db.database import get_db
from app.models.trade import Trade, OrderBook
from app.models.position import Position
from app.models.stock import Stock
from app.models.option import Option
from app.models.portfolio import Portfolio
from app.core.security import get_current_active_user
from app.schemas.trade import (
    TradeResponse, 
    OrderRequest, 
    OrderResponse,
    PositionResponse,
    OrderBookResponse
)
from app.services.trading_engine import TradingEngine

router = APIRouter()


@router.post("/order", response_model=OrderResponse)
async def place_order(
    order: OrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Place a new order."""
    # Validate the asset exists
    if order.asset_type == "stock":
        stock_result = await db.execute(
            select(Stock).where(Stock.symbol == order.symbol.upper())
        )
        asset = stock_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock {order.symbol} not found"
            )
    elif order.asset_type == "option" and order.option_id:
        option_result = await db.execute(
            select(Option).where(Option.id == order.option_id)
        )
        asset = option_result.scalar_one_or_none()
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Option not found"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid asset type or missing option_id"
        )
    
    # Get user's default portfolio
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.user_id == current_user.id,
                Portfolio.is_default == True
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    
    if not portfolio:
        # Create default portfolio if none exists
        portfolio = Portfolio(
            user_id=current_user.id,
            name="Default Portfolio",
            portfolio_type="trading",
            is_default=True,
            cash_balance=current_user.account_balance
        )
        db.add(portfolio)
        await db.commit()
        await db.refresh(portfolio)
    
    # Check buying power
    if order.side == "buy":
        estimated_cost = order.quantity * (order.limit_price or asset.current_price)
        if estimated_cost > portfolio.buying_power:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient buying power"
            )
    
    # Create order in order book
    order_id = str(uuid.uuid4())
    order_book_entry = OrderBook(
        user_id=current_user.id,
        order_id=order_id,
        symbol=order.symbol.upper(),
        asset_type=order.asset_type,
        side=order.side,
        quantity=order.quantity,
        order_type=order.order_type,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        time_in_force=order.time_in_force or "day",
        status="pending"
    )
    db.add(order_book_entry)
    
    # For paper trading, execute immediately if market order
    if current_user.account_type == "paper" and order.order_type == "market":
        # Execute the trade
        trade = Trade(
            user_id=current_user.id,
            portfolio_id=portfolio.id,
            trade_id=str(uuid.uuid4()),
            order_id=order_id,
            symbol=order.symbol.upper(),
            asset_type=order.asset_type,
            option_id=order.option_id,
            side=order.side,
            quantity=order.quantity,
            price=asset.current_price,
            total_amount=order.quantity * asset.current_price,
            order_type=order.order_type,
            time_in_force=order.time_in_force or "day",
            status="filled",
            fill_price=asset.current_price,
            filled_quantity=order.quantity,
            is_paper=True,
            executed_at=datetime.utcnow()
        )
        db.add(trade)
        
        # Update order book
        order_book_entry.status = "filled"
        order_book_entry.filled_quantity = order.quantity
        
        # Update portfolio cash
        if order.side == "buy":
            portfolio.cash_balance -= trade.total_amount
        else:
            portfolio.cash_balance += trade.total_amount
        
        await db.commit()
        
        return {
            "order_id": order_id,
            "status": "filled",
            "message": "Order executed successfully"
        }
    
    await db.commit()
    
    return {
        "order_id": order_id,
        "status": "pending",
        "message": "Order placed successfully"
    }


@router.delete("/order/{order_id}", response_model=dict)
async def cancel_order(
    order_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Cancel a pending order."""
    result = await db.execute(
        select(OrderBook).where(
            and_(
                OrderBook.order_id == order_id,
                OrderBook.user_id == current_user.id,
                OrderBook.status == "pending"
            )
        )
    )
    order = result.scalar_one_or_none()
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found or already executed"
        )
    
    order.status = "cancelled"
    await db.commit()
    
    return {"message": "Order cancelled successfully"}


@router.get("/", response_model=List[TradeResponse])
async def get_trades(
    portfolio_id: Optional[str] = None,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user's trade history."""
    query = select(Trade).where(Trade.user_id == current_user.id)
    
    if portfolio_id:
        query = query.where(Trade.portfolio_id == portfolio_id)
    
    if symbol:
        query = query.where(Trade.symbol == symbol.upper())
    
    if status:
        query = query.where(Trade.status == status)
    
    query = query.order_by(Trade.created_at.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    trades = result.scalars().all()
    
    return trades


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(
    trade_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific trade details."""
    result = await db.execute(
        select(Trade).where(
            and_(
                Trade.id == trade_id,
                Trade.user_id == current_user.id
            )
        )
    )
    trade = result.scalar_one_or_none()
    
    if not trade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trade not found"
        )
    
    return trade


@router.get("/orders/open", response_model=List[OrderBookResponse])
async def get_open_orders(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get all open orders."""
    result = await db.execute(
        select(OrderBook).where(
            and_(
                OrderBook.user_id == current_user.id,
                OrderBook.status.in_(["pending", "partial"])
            )
        ).order_by(OrderBook.created_at.desc())
    )
    orders = result.scalars().all()
    
    return orders


@router.get("/orders/history", response_model=List[OrderBookResponse])
async def get_order_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get order history."""
    result = await db.execute(
        select(OrderBook).where(
            OrderBook.user_id == current_user.id
        ).order_by(OrderBook.created_at.desc()).offset(offset).limit(limit)
    )
    orders = result.scalars().all()
    
    return orders
