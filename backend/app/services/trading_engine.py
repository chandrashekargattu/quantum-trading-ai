"""Trading engine for order execution."""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from app.models.trade import Trade, OrderBook
from app.models.position import Position
from app.models.portfolio import Portfolio
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class TradingEngine:
    """Engine for executing trades and managing positions."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def execute_order(
        self,
        order: OrderBook,
        current_price: float
    ) -> Optional[Trade]:
        """Execute an order based on current market conditions."""
        try:
            # Check order conditions
            can_execute = False
            execution_price = current_price
            
            if order.order_type == "market":
                can_execute = True
            elif order.order_type == "limit":
                if order.side == "buy" and current_price <= order.limit_price:
                    can_execute = True
                    execution_price = order.limit_price
                elif order.side == "sell" and current_price >= order.limit_price:
                    can_execute = True
                    execution_price = order.limit_price
            elif order.order_type == "stop":
                if order.side == "buy" and current_price >= order.stop_price:
                    can_execute = True
                elif order.side == "sell" and current_price <= order.stop_price:
                    can_execute = True
            
            if not can_execute:
                return None
            
            # Create trade record
            trade = Trade(
                user_id=order.user_id,
                trade_id=f"T{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                order_id=order.order_id,
                symbol=order.symbol,
                asset_type=order.asset_type,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                total_amount=order.quantity * execution_price,
                order_type=order.order_type,
                status="filled",
                fill_price=execution_price,
                filled_quantity=order.quantity,
                executed_at=datetime.utcnow()
            )
            
            # Update order status
            order.status = "filled"
            order.filled_quantity = order.quantity
            
            # Add to database
            self.db.add(trade)
            
            # Update position
            await self._update_position(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            return None
    
    async def _update_position(self, trade: Trade):
        """Update or create position based on trade."""
        # Find existing position
        from sqlalchemy import select, and_
        
        result = await self.db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == trade.portfolio_id,
                    Position.symbol == trade.symbol,
                    Position.is_open == True
                )
            )
        )
        position = result.scalar_one_or_none()
        
        if trade.side == "buy":
            if position:
                # Update existing position
                new_quantity = position.quantity + trade.quantity
                new_cost = (position.avg_cost * position.quantity + trade.total_amount) / new_quantity
                position.quantity = new_quantity
                position.avg_cost = new_cost
            else:
                # Create new position
                position = Position(
                    portfolio_id=trade.portfolio_id,
                    symbol=trade.symbol,
                    asset_type=trade.asset_type,
                    quantity=trade.quantity,
                    avg_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.total_amount,
                    unrealized_pnl=0,
                    unrealized_pnl_percent=0
                )
                self.db.add(position)
        
        elif trade.side == "sell" and position:
            # Reduce or close position
            if position.quantity <= trade.quantity:
                # Close position
                position.is_open = False
                position.closed_at = datetime.utcnow()
                position.realized_pnl = (trade.price - position.avg_cost) * position.quantity
            else:
                # Reduce position
                position.quantity -= trade.quantity
                # Realized P&L for the sold portion
                realized = (trade.price - position.avg_cost) * trade.quantity
                position.realized_pnl = (position.realized_pnl or 0) + realized
    
    async def calculate_position_value(
        self,
        position: Position,
        current_price: float
    ):
        """Calculate current position value and P&L."""
        position.current_price = current_price
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
        position.unrealized_pnl_percent = (position.unrealized_pnl / (position.quantity * position.avg_cost)) * 100
        position.last_updated = datetime.utcnow()
