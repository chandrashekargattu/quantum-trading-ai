"""Trading strategy endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.alert import Strategy
from app.models.trade import Trade
from app.core.security import get_current_active_user
from app.schemas.strategy import StrategyCreate, StrategyUpdate, StrategyResponse
from app.services.backtest import BacktestEngine

router = APIRouter()


@router.get("/", response_model=List[StrategyResponse])
async def get_strategies(
    active_only: bool = Query(False, description="Show only active strategies"),
    include_public: bool = Query(False, description="Include public strategies"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user's strategies."""
    query = select(Strategy)
    
    if include_public:
        query = query.where(
            or_(
                Strategy.user_id == current_user.id,
                Strategy.is_public == True
            )
        )
    else:
        query = query.where(Strategy.user_id == current_user.id)
    
    if active_only:
        query = query.where(Strategy.is_active == True)
    
    query = query.order_by(Strategy.created_at.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    strategies = result.scalars().all()
    
    return strategies


@router.post("/", response_model=StrategyResponse)
async def create_strategy(
    strategy_data: StrategyCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create a new trading strategy."""
    # Create strategy
    strategy = Strategy(
        user_id=current_user.id,
        name=strategy_data.name,
        description=strategy_data.description,
        strategy_type=strategy_data.strategy_type,
        config=strategy_data.config,
        max_position_size=strategy_data.max_position_size,
        stop_loss_percent=strategy_data.stop_loss_percent,
        take_profit_percent=strategy_data.take_profit_percent,
        is_public=strategy_data.is_public
    )
    
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    
    return strategy


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific strategy details."""
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                or_(
                    Strategy.user_id == current_user.id,
                    Strategy.is_public == True
                )
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    return strategy


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    strategy_update: StrategyUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Update a strategy."""
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Update fields
    update_data = strategy_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(strategy, field, value)
    
    strategy.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(strategy)
    
    return strategy


@router.delete("/{strategy_id}")
async def delete_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Delete a strategy."""
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    if strategy.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete active strategy. Deactivate it first."
        )
    
    await db.delete(strategy)
    await db.commit()
    
    return {"message": "Strategy deleted successfully"}


@router.post("/{strategy_id}/activate")
async def activate_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Activate a strategy for live/paper trading."""
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    if not strategy.backtest_results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy must be backtested before activation"
        )
    
    strategy.is_active = True
    strategy.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Strategy activated successfully"}


@router.post("/{strategy_id}/deactivate")
async def deactivate_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Deactivate a strategy."""
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                Strategy.user_id == current_user.id
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    strategy.is_active = False
    strategy.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Strategy deactivated successfully"}


@router.get("/{strategy_id}/performance")
async def get_strategy_performance(
    strategy_id: str,
    period: str = Query("all", regex="^(1w|1mo|3mo|6mo|1y|all)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get strategy performance metrics."""
    # Verify strategy access
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == strategy_id,
                or_(
                    Strategy.user_id == current_user.id,
                    Strategy.is_public == True
                )
            )
        )
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Get trades for this strategy
    trades_query = select(Trade).where(
        Trade.strategy_id == strategy_id
    )
    
    # Filter by period if not "all"
    if period != "all":
        period_map = {
            "1w": timedelta(days=7),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365)
        }
        start_date = datetime.utcnow() - period_map[period]
        trades_query = trades_query.where(Trade.created_at >= start_date)
    
    trades_result = await db.execute(trades_query)
    trades = trades_result.scalars().all()
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.realized_pnl > 0])
    losing_trades = len([t for t in trades if t.realized_pnl < 0])
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades
        avg_win = sum(t.realized_pnl for t in trades if t.realized_pnl > 0) / max(winning_trades, 1)
        avg_loss = sum(t.realized_pnl for t in trades if t.realized_pnl < 0) / max(losing_trades, 1)
        total_pnl = sum(t.realized_pnl for t in trades)
    else:
        win_rate = avg_win = avg_loss = total_pnl = 0
    
    return {
        "strategy_id": strategy_id,
        "period": period,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "sharpe_ratio": strategy.sharpe_ratio,
        "max_drawdown": 0,  # Would calculate from equity curve
        "backtest_results": strategy.backtest_results
    }


@router.get("/templates/")
async def get_strategy_templates(
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get predefined strategy templates."""
    templates = [
        {
            "name": "SMA Crossover",
            "description": "Buy when fast SMA crosses above slow SMA",
            "strategy_type": "trend_following",
            "config": {
                "fast_period": 20,
                "slow_period": 50,
                "position_size": 0.1
            }
        },
        {
            "name": "RSI Mean Reversion",
            "description": "Buy oversold, sell overbought",
            "strategy_type": "mean_reversion",
            "config": {
                "rsi_period": 14,
                "oversold_level": 30,
                "overbought_level": 70,
                "position_size": 0.1
            }
        },
        {
            "name": "Iron Condor",
            "description": "Options strategy for range-bound markets",
            "strategy_type": "options_spread",
            "config": {
                "strike_spacing": 5,
                "days_to_expiration": 30,
                "delta_target": 0.3
            }
        },
        {
            "name": "Covered Call",
            "description": "Generate income from stock holdings",
            "strategy_type": "options_spread",
            "config": {
                "strike_offset": 0.05,
                "days_to_expiration": 30,
                "min_premium": 0.01
            }
        }
    ]
    
    return templates
