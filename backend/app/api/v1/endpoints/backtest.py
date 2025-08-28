"""Backtesting endpoints for strategy validation."""

from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime
import uuid

from app.db.database import get_db
from app.models.alert import Strategy
from app.core.security import get_current_active_user
from app.schemas.strategy import BacktestRequest, BacktestResult
from app.services.backtest import BacktestEngine
from app.tasks.backtest_tasks import run_backtest_task

router = APIRouter()


@router.post("/run", response_model=dict)
async def run_backtest(
    backtest_params: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Run a backtest for a strategy."""
    # Verify strategy ownership
    result = await db.execute(
        select(Strategy).where(
            and_(
                Strategy.id == backtest_params.strategy_id,
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
    
    # Validate date range
    if backtest_params.start_date >= backtest_params.end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start date must be before end date"
        )
    
    # Create backtest job ID
    job_id = str(uuid.uuid4())
    
    # Run backtest in background
    background_tasks.add_task(
        run_backtest_task,
        job_id=job_id,
        strategy_id=str(strategy.id),
        params=backtest_params.dict()
    )
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Backtest started. Use the job ID to check status."
    }


@router.get("/status/{job_id}")
async def get_backtest_status(
    job_id: str,
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get backtest job status."""
    # In a real implementation, this would check a job queue or database
    # For now, return a mock response
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "message": "Backtest completed successfully"
    }


@router.get("/result/{strategy_id}", response_model=BacktestResult)
async def get_backtest_result(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get latest backtest result for a strategy."""
    # Verify strategy ownership
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No backtest results found for this strategy"
        )
    
    # Convert stored results to response format
    results = strategy.backtest_results
    
    return BacktestResult(
        strategy_id=strategy.id,
        period={
            "start": results.get("start_date", ""),
            "end": results.get("end_date", "")
        },
        initial_capital=results.get("initial_capital", 100000),
        final_capital=results.get("final_capital", 100000),
        total_return=results.get("total_return", 0),
        total_return_percent=results.get("total_return_percent", 0),
        total_trades=results.get("total_trades", 0),
        winning_trades=results.get("winning_trades", 0),
        losing_trades=results.get("losing_trades", 0),
        win_rate=results.get("win_rate", 0),
        avg_win=results.get("avg_win", 0),
        avg_loss=results.get("avg_loss", 0),
        profit_factor=results.get("profit_factor", 0),
        sharpe_ratio=results.get("sharpe_ratio", 0),
        sortino_ratio=results.get("sortino_ratio", 0),
        max_drawdown=results.get("max_drawdown", 0),
        max_drawdown_duration=results.get("max_drawdown_duration", 0),
        trades=results.get("trades", []),
        equity_curve=results.get("equity_curve", []),
        monthly_returns=results.get("monthly_returns", {}),
        completed_at=datetime.fromisoformat(results.get("completed_at", datetime.utcnow().isoformat()))
    )


@router.post("/quick-test")
async def quick_backtest(
    strategy_type: str,
    symbol: str,
    config: dict,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Run a quick backtest without saving the strategy."""
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Create temporary strategy config
    temp_strategy = {
        "strategy_type": strategy_type,
        "config": config
    }
    
    # Run quick backtest (last 3 months)
    end_date = datetime.utcnow()
    start_date = end_date.replace(month=end_date.month - 3 if end_date.month > 3 else 12 - (3 - end_date.month))
    
    try:
        results = await engine.run_backtest(
            strategy=temp_strategy,
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000
        )
        
        return {
            "symbol": symbol,
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_return_percent": results.get("total_return_percent", 0),
            "win_rate": results.get("win_rate", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "total_trades": results.get("total_trades", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}"
        )


@router.get("/performance-comparison")
async def compare_strategies(
    strategy_ids: List[str],
    period: str = "6mo",
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Compare performance of multiple strategies."""
    if len(strategy_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 strategies can be compared at once"
        )
    
    comparisons = []
    
    for strategy_id in strategy_ids:
        # Verify ownership
        result = await db.execute(
            select(Strategy).where(
                and_(
                    Strategy.id == strategy_id,
                    Strategy.user_id == current_user.id
                )
            )
        )
        strategy = result.scalar_one_or_none()
        
        if strategy and strategy.backtest_results:
            comparisons.append({
                "strategy_id": strategy.id,
                "name": strategy.name,
                "total_return": strategy.backtest_results.get("total_return_percent", 0),
                "sharpe_ratio": strategy.backtest_results.get("sharpe_ratio", 0),
                "win_rate": strategy.backtest_results.get("win_rate", 0),
                "max_drawdown": strategy.backtest_results.get("max_drawdown", 0),
                "total_trades": strategy.backtest_results.get("total_trades", 0)
            })
    
    return {
        "period": period,
        "strategies": comparisons
    }
