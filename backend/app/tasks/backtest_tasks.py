"""Background tasks for backtesting."""

import asyncio
from typing import Dict, Any
import logging

from app.services.backtest import BacktestEngine
from app.db.database import AsyncSessionLocal
from app.models.alert import Strategy
from sqlalchemy import select

logger = logging.getLogger(__name__)


async def run_backtest_task(
    job_id: str,
    strategy_id: str,
    params: Dict[str, Any]
):
    """Run backtest as a background task."""
    try:
        logger.info(f"Starting backtest job {job_id} for strategy {strategy_id}")
        
        # Create database session
        async with AsyncSessionLocal() as db:
            # Get strategy
            result = await db.execute(
                select(Strategy).where(Strategy.id == strategy_id)
            )
            strategy = result.scalar_one_or_none()
            
            if not strategy:
                logger.error(f"Strategy {strategy_id} not found")
                return
            
            # Initialize backtest engine
            engine = BacktestEngine()
            
            # Run backtest
            results = await engine.run_backtest(
                strategy={
                    "strategy_type": strategy.strategy_type,
                    "config": strategy.config
                },
                symbols=params["symbols"],
                start_date=params["start_date"],
                end_date=params["end_date"],
                initial_capital=params.get("initial_capital", 100000)
            )
            
            # Save results to strategy
            strategy.backtest_results = results
            
            # Update strategy metrics
            strategy.sharpe_ratio = results.get("sharpe_ratio", 0)
            strategy.win_rate = results.get("win_rate", 0)
            strategy.avg_return = results.get("total_return_percent", 0) / results.get("total_trades", 1)
            
            await db.commit()
            
            logger.info(f"Backtest job {job_id} completed successfully")
            
    except Exception as e:
        logger.error(f"Backtest job {job_id} failed: {e}")
        raise
