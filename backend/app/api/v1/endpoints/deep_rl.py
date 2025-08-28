"""Deep reinforcement learning endpoints for trading strategies."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.db.database import get_db
from app.core.security import get_current_active_user
from app.models import User
from app.ml.deep_rl_trading import DeepRLTradingEngine
import numpy as np

router = APIRouter()

# Store active RL engines (in production, use Redis or similar)
rl_engines = {}


@router.post("/train-rl-agent")
async def train_rl_agent(
    symbols: List[str],
    market_data: Dict[str, List[float]],
    agent_type: str = "PPO",
    n_episodes: int = 100,
    hyperparameters: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Train a deep reinforcement learning agent for trading.
    
    - **symbols**: List of symbols to trade
    - **market_data**: Historical market data for training
    - **agent_type**: Type of RL agent (PPO, DQN, etc.)
    - **n_episodes**: Number of training episodes
    - **hyperparameters**: Agent hyperparameters
    """
    try:
        # Create unique engine ID
        engine_id = f"{current_user.id}_{datetime.now().timestamp()}"
        
        # Initialize trading engine
        engine = DeepRLTradingEngine(
            market_data=market_data,
            agent_type=agent_type,
            hyperparameters=hyperparameters or {}
        )
        
        # Store engine
        rl_engines[engine_id] = engine
        
        # Train in background
        background_tasks.add_task(
            engine.train,
            n_episodes=n_episodes
        )
        
        return {
            "engine_id": engine_id,
            "status": "training_started",
            "agent_type": agent_type,
            "n_episodes": n_episodes,
            "symbols": symbols
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start RL training: {str(e)}"
        )


@router.get("/training-status/{engine_id}")
async def get_training_status(
    engine_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get training status and metrics for an RL agent."""
    if engine_id not in rl_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Engine not found"
        )
    
    engine = rl_engines[engine_id]
    
    return {
        "engine_id": engine_id,
        "training_history": {
            "rewards": engine.training_history.get("rewards", []),
            "portfolio_values": engine.training_history.get("portfolio_values", []),
            "sharpe_ratios": engine.training_history.get("sharpe_ratios", []),
            "max_drawdowns": engine.training_history.get("max_drawdowns", [])
        },
        "episodes_completed": len(engine.training_history.get("rewards", []))
    }


@router.post("/backtest-rl-strategy")
async def backtest_rl_strategy(
    engine_id: str,
    test_data: Dict[str, List[float]],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Backtest a trained RL agent on new data.
    
    - **engine_id**: ID of the trained engine
    - **test_data**: Market data for backtesting
    """
    if engine_id not in rl_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Engine not found"
        )
    
    try:
        engine = rl_engines[engine_id]
        results = engine.backtest(test_data)
        
        return {
            "total_return": results["total_return"],
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"],
            "win_rate": results["win_rate"],
            "total_trades": len(results["trades"]),
            "portfolio_values": results["portfolio_values"],
            "trades": results["trades"][:100],  # Limit to recent 100 trades
            "final_positions": results["final_positions"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}"
        )


@router.post("/execute-rl-trade")
async def execute_rl_trade(
    engine_id: str,
    current_prices: Dict[str, float],
    current_positions: Dict[str, float],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get trading action from trained RL agent.
    
    - **engine_id**: ID of the trained engine
    - **current_prices**: Current market prices
    - **current_positions**: Current portfolio positions
    """
    if engine_id not in rl_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Engine not found"
        )
    
    try:
        engine = rl_engines[engine_id]
        
        # Get action from agent
        # This is simplified - in production would need proper state construction
        state = engine.env.get_observation()
        action, _ = engine.agent.select_action(state)
        
        # Convert action to trading signals
        signals = engine._action_to_signals(action)
        
        return {
            "recommended_actions": signals,
            "confidence": 0.8,  # Placeholder
            "expected_return": 0.01,  # Placeholder
            "risk_assessment": "moderate"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute trade: {str(e)}"
        )


@router.delete("/delete-engine/{engine_id}")
async def delete_engine(
    engine_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, str]:
    """Delete a trained RL engine."""
    if engine_id not in rl_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Engine not found"
        )
    
    del rl_engines[engine_id]
    
    return {"message": "Engine deleted successfully"}
