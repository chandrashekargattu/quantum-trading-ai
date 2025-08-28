"""
Zerodha integration API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import logging

from app.core.auth import get_current_user
from app.models.user import User
from app.services.zerodha_integration import ZerodhaIntegrationService
from app.services.loss_recovery_analyzer import LossRecoveryAnalyzer
from app.services.indian_market_service import IndianMarketService
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class ZerodhaConnectRequest(BaseModel):
    """Request model for Zerodha connection."""
    api_key: str = Field(..., description="Zerodha API key")
    api_secret: str = Field(..., description="Zerodha API secret")
    request_token: Optional[str] = Field(None, description="Request token from Zerodha login")


class SmartOrderRequest(BaseModel):
    """Request model for smart order execution."""
    symbol: str
    quantity: int
    order_type: str = Field(..., description="MARKET, LIMIT, SL, SL-M")
    transaction_type: str = Field(..., description="BUY or SELL")
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    
class RecoveryPlanRequest(BaseModel):
    """Request model for recovery plan generation."""
    capital: float = Field(..., gt=0, description="Available capital for trading")
    risk_tolerance: str = Field("moderate", description="low, moderate, high")
    time_horizon: int = Field(180, description="Recovery timeline in days")


class StrategyActivationRequest(BaseModel):
    """Request model for strategy activation."""
    strategy_id: str
    capital_allocation: float
    paper_trading: bool = Field(False, description="Use paper trading mode")


# Initialize services
zerodha_service = ZerodhaIntegrationService()
recovery_analyzer = LossRecoveryAnalyzer()
indian_market = IndianMarketService()


@router.post("/connect")
async def connect_zerodha(
    request: ZerodhaConnectRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Connect to Zerodha account using API credentials.
    """
    try:
        success = await zerodha_service.initialize(
            api_key=request.api_key,
            api_secret=request.api_secret,
            request_token=request.request_token
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to connect to Zerodha. Please check credentials."
            )
        
        return {
            "success": True,
            "message": "Successfully connected to Zerodha",
            "user_id": zerodha_service.user_id
        }
        
    except Exception as e:
        logger.error(f"Zerodha connection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/analysis")
async def get_portfolio_analysis(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive portfolio analysis with AI recommendations.
    """
    try:
        analysis = await zerodha_service.get_portfolio_analysis()
        
        # Add recovery suggestions if in loss
        if analysis['total_pnl'] < 0:
            # Get AI recommendations
            recommendations = await zerodha_service.get_ai_recommendations({
                'total_loss': abs(analysis['total_pnl']),
                'positions': len(analysis['positions']),
                'holdings': len(analysis['holdings'])
            })
            analysis['ai_recommendations'] = recommendations
        
        return analysis
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recovery/analyze")
async def analyze_losses(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze trading losses and patterns.
    """
    try:
        # In real implementation, fetch trading history from DB
        # For now, using mock data
        import pandas as pd
        import numpy as np
        
        # Generate mock trading history
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        trading_history = pd.DataFrame({
            'date': dates,
            'symbol': np.random.choice(['RELIANCE', 'TCS', 'INFY', 'HDFC'], 100),
            'pnl': np.random.normal(-2500, 5000, 100),  # Negative mean = losses
            'position_size': np.random.uniform(10000, 50000, 100),
            'time': dates,
            'stop_loss_hit': np.random.choice([True, False], 100, p=[0.2, 0.8])
        })
        
        # Analyze profile
        profile = await recovery_analyzer.analyze_loss_profile(trading_history)
        
        return {
            'total_loss': float(profile.total_loss),
            'loss_timeline': profile.loss_timeline,
            'biggest_loss_trade': profile.biggest_loss_trade,
            'loss_patterns': profile.loss_patterns,
            'emotional_triggers': profile.emotional_triggers,
            'risk_score': profile.risk_score,
            'recovery_timeline_days': profile.recovery_timeline,
            'recommended_strategy': profile.recommended_strategy.value
        }
        
    except Exception as e:
        logger.error(f"Loss analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recovery/plan")
async def generate_recovery_plan(
    request: RecoveryPlanRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate personalized recovery plan.
    """
    try:
        # Get loss profile first (in real app, would be stored)
        # Using mock profile for demonstration
        from app.services.loss_recovery_analyzer import LossProfile, RecoveryStrategy
        
        profile = LossProfile(
            total_loss=Decimal('-250000'),
            loss_timeline=[],
            biggest_loss_trade={'symbol': 'RELIANCE', 'pnl': -50000},
            loss_patterns=['revenge_trading', 'no_stop_loss'],
            emotional_triggers=['early_morning_impulsiveness'],
            risk_score=0.65,
            recovery_timeline=request.time_horizon,
            recommended_strategy=RecoveryStrategy.SYSTEMATIC
        )
        
        # Generate plan
        plan = await recovery_analyzer.generate_recovery_plan(
            profile=profile,
            capital=Decimal(str(request.capital))
        )
        
        return plan
        
    except Exception as e:
        logger.error(f"Recovery plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/order/smart")
async def execute_smart_order(
    request: SmartOrderRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute order with AI-powered enhancements.
    """
    try:
        result = await zerodha_service.execute_smart_order(
            symbol=request.symbol,
            quantity=request.quantity,
            order_type=request.order_type,
            transaction_type=request.transaction_type,
            price=request.price
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Smart order execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy/activate")
async def activate_strategy(
    request: StrategyActivationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Activate a trading strategy.
    """
    try:
        # Map strategy IDs to strategy types
        strategy_map = {
            'index_option_selling': 'option_selling',
            'momentum_breakout': 'momentum',
            'arbitrage_trading': 'arbitrage',
            'ai_ml_signals': 'recovery_mode'
        }
        
        strategy_type = strategy_map.get(request.strategy_id, 'recovery_mode')
        
        result = await zerodha_service.start_algo_trading(
            strategy=strategy_type,
            capital=request.capital_allocation
        )
        
        # Store activation in database
        # TODO: Implement strategy activation storage
        
        return {
            **result,
            'strategy_id': request.strategy_id,
            'paper_trading': request.paper_trading
        }
        
    except Exception as e:
        logger.error(f"Strategy activation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy/performance/{strategy_id}")
async def get_strategy_performance(
    strategy_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get performance metrics for active strategy.
    """
    try:
        # In real implementation, fetch from database
        # Mock data for demonstration
        return {
            'strategy_id': strategy_id,
            'status': 'active',
            'start_date': datetime.now().isoformat(),
            'total_trades': 42,
            'winning_trades': 28,
            'losing_trades': 14,
            'win_rate': 0.67,
            'total_pnl': 35000,
            'average_profit': 2500,
            'average_loss': -1200,
            'risk_reward_ratio': 2.08,
            'sharpe_ratio': 1.85,
            'max_drawdown': -8500,
            'daily_pnl': [
                {'date': '2024-01-15', 'pnl': 5000},
                {'date': '2024-01-16', 'pnl': -2000},
                {'date': '2024-01-17', 'pnl': 3500}
            ]
        }
        
    except Exception as e:
        logger.error(f"Strategy performance fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/opportunities")
async def get_market_opportunities(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current market opportunities from AI analysis.
    """
    try:
        # Get various opportunity types
        opportunities = {
            'breakout_stocks': [
                {
                    'symbol': 'TATAMOTORS',
                    'signal': 'Breakout above 650',
                    'entry': 652,
                    'target': 680,
                    'stop_loss': 640,
                    'confidence': 0.78
                },
                {
                    'symbol': 'BAJFINANCE',
                    'signal': 'Flag pattern breakout',
                    'entry': 7250,
                    'target': 7500,
                    'stop_loss': 7100,
                    'confidence': 0.72
                }
            ],
            'option_trades': [
                {
                    'underlying': 'NIFTY',
                    'strategy': 'Iron Condor',
                    'strikes': '21500-21700 Call, 21000-20800 Put',
                    'premium': 8500,
                    'max_loss': 11500,
                    'probability_profit': 0.82
                }
            ],
            'arbitrage': [
                {
                    'symbol': 'RELIANCE',
                    'nse_price': 2456.50,
                    'bse_price': 2457.80,
                    'spread': 1.30,
                    'profit_100_shares': 130
                }
            ],
            'mean_reversion': [
                {
                    'symbol': 'INFY',
                    'rsi': 28,
                    'deviation': -2.3,
                    'entry': 1420,
                    'mean_price': 1455,
                    'confidence': 0.69
                }
            ]
        }
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Market opportunities fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recovery/progress")
async def get_recovery_progress(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Monitor recovery progress.
    """
    try:
        # In real implementation, calculate from actual trades
        start_date = datetime.now() - timedelta(days=30)
        
        progress = await recovery_analyzer.monitor_recovery_progress(
            user_id=str(current_user.id),
            start_date=start_date
        )
        
        return progress
        
    except Exception as e:
        logger.error(f"Recovery progress fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper-trading/toggle")
async def toggle_paper_trading(
    enabled: bool = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Enable or disable paper trading mode.
    """
    try:
        # In real implementation, update user settings
        return {
            'paper_trading_enabled': enabled,
            'message': f"Paper trading {'enabled' if enabled else 'disabled'}"
        }
        
    except Exception as e:
        logger.error(f"Paper trading toggle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/education/next-lesson")
async def get_next_lesson(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get next recommended lesson based on progress.
    """
    try:
        # In real implementation, track user progress
        return {
            'lesson_id': 'risk_management_basics',
            'title': 'Risk Management Fundamentals',
            'duration_minutes': 45,
            'type': 'video',
            'description': 'Learn position sizing, stop losses, and risk-reward ratios',
            'url': '/education/risk-management-basics',
            'quiz_available': True,
            'completion_reward': 'Unlock advanced position calculator'
        }
        
    except Exception as e:
        logger.error(f"Next lesson fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
