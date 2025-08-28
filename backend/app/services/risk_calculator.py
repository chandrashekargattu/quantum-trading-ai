"""Risk calculation service for portfolios."""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.portfolio import Portfolio, RiskMetrics
from app.models.position import Position

async def calculate_portfolio_risk(
    portfolio: Portfolio,
    db: AsyncSession
) -> RiskMetrics:
    """Calculate risk metrics for a portfolio."""
    # Get all open positions
    result = await db.execute(
        select(Position).where(
            and_(
                Position.portfolio_id == portfolio.id,
                Position.is_open == True
            )
        )
    )
    positions = result.scalars().all()
    
    # Calculate exposures
    long_exposure = sum(p.market_value for p in positions if p.quantity > 0)
    short_exposure = abs(sum(p.market_value for p in positions if p.quantity < 0))
    gross_exposure = long_exposure + short_exposure
    net_exposure = long_exposure - short_exposure
    
    # Calculate concentration metrics
    total_value = portfolio.total_value
    position_weights = []
    sector_weights = {}
    
    for position in positions:
        weight = position.market_value / total_value if total_value > 0 else 0
        position_weights.append(abs(weight))
        
        # Would need sector information from stock data
        sector = "Unknown"  # Placeholder
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Sort weights for concentration metrics
    position_weights.sort(reverse=True)
    
    largest_position_weight = position_weights[0] if position_weights else 0
    top_5_concentration = sum(position_weights[:5]) if len(position_weights) >= 5 else sum(position_weights)
    
    # Calculate VaR (simplified)
    returns = []
    for position in positions:
        if position.unrealized_pnl_percent:
            returns.append(position.unrealized_pnl_percent / 100)
    
    if returns:
        returns_array = np.array(returns)
        var_95 = np.percentile(returns_array, 5) * total_value
        var_99 = np.percentile(returns_array, 1) * total_value
        cvar_95 = returns_array[returns_array <= np.percentile(returns_array, 5)].mean() * total_value
    else:
        var_95 = var_99 = cvar_95 = 0
    
    # Calculate max loss (simplified - sum of all position values)
    max_loss = sum(p.market_value for p in positions)
    
    # Margin requirement (simplified - 50% of gross exposure for stocks)
    margin_requirement = gross_exposure * 0.5
    
    # Create risk metrics
    risk_metrics = RiskMetrics(
        portfolio_id=portfolio.id,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        largest_position_weight=largest_position_weight,
        top_5_concentration=top_5_concentration,
        sector_concentration=sector_weights,
        max_loss=max_loss,
        margin_requirement=margin_requirement,
        calculated_at=datetime.utcnow()
    )
    
    return risk_metrics
