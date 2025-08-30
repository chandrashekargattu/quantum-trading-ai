"""Portfolio management endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from datetime import datetime, timedelta, date

from app.db.database import get_db
from app.models.portfolio import Portfolio, PortfolioPerformance, RiskMetrics
from app.models.position import Position
from app.core.security import get_current_active_user
from app.schemas.portfolio import (
    PortfolioResponse,
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioPerformanceResponse,
    RiskMetricsResponse
)
from app.schemas.trade import PositionResponse

router = APIRouter()


@router.get("/", response_model=List[PortfolioResponse])
async def get_portfolios(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get all user portfolios."""
    result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.user_id == current_user.id,
                Portfolio.is_active == True
            )
        )
    )
    portfolios = result.scalars().all()
    
    # Update portfolio values
    for portfolio in portfolios:
        await update_portfolio_value(portfolio, db)
    
    await db.commit()
    
    return portfolios


@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create a new portfolio."""
    # Check if name already exists
    result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.user_id == current_user.id,
                Portfolio.name == portfolio_data.name
            )
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio with this name already exists"
        )
    
    # Create portfolio
    portfolio = Portfolio(
        user_id=current_user.id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        portfolio_type=portfolio_data.portfolio_type or "trading",
        cash_balance=portfolio_data.initial_cash or 100000.0,
        buying_power=portfolio_data.initial_cash or 100000.0,
        total_value=portfolio_data.initial_cash or 100000.0,
        is_default=False  # Only first portfolio is default
    )
    
    # Set as default if it's the first portfolio
    existing_count = await db.execute(
        select(func.count(Portfolio.id)).where(
            Portfolio.user_id == current_user.id
        )
    )
    if existing_count.scalar() == 0:
        portfolio.is_default = True
    
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    
    return portfolio


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific portfolio details."""
    result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Update portfolio value
    await update_portfolio_value(portfolio, db)
    await db.commit()
    
    return portfolio


@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: str,
    portfolio_update: PortfolioUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Update portfolio details."""
    result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Update fields
    update_data = portfolio_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(portfolio, field, value)
    
    portfolio.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(portfolio)
    
    return portfolio


@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Delete a portfolio (soft delete)."""
    result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default portfolio"
        )
    
    # Check if portfolio has open positions
    positions_result = await db.execute(
        select(Position).where(
            and_(
                Position.portfolio_id == portfolio_id,
                Position.is_open == True
            )
        )
    )
    if positions_result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete portfolio with open positions"
        )
    
    portfolio.is_active = False
    await db.commit()
    
    return {"message": "Portfolio deleted successfully"}


@router.get("/{portfolio_id}/positions", response_model=List[PositionResponse])
async def get_portfolio_positions(
    portfolio_id: str,
    is_open: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get all positions in a portfolio."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    if not portfolio_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get positions
    query = select(Position).where(Position.portfolio_id == portfolio_id)
    
    if is_open is not None:
        query = query.where(Position.is_open == is_open)
    
    result = await db.execute(query.order_by(Position.opened_at.desc()))
    positions = result.scalars().all()
    
    return positions


@router.get("/{portfolio_id}/performance", response_model=dict)
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = Query("1mo", regex="^(1d|1w|1mo|3mo|6mo|1y|all)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get portfolio performance history."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Calculate date range
    end_date = datetime.utcnow()
    period_map = {
        "1d": timedelta(days=1),
        "1w": timedelta(days=7),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
    }
    
    if period == "all":
        start_date = portfolio.created_at
    else:
        start_date = end_date - period_map[period]
    
    # Get performance history
    result = await db.execute(
        select(PortfolioPerformance).where(
            and_(
                PortfolioPerformance.portfolio_id == portfolio_id,
                PortfolioPerformance.date >= start_date
            )
        ).order_by(PortfolioPerformance.date.asc())
    )
    performance_data = result.scalars().all()
    
    # Format response
    dates = []
    values = []
    returns = []
    
    for perf in performance_data:
        dates.append(perf.date.isoformat())
        values.append(perf.total_value)
        returns.append(perf.cumulative_return_percent)
    
    return {
        "portfolio_id": portfolio_id,
        "period": period,
        "dates": dates,
        "values": values,
        "returns": returns,
        "current_value": portfolio.total_value,
        "total_return": portfolio.total_return,
        "total_return_percent": portfolio.total_return_percent
    }


@router.get("/{portfolio_id}/risk", response_model=RiskMetricsResponse)
async def get_portfolio_risk_metrics(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get portfolio risk metrics."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == current_user.id
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get latest risk metrics
    result = await db.execute(
        select(RiskMetrics).where(
            RiskMetrics.portfolio_id == portfolio_id
        ).order_by(RiskMetrics.calculated_at.desc()).limit(1)
    )
    risk_metrics = result.scalar_one_or_none()
    
    if not risk_metrics:
        # Calculate risk metrics
        from app.services.risk_calculator import calculate_portfolio_risk
        risk_metrics = await calculate_portfolio_risk(portfolio, db)
        db.add(risk_metrics)
        await db.commit()
    
    return risk_metrics


async def update_portfolio_value(portfolio: Portfolio, db: AsyncSession) -> None:
    """Update portfolio total value based on positions."""
    try:
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
        
        # Calculate total market value
        total_market_value = sum(pos.market_value or 0 for pos in positions)
        
        # Update portfolio values
        portfolio.total_value = portfolio.cash_balance + total_market_value
        portfolio.buying_power = portfolio.cash_balance  # Simplified - would include margin
        
        # Calculate returns - use cash_balance as initial if no positions
        initial_value = portfolio.cash_balance if not positions else 100000.0
        portfolio.total_return = portfolio.total_value - initial_value
        portfolio.total_return_percent = (portfolio.total_return / initial_value) * 100 if initial_value > 0 else 0.0
        
        # Set daily returns to 0 for now
        portfolio.daily_return = portfolio.daily_return or 0.0
        portfolio.daily_return_percent = portfolio.daily_return_percent or 0.0
        
        portfolio.updated_at = datetime.utcnow()
    except Exception as e:
        # If error, just use cash balance
        portfolio.total_value = portfolio.cash_balance
        portfolio.buying_power = portfolio.cash_balance
        portfolio.total_return = 0.0
        portfolio.total_return_percent = 0.0
        portfolio.daily_return = 0.0
        portfolio.daily_return_percent = 0.0
        portfolio.updated_at = datetime.utcnow()
