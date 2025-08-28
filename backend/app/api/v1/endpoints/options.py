"""Options trading endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from datetime import datetime, date

from app.db.database import get_db
from app.models.option import Option, OptionChain
from app.models.stock import Stock
from app.core.security import get_current_active_user
from app.schemas.option import OptionResponse, OptionChainResponse, OptionGreeksResponse
from app.services.market_data import MarketDataService
from app.ml.model_manager import ModelManager

router = APIRouter()


@router.get("/chain/{symbol}", response_model=OptionChainResponse)
async def get_option_chain(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Filter by expiration date (YYYY-MM-DD)"),
    min_strike: Optional[float] = Query(None, description="Minimum strike price"),
    max_strike: Optional[float] = Query(None, description="Maximum strike price"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get option chain for a symbol."""
    # Verify stock exists and is optionable
    stock_result = await db.execute(
        select(Stock).where(
            and_(
                Stock.symbol == symbol.upper(),
                Stock.is_optionable == True
            )
        )
    )
    stock = stock_result.scalar_one_or_none()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock {symbol} not found or not optionable"
        )
    
    # Build query
    query = select(Option).where(
        Option.underlying_symbol == symbol.upper()
    )
    
    if expiration:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        query = query.where(Option.expiration == exp_date)
    
    if min_strike is not None:
        query = query.where(Option.strike >= min_strike)
    
    if max_strike is not None:
        query = query.where(Option.strike <= max_strike)
    
    # Execute query
    result = await db.execute(query.order_by(Option.expiration, Option.strike))
    options = result.scalars().all()
    
    # If no options found, fetch from external API
    if not options:
        market_service = MarketDataService()
        chain_data = await market_service.fetch_option_chain(symbol)
        
        if chain_data:
            # Save to database
            for call_data in chain_data['calls']:
                call = Option(**call_data)
                db.add(call)
            
            for put_data in chain_data['puts']:
                put = Option(**put_data)
                db.add(put)
            
            await db.commit()
            
            # Re-query
            result = await db.execute(query.order_by(Option.expiration, Option.strike))
            options = result.scalars().all()
    
    # Separate calls and puts
    calls = [opt for opt in options if opt.option_type == 'call']
    puts = [opt for opt in options if opt.option_type == 'put']
    
    # Get unique expirations and strikes
    expirations = sorted(list(set([opt.expiration.isoformat() for opt in options])))
    strikes = sorted(list(set([opt.strike for opt in options])))
    
    return {
        'underlying_symbol': symbol.upper(),
        'calls': calls,
        'puts': puts,
        'expirations': expirations,
        'strikes': strikes
    }


@router.get("/{option_id}", response_model=OptionResponse)
async def get_option(
    option_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific option details."""
    result = await db.execute(
        select(Option).where(Option.id == option_id)
    )
    option = result.scalar_one_or_none()
    
    if not option:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Option not found"
        )
    
    return option


@router.get("/{symbol}/{expiration}", response_model=dict)
async def get_options_by_expiration(
    symbol: str,
    expiration: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get all options for a symbol and expiration date."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid expiration date format. Use YYYY-MM-DD"
        )
    
    result = await db.execute(
        select(Option).where(
            and_(
                Option.underlying_symbol == symbol.upper(),
                Option.expiration == exp_date
            )
        ).order_by(Option.strike)
    )
    options = result.scalars().all()
    
    calls = [opt for opt in options if opt.option_type == 'call']
    puts = [opt for opt in options if opt.option_type == 'put']
    
    return {
        'symbol': symbol.upper(),
        'expiration': expiration,
        'calls': calls,
        'puts': puts
    }


@router.post("/calculate-greeks", response_model=OptionGreeksResponse)
async def calculate_option_greeks(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = Query(..., regex="^(call|put)$"),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Calculate option Greeks using Black-Scholes model."""
    model_manager = ModelManager()
    
    greeks = await model_manager.calculate_option_greeks(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=option_type
    )
    
    return greeks


@router.get("/screener/high-volume")
async def screen_high_volume_options(
    min_volume: int = Query(100, description="Minimum volume"),
    min_oi: int = Query(100, description="Minimum open interest"),
    option_type: Optional[str] = Query(None, regex="^(call|put)$"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Screen for high volume options."""
    query = select(Option).where(
        and_(
            Option.volume >= min_volume,
            Option.open_interest >= min_oi
        )
    )
    
    if option_type:
        query = query.where(Option.option_type == option_type)
    
    result = await db.execute(
        query.order_by(Option.volume.desc()).limit(50)
    )
    options = result.scalars().all()
    
    return options


@router.get("/screener/unusual-activity")
async def screen_unusual_activity(
    volume_oi_ratio: float = Query(2.0, description="Min volume/OI ratio"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Screen for unusual options activity."""
    # This would typically use a more sophisticated query
    # For now, we'll use a simple volume/OI ratio
    result = await db.execute(
        select(Option).where(
            and_(
                Option.open_interest > 0,
                Option.volume > Option.open_interest * volume_oi_ratio
            )
        ).order_by(Option.volume.desc()).limit(50)
    )
    options = result.scalars().all()
    
    return options


@router.get("/strategies/spreads/{symbol}")
async def find_option_spreads(
    symbol: str,
    strategy_type: str = Query(..., regex="^(bull_call|bear_put|iron_condor|butterfly)$"),
    expiration: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Find potential option spread strategies."""
    # Get option chain
    query = select(Option).where(
        Option.underlying_symbol == symbol.upper()
    )
    
    if expiration:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        query = query.where(Option.expiration == exp_date)
    
    result = await db.execute(query.order_by(Option.strike))
    options = result.scalars().all()
    
    if not options:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No options found for {symbol}"
        )
    
    # Find spreads based on strategy type
    spreads = []
    
    if strategy_type == "bull_call":
        # Find bull call spreads
        calls = [opt for opt in options if opt.option_type == 'call']
        for i in range(len(calls) - 1):
            long_call = calls[i]
            short_call = calls[i + 1]
            
            spread = {
                'strategy': 'Bull Call Spread',
                'legs': [
                    {'action': 'buy', 'option': long_call, 'quantity': 1},
                    {'action': 'sell', 'option': short_call, 'quantity': 1}
                ],
                'max_profit': short_call.strike - long_call.strike - (long_call.ask - short_call.bid),
                'max_loss': long_call.ask - short_call.bid,
                'breakeven': long_call.strike + (long_call.ask - short_call.bid)
            }
            spreads.append(spread)
            
            if len(spreads) >= 10:  # Limit results
                break
    
    # Add other strategy types...
    
    return spreads
