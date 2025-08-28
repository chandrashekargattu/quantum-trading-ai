"""Options trading endpoints."""

from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, or_
from datetime import datetime, timedelta, date
import numpy as np

from app.db.database import get_db
from app.models.options import (
    Option, OptionOrder, OptionPosition, 
    OptionStrategy, OptionStrategyLeg,
    OptionType, OptionStyle, OptionAction
)
from app.models.portfolio import Portfolio
from app.models.option import OptionChain, OptionGreeksHistory
from app.core.security import get_current_active_user
from app.schemas.options import (
    OptionChainResponse,
    OptionOrderCreate,
    OptionOrderResponse,
    OptionStrategyCreate,
    OptionStrategyResponse,
    GreeksResponse
)
from app.services.options_pricing import OptionsPricingService
from app.services.market_data import MarketDataService
from app.api.v1.websocket import manager

router = APIRouter()


@router.get("/chains/{symbol}", response_model=OptionChainResponse)
async def get_option_chain(
    symbol: str,
    expiration_date: Optional[date] = None,
    strike_range: Optional[int] = Query(10, description="Number of strikes above/below current price"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get option chain for a symbol."""
    market_service = MarketDataService()
    
    # Get current stock price
    stock_data = await market_service.fetch_stock_data(symbol)
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock data not found for {symbol}"
        )
    
    current_price = stock_data['current_price']
    
    # Get option chain from market data service
    options_data = await market_service.fetch_option_chain(
        symbol=symbol,
        expiration_date=expiration_date
    )
    
    if not options_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Option chain not available for {symbol}"
        )
    
    # Filter strikes around current price
    if strike_range:
        strike_min = current_price * 0.8  # 20% below
        strike_max = current_price * 1.2  # 20% above
        
        options_data['calls'] = [
            opt for opt in options_data['calls']
            if strike_min <= opt['strike'] <= strike_max
        ][:strike_range]
        
        options_data['puts'] = [
            opt for opt in options_data['puts']
            if strike_min <= opt['strike'] <= strike_max
        ][:strike_range]
    
    # Calculate Greeks for each option
    pricing_service = OptionsPricingService()
    
    for option in options_data['calls'] + options_data['puts']:
        greeks = await pricing_service.calculate_greeks(
            spot_price=current_price,
            strike_price=option['strike'],
            time_to_expiry=option['days_to_expiry'] / 365.0,
            volatility=option['implied_volatility'],
            risk_free_rate=0.05,  # Current risk-free rate
            option_type='call' if option in options_data['calls'] else 'put'
        )
        option['greeks'] = greeks
    
    return OptionChainResponse(
        symbol=symbol.upper(),
        underlying_price=current_price,
        timestamp=datetime.utcnow(),
        expirations=options_data['expirations'],
        calls=options_data['calls'],
        puts=options_data['puts']
    )


@router.post("/orders", response_model=OptionOrderResponse)
async def create_option_order(
    order_data: OptionOrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create an option order."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == order_data.portfolio_id,
                Portfolio.user_id == current_user.id,
                Portfolio.is_active == True
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get or create option contract
    option_result = await db.execute(
        select(Option).where(
            and_(
                Option.underlying_symbol == order_data.underlying_symbol,
                Option.strike_price == order_data.strike_price,
                Option.expiration_date == order_data.expiration_date,
                Option.option_type == order_data.option_type
            )
        )
    )
    option = option_result.scalar_one_or_none()
    
    if not option:
        # Create new option contract
        option = Option(
            underlying_symbol=order_data.underlying_symbol.upper(),
            strike_price=order_data.strike_price,
            expiration_date=order_data.expiration_date,
            option_type=order_data.option_type,
            option_style=OptionStyle.AMERICAN,  # Default to American style
            contract_size=100  # Standard contract size
        )
        db.add(option)
        await db.commit()
        await db.refresh(option)
    
    # Get current option price
    market_service = MarketDataService()
    option_quote = await market_service.fetch_option_quote(
        symbol=order_data.underlying_symbol,
        strike=order_data.strike_price,
        expiration=order_data.expiration_date,
        option_type=order_data.option_type.value
    )
    
    if not option_quote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to get option quote"
        )
    
    # Calculate order value
    contracts = order_data.quantity
    price_per_contract = order_data.limit_price or option_quote['mid_price']
    order_value = contracts * price_per_contract * 100  # 100 shares per contract
    commission = calculate_option_commission(contracts)
    
    # Check buying power for buy orders
    if order_data.action == OptionAction.BUY_TO_OPEN:
        total_cost = order_value + commission
        if total_cost > portfolio.buying_power:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient buying power. Required: ${total_cost:.2f}"
            )
    
    # Check position for sell to close
    if order_data.action == OptionAction.SELL_TO_CLOSE:
        position_result = await db.execute(
            select(OptionPosition).where(
                and_(
                    OptionPosition.portfolio_id == portfolio.id,
                    OptionPosition.option_id == option.id,
                    OptionPosition.is_open == True
                )
            )
        )
        position = position_result.scalar_one_or_none()
        if not position or position.quantity < contracts:
            available = position.quantity if position else 0
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient contracts. Available: {available}"
            )
    
    # Create option order
    option_order = OptionOrder(
        portfolio_id=portfolio.id,
        option_id=option.id,
        action=order_data.action,
        quantity=contracts,
        order_type=order_data.order_type or "LIMIT",
        limit_price=price_per_contract,
        status="PENDING"
    )
    
    db.add(option_order)
    await db.commit()
    await db.refresh(option_order)
    
    # Execute order in background
    background_tasks.add_task(
        execute_option_order,
        order_id=str(option_order.id),
        db=db,
        user_id=str(current_user.id)
    )
    
    return option_order


@router.get("/positions", response_model=List[Dict[str, Any]])
async def get_option_positions(
    portfolio_id: Optional[str] = None,
    is_open: Optional[bool] = True,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get option positions."""
    query = select(OptionPosition).join(Portfolio).where(
        Portfolio.user_id == current_user.id
    )
    
    if portfolio_id:
        query = query.where(OptionPosition.portfolio_id == portfolio_id)
    
    if is_open is not None:
        query = query.where(OptionPosition.is_open == is_open)
    
    result = await db.execute(query)
    positions = result.scalars().all()
    
    # Enrich with current prices and Greeks
    market_service = MarketDataService()
    pricing_service = OptionsPricingService()
    enriched_positions = []
    
    for position in positions:
        option = position.option
        
        # Get current underlying price
        stock_data = await market_service.fetch_stock_data(option.underlying_symbol)
        if not stock_data:
            continue
        
        current_price = stock_data['current_price']
        
        # Get option quote
        option_quote = await market_service.fetch_option_quote(
            symbol=option.underlying_symbol,
            strike=option.strike_price,
            expiration=option.expiration_date,
            option_type=option.option_type.value
        )
        
        if option_quote:
            # Calculate Greeks
            days_to_expiry = (option.expiration_date - date.today()).days
            greeks = await pricing_service.calculate_greeks(
                spot_price=current_price,
                strike_price=option.strike_price,
                time_to_expiry=days_to_expiry / 365.0,
                volatility=option_quote['implied_volatility'],
                risk_free_rate=0.05,
                option_type=option.option_type.value
            )
            
            # Calculate P&L
            current_value = position.quantity * option_quote['mid_price'] * 100
            cost_basis = position.quantity * position.avg_cost * 100
            unrealized_pnl = current_value - cost_basis
            
            enriched_positions.append({
                "position_id": str(position.id),
                "portfolio_id": str(position.portfolio_id),
                "symbol": f"{option.underlying_symbol}_{option.expiration_date}_{option.option_type.value}_{option.strike_price}",
                "underlying_symbol": option.underlying_symbol,
                "strike": option.strike_price,
                "expiration": option.expiration_date,
                "option_type": option.option_type.value,
                "quantity": position.quantity,
                "avg_cost": position.avg_cost,
                "current_price": option_quote['mid_price'],
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": (unrealized_pnl / cost_basis) * 100,
                "greeks": greeks,
                "days_to_expiry": days_to_expiry
            })
    
    return enriched_positions


@router.post("/strategies", response_model=OptionStrategyResponse)
async def create_option_strategy(
    strategy_data: OptionStrategyCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create a multi-leg option strategy."""
    # Verify portfolio ownership
    portfolio_result = await db.execute(
        select(Portfolio).where(
            and_(
                Portfolio.id == strategy_data.portfolio_id,
                Portfolio.user_id == current_user.id,
                Portfolio.is_active == True
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Validate strategy
    if len(strategy_data.legs) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy must have at least 2 legs"
        )
    
    # Create strategy
    strategy = OptionStrategy(
        portfolio_id=portfolio.id,
        strategy_type=strategy_data.strategy_type,
        underlying_symbol=strategy_data.underlying_symbol.upper(),
        max_profit=0,  # Will be calculated
        max_loss=0,    # Will be calculated
        breakeven_points=[],  # Will be calculated
        status="PENDING"
    )
    
    db.add(strategy)
    await db.commit()
    
    # Create strategy legs
    total_debit = 0
    total_credit = 0
    
    for leg_data in strategy_data.legs:
        # Get or create option
        option_result = await db.execute(
            select(Option).where(
                and_(
                    Option.underlying_symbol == strategy_data.underlying_symbol,
                    Option.strike_price == leg_data.strike_price,
                    Option.expiration_date == leg_data.expiration_date,
                    Option.option_type == leg_data.option_type
                )
            )
        )
        option = option_result.scalar_one_or_none()
        
        if not option:
            option = Option(
                underlying_symbol=strategy_data.underlying_symbol.upper(),
                strike_price=leg_data.strike_price,
                expiration_date=leg_data.expiration_date,
                option_type=leg_data.option_type,
                option_style=OptionStyle.AMERICAN,
                contract_size=100
            )
            db.add(option)
            await db.commit()
            await db.refresh(option)
        
        # Create leg
        leg = OptionStrategyLeg(
            strategy_id=strategy.id,
            option_id=option.id,
            action=leg_data.action,
            quantity=leg_data.quantity,
            order_type="LIMIT",
            limit_price=leg_data.limit_price
        )
        
        db.add(leg)
        
        # Calculate debit/credit
        if leg_data.action in [OptionAction.BUY_TO_OPEN, OptionAction.BUY_TO_CLOSE]:
            total_debit += leg_data.quantity * leg_data.limit_price * 100
        else:
            total_credit += leg_data.quantity * leg_data.limit_price * 100
    
    # Update strategy calculations
    net_cost = total_debit - total_credit
    
    # Simple P&L calculation (would be more complex for different strategies)
    if strategy_data.strategy_type == "IRON_CONDOR":
        strategy.max_profit = total_credit - total_debit
        strategy.max_loss = calculate_iron_condor_max_loss(strategy_data.legs)
    elif strategy_data.strategy_type == "BULL_CALL_SPREAD":
        strategy.max_profit = calculate_spread_max_profit(strategy_data.legs, "CALL")
        strategy.max_loss = net_cost
    # Add more strategy calculations...
    
    await db.commit()
    await db.refresh(strategy)
    
    # Execute strategy in background
    background_tasks.add_task(
        execute_option_strategy,
        strategy_id=str(strategy.id),
        db=db,
        user_id=str(current_user.id)
    )
    
    return strategy


@router.get("/strategies/templates", response_model=List[Dict[str, Any]])
async def get_strategy_templates(
    underlying_symbol: str,
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get recommended option strategy templates based on market conditions."""
    market_service = MarketDataService()
    pricing_service = OptionsPricingService()
    
    # Get current market data
    stock_data = await market_service.fetch_stock_data(underlying_symbol)
    if not stock_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock data not found for {underlying_symbol}"
        )
    
    current_price = stock_data['current_price']
    volatility = stock_data.get('volatility', 0.2)
    
    # Generate strategy templates
    templates = []
    
    # Bull Call Spread
    templates.append({
        "strategy_type": "BULL_CALL_SPREAD",
        "name": "Bull Call Spread",
        "market_outlook": "Moderately Bullish",
        "description": "Profit from moderate price increase with limited risk",
        "legs": [
            {
                "action": "BUY_TO_OPEN",
                "option_type": "CALL",
                "strike_price": round(current_price * 0.98, 2),  # ITM
                "quantity": 1
            },
            {
                "action": "SELL_TO_OPEN",
                "option_type": "CALL",
                "strike_price": round(current_price * 1.02, 2),  # OTM
                "quantity": 1
            }
        ],
        "max_profit": "Limited",
        "max_loss": "Limited to net debit"
    })
    
    # Iron Condor
    templates.append({
        "strategy_type": "IRON_CONDOR",
        "name": "Iron Condor",
        "market_outlook": "Neutral",
        "description": "Profit from low volatility with limited risk",
        "legs": [
            {
                "action": "SELL_TO_OPEN",
                "option_type": "PUT",
                "strike_price": round(current_price * 0.95, 2),
                "quantity": 1
            },
            {
                "action": "BUY_TO_OPEN",
                "option_type": "PUT",
                "strike_price": round(current_price * 0.90, 2),
                "quantity": 1
            },
            {
                "action": "SELL_TO_OPEN",
                "option_type": "CALL",
                "strike_price": round(current_price * 1.05, 2),
                "quantity": 1
            },
            {
                "action": "BUY_TO_OPEN",
                "option_type": "CALL",
                "strike_price": round(current_price * 1.10, 2),
                "quantity": 1
            }
        ],
        "max_profit": "Net credit received",
        "max_loss": "Limited"
    })
    
    # Add more strategy templates based on market conditions...
    
    return templates


@router.get("/greeks/{symbol}", response_model=List[GreeksResponse])
async def get_option_greeks(
    symbol: str,
    expiration_date: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get Greeks for options."""
    market_service = MarketDataService()
    pricing_service = OptionsPricingService()
    
    # Get option chain
    options_data = await market_service.fetch_option_chain(
        symbol=symbol,
        expiration_date=expiration_date
    )
    
    if not options_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Options not found for {symbol}"
        )
    
    # Get current stock price
    stock_data = await market_service.fetch_stock_data(symbol)
    current_price = stock_data['current_price']
    
    greeks_list = []
    
    for option in options_data['calls'] + options_data['puts']:
        days_to_expiry = (option['expiration_date'] - date.today()).days
        
        greeks = await pricing_service.calculate_greeks(
            spot_price=current_price,
            strike_price=option['strike'],
            time_to_expiry=days_to_expiry / 365.0,
            volatility=option['implied_volatility'],
            risk_free_rate=0.05,
            option_type='call' if option in options_data['calls'] else 'put'
        )
        
        greeks_list.append(GreeksResponse(
            symbol=f"{symbol}_{option['expiration_date']}_{option['type']}_{option['strike']}",
            underlying_symbol=symbol,
            strike_price=option['strike'],
            expiration_date=option['expiration_date'],
            option_type=option['type'],
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            rho=greeks['rho'],
            implied_volatility=option['implied_volatility'],
            theoretical_value=greeks['theoretical_value']
        ))
    
    return greeks_list


def calculate_option_commission(contracts: int) -> float:
    """Calculate option trading commission."""
    # Simple commission model
    base_fee = 0.65  # Per contract
    return contracts * base_fee


def calculate_iron_condor_max_loss(legs: List[Any]) -> float:
    """Calculate maximum loss for iron condor."""
    # Implementation depends on specific leg structure
    return 0.0


def calculate_spread_max_profit(legs: List[Any], spread_type: str) -> float:
    """Calculate maximum profit for spread strategies."""
    # Implementation depends on specific spread type
    return 0.0


async def execute_option_order(order_id: str, db: AsyncSession, user_id: str):
    """Execute an option order."""
    # Implementation would connect to options exchange
    pass


async def execute_option_strategy(strategy_id: str, db: AsyncSession, user_id: str):
    """Execute a multi-leg option strategy."""
    # Implementation would execute all legs atomically
    pass