"""
Indian Options & F&O Strategies

Specialized strategies for Indian derivatives market:
- Index options (NIFTY, BANKNIFTY, FINNIFTY)
- Stock options strategies
- Expiry day strategies
- Event-based strategies (Budget, RBI policy)
- Indian market-specific adjustments
- SEBI margin requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy.stats import norm
import math

from app.services.indian_market_service import IndianMarketService, FNOData
from app.core.config import settings


class IndianOptionStrategy(Enum):
    """Indian market specific option strategies"""
    # Directional
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    
    # Spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    
    # Neutral strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    
    # Indian favorites
    CALL_RATIO_SPREAD = "call_ratio_spread"
    PUT_RATIO_SPREAD = "put_ratio_spread"
    JADE_LIZARD = "jade_lizard"
    TWISTED_SISTER = "twisted_sister"
    
    # Expiry specials
    EXPIRY_STRADDLE = "expiry_straddle"
    EXPIRY_STRANGLE = "expiry_strangle"
    ZERO_DTE = "zero_dte"  # 0 Days to Expiry
    
    # Calendar spreads
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"


@dataclass
class OptionPosition:
    """Option position details"""
    symbol: str
    strike: float
    option_type: str  # CE or PE
    expiry: datetime
    action: str  # buy or sell
    quantity: int
    premium: float
    lot_size: int
    margin_required: Optional[float] = None
    break_even: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with Indian market context"""
    strategy: IndianOptionStrategy
    positions: List[OptionPosition]
    total_margin: float
    max_profit: float
    max_loss: float
    break_even_points: List[float]
    probability_of_profit: float
    expected_return: float
    risk_reward_ratio: float
    market_outlook: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    adjustments: List[str]
    indian_context: Dict[str, Any]  # Expiry effects, events, etc.


class IndianOptionsCalculator:
    """Options calculations for Indian markets"""
    
    def __init__(self):
        self.risk_free_rate = 0.065  # Indian risk-free rate ~6.5%
        
        # SEBI margin requirements
        self.margin_requirements = {
            'index_futures': 0.10,  # 10% for index futures
            'stock_futures': 0.15,  # 15% for stock futures
            'short_option': 0.16,   # 16% for short options
            'spread_benefit': 0.70  # 70% margin benefit for spreads
        }
        
        # Indian market specifics
        self.index_lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'FINNIFTY': 40,
            'MIDCPNIFTY': 75
        }
        
        # Expiry days
        self.weekly_expiry = {
            'NIFTY': 3,      # Thursday
            'BANKNIFTY': 2,  # Wednesday  
            'FINNIFTY': 1,   # Tuesday
            'MIDCPNIFTY': 0  # Monday
        }
    
    def calculate_option_price(
        self, spot: float, strike: float, time_to_expiry: float,
        volatility: float, option_type: str, rate: Optional[float] = None
    ) -> Dict[str, float]:
        """Black-Scholes option pricing for Indian markets"""
        
        r = rate or self.risk_free_rate
        
        # Convert time to years
        t = time_to_expiry / 365
        
        if t <= 0:
            # Expiry day calculation
            intrinsic = max(0, spot - strike) if option_type == 'CE' else max(0, strike - spot)
            return {
                'price': intrinsic,
                'delta': 1.0 if (option_type == 'CE' and spot > strike) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Black-Scholes calculations
        d1 = (np.log(spot / strike) + (r + 0.5 * volatility ** 2) * t) / (volatility * np.sqrt(t))
        d2 = d1 - volatility * np.sqrt(t)
        
        if option_type == 'CE':  # Call option
            price = spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
            delta = norm.cdf(d1)
            
        else:  # Put option
            price = strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        
        # Greeks
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(t))
        theta = self._calculate_theta(spot, strike, t, r, volatility, d1, d2, option_type)
        vega = spot * norm.pdf(d1) * np.sqrt(t) / 100  # Per 1% change in IV
        rho = self._calculate_rho(strike, t, r, d2, option_type) / 100  # Per 1% change in rate
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega,
            'rho': rho
        }
    
    def _calculate_theta(
        self, S: float, K: float, t: float, r: float, 
        sigma: float, d1: float, d2: float, option_type: str
    ) -> float:
        """Calculate theta"""
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
        
        if option_type == 'CE':
            term2 = r * K * np.exp(-r * t) * norm.cdf(d2)
            theta = (term1 - term2) 
        else:
            term2 = r * K * np.exp(-r * t) * norm.cdf(-d2)
            theta = (term1 + term2)
        
        return theta
    
    def _calculate_rho(self, K: float, t: float, r: float, d2: float, option_type: str) -> float:
        """Calculate rho"""
        if option_type == 'CE':
            rho = K * t * np.exp(-r * t) * norm.cdf(d2)
        else:
            rho = -K * t * np.exp(-r * t) * norm.cdf(-d2)
        
        return rho
    
    def calculate_margin(
        self, strategy: IndianOptionStrategy, positions: List[OptionPosition]
    ) -> float:
        """Calculate SEBI margin requirements"""
        
        total_margin = 0
        
        for position in positions:
            if position.action == 'sell':
                # Short option margin
                if 'NIFTY' in position.symbol:
                    margin_percent = self.margin_requirements['index_futures']
                else:
                    margin_percent = self.margin_requirements['stock_futures']
                
                notional = position.strike * position.lot_size * position.quantity
                margin = notional * margin_percent
                
                # Add premium for short options
                margin += position.premium * position.lot_size * position.quantity
                
                total_margin += margin
        
        # Apply spread benefit
        if self._is_spread_strategy(strategy):
            total_margin *= self.margin_requirements['spread_benefit']
        
        return total_margin
    
    def _is_spread_strategy(self, strategy: IndianOptionStrategy) -> bool:
        """Check if strategy is a spread"""
        spread_strategies = [
            IndianOptionStrategy.BULL_CALL_SPREAD,
            IndianOptionStrategy.BEAR_PUT_SPREAD,
            IndianOptionStrategy.BULL_PUT_SPREAD,
            IndianOptionStrategy.BEAR_CALL_SPREAD,
            IndianOptionStrategy.IRON_CONDOR,
            IndianOptionStrategy.IRON_BUTTERFLY,
            IndianOptionStrategy.CALENDAR_SPREAD,
            IndianOptionStrategy.DIAGONAL_SPREAD
        ]
        return strategy in spread_strategies
    
    def calculate_payoff(
        self, positions: List[OptionPosition], spot_price: float
    ) -> float:
        """Calculate strategy payoff at given spot price"""
        
        total_payoff = 0
        
        for position in positions:
            if position.option_type == 'CE':
                intrinsic = max(0, spot_price - position.strike)
            else:
                intrinsic = max(0, position.strike - spot_price)
            
            if position.action == 'buy':
                payoff = intrinsic - position.premium
            else:  # sell
                payoff = position.premium - intrinsic
            
            total_payoff += payoff * position.lot_size * position.quantity
        
        return total_payoff
    
    def find_breakeven_points(
        self, positions: List[OptionPosition], min_price: float, max_price: float
    ) -> List[float]:
        """Find breakeven points for the strategy"""
        
        breakevens = []
        
        # Test prices at small intervals
        test_prices = np.linspace(min_price, max_price, 1000)
        payoffs = [self.calculate_payoff(positions, price) for price in test_prices]
        
        # Find where payoff crosses zero
        for i in range(1, len(payoffs)):
            if payoffs[i-1] * payoffs[i] < 0:  # Sign change
                # Linear interpolation for more accurate breakeven
                price1, price2 = test_prices[i-1], test_prices[i]
                payoff1, payoff2 = payoffs[i-1], payoffs[i]
                
                breakeven = price1 - payoff1 * (price2 - price1) / (payoff2 - payoff1)
                breakevens.append(round(breakeven, 2))
        
        return breakevens
    
    def calculate_probability_of_profit(
        self, positions: List[OptionPosition], spot: float, 
        volatility: float, days_to_expiry: int
    ) -> float:
        """Calculate probability of profit using log-normal distribution"""
        
        # Find breakeven points
        min_price = spot * 0.8
        max_price = spot * 1.2
        breakevens = self.find_breakeven_points(positions, min_price, max_price)
        
        if not breakevens:
            return 0.5  # Default if no breakeven found
        
        # Calculate probability for each breakeven
        t = days_to_expiry / 365
        
        if len(breakevens) == 1:
            # Single breakeven
            breakeven = breakevens[0]
            
            # Check if profit is above or below breakeven
            test_payoff = self.calculate_payoff(positions, breakeven + 1)
            
            if test_payoff > 0:
                # Profit above breakeven
                d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * volatility ** 2) * t) / (volatility * np.sqrt(t))
                prob = 1 - norm.cdf(d)
            else:
                # Profit below breakeven
                d = (np.log(spot / breakeven) + (self.risk_free_rate - 0.5 * volatility ** 2) * t) / (volatility * np.sqrt(t))
                prob = norm.cdf(d)
        
        elif len(breakevens) == 2:
            # Two breakevens (profit in between)
            lower_be = min(breakevens)
            upper_be = max(breakevens)
            
            d1 = (np.log(spot / lower_be) + (self.risk_free_rate - 0.5 * volatility ** 2) * t) / (volatility * np.sqrt(t))
            d2 = (np.log(spot / upper_be) + (self.risk_free_rate - 0.5 * volatility ** 2) * t) / (volatility * np.sqrt(t))
            
            prob = norm.cdf(d1) - norm.cdf(d2)
        
        else:
            # Multiple breakevens - use simulation
            prob = self._simulate_probability_of_profit(positions, spot, volatility, days_to_expiry)
        
        return min(max(prob, 0), 1)  # Ensure between 0 and 1
    
    def _simulate_probability_of_profit(
        self, positions: List[OptionPosition], spot: float, 
        volatility: float, days_to_expiry: int, simulations: int = 10000
    ) -> float:
        """Monte Carlo simulation for probability of profit"""
        
        t = days_to_expiry / 365
        
        # Generate random price paths
        random_shocks = np.random.normal(0, 1, simulations)
        
        # Log-normal distribution
        future_prices = spot * np.exp(
            (self.risk_free_rate - 0.5 * volatility ** 2) * t + 
            volatility * np.sqrt(t) * random_shocks
        )
        
        # Calculate payoffs
        profits = 0
        for price in future_prices:
            if self.calculate_payoff(positions, price) > 0:
                profits += 1
        
        return profits / simulations


class IndianOptionsStrategist:
    """Indian market options strategist"""
    
    def __init__(self):
        self.market_service = IndianMarketService()
        self.calculator = IndianOptionsCalculator()
        
        # Strategy selection criteria
        self.market_conditions = {
            'trending_up': [
                IndianOptionStrategy.BULL_CALL_SPREAD,
                IndianOptionStrategy.BULL_PUT_SPREAD,
                IndianOptionStrategy.LONG_CALL
            ],
            'trending_down': [
                IndianOptionStrategy.BEAR_PUT_SPREAD,
                IndianOptionStrategy.BEAR_CALL_SPREAD,
                IndianOptionStrategy.LONG_PUT
            ],
            'range_bound': [
                IndianOptionStrategy.SHORT_STRADDLE,
                IndianOptionStrategy.SHORT_STRANGLE,
                IndianOptionStrategy.IRON_CONDOR,
                IndianOptionStrategy.IRON_BUTTERFLY
            ],
            'high_volatility': [
                IndianOptionStrategy.LONG_STRADDLE,
                IndianOptionStrategy.LONG_STRANGLE
            ],
            'expiry_day': [
                IndianOptionStrategy.EXPIRY_STRADDLE,
                IndianOptionStrategy.EXPIRY_STRANGLE,
                IndianOptionStrategy.ZERO_DTE
            ]
        }
        
        # Indian events calendar
        self.event_calendar = {
            'rbi_policy': [
                datetime(2024, 2, 8),
                datetime(2024, 4, 5),
                datetime(2024, 6, 7),
                datetime(2024, 8, 8),
                datetime(2024, 10, 9),
                datetime(2024, 12, 6)
            ],
            'budget': datetime(2024, 2, 1),
            'quarterly_results': self._get_results_dates(),
            'monthly_expiry': self._get_monthly_expiries()
        }
    
    async def recommend_strategy(
        self, symbol: str, capital: float, risk_tolerance: str = 'medium'
    ) -> StrategyRecommendation:
        """Recommend best options strategy for Indian market"""
        
        # Get market data
        market_data = await self._analyze_market_conditions(symbol)
        
        # Check for special events
        event_context = self._check_upcoming_events()
        
        # Select appropriate strategies
        suitable_strategies = self._select_strategies(
            market_data, event_context, risk_tolerance
        )
        
        # Evaluate each strategy
        recommendations = []
        
        for strategy in suitable_strategies:
            recommendation = await self._build_strategy(
                symbol, strategy, capital, market_data
            )
            if recommendation:
                recommendations.append(recommendation)
        
        # Select best strategy
        if recommendations:
            best = max(recommendations, key=lambda x: self._score_strategy(x))
            return best
        
        # Default to covered strategy
        return await self._build_default_strategy(symbol, capital, market_data)
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions"""
        
        # Get current data
        stock_data = await self.market_service.get_stock_data(symbol)
        option_chain = await self.market_service.get_option_chain_analysis(symbol)
        
        # Calculate technical indicators
        volatility = self._calculate_implied_volatility(option_chain)
        trend = self._identify_trend(stock_data)
        
        # Check if it's expiry day
        is_expiry = self._is_expiry_day(symbol)
        
        return {
            'spot_price': stock_data.price,
            'volatility': volatility,
            'trend': trend,
            'pcr': option_chain.get('pcr', 1.0),
            'max_pain': option_chain.get('max_pain', stock_data.price),
            'iv_percentile': option_chain.get('iv_percentile', 50),
            'is_expiry': is_expiry,
            'support_resistance': self._find_support_resistance(stock_data, option_chain),
            'market_sentiment': option_chain.get('trading_signal', 'neutral')
        }
    
    def _identify_trend(self, stock_data) -> str:
        """Identify current trend"""
        
        # Simple trend identification
        change_percent = stock_data.change_percent
        
        if change_percent > 1.5:
            return 'trending_up'
        elif change_percent < -1.5:
            return 'trending_down'
        else:
            # Check week range
            price_position = (stock_data.price - stock_data.week_52_low) / (
                stock_data.week_52_high - stock_data.week_52_low
            )
            
            if price_position > 0.7:
                return 'trending_up'
            elif price_position < 0.3:
                return 'trending_down'
            else:
                return 'range_bound'
    
    def _calculate_implied_volatility(self, option_chain: Dict) -> float:
        """Calculate average implied volatility"""
        
        avg_iv = option_chain.get('avg_iv', 20)
        
        # Adjust for Indian market characteristics
        # Indian markets typically have higher volatility
        if avg_iv < 15:
            return 15  # Minimum IV
        elif avg_iv > 50:
            return 50  # Cap extreme IV
        
        return avg_iv
    
    def _is_expiry_day(self, symbol: str) -> bool:
        """Check if today is expiry day"""
        
        today = datetime.now()
        
        # Check weekly expiry
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            expiry_weekday = self.calculator.weekly_expiry.get(symbol)
            return today.weekday() == expiry_weekday
        
        # Check monthly expiry (last Thursday)
        # This is simplified - actual logic would be more complex
        return today.weekday() == 3 and today.day > 24
    
    def _find_support_resistance(self, stock_data, option_chain: Dict) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        
        support = option_chain.get('support_levels', [])
        resistance = option_chain.get('resistance_levels', [])
        
        # Add price-based levels
        if not support:
            support = [
                stock_data.price * 0.98,
                stock_data.price * 0.96,
                stock_data.week_52_low
            ]
        
        if not resistance:
            resistance = [
                stock_data.price * 1.02,
                stock_data.price * 1.04,
                stock_data.week_52_high
            ]
        
        return {
            'support': sorted(support, reverse=True)[:3],
            'resistance': sorted(resistance)[:3]
        }
    
    def _check_upcoming_events(self) -> Dict[str, Any]:
        """Check for upcoming market events"""
        
        today = datetime.now()
        upcoming_events = {}
        
        # Check RBI policy
        for policy_date in self.event_calendar['rbi_policy']:
            if 0 <= (policy_date - today).days <= 7:
                upcoming_events['rbi_policy'] = policy_date
                break
        
        # Check budget
        if 0 <= (self.event_calendar['budget'] - today).days <= 7:
            upcoming_events['budget'] = self.event_calendar['budget']
        
        # Check monthly expiry
        next_expiry = self._get_next_expiry()
        days_to_expiry = (next_expiry - today).days
        
        if days_to_expiry <= 3:
            upcoming_events['expiry'] = next_expiry
            upcoming_events['expiry_effect'] = True
        
        return upcoming_events
    
    def _select_strategies(
        self, market_data: Dict, events: Dict, risk_tolerance: str
    ) -> List[IndianOptionStrategy]:
        """Select suitable strategies based on conditions"""
        
        strategies = []
        
        # Market condition based selection
        trend = market_data['trend']
        if trend in self.market_conditions:
            strategies.extend(self.market_conditions[trend])
        
        # Volatility based
        if market_data['volatility'] > 25:
            strategies.extend(self.market_conditions['high_volatility'])
        
        # Expiry day strategies
        if market_data['is_expiry']:
            strategies.extend(self.market_conditions['expiry_day'])
        
        # Event based adjustments
        if 'rbi_policy' in events or 'budget' in events:
            # Add volatility strategies before events
            strategies.append(IndianOptionStrategy.LONG_STRADDLE)
            strategies.append(IndianOptionStrategy.LONG_STRANGLE)
        
        # Risk tolerance filter
        if risk_tolerance == 'low':
            # Remove naked selling strategies
            risky_strategies = [
                IndianOptionStrategy.SHORT_STRADDLE,
                IndianOptionStrategy.SHORT_STRANGLE,
                IndianOptionStrategy.SHORT_CALL,
                IndianOptionStrategy.SHORT_PUT
            ]
            strategies = [s for s in strategies if s not in risky_strategies]
        
        # Remove duplicates and return
        return list(set(strategies))[:5]  # Top 5 strategies
    
    async def _build_strategy(
        self, symbol: str, strategy: IndianOptionStrategy, 
        capital: float, market_data: Dict
    ) -> Optional[StrategyRecommendation]:
        """Build specific strategy"""
        
        try:
            if strategy == IndianOptionStrategy.BULL_CALL_SPREAD:
                return await self._build_bull_call_spread(symbol, capital, market_data)
            elif strategy == IndianOptionStrategy.IRON_CONDOR:
                return await self._build_iron_condor(symbol, capital, market_data)
            elif strategy == IndianOptionStrategy.EXPIRY_STRADDLE:
                return await self._build_expiry_straddle(symbol, capital, market_data)
            # Add more strategies...
            
        except Exception as e:
            print(f"Error building {strategy}: {str(e)}")
            
        return None
    
    async def _build_bull_call_spread(
        self, symbol: str, capital: float, market_data: Dict
    ) -> StrategyRecommendation:
        """Build bull call spread strategy"""
        
        spot = market_data['spot_price']
        
        # Select strikes
        buy_strike = self._round_to_strike(spot * 0.99, symbol)  # Slightly ITM
        sell_strike = self._round_to_strike(spot * 1.02, symbol)  # OTM
        
        # Get next expiry
        expiry = self._get_next_expiry()
        days_to_expiry = (expiry - datetime.now()).days
        
        # Calculate premiums
        buy_premium = self.calculator.calculate_option_price(
            spot, buy_strike, days_to_expiry, 
            market_data['volatility'] / 100, 'CE'
        )['price']
        
        sell_premium = self.calculator.calculate_option_price(
            spot, sell_strike, days_to_expiry,
            market_data['volatility'] / 100, 'CE'
        )['price']
        
        # Get lot size
        lot_size = self._get_lot_size(symbol)
        
        # Calculate number of lots based on capital
        net_debit = (buy_premium - sell_premium) * lot_size
        max_lots = int(capital / net_debit)
        lots = min(max_lots, 10)  # Cap at 10 lots
        
        # Create positions
        positions = [
            OptionPosition(
                symbol=symbol,
                strike=buy_strike,
                option_type='CE',
                expiry=expiry,
                action='buy',
                quantity=lots,
                premium=buy_premium,
                lot_size=lot_size
            ),
            OptionPosition(
                symbol=symbol,
                strike=sell_strike,
                option_type='CE',
                expiry=expiry,
                action='sell',
                quantity=lots,
                premium=sell_premium,
                lot_size=lot_size
            )
        ]
        
        # Calculate strategy metrics
        max_profit = (sell_strike - buy_strike - buy_premium + sell_premium) * lot_size * lots
        max_loss = net_debit * lots
        breakeven = buy_strike + buy_premium - sell_premium
        
        # Calculate margin
        margin = self.calculator.calculate_margin(
            IndianOptionStrategy.BULL_CALL_SPREAD, positions
        )
        
        # Probability of profit
        prob_profit = self.calculator.calculate_probability_of_profit(
            positions, spot, market_data['volatility'] / 100, days_to_expiry
        )
        
        return StrategyRecommendation(
            strategy=IndianOptionStrategy.BULL_CALL_SPREAD,
            positions=positions,
            total_margin=margin,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_points=[breakeven],
            probability_of_profit=prob_profit,
            expected_return=(max_profit * prob_profit - max_loss * (1 - prob_profit)) / margin,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else float('inf'),
            market_outlook='Moderately Bullish',
            entry_conditions=[
                f"Enter when {symbol} is above {spot * 0.99:.2f}",
                f"IV below {market_data['volatility'] * 1.2:.1f}%",
                "Positive market sentiment"
            ],
            exit_conditions=[
                f"Exit if {symbol} falls below {buy_strike}",
                "Exit 1 day before expiry",
                "Book profit at 70% of max profit"
            ],
            adjustments=[
                "Roll up spread if stock moves strongly",
                "Convert to butterfly if outlook changes"
            ],
            indian_context={
                'suitable_for': 'Trending markets post results',
                'avoid_during': 'High volatility events like RBI policy',
                'tax_treatment': 'Business income if frequently traded'
            }
        )
    
    async def _build_iron_condor(
        self, symbol: str, capital: float, market_data: Dict
    ) -> StrategyRecommendation:
        """Build iron condor strategy"""
        
        spot = market_data['spot_price']
        support = market_data['support_resistance']['support'][0]
        resistance = market_data['support_resistance']['resistance'][0]
        
        # Select strikes outside expected range
        put_sell_strike = self._round_to_strike(support, symbol)
        put_buy_strike = self._round_to_strike(put_sell_strike * 0.97, symbol)
        call_sell_strike = self._round_to_strike(resistance, symbol)
        call_buy_strike = self._round_to_strike(call_sell_strike * 1.03, symbol)
        
        # Get expiry
        expiry = self._get_next_expiry()
        days_to_expiry = (expiry - datetime.now()).days
        
        # Calculate premiums for all legs
        positions = []
        total_credit = 0
        
        strikes_actions = [
            (put_buy_strike, 'PE', 'buy'),
            (put_sell_strike, 'PE', 'sell'),
            (call_sell_strike, 'CE', 'sell'),
            (call_buy_strike, 'CE', 'buy')
        ]
        
        lot_size = self._get_lot_size(symbol)
        
        for strike, opt_type, action in strikes_actions:
            premium = self.calculator.calculate_option_price(
                spot, strike, days_to_expiry,
                market_data['volatility'] / 100, opt_type
            )['price']
            
            positions.append(OptionPosition(
                symbol=symbol,
                strike=strike,
                option_type=opt_type,
                expiry=expiry,
                action=action,
                quantity=1,
                premium=premium,
                lot_size=lot_size
            ))
            
            if action == 'sell':
                total_credit += premium * lot_size
            else:
                total_credit -= premium * lot_size
        
        # Calculate metrics
        max_profit = total_credit
        max_loss = min(
            (put_sell_strike - put_buy_strike) * lot_size - total_credit,
            (call_buy_strike - call_sell_strike) * lot_size - total_credit
        )
        
        breakevens = [
            put_sell_strike - total_credit / lot_size,
            call_sell_strike + total_credit / lot_size
        ]
        
        margin = self.calculator.calculate_margin(
            IndianOptionStrategy.IRON_CONDOR, positions
        )
        
        prob_profit = self.calculator.calculate_probability_of_profit(
            positions, spot, market_data['volatility'] / 100, days_to_expiry
        )
        
        return StrategyRecommendation(
            strategy=IndianOptionStrategy.IRON_CONDOR,
            positions=positions,
            total_margin=margin,
            max_profit=max_profit,
            max_loss=abs(max_loss),
            break_even_points=breakevens,
            probability_of_profit=prob_profit,
            expected_return=(max_profit * prob_profit - abs(max_loss) * (1 - prob_profit)) / margin,
            risk_reward_ratio=max_profit / abs(max_loss) if max_loss != 0 else float('inf'),
            market_outlook='Range Bound',
            entry_conditions=[
                f"Enter when {symbol} is between {put_sell_strike} and {call_sell_strike}",
                f"IV above {market_data['iv_percentile']}th percentile",
                "Low expected movement"
            ],
            exit_conditions=[
                f"Exit if {symbol} breaches {put_sell_strike} or {call_sell_strike}",
                "Close at 50% profit",
                "Exit 2 days before expiry"
            ],
            adjustments=[
                "Roll untested side if one side is threatened",
                "Convert to iron butterfly to reduce risk"
            ],
            indian_context={
                'suitable_for': 'Low volatility periods, post-event',
                'avoid_during': 'Results season, RBI policy week',
                'margin_benefit': 'SEBI allows 70% margin benefit for spreads'
            }
        )
    
    async def _build_expiry_straddle(
        self, symbol: str, capital: float, market_data: Dict
    ) -> StrategyRecommendation:
        """Build expiry day straddle strategy"""
        
        if not market_data['is_expiry']:
            # Switch to next expiry if not expiry day
            expiry = self._get_next_expiry()
        else:
            expiry = datetime.now().replace(hour=15, minute=30)
        
        spot = market_data['spot_price']
        strike = self._round_to_strike(spot, symbol)  # ATM
        
        days_to_expiry = max((expiry - datetime.now()).days, 0.1)  # Minimum 0.1 for same day
        
        # Higher IV on expiry day
        expiry_iv = market_data['volatility'] * 1.2
        
        # Calculate premiums
        call_premium = self.calculator.calculate_option_price(
            spot, strike, days_to_expiry, expiry_iv / 100, 'CE'
        )['price']
        
        put_premium = self.calculator.calculate_option_price(
            spot, strike, days_to_expiry, expiry_iv / 100, 'PE'
        )['price']
        
        lot_size = self._get_lot_size(symbol)
        
        # For expiry straddle, we buy both
        positions = [
            OptionPosition(
                symbol=symbol,
                strike=strike,
                option_type='CE',
                expiry=expiry,
                action='buy',
                quantity=1,
                premium=call_premium,
                lot_size=lot_size
            ),
            OptionPosition(
                symbol=symbol,
                strike=strike,
                option_type='PE',
                expiry=expiry,
                action='buy',
                quantity=1,
                premium=put_premium,
                lot_size=lot_size
            )
        ]
        
        total_debit = (call_premium + put_premium) * lot_size
        
        # Expiry day specific calculations
        breakevens = [
            strike - call_premium - put_premium,
            strike + call_premium + put_premium
        ]
        
        # Need larger movement on expiry
        required_move = (call_premium + put_premium) / strike
        
        # Lower probability on expiry due to theta
        prob_profit = 0.35 if days_to_expiry < 1 else self.calculator.calculate_probability_of_profit(
            positions, spot, expiry_iv / 100, days_to_expiry
        )
        
        return StrategyRecommendation(
            strategy=IndianOptionStrategy.EXPIRY_STRADDLE,
            positions=positions,
            total_margin=total_debit,  # Full premium for long positions
            max_profit=float('inf'),  # Unlimited
            max_loss=total_debit,
            break_even_points=breakevens,
            probability_of_profit=prob_profit,
            expected_return=-0.5 if days_to_expiry < 1 else 0.2,  # Negative expected on expiry
            risk_reward_ratio=float('inf'),
            market_outlook='High Volatility Expected',
            entry_conditions=[
                "Enter at 9:30 AM on expiry day",
                f"Required move: {required_move*100:.1f}%",
                "Check for news/events"
            ],
            exit_conditions=[
                "Exit by 2:30 PM on expiry",
                "Book profit at 30% gain",
                "Cut loss at 50%"
            ],
            adjustments=[
                "Convert to strangle if big move happens",
                "Hedge with futures if directional"
            ],
            indian_context={
                'expiry_behavior': 'High gamma risk, rapid time decay',
                'typical_pattern': 'Range till 2 PM, then directional',
                'caution': 'STT on ITM options can eat profits'
            }
        )
    
    def _round_to_strike(self, price: float, symbol: str) -> float:
        """Round price to valid strike price"""
        
        # Strike intervals for Indian markets
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            if price < 100:
                interval = 2.5
            elif price < 500:
                interval = 5
            elif price < 1000:
                interval = 10
            elif price < 5000:
                interval = 50
            else:
                interval = 100
        else:
            # Stocks
            if price < 100:
                interval = 2.5
            elif price < 500:
                interval = 5
            elif price < 1000:
                interval = 10
            elif price < 2500:
                interval = 25
            else:
                interval = 50
        
        return round(price / interval) * interval
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for symbol"""
        
        # Index lot sizes
        if symbol in self.calculator.index_lot_sizes:
            return self.calculator.index_lot_sizes[symbol]
        
        # Stock lot sizes (simplified)
        return 1000  # Default
    
    def _get_next_expiry(self) -> datetime:
        """Get next expiry date"""
        
        today = datetime.now()
        
        # For weekly expiry (simplified)
        days_ahead = 3 - today.weekday()  # Thursday
        if days_ahead <= 0:
            days_ahead += 7
        
        return today + timedelta(days=days_ahead)
    
    def _get_results_dates(self) -> List[datetime]:
        """Get quarterly results dates"""
        
        # Typical Indian results season
        results_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        results_dates = []
        
        current_year = datetime.now().year
        
        for month in results_months:
            # Results typically in 2nd-3rd week
            results_dates.extend([
                datetime(current_year, month, 10),
                datetime(current_year, month, 17),
                datetime(current_year, month, 24)
            ])
        
        return results_dates
    
    def _get_monthly_expiries(self) -> List[datetime]:
        """Get monthly expiry dates"""
        
        expiries = []
        current = datetime.now()
        
        for i in range(12):
            # Last Thursday of each month
            month = current.month + i
            year = current.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            
            # Find last Thursday
            last_day = 31
            while last_day > 0:
                try:
                    date = datetime(year, month, last_day)
                    if date.weekday() == 3:  # Thursday
                        expiries.append(date)
                        break
                except ValueError:
                    pass
                last_day -= 1
        
        return expiries
    
    def _score_strategy(self, recommendation: StrategyRecommendation) -> float:
        """Score a strategy for ranking"""
        
        score = 0
        
        # Probability of profit (40% weight)
        score += recommendation.probability_of_profit * 40
        
        # Risk-reward ratio (30% weight)
        if recommendation.risk_reward_ratio != float('inf'):
            score += min(recommendation.risk_reward_ratio * 10, 30)
        else:
            score += 30
        
        # Expected return (20% weight)
        score += min(max(recommendation.expected_return * 100, -20), 20)
        
        # Margin efficiency (10% weight)
        margin_efficiency = recommendation.max_profit / recommendation.total_margin
        score += min(margin_efficiency * 10, 10)
        
        return score
    
    async def _build_default_strategy(
        self, symbol: str, capital: float, market_data: Dict
    ) -> StrategyRecommendation:
        """Build a default conservative strategy"""
        
        # Default to bull put spread - limited risk, positive theta
        return await self._build_bull_put_spread(symbol, capital, market_data)
    
    async def _build_bull_put_spread(
        self, symbol: str, capital: float, market_data: Dict
    ) -> StrategyRecommendation:
        """Build bull put spread - credit strategy"""
        
        spot = market_data['spot_price']
        support = market_data['support_resistance']['support'][0]
        
        # Select strikes below support
        sell_strike = self._round_to_strike(support, symbol)
        buy_strike = self._round_to_strike(sell_strike * 0.95, symbol)
        
        expiry = self._get_next_expiry()
        days_to_expiry = (expiry - datetime.now()).days
        
        # Calculate premiums
        sell_premium = self.calculator.calculate_option_price(
            spot, sell_strike, days_to_expiry,
            market_data['volatility'] / 100, 'PE'
        )['price']
        
        buy_premium = self.calculator.calculate_option_price(
            spot, buy_strike, days_to_expiry,
            market_data['volatility'] / 100, 'PE'
        )['price']
        
        lot_size = self._get_lot_size(symbol)
        net_credit = (sell_premium - buy_premium) * lot_size
        
        positions = [
            OptionPosition(
                symbol=symbol,
                strike=sell_strike,
                option_type='PE',
                expiry=expiry,
                action='sell',
                quantity=1,
                premium=sell_premium,
                lot_size=lot_size
            ),
            OptionPosition(
                symbol=symbol,
                strike=buy_strike,
                option_type='PE',
                expiry=expiry,
                action='buy',
                quantity=1,
                premium=buy_premium,
                lot_size=lot_size
            )
        ]
        
        max_profit = net_credit
        max_loss = (sell_strike - buy_strike) * lot_size - net_credit
        breakeven = sell_strike - net_credit / lot_size
        
        margin = self.calculator.calculate_margin(
            IndianOptionStrategy.BULL_PUT_SPREAD, positions
        )
        
        prob_profit = self.calculator.calculate_probability_of_profit(
            positions, spot, market_data['volatility'] / 100, days_to_expiry
        )
        
        return StrategyRecommendation(
            strategy=IndianOptionStrategy.BULL_PUT_SPREAD,
            positions=positions,
            total_margin=margin,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_points=[breakeven],
            probability_of_profit=prob_profit,
            expected_return=(max_profit * prob_profit - max_loss * (1 - prob_profit)) / margin,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else float('inf'),
            market_outlook='Neutral to Bullish',
            entry_conditions=[
                f"Enter when {symbol} is above {sell_strike}",
                "Positive market breadth",
                "Support holding"
            ],
            exit_conditions=[
                "Exit at 50% of max profit",
                f"Exit if {symbol} breaks {sell_strike}",
                "Exit 3 days before expiry"
            ],
            adjustments=[
                "Roll down if market rallies strongly",
                "Convert to butterfly to lock profits"
            ],
            indian_context={
                'suitable_for': 'Regular income generation',
                'tax_benefit': 'Premium received is business income',
                'margin_note': 'Lower margin due to spread benefit'
            }
        )
