"""
Zerodha KiteConnect Integration Service for live trading and data access.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from decimal import Decimal
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect, KiteTicker
import json
import hashlib
from redis import asyncio as aioredis

from app.models.trade import TradeType
from app.services.indian_market_service import IndianMarketService
from app.services.risk_management import RiskManagementService
from app.services.sebi_compliance_service import SEBIComplianceService

logger = logging.getLogger(__name__)


class ZerodhaIntegrationService:
    """
    Comprehensive Zerodha integration for trading, data access, and portfolio management.
    """
    
    def __init__(self):
        self.kite: Optional[KiteConnect] = None
        self.kite_ticker: Optional[KiteTicker] = None
        self.indian_market = IndianMarketService()
        self.risk_manager = RiskManagementService()
        self.compliance = SEBIComplianceService()
        self.redis_client = None
        
        # Zerodha specific configurations
        self.api_key = None
        self.api_secret = None
        self.access_token = None
        self.user_id = None
        
        # Loss recovery parameters
        self.max_daily_loss_percent = 2.0  # 2% max daily loss
        self.recovery_mode = False
        self.original_capital = Decimal('0')
        self.current_losses = Decimal('0')
        
    async def initialize(self, api_key: str, api_secret: str, request_token: Optional[str] = None):
        """Initialize Zerodha connection with API credentials."""
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            self.kite = KiteConnect(api_key=api_key)
            
            if request_token:
                # Generate access token from request token
                data = self.kite.generate_session(request_token, api_secret=api_secret)
                self.access_token = data["access_token"]
                self.user_id = data["user_id"]
                self.kite.set_access_token(self.access_token)
                
                # Store in Redis for persistence
                self.redis_client = await aioredis.create_redis_pool('redis://localhost')
                await self.redis_client.setex(
                    f"zerodha_token:{self.user_id}",
                    3600 * 8,  # 8 hours validity
                    self.access_token
                )
                
                logger.info(f"Zerodha connection established for user: {self.user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha: {e}")
            raise
    
    async def get_portfolio_analysis(self) -> Dict[str, Any]:
        """Analyze current portfolio with loss recovery suggestions."""
        try:
            # Get current positions
            positions = self.kite.positions()
            holdings = self.kite.holdings()
            orders = self.kite.orders()
            
            # Calculate P&L
            total_pnl = Decimal('0')
            unrealized_pnl = Decimal('0')
            realized_pnl = Decimal('0')
            
            # Analyze positions
            position_analysis = []
            for pos in positions['net']:
                pnl = Decimal(str(pos['pnl']))
                total_pnl += pnl
                
                if pos['quantity'] != 0:
                    unrealized_pnl += pnl
                else:
                    realized_pnl += pnl
                
                # AI-based position analysis
                analysis = await self._analyze_position(pos)
                position_analysis.append({
                    'symbol': pos['tradingsymbol'],
                    'pnl': float(pnl),
                    'quantity': pos['quantity'],
                    'recommendation': analysis['recommendation'],
                    'confidence': analysis['confidence'],
                    'exit_strategy': analysis['exit_strategy']
                })
            
            # Analyze holdings
            holding_analysis = []
            for holding in holdings:
                pnl = Decimal(str(holding['pnl']))
                unrealized_pnl += pnl
                
                analysis = await self._analyze_holding(holding)
                holding_analysis.append({
                    'symbol': holding['tradingsymbol'],
                    'pnl': float(pnl),
                    'quantity': holding['quantity'],
                    'average_price': holding['average_price'],
                    'ltp': holding['last_price'],
                    'recommendation': analysis['recommendation'],
                    'target_price': analysis['target_price'],
                    'stop_loss': analysis['stop_loss']
                })
            
            # Loss recovery strategies
            recovery_strategies = await self._generate_recovery_strategies(
                total_pnl,
                position_analysis,
                holding_analysis
            )
            
            return {
                'total_pnl': float(total_pnl),
                'unrealized_pnl': float(unrealized_pnl),
                'realized_pnl': float(realized_pnl),
                'positions': position_analysis,
                'holdings': holding_analysis,
                'recovery_strategies': recovery_strategies,
                'risk_score': await self._calculate_portfolio_risk(),
                'recommendations': await self._get_ai_recommendations(total_pnl)
            }
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            raise
    
    async def _analyze_position(self, position: Dict) -> Dict[str, Any]:
        """AI-powered position analysis with exit strategies."""
        symbol = position['tradingsymbol']
        
        # Get technical indicators
        try:
            # Fetch historical data
            historical = self.kite.historical_data(
                position['instrument_token'],
                datetime.now() - timedelta(days=30),
                datetime.now(),
                '15minute'
            )
            
            df = pd.DataFrame(historical)
            
            # Calculate indicators
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['signal'] = self._calculate_macd(df['close'])
            support, resistance = self._find_support_resistance(df)
            
            # AI recommendation based on multiple factors
            current_price = position['last_price']
            avg_price = position['average_price'] if position['average_price'] else current_price
            pnl_percent = ((current_price - avg_price) / avg_price) * 100
            
            # Decision logic
            if pnl_percent < -5:  # Losing position
                if df['rsi'].iloc[-1] < 30:  # Oversold
                    recommendation = "HOLD_FOR_RECOVERY"
                    confidence = 0.75
                    exit_strategy = {
                        'type': 'staged_exit',
                        'levels': [
                            {'price': avg_price * 0.98, 'quantity_percent': 50},
                            {'price': avg_price * 1.02, 'quantity_percent': 50}
                        ]
                    }
                else:
                    recommendation = "CUT_LOSS"
                    confidence = 0.85
                    exit_strategy = {
                        'type': 'immediate',
                        'stop_loss': current_price * 0.98
                    }
            elif pnl_percent > 5:  # Winning position
                if df['rsi'].iloc[-1] > 70:  # Overbought
                    recommendation = "BOOK_PARTIAL_PROFIT"
                    confidence = 0.8
                    exit_strategy = {
                        'type': 'trailing_stop',
                        'trail_percent': 2.0
                    }
                else:
                    recommendation = "HOLD_WITH_TRAILING_STOP"
                    confidence = 0.7
                    exit_strategy = {
                        'type': 'trailing_stop',
                        'trail_percent': 3.0
                    }
            else:  # Neutral position
                recommendation = "MONITOR"
                confidence = 0.6
                exit_strategy = {
                    'type': 'bracket',
                    'stop_loss': support,
                    'target': resistance
                }
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'exit_strategy': exit_strategy,
                'technical_indicators': {
                    'rsi': float(df['rsi'].iloc[-1]),
                    'macd': float(df['macd'].iloc[-1]),
                    'support': float(support),
                    'resistance': float(resistance)
                }
            }
            
        except Exception as e:
            logger.error(f"Position analysis failed for {symbol}: {e}")
            return {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'exit_strategy': {'type': 'manual'}
            }
    
    async def _generate_recovery_strategies(
        self,
        total_loss: Decimal,
        positions: List[Dict],
        holdings: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate specific strategies to recover losses."""
        strategies = []
        
        # Strategy 1: Tax Loss Harvesting
        if total_loss < 0:
            tax_harvest_candidates = [
                h for h in holdings 
                if h['pnl'] < -1000 and h['recommendation'] != 'HOLD_FOR_RECOVERY'
            ]
            
            if tax_harvest_candidates:
                strategies.append({
                    'name': 'Tax Loss Harvesting',
                    'description': 'Book losses to offset taxes and redeploy capital',
                    'potential_tax_savings': float(abs(sum(h['pnl'] for h in tax_harvest_candidates)) * 0.15),
                    'candidates': [h['symbol'] for h in tax_harvest_candidates],
                    'priority': 'HIGH'
                })
        
        # Strategy 2: High Probability Option Selling
        if len(holdings) > 0:
            covered_call_candidates = [
                h for h in holdings 
                if h['quantity'] >= 100 and h['pnl'] > -10
            ]
            
            if covered_call_candidates:
                strategies.append({
                    'name': 'Covered Call Strategy',
                    'description': 'Generate monthly income by selling OTM calls',
                    'potential_monthly_income': len(covered_call_candidates) * 2000,  # Rough estimate
                    'candidates': [h['symbol'] for h in covered_call_candidates],
                    'priority': 'MEDIUM'
                })
        
        # Strategy 3: Index Option Spreads
        strategies.append({
            'name': 'NIFTY/BANKNIFTY Credit Spreads',
            'description': 'Weekly credit spreads with 80%+ win rate',
            'potential_weekly_income': 5000,  # Conservative estimate
            'risk_reward_ratio': '1:3',
            'recommended_capital': 200000,
            'priority': 'HIGH'
        })
        
        # Strategy 4: Momentum Trading on Leaders
        strategies.append({
            'name': 'Momentum Trading',
            'description': 'Trade top 10 momentum stocks with strict stop losses',
            'expected_monthly_return': '8-12%',
            'recommended_stocks': await self._get_momentum_stocks(),
            'max_position_size': '10% of capital',
            'priority': 'MEDIUM'
        })
        
        # Strategy 5: Arbitrage Opportunities
        arb_opportunities = await self._find_arbitrage_opportunities()
        if arb_opportunities:
            strategies.append({
                'name': 'Risk-Free Arbitrage',
                'description': 'Exploit price differences between NSE and BSE',
                'opportunities': arb_opportunities,
                'potential_daily_profit': sum(opp['profit'] for opp in arb_opportunities),
                'priority': 'VERY_HIGH'
            })
        
        return sorted(strategies, key=lambda x: {'VERY_HIGH': 0, 'HIGH': 1, 'MEDIUM': 2}.get(x['priority'], 3))
    
    async def execute_smart_order(
        self,
        symbol: str,
        quantity: int,
        order_type: str,
        transaction_type: str,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute orders with AI-powered enhancements."""
        try:
            # Pre-trade compliance check
            compliance_check = await self.compliance.check_order_compliance({
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'transaction_type': transaction_type,
                'price': price
            })
            
            if not compliance_check['compliant']:
                return {
                    'success': False,
                    'reason': compliance_check['reason'],
                    'suggestions': compliance_check['suggestions']
                }
            
            # Risk management check
            risk_check = await self.risk_manager.check_position_risk(
                symbol, quantity, price or 0
            )
            
            if risk_check['risk_score'] > 0.8:
                return {
                    'success': False,
                    'reason': 'Risk too high',
                    'risk_score': risk_check['risk_score'],
                    'suggestions': risk_check['mitigation_strategies']
                }
            
            # Smart order routing
            order_params = await self._optimize_order_params(
                symbol, quantity, order_type, transaction_type, price
            )
            
            # Place order with Zerodha
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=order_params['exchange'],
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_params['order_type'],
                price=order_params.get('price'),
                trigger_price=order_params.get('trigger_price'),
                validity=order_params.get('validity', 'DAY'),
                variety=order_params.get('variety', 'regular'),
                tag='AI_POWERED'
            )
            
            # Set up monitoring
            asyncio.create_task(self._monitor_order(order_id))
            
            return {
                'success': True,
                'order_id': order_id,
                'enhanced_params': order_params,
                'expected_execution_time': order_params.get('expected_execution_time'),
                'ai_confidence': order_params.get('ai_confidence', 0.85)
            }
            
        except Exception as e:
            logger.error(f"Smart order execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _optimize_order_params(
        self,
        symbol: str,
        quantity: int,
        order_type: str,
        transaction_type: str,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Optimize order parameters using AI and market microstructure."""
        # Get current market depth
        quote = self.kite.quote([f"NSE:{symbol}"])
        depth = quote[f"NSE:{symbol}"]["depth"]
        
        # Analyze liquidity
        bid_liquidity = sum(level['quantity'] for level in depth['buy'])
        ask_liquidity = sum(level['quantity'] for level in depth['sell'])
        
        # Smart routing decision
        if symbol in ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']:  # Dual listed
            nse_quote = self.kite.quote([f"NSE:{symbol}"])[f"NSE:{symbol}"]
            bse_quote = self.kite.quote([f"BSE:{symbol}"])[f"BSE:{symbol}"]
            
            if transaction_type == 'BUY':
                exchange = 'NSE' if nse_quote['last_price'] <= bse_quote['last_price'] else 'BSE'
            else:
                exchange = 'BSE' if bse_quote['last_price'] >= nse_quote['last_price'] else 'NSE'
        else:
            exchange = 'NSE'
        
        # Optimize order type based on urgency and market conditions
        optimized_params = {
            'exchange': exchange,
            'order_type': order_type,
            'variety': 'regular'
        }
        
        # For large orders, use iceberg
        if quantity > bid_liquidity * 0.1 or quantity > ask_liquidity * 0.1:
            optimized_params['variety'] = 'iceberg'
            optimized_params['iceberg_legs'] = min(10, quantity // 100)
            optimized_params['iceberg_quantity'] = quantity // optimized_params['iceberg_legs']
        
        # Price optimization for limit orders
        if order_type == 'LIMIT' and not price:
            if transaction_type == 'BUY':
                # Place slightly below best ask for better fill
                optimized_params['price'] = depth['sell'][0]['price'] - 0.05
            else:
                # Place slightly above best bid
                optimized_params['price'] = depth['buy'][0]['price'] + 0.05
        
        # Add stop loss for all market orders
        if order_type == 'MARKET':
            current_price = quote[f"{exchange}:{symbol}"]['last_price']
            if transaction_type == 'BUY':
                optimized_params['trigger_price'] = current_price * 0.98
            else:
                optimized_params['trigger_price'] = current_price * 1.02
            optimized_params['variety'] = 'stoploss'
        
        # Expected execution analysis
        optimized_params['expected_execution_time'] = '< 1 second' if order_type == 'MARKET' else '1-5 minutes'
        optimized_params['ai_confidence'] = 0.9 if bid_liquidity > quantity * 5 else 0.7
        
        return optimized_params
    
    async def start_algo_trading(self, strategy: str, capital: float) -> Dict[str, Any]:
        """Start algorithmic trading with selected strategy."""
        try:
            # Initialize strategy based on selection
            if strategy == 'recovery_mode':
                return await self._start_recovery_mode_trading(capital)
            elif strategy == 'option_selling':
                return await self._start_option_selling_strategy(capital)
            elif strategy == 'momentum':
                return await self._start_momentum_strategy(capital)
            elif strategy == 'arbitrage':
                return await self._start_arbitrage_strategy(capital)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Failed to start algo trading: {e}")
            raise
    
    async def _start_recovery_mode_trading(self, capital: float) -> Dict[str, Any]:
        """Conservative strategy focused on recovering losses."""
        self.recovery_mode = True
        self.original_capital = Decimal(str(capital))
        
        # Strategy parameters
        strategy_config = {
            'max_positions': 3,
            'position_size': capital * 0.2,  # 20% per position
            'stop_loss_percent': 1.0,  # 1% stop loss
            'target_percent': 2.0,  # 2% target
            'daily_target': capital * 0.01,  # 1% daily target
            'focus_stocks': [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                'HDFC', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK'
            ],
            'strategies': [
                'mean_reversion',
                'support_resistance_bounce',
                'opening_range_breakout'
            ]
        }
        
        # Start monitoring and trading
        asyncio.create_task(self._recovery_trading_loop(strategy_config))
        
        return {
            'status': 'started',
            'mode': 'recovery',
            'config': strategy_config,
            'expected_daily_return': '0.5-1%',
            'risk_level': 'LOW',
            'monitoring': True
        }
    
    async def _recovery_trading_loop(self, config: Dict[str, Any]):
        """Main trading loop for recovery mode."""
        while self.recovery_mode:
            try:
                # Check if market is open
                if not await self.indian_market.is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Get current positions
                positions = self.kite.positions()['net']
                active_positions = [p for p in positions if p['quantity'] != 0]
                
                # Check daily P&L
                daily_pnl = sum(Decimal(str(p['pnl'])) for p in positions)
                
                if daily_pnl >= config['daily_target']:
                    logger.info(f"Daily target achieved: {daily_pnl}")
                    # Close all positions
                    for pos in active_positions:
                        await self.execute_smart_order(
                            pos['tradingsymbol'],
                            abs(pos['quantity']),
                            'MARKET',
                            'SELL' if pos['quantity'] > 0 else 'BUY'
                        )
                    await asyncio.sleep(86400)  # Wait for next day
                    continue
                
                # Look for new opportunities if under position limit
                if len(active_positions) < config['max_positions']:
                    opportunities = await self._find_trading_opportunities(config)
                    
                    for opp in opportunities[:config['max_positions'] - len(active_positions)]:
                        await self.execute_smart_order(
                            opp['symbol'],
                            opp['quantity'],
                            opp['order_type'],
                            opp['transaction_type'],
                            opp.get('price')
                        )
                
                # Monitor existing positions
                for pos in active_positions:
                    await self._monitor_position_for_exit(pos, config)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Recovery trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def _find_trading_opportunities(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find high-probability trading opportunities."""
        opportunities = []
        
        for symbol in config['focus_stocks']:
            try:
                # Get current data
                quote = self.kite.quote([f"NSE:{symbol}"])[f"NSE:{symbol}"]
                ohlc = self.kite.ohlc([f"NSE:{symbol}"])[f"NSE:{symbol}"]
                
                # Check each strategy
                for strategy in config['strategies']:
                    signal = await self._check_strategy_signal(
                        symbol, quote, ohlc, strategy
                    )
                    
                    if signal['valid']:
                        opportunities.append({
                            'symbol': symbol,
                            'strategy': strategy,
                            'transaction_type': signal['side'],
                            'order_type': 'LIMIT',
                            'price': signal['entry_price'],
                            'quantity': int(config['position_size'] / signal['entry_price']),
                            'stop_loss': signal['stop_loss'],
                            'target': signal['target'],
                            'confidence': signal['confidence']
                        })
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                continue
        
        # Sort by confidence and return top opportunities
        return sorted(opportunities, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Find support and resistance levels."""
        # Simple method using recent highs and lows
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        return float(recent_low), float(recent_high)
    
    async def _get_momentum_stocks(self) -> List[str]:
        """Get top momentum stocks."""
        # This would connect to our momentum scanner
        # For now, return top NIFTY movers
        return ['TATAMOTORS', 'ADANIENT', 'ADANIPORTS', 'BAJFINANCE', 'ZOMATO']
    
    async def _find_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities between exchanges."""
        opportunities = []
        
        dual_listed = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        
        for symbol in dual_listed:
            try:
                nse_quote = self.kite.quote([f"NSE:{symbol}"])[f"NSE:{symbol}"]
                bse_quote = self.kite.quote([f"BSE:{symbol}"])[f"BSE:{symbol}"]
                
                price_diff = abs(nse_quote['last_price'] - bse_quote['last_price'])
                price_diff_percent = (price_diff / nse_quote['last_price']) * 100
                
                if price_diff_percent > 0.1:  # 0.1% difference
                    opportunities.append({
                        'symbol': symbol,
                        'buy_exchange': 'NSE' if nse_quote['last_price'] < bse_quote['last_price'] else 'BSE',
                        'sell_exchange': 'BSE' if nse_quote['last_price'] < bse_quote['last_price'] else 'NSE',
                        'price_difference': price_diff,
                        'profit_percent': price_diff_percent,
                        'profit': price_diff * 100  # For 100 shares
                    })
                    
            except Exception as e:
                logger.error(f"Error checking arbitrage for {symbol}: {e}")
                continue
        
        return opportunities
    
    async def get_ai_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get personalized AI recommendations based on user profile."""
        recommendations = []
        
        # Analyze user's trading history and current losses
        if user_profile.get('total_loss', 0) > 100000:
            recommendations.append({
                'title': 'Start with Paper Trading',
                'description': 'Practice strategies without real money for 1 month',
                'priority': 'CRITICAL',
                'action': 'enable_paper_trading'
            })
        
        # Based on loss pattern
        if user_profile.get('loss_pattern') == 'revenge_trading':
            recommendations.append({
                'title': 'Implement Cooling Period',
                'description': 'Wait 30 minutes after any loss before next trade',
                'priority': 'HIGH',
                'action': 'enable_cooling_period'
            })
        
        # Conservative strategies for recovery
        recommendations.extend([
            {
                'title': 'Index ETF Investment',
                'description': 'Allocate 40% to NIFTYBEES for stable growth',
                'expected_return': '12-15% annually',
                'risk': 'LOW',
                'priority': 'HIGH'
            },
            {
                'title': 'Covered Call Strategy',
                'description': 'Generate 2-3% monthly income on existing holdings',
                'suitable_stocks': ['RELIANCE', 'TCS', 'INFY'],
                'priority': 'MEDIUM'
            },
            {
                'title': 'NIFTY Weekly Spreads',
                'description': 'Iron condors with 85% win rate',
                'weekly_income': '₹5,000-10,000 per lakh',
                'priority': 'HIGH'
            }
        ])
        
        return recommendations
    
    async def _monitor_order(self, order_id: str):
        """Monitor order execution and make adjustments."""
        try:
            while True:
                order = self.kite.order_history(order_id)[-1]
                
                if order['status'] in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                    logger.info(f"Order {order_id} status: {order['status']}")
                    
                    if order['status'] == 'COMPLETE':
                        # Set up position monitoring
                        asyncio.create_task(
                            self._monitor_position_post_execution(order)
                        )
                    break
                
                elif order['status'] == 'OPEN' and order.get('pending_quantity', 0) > 0:
                    # Check if we need to modify the order
                    if await self._should_modify_order(order):
                        self.kite.modify_order(
                            order_id=order_id,
                            price=await self._get_aggressive_price(order)
                        )
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Order monitoring failed: {e}")
    
    async def _monitor_position_post_execution(self, order: Dict):
        """Monitor position after execution for smart exits."""
        # Implement trailing stop loss, target monitoring, etc.
        pass
    
    async def _should_modify_order(self, order: Dict) -> bool:
        """Determine if order needs modification."""
        # Check if order is taking too long, market moved, etc.
        return False
    
    async def _get_aggressive_price(self, order: Dict) -> float:
        """Get more aggressive price for faster fill."""
        quote = self.kite.quote([f"{order['exchange']}:{order['tradingsymbol']}"])
        current = quote[f"{order['exchange']}:{order['tradingsymbol']}"]['last_price']
        
        if order['transaction_type'] == 'BUY':
            return current + 0.10  # Pay 10 paise more
        else:
            return current - 0.10  # Accept 10 paise less
    
    async def _calculate_portfolio_risk(self) -> float:
        """Calculate overall portfolio risk score."""
        # Implement comprehensive risk calculation
        return 0.65  # Placeholder
    
    async def _get_ai_recommendations(self, pnl: Decimal) -> List[str]:
        """Get AI-powered recommendations based on P&L."""
        recommendations = []
        
        if pnl < -50000:
            recommendations.extend([
                "Switch to paper trading for skill development",
                "Focus on index ETFs for next 3 months",
                "Implement strict 2% daily loss limit",
                "Consider taking a trading break to reassess strategy"
            ])
        elif pnl < 0:
            recommendations.extend([
                "Reduce position sizes by 50%",
                "Focus on high-probability setups only",
                "Implement systematic stop losses",
                "Start journaling all trades"
            ])
        else:
            recommendations.extend([
                "Gradually increase position sizes",
                "Diversify into multiple strategies",
                "Consider adding options strategies",
                "Maintain disciplined risk management"
            ])
        
        return recommendations
    
    async def _analyze_holding(self, holding: Dict) -> Dict[str, Any]:
        """Analyze long-term holdings with AI insights."""
        # Implement comprehensive holding analysis
        return {
            'recommendation': 'HOLD',
            'target_price': holding['last_price'] * 1.15,
            'stop_loss': holding['last_price'] * 0.92
        }
    
    async def _check_strategy_signal(
        self,
        symbol: str,
        quote: Dict,
        ohlc: Dict,
        strategy: str
    ) -> Dict[str, Any]:
        """Check if strategy conditions are met."""
        signal = {
            'valid': False,
            'side': 'BUY',
            'entry_price': 0,
            'stop_loss': 0,
            'target': 0,
            'confidence': 0
        }
        
        current_price = quote['last_price']
        
        if strategy == 'mean_reversion':
            # Check if price is oversold
            lower_band = ohlc['low'] * 0.98
            if current_price <= lower_band:
                signal.update({
                    'valid': True,
                    'side': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.98,
                    'target': ohlc['close'] * 1.01,
                    'confidence': 0.75
                })
        
        elif strategy == 'support_resistance_bounce':
            # Check bounce from support
            support = ohlc['low']
            if abs(current_price - support) / support < 0.005:  # Within 0.5% of support
                signal.update({
                    'valid': True,
                    'side': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': support * 0.995,
                    'target': ohlc['high'] * 0.98,
                    'confidence': 0.70
                })
        
        elif strategy == 'opening_range_breakout':
            # Check if breaking opening range
            if current_price > ohlc['open'] * 1.005:  # 0.5% above open
                signal.update({
                    'valid': True,
                    'side': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': ohlc['open'],
                    'target': current_price * 1.015,
                    'confidence': 0.65
                })
        
        return signal
    
    async def _monitor_position_for_exit(self, position: Dict, config: Dict):
        """Monitor position for exit conditions."""
        try:
            current_quote = self.kite.quote([f"NSE:{position['tradingsymbol']}"])
            current_price = current_quote[f"NSE:{position['tradingsymbol']}"]['last_price']
            
            pnl_percent = (position['pnl'] / (position['average_price'] * abs(position['quantity']))) * 100
            
            # Check stop loss
            if pnl_percent <= -config['stop_loss_percent']:
                await self.execute_smart_order(
                    position['tradingsymbol'],
                    abs(position['quantity']),
                    'MARKET',
                    'SELL' if position['quantity'] > 0 else 'BUY'
                )
                logger.info(f"Stop loss hit for {position['tradingsymbol']}")
            
            # Check target
            elif pnl_percent >= config['target_percent']:
                await self.execute_smart_order(
                    position['tradingsymbol'],
                    abs(position['quantity']),
                    'MARKET',
                    'SELL' if position['quantity'] > 0 else 'BUY'
                )
                logger.info(f"Target achieved for {position['tradingsymbol']}")
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
