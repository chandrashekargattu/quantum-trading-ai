"""Advanced backtesting engine with event-driven architecture."""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from concurrent.futures import ProcessPoolExecutor
import pickle
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.models.backtest import (
    Backtest, BacktestResult, BacktestTrade,
    OptimizationRun, OptimizationResult
)
from app.models.trading import OrderSide, OrderType
from app.models.portfolio import Portfolio
from app.models.stock import Stock, PriceHistory
from app.models.option import Option, OptionChain
from app.services.market_data import MarketDataService
from app.services.risk_management import RiskManagementService
from app.services.options_pricing import OptionsPricingService

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in backtesting."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK = "RISK"
    REBALANCE = "REBALANCE"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]


@dataclass
class MarketEvent(Event):
    """Market data update event."""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    
    def __post_init__(self):
        self.event_type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # -1 to 1
    strategy_id: str
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Order placement event."""
    symbol: str
    order_type: OrderType
    quantity: int
    side: OrderSide
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Order fill/execution event."""
    symbol: str
    quantity: int
    side: OrderSide
    fill_price: float
    commission: float
    slippage: float
    
    def __post_init__(self):
        self.event_type = EventType.FILL


class BacktestingEngine:
    """Advanced event-driven backtesting engine."""
    
    def __init__(self):
        self.events_queue = asyncio.Queue()
        self.market_data = {}
        self.portfolio = None
        self.strategies = {}
        self.execution_handler = ExecutionHandler()
        self.performance_tracker = PerformanceTracker()
        self.risk_manager = BacktestRiskManager()
        self.market_service = MarketDataService()
        self._running = False
        
    async def run_backtest(
        self,
        strategy: 'TradingStrategy',
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        db: AsyncSession,
        **kwargs
    ) -> BacktestResult:
        """Run a backtest for a given strategy."""
        logger.info(f"Starting backtest: {strategy.name} from {start_date} to {end_date}")
        
        # Initialize backtest
        backtest = Backtest(
            strategy_name=strategy.name,
            strategy_config=strategy.get_config(),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            status="RUNNING"
        )
        db.add(backtest)
        await db.commit()
        
        try:
            # Initialize components
            self.portfolio = BacktestPortfolio(initial_capital)
            self.strategies[strategy.name] = strategy
            
            # Load historical data
            await self._load_historical_data(symbols, start_date, end_date, db)
            
            # Run event loop
            await self._run_event_loop()
            
            # Calculate final metrics
            metrics = self.performance_tracker.calculate_metrics(
                self.portfolio.equity_curve,
                self.portfolio.trades
            )
            
            # Save results
            result = BacktestResult(
                backtest_id=backtest.id,
                final_value=self.portfolio.total_value,
                total_return=metrics['total_return'],
                total_return_percent=metrics['total_return_percent'],
                sharpe_ratio=metrics['sharpe_ratio'],
                sortino_ratio=metrics['sortino_ratio'],
                calmar_ratio=metrics['calmar_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_trades=len(self.portfolio.trades),
                equity_curve=self.portfolio.equity_curve,
                drawdown_curve=metrics['drawdown_curve'],
                metrics=metrics
            )
            
            db.add(result)
            
            # Save individual trades
            for trade in self.portfolio.trades:
                backtest_trade = BacktestTrade(
                    backtest_id=backtest.id,
                    symbol=trade['symbol'],
                    side=trade['side'],
                    quantity=trade['quantity'],
                    entry_price=trade['entry_price'],
                    exit_price=trade.get('exit_price'),
                    entry_time=trade['entry_time'],
                    exit_time=trade.get('exit_time'),
                    pnl=trade.get('pnl', 0),
                    pnl_percent=trade.get('pnl_percent', 0)
                )
                db.add(backtest_trade)
            
            backtest.status = "COMPLETED"
            await db.commit()
            
            logger.info(f"Backtest completed. Total return: {metrics['total_return_percent']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            backtest.status = "FAILED"
            backtest.error_message = str(e)
            await db.commit()
            raise
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        db: AsyncSession
    ):
        """Load historical market data."""
        for symbol in symbols:
            # Load price data
            price_result = await db.execute(
                select(PriceHistory).join(Stock).where(
                    and_(
                        Stock.symbol == symbol,
                        PriceHistory.date >= start_date,
                        PriceHistory.date <= end_date
                    )
                ).order_by(PriceHistory.date.asc())
            )
            price_history = price_result.scalars().all()
            
            # Convert to DataFrame for easier manipulation
            if price_history:
                df = pd.DataFrame([{
                    'date': p.date,
                    'open': p.open_price,
                    'high': p.high_price,
                    'low': p.low_price,
                    'close': p.close_price,
                    'volume': p.volume
                } for p in price_history])
                df.set_index('date', inplace=True)
                self.market_data[symbol] = df
            else:
                # Fetch from external source if not in database
                data = await self.market_service.fetch_historical_data(
                    symbol, start_date, end_date
                )
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    self.market_data[symbol] = df
    
    async def _run_event_loop(self):
        """Main event processing loop."""
        self._running = True
        
        # Create market events from historical data
        await self._generate_market_events()
        
        while self._running:
            try:
                # Get next event
                event = await asyncio.wait_for(
                    self.events_queue.get(),
                    timeout=1.0
                )
                
                # Process event based on type
                if event.event_type == EventType.MARKET:
                    await self._handle_market_event(event)
                elif event.event_type == EventType.SIGNAL:
                    await self._handle_signal_event(event)
                elif event.event_type == EventType.ORDER:
                    await self._handle_order_event(event)
                elif event.event_type == EventType.FILL:
                    await self._handle_fill_event(event)
                elif event.event_type == EventType.RISK:
                    await self._handle_risk_event(event)
                elif event.event_type == EventType.REBALANCE:
                    await self._handle_rebalance_event(event)
                    
            except asyncio.TimeoutError:
                # Check if we're done processing
                if self.events_queue.empty():
                    self._running = False
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _generate_market_events(self):
        """Generate market events from historical data."""
        # Get all unique timestamps
        all_timestamps = set()
        for df in self.market_data.values():
            all_timestamps.update(df.index)
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Generate events for each timestamp
        for timestamp in sorted_timestamps:
            for symbol, df in self.market_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    event = MarketEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        price=row['close'],
                        volume=row['volume'],
                        bid=row['close'] * 0.9999,  # Simulate bid
                        ask=row['close'] * 1.0001,  # Simulate ask
                        data={'ohlcv': row.to_dict()}
                    )
                    await self.events_queue.put(event)
    
    async def _handle_market_event(self, event: MarketEvent):
        """Handle market data update."""
        # Update portfolio with latest prices
        self.portfolio.update_market_price(event.symbol, event.price)
        
        # Update strategies with market data
        for strategy in self.strategies.values():
            signal = await strategy.calculate_signal(
                event.symbol,
                event.data['ohlcv'],
                self.portfolio
            )
            
            if signal:
                signal_event = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    signal_type=signal['type'],
                    strength=signal['strength'],
                    strategy_id=strategy.name,
                    data=signal
                )
                await self.events_queue.put(signal_event)
        
        # Check risk limits
        risk_check = self.risk_manager.check_portfolio_risk(self.portfolio)
        if risk_check['action_required']:
            risk_event = Event(
                event_type=EventType.RISK,
                timestamp=event.timestamp,
                data=risk_check
            )
            await self.events_queue.put(risk_event)
    
    async def _handle_signal_event(self, event: SignalEvent):
        """Handle trading signal."""
        # Apply portfolio construction rules
        position_size = self._calculate_position_size(
            event.symbol,
            event.strength,
            self.portfolio
        )
        
        if position_size > 0:
            # Generate order
            current_position = self.portfolio.get_position(event.symbol)
            
            if event.signal_type == "BUY" and current_position <= 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    side=OrderSide.BUY,
                    data={'strategy_id': event.strategy_id}
                )
                await self.events_queue.put(order)
                
            elif event.signal_type == "SELL" and current_position >= 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    side=OrderSide.SELL,
                    data={'strategy_id': event.strategy_id}
                )
                await self.events_queue.put(order)
    
    async def _handle_order_event(self, event: OrderEvent):
        """Handle order placement."""
        # Simulate order execution
        fill = self.execution_handler.execute_order(
            event,
            self.market_data[event.symbol],
            event.timestamp
        )
        
        if fill:
            await self.events_queue.put(fill)
    
    async def _handle_fill_event(self, event: FillEvent):
        """Handle order fill."""
        # Update portfolio
        self.portfolio.process_fill(event)
        
        # Track for performance
        self.performance_tracker.record_trade(event)
        
        # Log
        logger.debug(f"Fill: {event.side} {event.quantity} {event.symbol} @ {event.fill_price}")
    
    async def _handle_risk_event(self, event: Event):
        """Handle risk management event."""
        risk_data = event.data
        
        if risk_data['reduce_exposure']:
            # Generate orders to reduce risk
            for symbol, reduction in risk_data['reductions'].items():
                current_position = self.portfolio.get_position(symbol)
                if current_position > 0:
                    order = OrderEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        quantity=int(current_position * reduction),
                        side=OrderSide.SELL,
                        data={'reason': 'risk_reduction'}
                    )
                    await self.events_queue.put(order)
    
    async def _handle_rebalance_event(self, event: Event):
        """Handle portfolio rebalancing."""
        target_weights = event.data['target_weights']
        current_weights = self.portfolio.get_weights()
        
        # Calculate trades needed
        trades = self._calculate_rebalance_trades(
            current_weights,
            target_weights,
            self.portfolio.total_value
        )
        
        # Generate orders
        for symbol, quantity in trades.items():
            if quantity > 0:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL
                quantity = abs(quantity)
            
            if quantity > 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    side=side,
                    data={'reason': 'rebalance'}
                )
                await self.events_queue.put(order)
    
    def _calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        portfolio: 'BacktestPortfolio'
    ) -> int:
        """Calculate position size based on signal and risk."""
        # Kelly Criterion with safety factor
        kelly_fraction = abs(signal_strength) * 0.25  # 25% of Kelly
        
        # Apply portfolio constraints
        max_position_value = portfolio.total_value * 0.1  # Max 10% per position
        
        # Get current price
        current_price = portfolio.current_prices.get(symbol, 100)
        
        # Calculate shares
        position_value = portfolio.total_value * kelly_fraction
        position_value = min(position_value, max_position_value)
        
        shares = int(position_value / current_price)
        
        return shares
    
    def _calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float
    ) -> Dict[str, int]:
        """Calculate trades needed for rebalancing."""
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold
                value_diff = weight_diff * total_value
                current_price = self.portfolio.current_prices.get(symbol, 100)
                shares = int(value_diff / current_price)
                
                if shares != 0:
                    trades[symbol] = shares
        
        return trades
    
    async def run_walk_forward_optimization(
        self,
        strategy_class: type,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        param_grid: Dict[str, List[Any]],
        in_sample_ratio: float,
        db: AsyncSession
    ) -> OptimizationResult:
        """Run walk-forward optimization."""
        optimization = OptimizationRun(
            strategy_name=strategy_class.__name__,
            optimization_type="walk_forward",
            param_grid=param_grid,
            start_date=start_date,
            end_date=end_date,
            status="RUNNING"
        )
        db.add(optimization)
        await db.commit()
        
        try:
            # Calculate periods
            total_days = (end_date - start_date).days
            in_sample_days = int(total_days * in_sample_ratio)
            out_sample_days = total_days - in_sample_days
            num_periods = total_days // out_sample_days
            
            results = []
            
            for period in range(num_periods):
                # Define period dates
                period_start = start_date + timedelta(days=period * out_sample_days)
                is_end = period_start + timedelta(days=in_sample_days)
                oos_start = is_end
                oos_end = min(oos_start + timedelta(days=out_sample_days), end_date)
                
                # In-sample optimization
                best_params = await self._optimize_parameters(
                    strategy_class,
                    symbols,
                    period_start,
                    is_end,
                    initial_capital,
                    param_grid,
                    db
                )
                
                # Out-of-sample testing
                strategy = strategy_class(**best_params)
                oos_result = await self.run_backtest(
                    strategy,
                    symbols,
                    oos_start,
                    oos_end,
                    initial_capital,
                    db
                )
                
                results.append({
                    'period': period,
                    'params': best_params,
                    'in_sample_end': is_end,
                    'out_sample_return': oos_result.total_return_percent,
                    'out_sample_sharpe': oos_result.sharpe_ratio
                })
            
            # Aggregate results
            avg_return = np.mean([r['out_sample_return'] for r in results])
            avg_sharpe = np.mean([r['out_sample_sharpe'] for r in results])
            
            optimization_result = OptimizationResult(
                optimization_id=optimization.id,
                best_params=best_params,  # Last period's best
                in_sample_sharpe=avg_sharpe,
                out_sample_sharpe=avg_sharpe,
                in_sample_return=avg_return,
                out_sample_return=avg_return,
                robustness_score=self._calculate_robustness(results),
                parameter_stability=self._calculate_parameter_stability(results)
            )
            
            optimization.status = "COMPLETED"
            optimization.best_params = best_params
            db.add(optimization_result)
            await db.commit()
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {e}")
            optimization.status = "FAILED"
            optimization.error_message = str(e)
            await db.commit()
            raise
    
    async def _optimize_parameters(
        self,
        strategy_class: type,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        param_grid: Dict[str, List[Any]],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        best_sharpe = -np.inf
        best_params = {}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Test each combination
        for params in param_combinations:
            strategy = strategy_class(**params)
            
            try:
                result = await self.run_backtest(
                    strategy,
                    symbols,
                    start_date,
                    end_date,
                    initial_capital,
                    db
                )
                
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Failed to test params {params}: {e}")
        
        return best_params
    
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_robustness(self, results: List[Dict]) -> float:
        """Calculate strategy robustness score."""
        returns = [r['out_sample_return'] for r in results]
        
        # Consistency of returns
        consistency = 1 - (np.std(returns) / (np.mean(returns) + 1e-10))
        
        # Percentage of profitable periods
        profitable = sum(1 for r in returns if r > 0) / len(returns)
        
        # Combined score
        robustness = (consistency + profitable) / 2
        
        return max(0, min(1, robustness))
    
    def _calculate_parameter_stability(
        self,
        results: List[Dict]
    ) -> Dict[str, float]:
        """Calculate parameter stability across periods."""
        param_values = defaultdict(list)
        
        for result in results:
            for param, value in result['params'].items():
                if isinstance(value, (int, float)):
                    param_values[param].append(value)
        
        stability = {}
        for param, values in param_values.items():
            if len(values) > 1:
                # Coefficient of variation
                cv = np.std(values) / (np.mean(values) + 1e-10)
                stability[param] = 1 - min(cv, 1)
        
        return stability
    
    async def run_monte_carlo_permutation(
        self,
        strategy: 'TradingStrategy',
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        num_simulations: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Run Monte Carlo permutation test for statistical significance."""
        # Run base backtest
        base_result = await self.run_backtest(
            strategy, symbols, start_date, end_date, initial_capital, db
        )
        base_return = base_result.total_return_percent
        
        # Run permutations
        permutation_returns = []
        
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for i in range(num_simulations):
                # Shuffle returns while preserving autocorrelation structure
                shuffled_data = self._block_bootstrap(self.market_data)
                
                future = executor.submit(
                    self._run_permutation_backtest,
                    strategy,
                    shuffled_data,
                    initial_capital
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    permutation_returns.append(result)
                except Exception as e:
                    logger.error(f"Permutation failed: {e}")
        
        # Calculate p-value
        p_value = sum(1 for r in permutation_returns if r >= base_return) / len(permutation_returns)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(permutation_returns, 2.5)
        ci_upper = np.percentile(permutation_returns, 97.5)
        
        return {
            'base_return': base_return,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05,
            'permutation_mean': np.mean(permutation_returns),
            'permutation_std': np.std(permutation_returns)
        }
    
    def _block_bootstrap(
        self,
        data: Dict[str, pd.DataFrame],
        block_size: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """Perform block bootstrap to preserve time series properties."""
        bootstrapped = {}
        
        for symbol, df in data.items():
            n_samples = len(df)
            n_blocks = n_samples // block_size
            
            # Generate random block indices
            block_indices = np.random.randint(0, n_blocks, n_blocks)
            
            # Reconstruct series from blocks
            new_data = []
            for idx in block_indices:
                start = idx * block_size
                end = min(start + block_size, n_samples)
                new_data.append(df.iloc[start:end])
            
            bootstrapped[symbol] = pd.concat(new_data).reset_index(drop=True)
        
        return bootstrapped
    
    def _run_permutation_backtest(
        self,
        strategy: 'TradingStrategy',
        data: Dict[str, pd.DataFrame],
        initial_capital: float
    ) -> float:
        """Run a single permutation backtest."""
        # Simplified backtest for permutation testing
        portfolio_value = initial_capital
        
        for timestamp in sorted(set().union(*[df.index for df in data.values()])):
            for symbol, df in data.items():
                if timestamp in df.index:
                    # Get signal
                    signal = strategy.calculate_signal_simple(
                        symbol,
                        df.loc[:timestamp]
                    )
                    
                    # Simple position sizing
                    if signal > 0:
                        portfolio_value *= 1 + (signal * 0.01)  # 1% per signal unit
        
        return ((portfolio_value - initial_capital) / initial_capital) * 100


class BacktestPortfolio:
    """Portfolio tracker for backtesting."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = defaultdict(int)
        self.current_prices = {}
        self.equity_curve = []
        self.trades = []
        self.open_trades = {}
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            self.positions[symbol] * self.current_prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + positions_value
    
    def update_market_price(self, symbol: str, price: float):
        """Update current market price."""
        self.current_prices[symbol] = price
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'value': self.total_value
        })
    
    def process_fill(self, fill: FillEvent):
        """Process order fill."""
        if fill.side == OrderSide.BUY:
            self.positions[fill.symbol] += fill.quantity
            self.cash -= (fill.fill_price * fill.quantity + fill.commission)
            
            # Track trade entry
            self.open_trades[fill.symbol] = {
                'entry_price': fill.fill_price,
                'entry_time': fill.timestamp,
                'quantity': fill.quantity,
                'side': 'LONG'
            }
        else:
            self.positions[fill.symbol] -= fill.quantity
            self.cash += (fill.fill_price * fill.quantity - fill.commission)
            
            # Track trade exit
            if fill.symbol in self.open_trades:
                entry = self.open_trades.pop(fill.symbol)
                pnl = (fill.fill_price - entry['entry_price']) * entry['quantity']
                pnl_percent = (pnl / (entry['entry_price'] * entry['quantity'])) * 100
                
                self.trades.append({
                    'symbol': fill.symbol,
                    'side': entry['side'],
                    'quantity': entry['quantity'],
                    'entry_price': entry['entry_price'],
                    'exit_price': fill.fill_price,
                    'entry_time': entry['entry_time'],
                    'exit_time': fill.timestamp,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent
                })
    
    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        total = self.total_value
        weights = {}
        
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                value = quantity * self.current_prices.get(symbol, 0)
                weights[symbol] = value / total
        
        return weights


class ExecutionHandler:
    """Handles order execution simulation."""
    
    def __init__(self):
        self.slippage_model = SlippageModel()
        self.commission_model = CommissionModel()
    
    def execute_order(
        self,
        order: OrderEvent,
        market_data: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[FillEvent]:
        """Simulate order execution."""
        if timestamp not in market_data.index:
            return None
        
        bar = market_data.loc[timestamp]
        
        # Calculate fill price with slippage
        if order.order_type == OrderType.MARKET:
            base_price = bar['close']
            slippage = self.slippage_model.calculate_slippage(
                order.side,
                order.quantity,
                bar['volume'],
                bar['close']
            )
            fill_price = base_price + slippage
        else:
            # Limit order logic
            if order.side == OrderSide.BUY and order.limit_price >= bar['low']:
                fill_price = min(order.limit_price, bar['high'])
            elif order.side == OrderSide.SELL and order.limit_price <= bar['high']:
                fill_price = max(order.limit_price, bar['low'])
            else:
                return None  # Order not filled
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(
            order.quantity,
            fill_price
        )
        
        return FillEvent(
            timestamp=timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            side=order.side,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage if order.order_type == OrderType.MARKET else 0,
            data=order.data
        )


class SlippageModel:
    """Market impact and slippage modeling."""
    
    def __init__(self, impact_coefficient: float = 0.1):
        self.impact_coefficient = impact_coefficient
    
    def calculate_slippage(
        self,
        side: OrderSide,
        quantity: int,
        volume: int,
        price: float
    ) -> float:
        """Calculate price slippage based on order size and market volume."""
        # Participation rate
        participation = quantity / max(volume, 1)
        
        # Square-root market impact model
        impact = self.impact_coefficient * np.sqrt(participation) * price
        
        # Apply based on side
        if side == OrderSide.BUY:
            return impact  # Pay more
        else:
            return -impact  # Receive less


class CommissionModel:
    """Trading commission calculation."""
    
    def __init__(self, rate: float = 0.001):
        self.rate = rate  # 0.1% default
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate trading commission."""
        return quantity * price * self.rate


class PerformanceTracker:
    """Track and calculate performance metrics."""
    
    def __init__(self):
        self.trades = []
        self.daily_returns = []
    
    def record_trade(self, fill: FillEvent):
        """Record executed trade."""
        self.trades.append(fill)
    
    def calculate_metrics(
        self,
        equity_curve: List[Dict],
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not equity_curve:
            return {}
        
        # Convert to series
        values = pd.Series(
            [e['value'] for e in equity_curve],
            index=[e['timestamp'] for e in equity_curve]
        )
        
        # Calculate returns
        returns = values.pct_change().dropna()
        
        # Basic metrics
        total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(values, total_return)
        
        # Drawdown analysis
        drawdown_series = self._calculate_drawdown_series(values)
        max_drawdown = drawdown_series.min()
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        return {
            'total_return': values.iloc[-1] - values.iloc[0],
            'total_return_percent': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': abs(max_drawdown),
            'drawdown_curve': drawdown_series.to_list(),
            **trade_stats
        }
    
    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def _calculate_calmar_ratio(
        self,
        values: pd.Series,
        total_return: float
    ) -> float:
        """Calculate Calmar ratio."""
        drawdown = self._calculate_drawdown_series(values)
        max_dd = abs(drawdown.min())
        
        if max_dd == 0:
            return 0.0
        
        # Annualized return / Max Drawdown
        days = (values.index[-1] - values.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        return annual_return / max_dd
    
    def _calculate_drawdown_series(self, values: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        return drawdown
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade-based statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
        
        # Separate wins and losses
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        # Calculate metrics
        win_rate = len(wins) / len(trades) if trades else 0
        
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': total_losses / len(losses) if losses else 0,
            'largest_win': max(t['pnl'] for t in wins) if wins else 0,
            'largest_loss': min(t['pnl'] for t in losses) if losses else 0
        }


class BacktestRiskManager:
    """Risk management for backtesting."""
    
    def __init__(self):
        self.max_drawdown_limit = 0.20  # 20%
        self.position_limit = 0.10  # 10% per position
        self.sector_limit = 0.30  # 30% per sector
    
    def check_portfolio_risk(
        self,
        portfolio: BacktestPortfolio
    ) -> Dict[str, Any]:
        """Check portfolio risk limits."""
        action_required = False
        reductions = {}
        
        # Check drawdown
        if portfolio.equity_curve:
            values = [e['value'] for e in portfolio.equity_curve]
            if len(values) > 1:
                peak = max(values)
                current = values[-1]
                drawdown = (peak - current) / peak
                
                if drawdown > self.max_drawdown_limit:
                    action_required = True
                    # Reduce all positions by 50%
                    for symbol in portfolio.positions:
                        reductions[symbol] = 0.5
        
        # Check position concentration
        total_value = portfolio.total_value
        for symbol, quantity in portfolio.positions.items():
            if quantity > 0:
                position_value = quantity * portfolio.current_prices.get(symbol, 0)
                weight = position_value / total_value
                
                if weight > self.position_limit:
                    action_required = True
                    excess = weight - self.position_limit
                    reductions[symbol] = excess / weight
        
        return {
            'action_required': action_required,
            'reduce_exposure': action_required,
            'reductions': reductions
        }


# Example Trading Strategy Base Class
class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
    
    async def calculate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio: BacktestPortfolio
    ) -> Optional[Dict[str, Any]]:
        """Calculate trading signal. Override in subclasses."""
        raise NotImplementedError
    
    def calculate_signal_simple(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> float:
        """Simple signal calculation for permutation testing."""
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return {
            'name': self.name,
            'params': self.params
        }
