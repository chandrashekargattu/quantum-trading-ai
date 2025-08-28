"""
Advanced Market Making Algorithms

Implements sophisticated market making strategies similar to those used by
Jane Street, Citadel Securities, and other top market makers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Deque
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
import asyncio
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import torch
import torch.nn as nn
from numba import jit, prange

logger = logging.getLogger(__name__)


@dataclass
class MarketMakingParameters:
    """Parameters for market making strategies."""
    
    # Inventory management
    max_inventory: Decimal
    target_inventory: Decimal = Decimal('0')
    inventory_skew_factor: Decimal = Decimal('0.5')
    
    # Risk parameters
    max_spread: Decimal = Decimal('0.01')
    min_spread: Decimal = Decimal('0.0001')
    risk_aversion: Decimal = Decimal('0.1')
    
    # Order parameters
    base_order_size: Decimal = Decimal('100')
    order_size_multiplier: Decimal = Decimal('1.0')
    max_order_size: Decimal = Decimal('10000')
    
    # Pricing parameters
    tick_size: Decimal = Decimal('0.01')
    min_edge: Decimal = Decimal('0.0001')
    
    # Time parameters
    quote_update_frequency: float = 0.001  # 1ms
    position_close_time: int = 3600  # 1 hour in seconds


@dataclass
class MarketRegime:
    """Market regime classification."""
    
    regime_type: str  # 'normal', 'volatile', 'trending', 'mean_reverting'
    confidence: float
    volatility_regime: str  # 'low', 'medium', 'high', 'extreme'
    liquidity_regime: str  # 'abundant', 'normal', 'scarce'
    trend_strength: float  # -1 to 1 (negative for downtrend)
    mean_reversion_speed: float


class AdaptiveSpreadModel:
    """
    Adaptive spread model that adjusts to market conditions.
    Uses reinforcement learning and statistical models.
    """
    
    def __init__(self, symbol: str, lookback_period: int = 1000):
        self.symbol = symbol
        self.lookback_period = lookback_period
        
        # Historical data storage
        self.spread_history: Deque[Decimal] = deque(maxlen=lookback_period)
        self.fill_rate_history: Deque[float] = deque(maxlen=lookback_period)
        self.adverse_selection_history: Deque[float] = deque(maxlen=lookback_period)
        
        # Gaussian Process for spread optimization
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        
        # Neural network for complex pattern recognition
        self.nn_model = self._build_neural_network()
        
        # Current optimal spread
        self.optimal_spread = Decimal('0.001')
        
    def _build_neural_network(self) -> nn.Module:
        """Build neural network for spread prediction."""
        class SpreadNet(nn.Module):
            def __init__(self, input_dim: int = 20):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return torch.sigmoid(x) * 0.01  # Max spread of 1%
        
        return SpreadNet()
    
    def update(
        self,
        market_data: Dict[str, Any],
        execution_data: Dict[str, Any]
    ) -> Decimal:
        """Update spread model with new data and return optimal spread."""
        # Extract features
        features = self._extract_features(market_data, execution_data)
        
        # Update histories
        current_spread = market_data.get('spread', Decimal('0.001'))
        fill_rate = execution_data.get('fill_rate', 0.5)
        adverse_selection = execution_data.get('adverse_selection', 0.0)
        
        self.spread_history.append(current_spread)
        self.fill_rate_history.append(fill_rate)
        self.adverse_selection_history.append(adverse_selection)
        
        # Update models if enough data
        if len(self.spread_history) >= 100:
            # Gaussian Process update
            X = np.array(features).reshape(-1, len(features))
            y = np.array([fill_rate - adverse_selection])  # Optimize for profit
            
            if len(self.spread_history) >= 200:
                # Retrain periodically
                historical_features = self._get_historical_features()
                historical_targets = self._calculate_historical_targets()
                self.gp_model.fit(historical_features, historical_targets)
            
            # Predict optimal spread
            gp_spread, gp_std = self.gp_model.predict(X, return_std=True)
            
            # Neural network prediction
            nn_input = torch.tensor(features, dtype=torch.float32)
            nn_spread = self.nn_model(nn_input).item()
            
            # Combine predictions
            self.optimal_spread = Decimal(str(
                0.7 * float(gp_spread[0]) + 0.3 * nn_spread
            ))
        
        # Apply constraints
        self.optimal_spread = max(
            Decimal('0.0001'),  # Min spread
            min(self.optimal_spread, Decimal('0.01'))  # Max spread
        )
        
        return self.optimal_spread
    
    def _extract_features(
        self,
        market_data: Dict[str, Any],
        execution_data: Dict[str, Any]
    ) -> List[float]:
        """Extract features for spread optimization."""
        features = []
        
        # Market features
        features.append(float(market_data.get('volatility', 0.02)))
        features.append(float(market_data.get('volume', 1000000)))
        features.append(float(market_data.get('order_imbalance', 0.0)))
        features.append(float(market_data.get('trade_intensity', 100)))
        features.append(float(market_data.get('quote_intensity', 1000)))
        
        # Execution features
        features.append(execution_data.get('fill_rate', 0.5))
        features.append(execution_data.get('adverse_selection', 0.0))
        features.append(execution_data.get('avg_queue_position', 0.5))
        
        # Historical features
        if len(self.spread_history) > 0:
            features.append(float(np.mean(list(self.spread_history))))
            features.append(float(np.std(list(self.spread_history))))
        else:
            features.extend([0.001, 0.0001])
        
        if len(self.fill_rate_history) > 0:
            features.append(np.mean(self.fill_rate_history))
            features.append(np.std(self.fill_rate_history))
        else:
            features.extend([0.5, 0.1])
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _get_historical_features(self) -> np.ndarray:
        """Get historical features for model training."""
        # Simplified - would be more complex in production
        n_samples = min(len(self.spread_history), 100)
        features = []
        
        for i in range(n_samples):
            feat = [
                float(self.spread_history[i]),
                self.fill_rate_history[i],
                self.adverse_selection_history[i]
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _calculate_historical_targets(self) -> np.ndarray:
        """Calculate historical optimization targets."""
        # Target is profitability metric
        targets = []
        
        for i in range(min(len(self.spread_history), 100)):
            profit = self.fill_rate_history[i] - self.adverse_selection_history[i]
            targets.append(profit)
        
        return np.array(targets)


class InventoryOptimizer:
    """
    Advanced inventory optimization using stochastic control theory.
    Implements Almgren-Chriss model and extensions.
    """
    
    def __init__(
        self,
        symbol: str,
        risk_aversion: float = 0.1,
        time_horizon: float = 3600  # 1 hour
    ):
        self.symbol = symbol
        self.risk_aversion = risk_aversion
        self.time_horizon = time_horizon
        
        # Model parameters
        self.permanent_impact = 0.1  # Kyle's lambda
        self.temporary_impact = 0.01
        self.volatility = 0.02
        
        # Optimization state
        self.current_inventory = 0.0
        self.target_inventory = 0.0
        self.optimal_trajectory = None
        
    def optimize_inventory_trajectory(
        self,
        current_inventory: float,
        target_inventory: float,
        market_conditions: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize inventory trajectory using Almgren-Chriss framework.
        Returns optimal trading trajectory.
        """
        self.current_inventory = current_inventory
        self.target_inventory = target_inventory
        
        # Update model parameters based on market conditions
        self._update_model_parameters(market_conditions)
        
        # Number of time steps
        n_steps = max(10, int(self.time_horizon / 60))  # At least 10 steps
        dt = self.time_horizon / n_steps
        
        # Initial guess: linear trajectory
        initial_trajectory = np.linspace(
            current_inventory,
            target_inventory,
            n_steps + 1
        )
        
        # Optimization constraints
        if constraints:
            max_trade_size = constraints.get('max_trade_size', float('inf'))
            max_participation = constraints.get('max_participation', 0.1)
        else:
            max_trade_size = float('inf')
            max_participation = 0.1
        
        # Optimize using scipy
        result = minimize(
            self._objective_function,
            initial_trajectory[1:-1],  # Optimize intermediate points
            method='SLSQP',
            bounds=[(target_inventory, current_inventory)] * (n_steps - 1),
            constraints=[
                # Trading rate constraints
                {
                    'type': 'ineq',
                    'fun': lambda x: max_trade_size - np.abs(np.diff(
                        np.concatenate([[current_inventory], x, [target_inventory]])
                    ))
                }
            ]
        )
        
        # Construct full trajectory
        self.optimal_trajectory = np.concatenate([
            [current_inventory],
            result.x,
            [target_inventory]
        ])
        
        return self.optimal_trajectory
    
    def _objective_function(self, trajectory: np.ndarray) -> float:
        """
        Objective function for trajectory optimization.
        Minimizes expected cost + risk penalty.
        """
        # Full trajectory including endpoints
        full_trajectory = np.concatenate([
            [self.current_inventory],
            trajectory,
            [self.target_inventory]
        ])
        
        # Trading rates
        trades = np.diff(full_trajectory)
        n_trades = len(trades)
        
        # Expected cost (temporary + permanent impact)
        temp_cost = self.temporary_impact * np.sum(trades ** 2)
        perm_cost = self.permanent_impact * np.sum(np.abs(trades))
        
        # Risk penalty (variance of implementation shortfall)
        positions = full_trajectory[:-1]
        risk = self.volatility ** 2 * np.sum(positions ** 2) / n_trades
        
        # Total objective
        return temp_cost + perm_cost + self.risk_aversion * risk
    
    def _update_model_parameters(self, market_conditions: Dict[str, float]):
        """Update model parameters based on current market conditions."""
        # Adjust impacts based on liquidity
        liquidity_factor = market_conditions.get('liquidity_factor', 1.0)
        self.temporary_impact = 0.01 / liquidity_factor
        self.permanent_impact = 0.1 / liquidity_factor
        
        # Adjust volatility
        self.volatility = market_conditions.get('volatility', 0.02)
    
    def get_next_trade(self, current_time: float) -> float:
        """Get next trade size based on optimal trajectory."""
        if self.optimal_trajectory is None:
            return 0.0
        
        # Find current position in trajectory
        time_fraction = current_time / self.time_horizon
        trajectory_index = int(time_fraction * (len(self.optimal_trajectory) - 1))
        
        if trajectory_index >= len(self.optimal_trajectory) - 1:
            return 0.0
        
        # Return trade to move to next trajectory point
        current_target = self.optimal_trajectory[trajectory_index + 1]
        trade_size = current_target - self.current_inventory
        
        return trade_size


class StatisticalArbitrageEngine:
    """
    Statistical arbitrage engine for market making.
    Identifies and exploits short-term mispricings.
    """
    
    def __init__(self, symbols: List[str], lookback: int = 1000):
        self.symbols = symbols
        self.lookback = lookback
        
        # Price histories
        self.price_histories = {
            symbol: deque(maxlen=lookback) for symbol in symbols
        }
        
        # Cointegration relationships
        self.cointegration_pairs = []
        self.pair_parameters = {}
        
        # Signal generation
        self.signals = defaultdict(float)
        self.signal_confidence = defaultdict(float)
        
    def update_prices(self, prices: Dict[str, float]):
        """Update price histories."""
        for symbol, price in prices.items():
            if symbol in self.price_histories:
                self.price_histories[symbol].append(price)
        
        # Update signals if enough data
        if all(len(hist) >= 100 for hist in self.price_histories.values()):
            self._update_signals()
    
    def _update_signals(self):
        """Update statistical arbitrage signals."""
        # Find cointegrated pairs
        self._find_cointegration_pairs()
        
        # Generate signals for each pair
        for pair in self.cointegration_pairs:
            signal, confidence = self._calculate_pair_signal(pair)
            pair_key = f"{pair[0]}_{pair[1]}"
            self.signals[pair_key] = signal
            self.signal_confidence[pair_key] = confidence
        
        # Mean reversion signals for individual assets
        for symbol in self.symbols:
            signal, confidence = self._calculate_mean_reversion_signal(symbol)
            self.signals[symbol] = signal
            self.signal_confidence[symbol] = confidence
    
    def _find_cointegration_pairs(self):
        """Find cointegrated pairs using Johansen test."""
        # Simplified version - would use proper cointegration tests
        from itertools import combinations
        
        self.cointegration_pairs = []
        
        for symbol1, symbol2 in combinations(self.symbols, 2):
            prices1 = np.array(list(self.price_histories[symbol1]))
            prices2 = np.array(list(self.price_histories[symbol2]))
            
            # Calculate correlation
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            
            # Simple cointegration check
            if abs(correlation) > 0.8:
                # Estimate parameters
                spread = prices1 - prices2
                mean_spread = np.mean(spread)
                std_spread = np.std(spread)
                
                self.cointegration_pairs.append((symbol1, symbol2))
                self.pair_parameters[f"{symbol1}_{symbol2}"] = {
                    'hedge_ratio': 1.0,  # Simplified
                    'mean_spread': mean_spread,
                    'std_spread': std_spread
                }
    
    def _calculate_pair_signal(
        self,
        pair: Tuple[str, str]
    ) -> Tuple[float, float]:
        """Calculate trading signal for a pair."""
        symbol1, symbol2 = pair
        prices1 = np.array(list(self.price_histories[symbol1]))
        prices2 = np.array(list(self.price_histories[symbol2]))
        
        params = self.pair_parameters[f"{symbol1}_{symbol2}"]
        
        # Current spread
        current_spread = prices1[-1] - prices2[-1]
        normalized_spread = (
            current_spread - params['mean_spread']
        ) / params['std_spread']
        
        # Signal strength (mean reversion)
        signal = -normalized_spread  # Negative for mean reversion
        
        # Confidence based on spread extremeness
        confidence = 1.0 - np.exp(-abs(normalized_spread))
        
        return signal, confidence
    
    def _calculate_mean_reversion_signal(
        self,
        symbol: str
    ) -> Tuple[float, float]:
        """Calculate mean reversion signal for individual asset."""
        prices = np.array(list(self.price_histories[symbol]))
        
        # Calculate various moving averages
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:])
        
        # Current price relative to MAs
        current_price = prices[-1]
        deviation_20 = (current_price - ma_20) / ma_20
        deviation_50 = (current_price - ma_50) / ma_50
        
        # Combined signal
        signal = -(deviation_20 + deviation_50) / 2
        
        # Confidence based on consistency
        confidence = min(1.0, abs(signal) * 10)
        
        return signal, confidence
    
    def get_trading_signals(self) -> Dict[str, Dict[str, float]]:
        """Get current trading signals."""
        result = {}
        
        for key, signal in self.signals.items():
            result[key] = {
                'signal': signal,
                'confidence': self.signal_confidence[key],
                'direction': 'buy' if signal > 0 else 'sell',
                'strength': abs(signal)
            }
        
        return result


@jit(nopython=True, parallel=True)
def calculate_optimal_quotes_numba(
    fair_value: float,
    inventory: float,
    max_inventory: float,
    risk_aversion: float,
    volatility: float,
    order_arrival_rate: float,
    fill_probability: float
) -> Tuple[float, float]:
    """
    Fast calculation of optimal quotes using Avellaneda-Stoikov model.
    JIT-compiled for performance.
    """
    # Reservation price
    inventory_risk = risk_aversion * volatility ** 2 * inventory
    reservation_price = fair_value - inventory_risk
    
    # Optimal spread
    gamma = -np.log(fill_probability) / order_arrival_rate
    half_spread = gamma + (1 / risk_aversion) * np.log(1 + risk_aversion / order_arrival_rate)
    
    # Adjust for inventory
    inventory_ratio = inventory / max_inventory
    skew = 0.5 * inventory_ratio * half_spread
    
    # Calculate quotes
    bid_price = reservation_price - half_spread + skew
    ask_price = reservation_price + half_spread + skew
    
    return bid_price, ask_price


class AdvancedMarketMaker:
    """
    Advanced market maker combining multiple strategies.
    """
    
    def __init__(
        self,
        symbol: str,
        parameters: MarketMakingParameters
    ):
        self.symbol = symbol
        self.parameters = parameters
        
        # Strategy components
        self.spread_model = AdaptiveSpreadModel(symbol)
        self.inventory_optimizer = InventoryOptimizer(
            symbol,
            float(parameters.risk_aversion)
        )
        self.stat_arb_engine = StatisticalArbitrageEngine([symbol])
        
        # State tracking
        self.inventory = Decimal('0')
        self.position_pnl = Decimal('0')
        self.total_volume = Decimal('0')
        
        # Performance metrics
        self.metrics = {
            'trades': 0,
            'winning_trades': 0,
            'total_pnl': Decimal('0'),
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
        # Quote generation
        self.last_bid = None
        self.last_ask = None
        
    async def generate_quotes(
        self,
        market_data: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Generate bid and ask quotes using all strategies."""
        # Update models
        execution_data = self._get_execution_metrics()
        optimal_spread = self.spread_model.update(market_data, execution_data)
        
        # Get fair value estimate
        fair_value = self._estimate_fair_value(market_data, predictions)
        
        # Calculate optimal quotes using Numba-accelerated function
        bid_price, ask_price = calculate_optimal_quotes_numba(
            float(fair_value),
            float(self.inventory),
            float(self.parameters.max_inventory),
            float(self.parameters.risk_aversion),
            market_data.get('volatility', 0.02),
            market_data.get('order_arrival_rate', 100),
            market_data.get('fill_probability', 0.5)
        )
        
        # Adjust for statistical arbitrage signals
        stat_arb_signals = self.stat_arb_engine.get_trading_signals()
        if self.symbol in stat_arb_signals:
            signal = stat_arb_signals[self.symbol]
            adjustment = Decimal(str(signal['signal'] * 0.0001))
            bid_price = float(Decimal(str(bid_price)) + adjustment)
            ask_price = float(Decimal(str(ask_price)) + adjustment)
        
        # Apply tick size rounding
        bid_price = self._round_to_tick(Decimal(str(bid_price)), ROUND_DOWN)
        ask_price = self._round_to_tick(Decimal(str(ask_price)), ROUND_UP)
        
        # Generate order objects
        bid_order = None
        ask_order = None
        
        if self.inventory < self.parameters.max_inventory:
            bid_size = self._calculate_order_size('buy', market_data)
            bid_order = {
                'symbol': self.symbol,
                'side': 'buy',
                'price': bid_price,
                'quantity': bid_size,
                'order_type': 'limit',
                'time_in_force': 'IOC'
            }
        
        if self.inventory > -self.parameters.max_inventory:
            ask_size = self._calculate_order_size('sell', market_data)
            ask_order = {
                'symbol': self.symbol,
                'side': 'sell',
                'price': ask_price,
                'quantity': ask_size,
                'order_type': 'limit',
                'time_in_force': 'IOC'
            }
        
        # Update last quotes
        self.last_bid = bid_price
        self.last_ask = ask_price
        
        return bid_order, ask_order
    
    def _estimate_fair_value(
        self,
        market_data: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Decimal:
        """Estimate fair value using multiple sources."""
        weights = {
            'mid_price': 0.3,
            'vwap': 0.2,
            'ml_prediction': 0.3,
            'microprice': 0.2
        }
        
        # Components
        mid_price = (
            market_data.get('best_bid', 100) +
            market_data.get('best_ask', 100)
        ) / 2
        
        vwap = market_data.get('vwap', mid_price)
        ml_pred = predictions.get('price_prediction', mid_price)
        
        # Microprice (order book imbalance weighted)
        bid_size = market_data.get('bid_size', 1000)
        ask_size = market_data.get('ask_size', 1000)
        total_size = bid_size + ask_size
        
        if total_size > 0:
            microprice = (
                market_data.get('best_bid', 100) * ask_size +
                market_data.get('best_ask', 100) * bid_size
            ) / total_size
        else:
            microprice = mid_price
        
        # Weighted average
        fair_value = (
            weights['mid_price'] * mid_price +
            weights['vwap'] * vwap +
            weights['ml_prediction'] * ml_pred +
            weights['microprice'] * microprice
        )
        
        return Decimal(str(fair_value))
    
    def _calculate_order_size(
        self,
        side: str,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate optimal order size."""
        base_size = self.parameters.base_order_size
        
        # Adjust for inventory
        inventory_ratio = abs(self.inventory) / self.parameters.max_inventory
        inventory_adjustment = 1 - inventory_ratio * 0.5
        
        # Adjust for market conditions
        volatility = market_data.get('volatility', 0.02)
        volatility_adjustment = 1 / (1 + volatility * 10)
        
        # Adjust for queue position
        queue_position = market_data.get('queue_position', 0.5)
        queue_adjustment = 1 + (1 - queue_position) * 0.5
        
        # Final size
        size = base_size * inventory_adjustment * volatility_adjustment * queue_adjustment
        size = size * self.parameters.order_size_multiplier
        
        # Apply limits
        size = max(
            self.parameters.base_order_size / 10,  # Min size
            min(size, self.parameters.max_order_size)  # Max size
        )
        
        return size
    
    def _round_to_tick(self, price: Decimal, rounding: str) -> Decimal:
        """Round price to tick size."""
        tick = self.parameters.tick_size
        return (price / tick).quantize(Decimal('1'), rounding) * tick
    
    def _get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for model updates."""
        return {
            'fill_rate': 0.6,  # Placeholder
            'adverse_selection': 0.001,  # Placeholder
            'avg_queue_position': 0.5,  # Placeholder
            'realized_spread': 0.0005  # Placeholder
        }
    
    def update_metrics(self, execution: Dict[str, Any]):
        """Update performance metrics."""
        # Update inventory
        if execution['side'] == 'buy':
            self.inventory += execution['quantity']
        else:
            self.inventory -= execution['quantity']
        
        # Update PnL
        if execution['side'] == 'buy':
            self.position_pnl -= execution['price'] * execution['quantity']
        else:
            self.position_pnl += execution['price'] * execution['quantity']
        
        # Update metrics
        self.metrics['trades'] += 1
        self.total_volume += execution['quantity']
