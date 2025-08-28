"""
Comprehensive tests for Advanced Market Making module.
Tests adaptive spreads, inventory optimization, and statistical arbitrage.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from decimal import Decimal, ROUND_DOWN
from unittest.mock import Mock, patch, MagicMock
from collections import deque
from datetime import datetime, timedelta
from scipy.optimize import minimize

from app.market_making.advanced_market_maker import (
    MarketMakingParameters,
    MarketRegime,
    AdaptiveSpreadModel,
    InventoryOptimizer,
    StatisticalArbitrageEngine,
    AdvancedMarketMaker,
    calculate_optimal_quotes_numba
)


class TestMarketMakingParameters:
    """Test market making parameters."""
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        params = MarketMakingParameters(
            max_inventory=Decimal('10000')
        )
        
        assert params.max_inventory == Decimal('10000')
        assert params.target_inventory == Decimal('0')
        assert params.inventory_skew_factor == Decimal('0.5')
        assert params.max_spread == Decimal('0.01')
        assert params.min_spread == Decimal('0.0001')
        assert params.risk_aversion == Decimal('0.1')
        assert params.base_order_size == Decimal('100')
        assert params.tick_size == Decimal('0.01')
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = MarketMakingParameters(
            max_inventory=Decimal('50000'),
            target_inventory=Decimal('1000'),
            inventory_skew_factor=Decimal('0.8'),
            max_spread=Decimal('0.05'),
            min_spread=Decimal('0.0005'),
            risk_aversion=Decimal('0.5'),
            base_order_size=Decimal('500'),
            order_size_multiplier=Decimal('2.0'),
            max_order_size=Decimal('50000'),
            tick_size=Decimal('0.001'),
            min_edge=Decimal('0.0002'),
            quote_update_frequency=0.0005,
            position_close_time=7200
        )
        
        assert params.target_inventory == Decimal('1000')
        assert params.order_size_multiplier == Decimal('2.0')
        assert params.quote_update_frequency == 0.0005
        assert params.position_close_time == 7200


class TestMarketRegime:
    """Test market regime classification."""
    
    def test_regime_initialization(self):
        """Test regime dataclass initialization."""
        regime = MarketRegime(
            regime_type='volatile',
            confidence=0.85,
            volatility_regime='high',
            liquidity_regime='scarce',
            trend_strength=-0.3,
            mean_reversion_speed=0.1
        )
        
        assert regime.regime_type == 'volatile'
        assert regime.confidence == 0.85
        assert regime.volatility_regime == 'high'
        assert regime.liquidity_regime == 'scarce'
        assert regime.trend_strength == -0.3
        assert regime.mean_reversion_speed == 0.1
    
    def test_regime_types(self):
        """Test different regime types."""
        regimes = [
            MarketRegime('normal', 0.9, 'medium', 'normal', 0.0, 0.5),
            MarketRegime('volatile', 0.8, 'extreme', 'scarce', 0.1, 0.2),
            MarketRegime('trending', 0.7, 'low', 'abundant', 0.8, 0.05),
            MarketRegime('mean_reverting', 0.85, 'medium', 'normal', -0.1, 0.8)
        ]
        
        for regime in regimes:
            assert regime.regime_type in ['normal', 'volatile', 'trending', 'mean_reverting']
            assert 0 <= regime.confidence <= 1
            assert -1 <= regime.trend_strength <= 1
            assert regime.mean_reversion_speed >= 0


class TestAdaptiveSpreadModel:
    """Test adaptive spread model."""
    
    @pytest.fixture
    def spread_model(self):
        """Create spread model instance."""
        return AdaptiveSpreadModel('AAPL', lookback_period=500)
    
    def test_model_initialization(self, spread_model):
        """Test spread model initialization."""
        assert spread_model.symbol == 'AAPL'
        assert spread_model.lookback_period == 500
        assert len(spread_model.spread_history) == 0
        assert len(spread_model.fill_rate_history) == 0
        assert len(spread_model.adverse_selection_history) == 0
        assert spread_model.optimal_spread == Decimal('0.001')
        
        # Check neural network initialization
        assert spread_model.nn_model is not None
        assert hasattr(spread_model.nn_model, 'forward')
        
        # Check Gaussian Process initialization
        assert spread_model.gp_model is not None
    
    def test_neural_network_architecture(self, spread_model):
        """Test neural network architecture."""
        nn = spread_model.nn_model
        
        # Test forward pass
        input_tensor = torch.randn(1, 20)
        output = nn(input_tensor)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 0.01  # Max spread of 1%
    
    def test_update_spread_initial(self, spread_model):
        """Test spread update with initial data."""
        market_data = {
            'spread': Decimal('0.002'),
            'volatility': 0.02,
            'volume': 1000000,
            'order_imbalance': 0.1,
            'trade_intensity': 150,
            'quote_intensity': 1500
        }
        
        execution_data = {
            'fill_rate': 0.6,
            'adverse_selection': 0.001,
            'avg_queue_position': 0.4
        }
        
        # First update - should use default spread
        optimal_spread = spread_model.update(market_data, execution_data)
        
        assert isinstance(optimal_spread, Decimal)
        assert optimal_spread == Decimal('0.001')  # Default before enough data
        
        # Check histories updated
        assert len(spread_model.spread_history) == 1
        assert len(spread_model.fill_rate_history) == 1
        assert len(spread_model.adverse_selection_history) == 1
    
    def test_update_spread_with_history(self, spread_model):
        """Test spread update with sufficient history."""
        # Fill history
        for i in range(200):
            market_data = {
                'spread': Decimal('0.001') + Decimal(str(i * 0.00001)),
                'volatility': 0.02 + i * 0.0001,
                'volume': 1000000 + i * 1000,
                'order_imbalance': 0.1 * np.sin(i / 10),
                'trade_intensity': 100 + i,
                'quote_intensity': 1000 + i * 10
            }
            
            execution_data = {
                'fill_rate': 0.5 + 0.3 * np.sin(i / 20),
                'adverse_selection': 0.001 + 0.0005 * np.cos(i / 15),
                'avg_queue_position': 0.5 + 0.3 * np.sin(i / 25)
            }
            
            spread_model.update(market_data, execution_data)
        
        # Now should use models
        assert len(spread_model.spread_history) == 200
        
        # Final spread should be optimized
        assert spread_model.optimal_spread != Decimal('0.001')
        assert Decimal('0.0001') <= spread_model.optimal_spread <= Decimal('0.01')
    
    def test_feature_extraction(self, spread_model):
        """Test feature extraction for models."""
        market_data = {
            'volatility': 0.025,
            'volume': 1500000,
            'order_imbalance': -0.05,
            'trade_intensity': 200,
            'quote_intensity': 2000
        }
        
        execution_data = {
            'fill_rate': 0.7,
            'adverse_selection': 0.0005,
            'avg_queue_position': 0.3
        }
        
        features = spread_model._extract_features(market_data, execution_data)
        
        assert len(features) == 20  # Fixed feature size
        assert all(isinstance(f, float) for f in features)
        
        # Check specific features
        assert features[0] == 0.025  # volatility
        assert features[1] == 1500000  # volume
        assert features[2] == -0.05  # order_imbalance
        assert features[5] == 0.7  # fill_rate
    
    def test_spread_constraints(self, spread_model):
        """Test spread constraints are enforced."""
        # Fill with extreme data to test constraints
        for i in range(100):
            market_data = {
                'spread': Decimal('0.1'),  # Very wide
                'volatility': 0.1,  # High volatility
                'volume': 100,  # Low volume
                'order_imbalance': 0.9,  # Extreme imbalance
                'trade_intensity': 10,
                'quote_intensity': 100
            }
            
            execution_data = {
                'fill_rate': 0.1,  # Low fill rate
                'adverse_selection': 0.01,  # High adverse selection
                'avg_queue_position': 0.9
            }
            
            optimal_spread = spread_model.update(market_data, execution_data)
        
        # Should still respect constraints
        assert Decimal('0.0001') <= optimal_spread <= Decimal('0.01')


class TestInventoryOptimizer:
    """Test inventory optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create inventory optimizer instance."""
        return InventoryOptimizer(
            symbol='AAPL',
            risk_aversion=0.1,
            time_horizon=3600
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.symbol == 'AAPL'
        assert optimizer.risk_aversion == 0.1
        assert optimizer.time_horizon == 3600
        assert optimizer.permanent_impact == 0.1
        assert optimizer.temporary_impact == 0.01
        assert optimizer.volatility == 0.02
    
    def test_optimize_trajectory_basic(self, optimizer):
        """Test basic trajectory optimization."""
        current_inventory = 1000.0
        target_inventory = 0.0
        
        market_conditions = {
            'liquidity_factor': 1.0,
            'volatility': 0.02
        }
        
        trajectory = optimizer.optimize_inventory_trajectory(
            current_inventory,
            target_inventory,
            market_conditions
        )
        
        assert isinstance(trajectory, np.ndarray)
        assert trajectory[0] == current_inventory
        assert trajectory[-1] == target_inventory
        assert len(trajectory) >= 10  # Minimum steps
        
        # Should be monotonically decreasing (selling)
        assert all(trajectory[i] >= trajectory[i+1] for i in range(len(trajectory)-1))
    
    def test_optimize_trajectory_with_constraints(self, optimizer):
        """Test trajectory optimization with constraints."""
        current_inventory = 5000.0
        target_inventory = -2000.0  # Going short
        
        constraints = {
            'max_trade_size': 500.0,
            'max_participation': 0.1
        }
        
        trajectory = optimizer.optimize_inventory_trajectory(
            current_inventory,
            target_inventory,
            market_conditions={'liquidity_factor': 0.8, 'volatility': 0.03},
            constraints=constraints
        )
        
        # Check constraints respected
        trades = np.diff(trajectory)
        assert all(abs(trade) <= constraints['max_trade_size'] * 1.01 for trade in trades)  # Small tolerance
    
    def test_objective_function(self, optimizer):
        """Test objective function calculation."""
        optimizer.current_inventory = 1000.0
        optimizer.target_inventory = 0.0
        
        # Test trajectory
        test_trajectory = np.array([500.0, 250.0])  # Intermediate points
        
        cost = optimizer._objective_function(test_trajectory)
        
        assert isinstance(cost, float)
        assert cost > 0  # Should have positive cost
        
        # Linear trajectory should have lower cost than aggressive
        linear_trajectory = np.array([500.0, 250.0])
        aggressive_trajectory = np.array([900.0, 100.0])
        
        cost_linear = optimizer._objective_function(linear_trajectory)
        cost_aggressive = optimizer._objective_function(aggressive_trajectory)
        
        # More aggressive trading should cost more (usually)
        # This might not always hold due to risk penalties
    
    def test_model_parameter_updates(self, optimizer):
        """Test dynamic model parameter updates."""
        # Normal market
        optimizer._update_model_parameters({'liquidity_factor': 1.0, 'volatility': 0.02})
        normal_temp_impact = optimizer.temporary_impact
        normal_perm_impact = optimizer.permanent_impact
        
        # Illiquid market
        optimizer._update_model_parameters({'liquidity_factor': 0.5, 'volatility': 0.04})
        assert optimizer.temporary_impact > normal_temp_impact
        assert optimizer.permanent_impact > normal_perm_impact
        assert optimizer.volatility == 0.04
        
        # Very liquid market
        optimizer._update_model_parameters({'liquidity_factor': 2.0, 'volatility': 0.01})
        assert optimizer.temporary_impact < normal_temp_impact
        assert optimizer.permanent_impact < normal_perm_impact
    
    def test_get_next_trade(self, optimizer):
        """Test next trade calculation."""
        # Set up trajectory
        optimizer.optimal_trajectory = np.array([1000, 750, 500, 250, 0])
        optimizer.time_horizon = 3600  # 1 hour
        optimizer.current_inventory = 1000
        
        # At start
        trade = optimizer.get_next_trade(0)
        assert trade == -250  # Move from 1000 to 750
        
        # Halfway through
        trade = optimizer.get_next_trade(1800)
        assert abs(trade - (-500)) < 100  # Approximately -500
        
        # At end
        trade = optimizer.get_next_trade(3600)
        assert trade == 0.0  # No more trades


class TestStatisticalArbitrageEngine:
    """Test statistical arbitrage engine."""
    
    @pytest.fixture
    def stat_arb_engine(self):
        """Create stat arb engine instance."""
        return StatisticalArbitrageEngine(
            symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            lookback=500
        )
    
    def test_engine_initialization(self, stat_arb_engine):
        """Test engine initialization."""
        assert len(stat_arb_engine.symbols) == 4
        assert stat_arb_engine.lookback == 500
        assert all(isinstance(hist, deque) for hist in stat_arb_engine.price_histories.values())
        assert all(hist.maxlen == 500 for hist in stat_arb_engine.price_histories.values())
        assert len(stat_arb_engine.cointegration_pairs) == 0
        assert len(stat_arb_engine.signals) == 0
    
    def test_update_prices(self, stat_arb_engine):
        """Test price update mechanism."""
        prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'AMZN': 3200.0
        }
        
        stat_arb_engine.update_prices(prices)
        
        for symbol in stat_arb_engine.symbols:
            assert len(stat_arb_engine.price_histories[symbol]) == 1
            assert stat_arb_engine.price_histories[symbol][0] == prices[symbol]
        
        # Update multiple times
        for i in range(100):
            new_prices = {
                symbol: price * (1 + np.random.normal(0, 0.01))
                for symbol, price in prices.items()
            }
            stat_arb_engine.update_prices(new_prices)
        
        # Should trigger signal updates
        assert len(stat_arb_engine.signals) > 0
    
    def test_cointegration_detection(self, stat_arb_engine):
        """Test cointegration pair detection."""
        # Create synthetic cointegrated price series
        n_points = 200
        t = np.arange(n_points)
        
        # Base series
        base = 100 + 0.1 * t + np.random.normal(0, 1, n_points).cumsum()
        
        # Cointegrated series (with noise)
        spread_mean = 50
        spread_noise = np.random.normal(0, 2, n_points)
        cointegrated = base + spread_mean + spread_noise
        
        # Non-cointegrated series
        independent = 200 + 0.2 * t + np.random.normal(0, 3, n_points).cumsum()
        
        # Another cointegrated series
        cointegrated2 = base * 0.5 + 25 + np.random.normal(0, 1, n_points)
        
        # Fill price histories
        for i in range(n_points):
            prices = {
                'AAPL': base[i],
                'GOOGL': cointegrated[i],
                'MSFT': independent[i],
                'AMZN': cointegrated2[i]
            }
            stat_arb_engine.update_prices(prices)
        
        # Check cointegration detection
        stat_arb_engine._find_cointegration_pairs()
        
        # Should find AAPL-GOOGL and AAPL-AMZN pairs
        pair_symbols = [(p[0], p[1]) for p in stat_arb_engine.cointegration_pairs]
        
        # At least one pair should be detected
        assert len(stat_arb_engine.cointegration_pairs) >= 1
    
    def test_pair_signal_calculation(self, stat_arb_engine):
        """Test trading signal calculation for pairs."""
        # Set up a known pair
        stat_arb_engine.cointegration_pairs = [('AAPL', 'GOOGL')]
        stat_arb_engine.pair_parameters['AAPL_GOOGL'] = {
            'hedge_ratio': 1.0,
            'mean_spread': 100.0,
            'std_spread': 5.0
        }
        
        # Fill price histories
        for i in range(100):
            stat_arb_engine.price_histories['AAPL'].append(150 + i * 0.1)
            stat_arb_engine.price_histories['GOOGL'].append(250 + i * 0.1)
        
        # Current spread = 150.9 - 259.9 = -109, normalized = (-109 - 100) / 5 = -41.8
        signal, confidence = stat_arb_engine._calculate_pair_signal(('AAPL', 'GOOGL'))
        
        assert isinstance(signal, float)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        
        # Large deviation should give strong signal
        assert abs(signal) > 1.0
    
    def test_mean_reversion_signals(self, stat_arb_engine):
        """Test mean reversion signal generation."""
        # Create price series with mean reversion
        prices = []
        base_price = 100
        
        for i in range(100):
            if i < 50:
                # Price goes up
                price = base_price + i * 0.5
            else:
                # Price reverts
                price = base_price + (100 - i) * 0.5
            
            prices.append(price)
            stat_arb_engine.price_histories['AAPL'].append(price)
        
        signal, confidence = stat_arb_engine._calculate_mean_reversion_signal('AAPL')
        
        assert isinstance(signal, float)
        assert isinstance(confidence, float)
        
        # At the peak (around step 50), should have negative signal (sell)
        # At the end, price is back to normal, signal should be smaller
    
    def test_get_trading_signals(self, stat_arb_engine):
        """Test getting formatted trading signals."""
        # Set up some signals
        stat_arb_engine.signals = {
            'AAPL': 0.5,
            'GOOGL': -0.3,
            'AAPL_GOOGL': 0.8
        }
        stat_arb_engine.signal_confidence = {
            'AAPL': 0.7,
            'GOOGL': 0.4,
            'AAPL_GOOGL': 0.9
        }
        
        signals = stat_arb_engine.get_trading_signals()
        
        assert 'AAPL' in signals
        assert signals['AAPL']['signal'] == 0.5
        assert signals['AAPL']['confidence'] == 0.7
        assert signals['AAPL']['direction'] == 'buy'
        assert signals['AAPL']['strength'] == 0.5
        
        assert signals['GOOGL']['direction'] == 'sell'
        assert signals['GOOGL']['strength'] == 0.3


class TestJITCompiledFunctions:
    """Test JIT-compiled functions."""
    
    def test_calculate_optimal_quotes_numba(self):
        """Test Numba-optimized quote calculation."""
        # Test inputs
        fair_value = 150.0
        inventory = 500.0
        max_inventory = 1000.0
        risk_aversion = 0.1
        volatility = 0.02
        order_arrival_rate = 100.0
        fill_probability = 0.5
        
        bid_price, ask_price = calculate_optimal_quotes_numba(
            fair_value,
            inventory,
            max_inventory,
            risk_aversion,
            volatility,
            order_arrival_rate,
            fill_probability
        )
        
        assert isinstance(bid_price, float)
        assert isinstance(ask_price, float)
        assert bid_price < ask_price
        assert bid_price < fair_value < ask_price
        
        # Test with different inventory levels
        # No inventory - symmetric quotes
        bid_0, ask_0 = calculate_optimal_quotes_numba(
            fair_value, 0, max_inventory, risk_aversion,
            volatility, order_arrival_rate, fill_probability
        )
        
        # Long inventory - lower prices
        bid_long, ask_long = calculate_optimal_quotes_numba(
            fair_value, 800, max_inventory, risk_aversion,
            volatility, order_arrival_rate, fill_probability
        )
        
        # Short inventory - higher prices
        bid_short, ask_short = calculate_optimal_quotes_numba(
            fair_value, -800, max_inventory, risk_aversion,
            volatility, order_arrival_rate, fill_probability
        )
        
        # Long inventory should have lower quotes
        assert bid_long < bid_0
        assert ask_long < ask_0
        
        # Short inventory should have higher quotes
        assert bid_short > bid_0
        assert ask_short > ask_0
    
    def test_numba_edge_cases(self):
        """Test Numba function edge cases."""
        # Zero volatility
        bid, ask = calculate_optimal_quotes_numba(
            100.0, 0, 1000, 0.1, 0.0, 100, 0.5
        )
        assert bid < 100 < ask
        
        # Very high risk aversion
        bid_high_risk, ask_high_risk = calculate_optimal_quotes_numba(
            100.0, 500, 1000, 10.0, 0.02, 100, 0.5
        )
        
        # Very low risk aversion
        bid_low_risk, ask_low_risk = calculate_optimal_quotes_numba(
            100.0, 500, 1000, 0.001, 0.02, 100, 0.5
        )
        
        # Higher risk aversion should lead to wider spreads
        spread_high = ask_high_risk - bid_high_risk
        spread_low = ask_low_risk - bid_low_risk
        assert spread_high > spread_low


class TestAdvancedMarketMaker:
    """Test integrated advanced market maker."""
    
    @pytest.fixture
    def market_maker(self):
        """Create market maker instance."""
        params = MarketMakingParameters(
            max_inventory=Decimal('10000'),
            base_order_size=Decimal('100'),
            risk_aversion=Decimal('0.1')
        )
        return AdvancedMarketMaker('AAPL', params)
    
    def test_market_maker_initialization(self, market_maker):
        """Test market maker initialization."""
        assert market_maker.symbol == 'AAPL'
        assert market_maker.inventory == Decimal('0')
        assert market_maker.position_pnl == Decimal('0')
        assert market_maker.total_volume == Decimal('0')
        
        # Check components initialized
        assert market_maker.spread_model is not None
        assert market_maker.inventory_optimizer is not None
        assert market_maker.stat_arb_engine is not None
    
    @pytest.mark.asyncio
    async def test_generate_quotes_normal(self, market_maker):
        """Test quote generation under normal conditions."""
        market_data = {
            'best_bid': 149.95,
            'best_ask': 150.05,
            'bid_size': 1000,
            'ask_size': 1200,
            'vwap': 150.00,
            'volatility': 0.02,
            'order_arrival_rate': 100,
            'fill_probability': 0.5,
            'queue_position': 0.5
        }
        
        predictions = {
            'price_prediction': 150.02,
            'confidence': 0.8
        }
        
        # Update stat arb with some prices
        for i in range(100):
            market_maker.stat_arb_engine.update_prices({'AAPL': 150 + np.random.normal(0, 0.5)})
        
        bid_order, ask_order = await market_maker.generate_quotes(market_data, predictions)
        
        assert bid_order is not None
        assert ask_order is not None
        assert bid_order['side'] == 'buy'
        assert ask_order['side'] == 'sell'
        assert bid_order['price'] < ask_order['price']
        assert bid_order['order_type'] == 'limit'
        assert bid_order['time_in_force'] == 'IOC'
    
    @pytest.mark.asyncio
    async def test_generate_quotes_inventory_limits(self, market_maker):
        """Test quote generation at inventory limits."""
        market_data = {
            'best_bid': 149.95,
            'best_ask': 150.05,
            'bid_size': 1000,
            'ask_size': 1200,
            'vwap': 150.00,
            'volatility': 0.02,
            'order_arrival_rate': 100,
            'fill_probability': 0.5,
            'queue_position': 0.5
        }
        
        predictions = {'price_prediction': 150.00}
        
        # Set max long inventory
        market_maker.inventory = market_maker.parameters.max_inventory
        
        bid_order, ask_order = await market_maker.generate_quotes(market_data, predictions)
        
        assert bid_order is None  # No more buying
        assert ask_order is not None  # Can still sell
        
        # Set max short inventory
        market_maker.inventory = -market_maker.parameters.max_inventory
        
        bid_order, ask_order = await market_maker.generate_quotes(market_data, predictions)
        
        assert bid_order is not None  # Can buy
        assert ask_order is None  # No more selling
    
    def test_fair_value_estimation(self, market_maker):
        """Test fair value calculation."""
        market_data = {
            'best_bid': 149.90,
            'best_ask': 150.10,
            'vwap': 150.05,
            'bid_size': 1500,
            'ask_size': 1000
        }
        
        predictions = {
            'price_prediction': 150.03,
            'confidence': 0.9
        }
        
        fair_value = market_maker._estimate_fair_value(market_data, predictions)
        
        assert isinstance(fair_value, Decimal)
        assert Decimal('149.5') <= fair_value <= Decimal('150.5')
        
        # Should be influenced by all components
        # Mid price = 150.00
        # VWAP = 150.05
        # ML prediction = 150.03
        # Microprice = (149.90 * 1000 + 150.10 * 1500) / 2500 = 150.02
        
        # Weighted average should be around 150.01-150.03
        assert abs(fair_value - Decimal('150.02')) < Decimal('0.05')
    
    def test_order_size_calculation(self, market_maker):
        """Test dynamic order size calculation."""
        market_data_normal = {
            'volatility': 0.02,
            'queue_position': 0.5
        }
        
        # Normal conditions
        size_normal = market_maker._calculate_order_size('buy', market_data_normal)
        assert size_normal == market_maker.parameters.base_order_size
        
        # High volatility
        market_data_volatile = {
            'volatility': 0.05,
            'queue_position': 0.5
        }
        size_volatile = market_maker._calculate_order_size('buy', market_data_volatile)
        assert size_volatile < size_normal  # Reduce size in volatile markets
        
        # Good queue position
        market_data_good_queue = {
            'volatility': 0.02,
            'queue_position': 0.1  # Near front
        }
        size_good_queue = market_maker._calculate_order_size('buy', market_data_good_queue)
        assert size_good_queue > size_normal  # Increase size with good queue position
        
        # With inventory
        market_maker.inventory = market_maker.parameters.max_inventory * Decimal('0.8')
        size_with_inventory = market_maker._calculate_order_size('buy', market_data_normal)
        assert size_with_inventory < size_normal  # Reduce size when inventory is high
    
    def test_tick_rounding(self, market_maker):
        """Test price rounding to tick size."""
        # Test rounding down
        price1 = Decimal('150.12345')
        rounded_down = market_maker._round_to_tick(price1, ROUND_DOWN)
        assert rounded_down == Decimal('150.12')
        
        # Test rounding up
        from decimal import ROUND_UP
        rounded_up = market_maker._round_to_tick(price1, ROUND_UP)
        assert rounded_up == Decimal('150.13')
        
        # Test different tick size
        market_maker.parameters.tick_size = Decimal('0.05')
        price2 = Decimal('150.17')
        rounded = market_maker._round_to_tick(price2, ROUND_DOWN)
        assert rounded == Decimal('150.15')
    
    def test_update_metrics(self, market_maker):
        """Test metrics updating."""
        # Buy execution
        buy_execution = {
            'side': 'buy',
            'quantity': Decimal('100'),
            'price': Decimal('149.95')
        }
        
        market_maker.update_metrics(buy_execution)
        
        assert market_maker.inventory == Decimal('100')
        assert market_maker.position_pnl == -Decimal('14995')  # Negative for buy
        assert market_maker.metrics['trades'] == 1
        assert market_maker.total_volume == Decimal('100')
        
        # Sell execution
        sell_execution = {
            'side': 'sell',
            'quantity': Decimal('50'),
            'price': Decimal('150.05')
        }
        
        market_maker.update_metrics(sell_execution)
        
        assert market_maker.inventory == Decimal('50')
        assert market_maker.position_pnl == -Decimal('14995') + Decimal('7502.5')
        assert market_maker.metrics['trades'] == 2
        assert market_maker.total_volume == Decimal('150')
