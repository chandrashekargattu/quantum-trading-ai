"""
Comprehensive tests for High-Frequency Trading Engine.
Tests lock-free order books, smart routing, and ultra-low latency execution.
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal, ROUND_DOWN
import time
from collections import deque
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from sortedcontainers import SortedDict

from app.hft.hft_engine import (
    OrderBookLevel,
    MarketMicrostructure,
    HFTOrder,
    Execution,
    LockFreeOrderBook,
    SmartOrderRouter,
    MarketMakingEngine,
    UltraLowLatencyExecutor,
    HFTEngine,
    calculate_vwap,
    calculate_order_imbalance
)


class TestOrderBookDataStructures:
    """Test order book data structures."""
    
    def test_order_book_level(self):
        """Test OrderBookLevel dataclass."""
        level = OrderBookLevel(
            price=Decimal('100.50'),
            quantity=Decimal('1000'),
            order_count=5,
            timestamp=1234567890123456789
        )
        
        assert level.price == Decimal('100.50')
        assert level.quantity == Decimal('1000')
        assert level.order_count == 5
        assert level.timestamp == 1234567890123456789
    
    def test_market_microstructure(self):
        """Test MarketMicrostructure dataclass."""
        micro = MarketMicrostructure(
            bid_ask_spread=Decimal('0.01'),
            effective_spread=Decimal('0.012'),
            realized_spread=Decimal('0.011'),
            price_impact=Decimal('0.0001'),
            order_imbalance=Decimal('0.1'),
            trade_intensity=100.5,
            quote_intensity=1000.2,
            volatility=Decimal('0.02'),
            tick_direction=1,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        assert micro.bid_ask_spread == Decimal('0.01')
        assert micro.tick_direction == 1
        assert micro.trade_intensity == 100.5
    
    def test_hft_order(self):
        """Test HFTOrder dataclass."""
        order = HFTOrder(
            order_id='TEST123',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        
        assert order.order_id == 'TEST123'
        assert order.symbol == 'AAPL'
        assert order.side == 'buy'
        assert order.timestamp > 0
        
        # Test optional fields
        order_full = HFTOrder(
            order_id='TEST456',
            symbol='GOOGL',
            side='sell',
            order_type='pegged',
            price=None,
            quantity=Decimal('50'),
            time_in_force='GTC',
            hidden_quantity=Decimal('30'),
            peg_offset=Decimal('-0.01'),
            min_quantity=Decimal('10'),
            client_order_id='CLIENT123',
            execution_instructions={'post_only': True}
        )
        
        assert order_full.hidden_quantity == Decimal('30')
        assert order_full.peg_offset == Decimal('-0.01')
        assert order_full.execution_instructions['post_only'] is True


class TestLockFreeOrderBook:
    """Test lock-free order book implementation."""
    
    @pytest.fixture
    def order_book(self):
        """Create order book instance."""
        return LockFreeOrderBook('AAPL')
    
    def test_order_book_initialization(self, order_book):
        """Test order book initialization."""
        assert order_book.symbol == 'AAPL'
        assert isinstance(order_book.bids, SortedDict)
        assert isinstance(order_book.asks, SortedDict)
        assert len(order_book.order_map) == 0
        assert order_book.last_update_time > 0
    
    def test_add_order(self, order_book):
        """Test adding orders to book."""
        # Add buy order
        buy_order = HFTOrder(
            order_id='BUY1',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('149.90'),
            quantity=Decimal('100'),
            time_in_force='GTC'
        )
        
        assert order_book.add_order(buy_order)
        assert Decimal('149.90') in order_book.bids
        assert 'BUY1' in order_book.order_map
        
        # Add sell order
        sell_order = HFTOrder(
            order_id='SELL1',
            symbol='AAPL',
            side='sell',
            order_type='limit',
            price=Decimal('150.10'),
            quantity=Decimal('200'),
            time_in_force='GTC'
        )
        
        assert order_book.add_order(sell_order)
        assert Decimal('150.10') in order_book.asks
        assert 'SELL1' in order_book.order_map
    
    def test_cancel_order(self, order_book):
        """Test order cancellation."""
        # Add order
        order = HFTOrder(
            order_id='CANCEL1',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('149.50'),
            quantity=Decimal('100'),
            time_in_force='GTC'
        )
        order_book.add_order(order)
        
        # Cancel order
        assert order_book.cancel_order('CANCEL1')
        assert 'CANCEL1' not in order_book.order_map
        assert Decimal('149.50') not in order_book.bids
        
        # Try to cancel non-existent order
        assert not order_book.cancel_order('NONEXISTENT')
    
    def test_best_bid_ask(self, order_book):
        """Test getting best bid and ask prices."""
        # Empty book
        best_bid, best_ask = order_book.get_best_bid_ask()
        assert best_bid is None
        assert best_ask is None
        
        # Add orders
        order_book.add_order(HFTOrder(
            order_id='B1', symbol='AAPL', side='buy',
            order_type='limit', price=Decimal('149.90'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        order_book.add_order(HFTOrder(
            order_id='B2', symbol='AAPL', side='buy',
            order_type='limit', price=Decimal('149.80'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        order_book.add_order(HFTOrder(
            order_id='S1', symbol='AAPL', side='sell',
            order_type='limit', price=Decimal('150.10'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        order_book.add_order(HFTOrder(
            order_id='S2', symbol='AAPL', side='sell',
            order_type='limit', price=Decimal('150.20'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        
        best_bid, best_ask = order_book.get_best_bid_ask()
        assert best_bid == Decimal('149.90')  # Highest bid
        assert best_ask == Decimal('150.10')  # Lowest ask
    
    def test_market_depth(self, order_book):
        """Test market depth retrieval."""
        # Add multiple orders at different price levels
        for i in range(10):
            # Bids
            order_book.add_order(HFTOrder(
                order_id=f'B{i}', symbol='AAPL', side='buy',
                order_type='limit', price=Decimal(f'149.{90-i}'),
                quantity=Decimal(f'{100*(i+1)}'), time_in_force='GTC'
            ))
            # Asks
            order_book.add_order(HFTOrder(
                order_id=f'S{i}', symbol='AAPL', side='sell',
                order_type='limit', price=Decimal(f'150.{10+i}'),
                quantity=Decimal(f'{100*(i+1)}'), time_in_force='GTC'
            ))
        
        # Get depth
        depth = order_book.get_market_depth(levels=5)
        
        assert len(depth['bids']) == 5
        assert len(depth['asks']) == 5
        
        # Check bid ordering (highest to lowest)
        bid_prices = [level.price for level in depth['bids']]
        assert bid_prices == sorted(bid_prices, reverse=True)
        
        # Check ask ordering (lowest to highest)
        ask_prices = [level.price for level in depth['asks']]
        assert ask_prices == sorted(ask_prices)
        
        # Check aggregation
        assert depth['bids'][0].quantity == Decimal('100')  # Single order
        assert depth['bids'][0].order_count == 1
    
    def test_order_book_thread_safety(self, order_book):
        """Test order book operations are atomic."""
        import threading
        
        def add_orders():
            for i in range(100):
                order = HFTOrder(
                    order_id=f'{threading.current_thread().name}_{i}',
                    symbol='AAPL',
                    side='buy' if i % 2 == 0 else 'sell',
                    order_type='limit',
                    price=Decimal(f'150.{i % 100:02d}'),
                    quantity=Decimal('100'),
                    time_in_force='GTC'
                )
                order_book.add_order(order)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_orders, name=f'Thread{i}')
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check all orders added
        assert len(order_book.order_map) == 500


class TestJITCompiledFunctions:
    """Test JIT-compiled numerical functions."""
    
    def test_calculate_vwap(self):
        """Test VWAP calculation."""
        prices = np.array([100.0, 101.0, 102.0, 101.5, 100.5])
        volumes = np.array([1000.0, 1500.0, 2000.0, 1200.0, 800.0])
        
        # Test different windows
        vwap_full = calculate_vwap(prices, volumes, window=5)
        expected_full = np.sum(prices * volumes) / np.sum(volumes)
        assert abs(vwap_full - expected_full) < 0.001
        
        # Test partial window
        vwap_partial = calculate_vwap(prices, volumes, window=3)
        expected_partial = np.sum(prices[-3:] * volumes[-3:]) / np.sum(volumes[-3:])
        assert abs(vwap_partial - expected_partial) < 0.001
        
        # Test edge cases
        vwap_single = calculate_vwap(np.array([100.0]), np.array([1000.0]), window=10)
        assert vwap_single == 100.0
        
        # Zero volume
        vwap_zero = calculate_vwap(prices, np.zeros(5), window=5)
        assert vwap_zero == prices[-1]  # Falls back to last price
    
    def test_calculate_order_imbalance(self):
        """Test order imbalance calculation."""
        bid_volumes = np.array([1000.0, 2000.0, 1500.0])
        ask_volumes = np.array([1500.0, 1000.0, 2000.0])
        
        imbalance = calculate_order_imbalance(bid_volumes, ask_volumes)
        
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        expected = (total_bid - total_ask) / (total_bid + total_ask)
        
        assert abs(imbalance - expected) < 0.001
        
        # Test edge cases
        # Equal volumes
        equal_imbalance = calculate_order_imbalance(
            np.array([1000.0]),
            np.array([1000.0])
        )
        assert equal_imbalance == 0.0
        
        # Zero volumes
        zero_imbalance = calculate_order_imbalance(
            np.array([0.0]),
            np.array([0.0])
        )
        assert zero_imbalance == 0.0
        
        # All bid
        bid_imbalance = calculate_order_imbalance(
            np.array([1000.0]),
            np.array([0.0])
        )
        assert bid_imbalance == 1.0
        
        # All ask
        ask_imbalance = calculate_order_imbalance(
            np.array([0.0]),
            np.array([1000.0])
        )
        assert ask_imbalance == -1.0


class TestSmartOrderRouter:
    """Test smart order routing functionality."""
    
    @pytest.fixture
    def router(self):
        """Create router instance."""
        venues = ['NYSE', 'NASDAQ', 'BATS', 'IEX']
        router = SmartOrderRouter(venues)
        
        # Set up mock venue characteristics
        router.venue_latencies = {
            'NYSE': 50,
            'NASDAQ': 40,
            'BATS': 30,
            'IEX': 350  # IEX has intentional delay
        }
        router.venue_fees = {
            'NYSE': 0.0028,
            'NASDAQ': 0.0030,
            'BATS': 0.0025,
            'IEX': 0.0009
        }
        
        return router
    
    @pytest.mark.asyncio
    async def test_route_order_basic(self, router):
        """Test basic order routing."""
        order = HFTOrder(
            order_id='TEST123',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('1000'),
            time_in_force='IOC'
        )
        
        # Mock liquidity distribution
        with patch.object(router, '_get_liquidity_distribution', 
                         return_value={'NYSE': 400000, 'NASDAQ': 500000, 'BATS': 300000, 'IEX': 200000}):
            
            sub_orders = await router.route_order(order, {'current_price': 150.0})
            
            assert len(sub_orders) > 0
            assert all(venue in router.venues for venue, _ in sub_orders)
            
            # Check total quantity matches
            total_quantity = sum(sub_order.quantity for _, sub_order in sub_orders)
            assert abs(total_quantity - order.quantity) < Decimal('0.01')
    
    def test_calculate_optimal_split(self, router):
        """Test optimal order splitting algorithm."""
        order = HFTOrder(
            order_id='TEST',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('1000'),
            time_in_force='IOC'
        )
        
        liquidity_map = {
            'NYSE': 400000,
            'NASDAQ': 500000,
            'BATS': 300000,
            'IEX': 200000
        }
        
        splits = router._calculate_optimal_split(
            order,
            liquidity_map,
            {'current_price': 150.0}
        )
        
        # Check all venues included
        assert set(splits.keys()) == set(router.venues)
        
        # Check splits sum to 1
        assert abs(sum(splits.values()) - 1.0) < 0.001
        
        # Check NASDAQ gets highest allocation (most liquidity, reasonable fees)
        assert max(splits.items(), key=lambda x: x[1])[0] in ['NASDAQ', 'BATS']
        
        # Check IEX gets lower allocation (high latency)
        assert splits['IEX'] < splits['NYSE']
    
    def test_edge_cases(self, router):
        """Test router edge cases."""
        order = HFTOrder(
            order_id='TEST',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('1'),  # Very small order
            time_in_force='IOC'
        )
        
        # No liquidity
        splits = router._calculate_optimal_split(
            order,
            {'NYSE': 0, 'NASDAQ': 0, 'BATS': 0, 'IEX': 0},
            {}
        )
        
        # Should split equally
        assert all(abs(split - 0.25) < 0.001 for split in splits.values())


class TestMarketMakingEngine:
    """Test market making engine."""
    
    @pytest.fixture
    def mm_engine(self):
        """Create market making engine."""
        return MarketMakingEngine(
            symbol='AAPL',
            risk_limits={'max_inventory': 10000, 'max_position': 1000000},
            target_spread=Decimal('0.0002')
        )
    
    def test_mm_initialization(self, mm_engine):
        """Test market making engine initialization."""
        assert mm_engine.symbol == 'AAPL'
        assert mm_engine.target_spread == Decimal('0.0002')
        assert mm_engine.inventory == Decimal('0')
        assert mm_engine.max_inventory == Decimal('10000')
    
    def test_calculate_quotes_normal(self, mm_engine):
        """Test quote calculation under normal conditions."""
        market_data = MarketMicrostructure(
            bid_ask_spread=Decimal('0.02'),
            effective_spread=Decimal('0.022'),
            realized_spread=Decimal('0.021'),
            price_impact=Decimal('0.0001'),
            order_imbalance=Decimal('0.05'),
            trade_intensity=100.0,
            quote_intensity=1000.0,
            volatility=Decimal('0.015'),
            tick_direction=1,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        predictions = {'price': 150.0}
        
        # Set fair value
        mm_engine.fair_value = Decimal('150.00')
        
        bid_order, ask_order = mm_engine.calculate_quotes(market_data, predictions)
        
        assert bid_order is not None
        assert ask_order is not None
        assert bid_order.side == 'buy'
        assert ask_order.side == 'sell'
        assert bid_order.price < ask_order.price
        assert bid_order.time_in_force == 'IOC'
    
    def test_inventory_skew(self, mm_engine):
        """Test inventory-based quote skewing."""
        market_data = MarketMicrostructure(
            bid_ask_spread=Decimal('0.01'),
            effective_spread=Decimal('0.01'),
            realized_spread=Decimal('0.01'),
            price_impact=Decimal('0.0001'),
            order_imbalance=Decimal('0'),
            trade_intensity=100.0,
            quote_intensity=1000.0,
            volatility=Decimal('0.01'),
            tick_direction=0,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        mm_engine.fair_value = Decimal('150.00')
        
        # No inventory - symmetric quotes
        bid1, ask1 = mm_engine.calculate_quotes(market_data, {'price': 150.0})
        
        # Long inventory - skew to encourage selling
        mm_engine.inventory = Decimal('5000')  # Half of max
        bid2, ask2 = mm_engine.calculate_quotes(market_data, {'price': 150.0})
        
        # Bid should be lower, ask should be lower
        assert bid2.price < bid1.price
        assert ask2.price < ask1.price
        
        # Test max inventory limits
        mm_engine.inventory = mm_engine.max_inventory
        bid3, ask3 = mm_engine.calculate_quotes(market_data, {'price': 150.0})
        
        assert bid3 is None  # No more buying at max inventory
        assert ask3 is not None
    
    def test_fair_value_calculation(self, mm_engine):
        """Test fair value calculation from multiple sources."""
        market_data = MarketMicrostructure(
            bid_ask_spread=Decimal('0.02'),
            effective_spread=Decimal('0.02'),
            realized_spread=Decimal('0.02'),
            price_impact=Decimal('0.0001'),
            order_imbalance=Decimal('0.1'),
            trade_intensity=100.0,
            quote_intensity=1000.0,
            volatility=Decimal('0.02'),
            tick_direction=1,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        predictions = {
            'price': 150.5,
            'confidence': 0.8
        }
        
        fair_value = mm_engine._calculate_fair_value(market_data, predictions)
        
        assert isinstance(fair_value, Decimal)
        assert fair_value > 0
        
        # Should be influenced by prediction
        assert abs(float(fair_value) - 150.5) < 2.0  # Reasonable range
    
    def test_optimal_spread_calculation(self, mm_engine):
        """Test dynamic spread calculation."""
        # Low volatility, low imbalance
        market_data_calm = MarketMicrostructure(
            bid_ask_spread=Decimal('0.01'),
            effective_spread=Decimal('0.01'),
            realized_spread=Decimal('0.01'),
            price_impact=Decimal('0.0001'),
            order_imbalance=Decimal('0.01'),
            trade_intensity=50.0,
            quote_intensity=500.0,
            volatility=Decimal('0.005'),
            tick_direction=0,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        spread_calm = mm_engine._calculate_optimal_spread(market_data_calm)
        
        # High volatility, high imbalance
        market_data_volatile = MarketMicrostructure(
            bid_ask_spread=Decimal('0.05'),
            effective_spread=Decimal('0.05'),
            realized_spread=Decimal('0.05'),
            price_impact=Decimal('0.001'),
            order_imbalance=Decimal('0.5'),
            trade_intensity=200.0,
            quote_intensity=2000.0,
            volatility=Decimal('0.05'),
            tick_direction=1,
            volume_clock=1000000.0,
            trade_clock=100.0
        )
        
        spread_volatile = mm_engine._calculate_optimal_spread(market_data_volatile)
        
        # Volatile market should have wider spread
        assert spread_volatile > spread_calm
        assert spread_volatile > mm_engine.target_spread
        
        # Check minimum spread
        assert spread_calm >= Decimal('0.0001')


class TestUltraLowLatencyExecutor:
    """Test ultra-low latency execution."""
    
    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return UltraLowLatencyExecutor()
    
    @pytest.mark.asyncio
    async def test_execute_order(self, executor):
        """Test order execution."""
        order = HFTOrder(
            order_id='FAST123',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        
        start = time.time_ns()
        execution = await executor.execute_order(order, 'NYSE', priority=1)
        latency_ns = time.time_ns() - start
        
        assert isinstance(execution, Execution)
        assert execution.order_id == 'test_order'  # Mocked
        assert execution.venue == 'NYSE'
        
        # Check latency tracking
        stats = executor.execution_stats['NYSE']
        assert stats['count'] == 1
        assert stats['min_latency'] > 0
        assert stats['max_latency'] > 0
    
    def test_order_serialization(self, executor):
        """Test fast order serialization."""
        order = HFTOrder(
            order_id='SER123',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        
        serialized = executor._serialize_order_fast(order)
        
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Test edge cases
        order_market = HFTOrder(
            order_id='MKT',
            symbol='X',  # Short symbol
            side='sell',
            order_type='market',
            price=None,  # No price for market order
            quantity=Decimal('1'),
            time_in_force='FOK'
        )
        
        serialized_market = executor._serialize_order_fast(order_market)
        assert isinstance(serialized_market, bytes)
    
    def test_execution_statistics(self, executor):
        """Test execution statistics tracking."""
        executor._update_stats('NYSE', 10.5)
        executor._update_stats('NYSE', 15.3)
        executor._update_stats('NYSE', 8.2)
        executor._update_stats('NASDAQ', 12.1)
        
        nyse_stats = executor.execution_stats['NYSE']
        assert nyse_stats['count'] == 3
        assert nyse_stats['min_latency'] == 8.2
        assert nyse_stats['max_latency'] == 15.3
        assert nyse_stats['total_latency'] == 10.5 + 15.3 + 8.2
        
        nasdaq_stats = executor.execution_stats['NASDAQ']
        assert nasdaq_stats['count'] == 1
        assert nasdaq_stats['min_latency'] == 12.1


class TestHFTEngine:
    """Test main HFT engine integration."""
    
    @pytest.fixture
    def hft_engine(self):
        """Create HFT engine instance."""
        return HFTEngine(
            symbols=['AAPL', 'GOOGL'],
            venues=['NYSE', 'NASDAQ'],
            risk_limits={
                'max_position': 100000,
                'max_loss': -5000,
                'max_orders_per_second': 1000
            }
        )
    
    def test_engine_initialization(self, hft_engine):
        """Test HFT engine initialization."""
        assert len(hft_engine.symbols) == 2
        assert len(hft_engine.venues) == 2
        assert len(hft_engine.order_books) == 2
        assert len(hft_engine.market_makers) == 2
        
        assert hft_engine.risk_limits['max_position'] == 100000
        assert hft_engine.metrics['orders_sent'] == 0
    
    @pytest.mark.asyncio
    async def test_send_order(self, hft_engine):
        """Test order sending through the engine."""
        order = HFTOrder(
            order_id='ENGINE123',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('150.00'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        
        # Mock router and executor
        with patch.object(hft_engine.smart_router, 'route_order',
                         return_value=[('NYSE', order)]):
            with patch.object(hft_engine.executor, 'execute_order',
                             return_value=Mock(quantity=Decimal('100'))):
                
                await hft_engine._send_order(order)
                
                assert hft_engine.metrics['orders_sent'] == 1
                assert hft_engine.metrics['orders_filled'] == 1
                assert hft_engine.metrics['total_volume'] == Decimal('100')
    
    def test_calculate_microstructure(self, hft_engine):
        """Test market microstructure calculation."""
        # Add some orders to order book
        order_book = hft_engine.order_books['AAPL']
        
        order_book.add_order(HFTOrder(
            order_id='B1', symbol='AAPL', side='buy',
            order_type='limit', price=Decimal('149.99'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        order_book.add_order(HFTOrder(
            order_id='S1', symbol='AAPL', side='sell',
            order_type='limit', price=Decimal('150.01'),
            quantity=Decimal('100'), time_in_force='GTC'
        ))
        
        microstructure = hft_engine._calculate_microstructure('AAPL')
        
        assert microstructure.bid_ask_spread == Decimal('0.02')
        assert microstructure.volatility == Decimal('0.02')  # Default
        assert microstructure.trade_intensity == 100.0  # Default
    
    @pytest.mark.asyncio
    async def test_risk_monitoring(self, hft_engine):
        """Test risk monitoring functionality."""
        # Set high position
        hft_engine.position_tracker['AAPL'] = Decimal('150000')  # Over limit
        
        # Capture log warning
        with patch('app.hft.hft_engine.logger.warning') as mock_warning:
            await hft_engine._risk_monitoring_loop()
            
            # Should log position limit warning
            mock_warning.assert_called()
            assert 'Position limit exceeded' in str(mock_warning.call_args)
        
        # Test loss limit
        hft_engine.pnl_tracker['AAPL'] = Decimal('-6000')  # Over loss limit
        
        with patch('app.hft.hft_engine.logger.error') as mock_error:
            await hft_engine._risk_monitoring_loop()
            
            # Should log loss limit error
            mock_error.assert_called()
            assert 'Loss limit exceeded' in str(mock_error.call_args)
    
    @pytest.mark.asyncio
    async def test_market_making_loop(self, hft_engine):
        """Test market making loop."""
        # Mock market maker quote generation
        mock_bid = HFTOrder(
            order_id='MM_BID',
            symbol='AAPL',
            side='buy',
            order_type='limit',
            price=Decimal('149.99'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        mock_ask = HFTOrder(
            order_id='MM_ASK',
            symbol='AAPL',
            side='sell',
            order_type='limit',
            price=Decimal('150.01'),
            quantity=Decimal('100'),
            time_in_force='IOC'
        )
        
        with patch.object(hft_engine.market_makers['AAPL'], 'calculate_quotes',
                         return_value=(mock_bid, mock_ask)):
            with patch.object(hft_engine, '_send_order') as mock_send:
                # Run one iteration
                await hft_engine._market_making_loop()
                
                # Should send both orders
                assert mock_send.call_count == 2
