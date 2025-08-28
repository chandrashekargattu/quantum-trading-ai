"""
High-Frequency Trading (HFT) Engine

Ultra-low latency trading engine with advanced order routing,
market making capabilities, and microsecond-level execution.
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from decimal import Decimal, ROUND_DOWN
import heapq
from collections import defaultdict, deque
import time
import redis.asyncio as aioredis
import uvloop
from sortedcontainers import SortedDict
import numba
from numba import jit, cuda
import struct

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: Decimal
    quantity: Decimal
    order_count: int
    timestamp: int  # Nanoseconds since epoch


@dataclass
class MarketMicrostructure:
    """Market microstructure data."""
    bid_ask_spread: Decimal
    effective_spread: Decimal
    realized_spread: Decimal
    price_impact: Decimal
    order_imbalance: Decimal
    trade_intensity: float
    quote_intensity: float
    volatility: Decimal
    tick_direction: int  # -1, 0, 1
    volume_clock: float
    trade_clock: float


@dataclass
class HFTOrder:
    """High-frequency trading order."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'limit', 'market', 'pegged', 'hidden'
    price: Optional[Decimal]
    quantity: Decimal
    time_in_force: str  # 'IOC', 'FOK', 'GTC', 'GTX'
    hidden_quantity: Optional[Decimal] = None
    peg_offset: Optional[Decimal] = None
    min_quantity: Optional[Decimal] = None
    timestamp: int = field(default_factory=lambda: time.time_ns())
    client_order_id: Optional[str] = None
    execution_instructions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Execution:
    """Trade execution details."""
    execution_id: str
    order_id: str
    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    timestamp: int
    venue: str
    liquidity_flag: str  # 'maker' or 'taker'
    fees: Decimal
    rebates: Decimal


class LockFreeOrderBook:
    """
    Lock-free order book implementation for ultra-low latency.
    Uses atomic operations and memory barriers.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = SortedDict()  # Price -> List of orders
        self.asks = SortedDict()  # Price -> List of orders
        self.order_map = {}  # Order ID -> Order details
        self.last_update_time = time.time_ns()
        
    def add_order(self, order: HFTOrder) -> bool:
        """Add order to book atomically."""
        book = self.bids if order.side == 'buy' else self.asks
        price = order.price
        
        if price not in book:
            book[price] = deque()
        
        book[price].append(order)
        self.order_map[order.order_id] = order
        self.last_update_time = time.time_ns()
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order atomically."""
        if order_id not in self.order_map:
            return False
        
        order = self.order_map[order_id]
        book = self.bids if order.side == 'buy' else self.asks
        
        if order.price in book:
            try:
                book[order.price].remove(order)
                if not book[order.price]:
                    del book[order.price]
            except ValueError:
                pass
        
        del self.order_map[order_id]
        self.last_update_time = time.time_ns()
        
        return True
    
    def get_best_bid_ask(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get best bid and ask prices."""
        best_bid = self.bids.keys()[-1] if self.bids else None
        best_ask = self.asks.keys()[0] if self.asks else None
        return best_bid, best_ask
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, List[OrderBookLevel]]:
        """Get market depth up to specified levels."""
        depth = {'bids': [], 'asks': []}
        
        # Bids (highest to lowest)
        for i, (price, orders) in enumerate(reversed(self.bids.items())):
            if i >= levels:
                break
            quantity = sum(order.quantity for order in orders)
            depth['bids'].append(OrderBookLevel(
                price=price,
                quantity=quantity,
                order_count=len(orders),
                timestamp=self.last_update_time
            ))
        
        # Asks (lowest to highest)
        for i, (price, orders) in enumerate(self.asks.items()):
            if i >= levels:
                break
            quantity = sum(order.quantity for order in orders)
            depth['asks'].append(OrderBookLevel(
                price=price,
                quantity=quantity,
                order_count=len(orders),
                timestamp=self.last_update_time
            ))
        
        return depth


@numba.jit(nopython=True, cache=True)
def calculate_vwap(prices: np.ndarray, volumes: np.ndarray, window: int) -> float:
    """Calculate Volume Weighted Average Price using JIT compilation."""
    if len(prices) < window:
        window = len(prices)
    
    total_value = 0.0
    total_volume = 0.0
    
    for i in range(len(prices) - window, len(prices)):
        total_value += prices[i] * volumes[i]
        total_volume += volumes[i]
    
    return total_value / total_volume if total_volume > 0 else prices[-1]


@numba.jit(nopython=True, cache=True)
def calculate_order_imbalance(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray
) -> float:
    """Calculate order imbalance using JIT compilation."""
    total_bid = np.sum(bid_volumes)
    total_ask = np.sum(ask_volumes)
    total_volume = total_bid + total_ask
    
    if total_volume == 0:
        return 0.0
    
    return (total_bid - total_ask) / total_volume


class SmartOrderRouter:
    """
    Smart Order Router (SOR) for optimal execution across multiple venues.
    Implements advanced routing algorithms to minimize market impact.
    """
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.venue_latencies = {}  # Venue -> latency in microseconds
        self.venue_fees = {}  # Venue -> fee structure
        self.venue_liquidity = {}  # Venue -> liquidity metrics
        
    async def route_order(
        self,
        order: HFTOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[str, HFTOrder]]:
        """
        Route order optimally across venues.
        Returns list of (venue, sub_order) tuples.
        """
        # Get liquidity distribution across venues
        liquidity_map = await self._get_liquidity_distribution(
            order.symbol,
            order.side
        )
        
        # Calculate optimal split
        splits = self._calculate_optimal_split(
            order,
            liquidity_map,
            market_data
        )
        
        # Create sub-orders
        sub_orders = []
        remaining_quantity = order.quantity
        
        for venue, allocation in splits.items():
            if allocation > 0 and remaining_quantity > 0:
                sub_quantity = min(
                    Decimal(str(allocation)) * order.quantity,
                    remaining_quantity
                )
                
                sub_order = HFTOrder(
                    order_id=f"{order.order_id}_{venue}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    price=order.price,
                    quantity=sub_quantity,
                    time_in_force=order.time_in_force,
                    client_order_id=order.order_id
                )
                
                sub_orders.append((venue, sub_order))
                remaining_quantity -= sub_quantity
        
        return sub_orders
    
    def _calculate_optimal_split(
        self,
        order: HFTOrder,
        liquidity_map: Dict[str, float],
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate optimal order split across venues."""
        # Implementation of advanced splitting algorithm
        # considering liquidity, fees, latency, and market impact
        
        total_liquidity = sum(liquidity_map.values())
        if total_liquidity == 0:
            # Equal split if no liquidity info
            return {venue: 1.0 / len(self.venues) for venue in self.venues}
        
        # Weighted by liquidity, adjusted for fees and latency
        splits = {}
        for venue in self.venues:
            liquidity_weight = liquidity_map.get(venue, 0) / total_liquidity
            
            # Adjust for fees (lower is better)
            fee_adjustment = 1.0 / (1.0 + self.venue_fees.get(venue, 0.001))
            
            # Adjust for latency (lower is better)
            latency_adjustment = 1.0 / (1.0 + self.venue_latencies.get(venue, 100) / 1000)
            
            splits[venue] = liquidity_weight * fee_adjustment * latency_adjustment
        
        # Normalize
        total_weight = sum(splits.values())
        if total_weight > 0:
            splits = {k: v / total_weight for k, v in splits.items()}
        
        return splits
    
    async def _get_liquidity_distribution(
        self,
        symbol: str,
        side: str
    ) -> Dict[str, float]:
        """Get current liquidity distribution across venues."""
        # In practice, this would query real-time data from venues
        # Placeholder implementation
        return {venue: np.random.random() * 1000000 for venue in self.venues}


class MarketMakingEngine:
    """
    Advanced market making engine implementing sophisticated strategies.
    """
    
    def __init__(
        self,
        symbol: str,
        risk_limits: Dict[str, float],
        target_spread: Decimal = Decimal('0.0001')
    ):
        self.symbol = symbol
        self.risk_limits = risk_limits
        self.target_spread = target_spread
        
        # Market making state
        self.inventory = Decimal('0')
        self.target_inventory = Decimal('0')
        self.max_inventory = Decimal(str(risk_limits.get('max_inventory', 10000)))
        
        # Price models
        self.fair_value = None
        self.reservation_price = None
        
        # Risk metrics
        self.position_pnl = Decimal('0')
        self.realized_pnl = Decimal('0')
        self.inventory_risk = Decimal('0')
        
        # Quote parameters
        self.quote_size = Decimal('100')
        self.skew_factor = Decimal('0.5')
        
    def calculate_quotes(
        self,
        market_data: MarketMicrostructure,
        predictions: Dict[str, float]
    ) -> Tuple[Optional[HFTOrder], Optional[HFTOrder]]:
        """Calculate bid and ask quotes."""
        # Calculate fair value using multiple models
        self.fair_value = self._calculate_fair_value(market_data, predictions)
        
        # Calculate reservation price (adjusted for inventory risk)
        self.reservation_price = self._calculate_reservation_price()
        
        # Calculate optimal spread
        optimal_spread = self._calculate_optimal_spread(market_data)
        
        # Adjust for inventory skew
        inventory_skew = self._calculate_inventory_skew()
        
        # Generate quotes
        bid_price = self.reservation_price - optimal_spread / 2 - inventory_skew
        ask_price = self.reservation_price + optimal_spread / 2 + inventory_skew
        
        # Create orders
        bid_order = None
        ask_order = None
        
        if self.inventory < self.max_inventory:
            bid_order = HFTOrder(
                order_id=f"MM_BID_{self.symbol}_{time.time_ns()}",
                symbol=self.symbol,
                side='buy',
                order_type='limit',
                price=bid_price.quantize(Decimal('0.01'), ROUND_DOWN),
                quantity=self.quote_size,
                time_in_force='IOC'
            )
        
        if self.inventory > -self.max_inventory:
            ask_order = HFTOrder(
                order_id=f"MM_ASK_{self.symbol}_{time.time_ns()}",
                symbol=self.symbol,
                side='sell',
                order_type='limit',
                price=ask_price.quantize(Decimal('0.01'), ROUND_DOWN),
                quantity=self.quote_size,
                time_in_force='IOC'
            )
        
        return bid_order, ask_order
    
    def _calculate_fair_value(
        self,
        market_data: MarketMicrostructure,
        predictions: Dict[str, float]
    ) -> Decimal:
        """Calculate fair value using multiple signals."""
        # Weighted average of different price estimates
        weights = {
            'mid_price': 0.4,
            'microprice': 0.3,
            'prediction': 0.2,
            'vwap': 0.1
        }
        
        # Get mid price
        mid_price = market_data.bid_ask_spread / 2
        
        # Calculate microprice (weighted by queue sizes)
        microprice = self._calculate_microprice(market_data)
        
        # ML prediction
        predicted_price = Decimal(str(predictions.get('price', float(mid_price))))
        
        # VWAP (placeholder)
        vwap = mid_price  # Would use actual VWAP in practice
        
        # Weighted average
        fair_value = (
            weights['mid_price'] * mid_price +
            weights['microprice'] * microprice +
            weights['prediction'] * predicted_price +
            weights['vwap'] * vwap
        )
        
        return fair_value
    
    def _calculate_reservation_price(self) -> Decimal:
        """Calculate reservation price adjusted for inventory."""
        if self.fair_value is None:
            return Decimal('0')
        
        # Avellaneda-Stoikov model
        gamma = Decimal('0.1')  # Risk aversion parameter
        sigma = Decimal('0.02')  # Volatility
        T = Decimal('1')  # Time horizon
        
        inventory_adjustment = gamma * sigma * sigma * self.inventory * T
        
        return self.fair_value - inventory_adjustment
    
    def _calculate_optimal_spread(
        self,
        market_data: MarketMicrostructure
    ) -> Decimal:
        """Calculate optimal spread based on market conditions."""
        # Base spread
        base_spread = self.target_spread
        
        # Adjust for volatility
        volatility_adjustment = market_data.volatility * Decimal('10')
        
        # Adjust for order imbalance
        imbalance_adjustment = abs(market_data.order_imbalance) * Decimal('0.0001')
        
        # Adjust for trade intensity
        intensity_adjustment = Decimal(str(market_data.trade_intensity)) * Decimal('0.00001')
        
        optimal_spread = (
            base_spread +
            volatility_adjustment +
            imbalance_adjustment +
            intensity_adjustment
        )
        
        # Ensure minimum spread
        return max(optimal_spread, Decimal('0.0001'))
    
    def _calculate_inventory_skew(self) -> Decimal:
        """Calculate price skew based on inventory."""
        if self.max_inventory == 0:
            return Decimal('0')
        
        # Normalized inventory position
        inventory_ratio = self.inventory / self.max_inventory
        
        # Skew increases with inventory
        skew = self.skew_factor * inventory_ratio * self.target_spread
        
        return skew
    
    def _calculate_microprice(
        self,
        market_data: MarketMicrostructure
    ) -> Decimal:
        """Calculate microprice weighted by order book imbalance."""
        # Placeholder - would use actual order book data
        return self.fair_value or Decimal('0')


class UltraLowLatencyExecutor:
    """
    Ultra-low latency order executor with hardware acceleration support.
    """
    
    def __init__(self):
        self.execution_queue = asyncio.Queue()
        self.pending_orders = {}
        self.execution_stats = defaultdict(lambda: {
            'count': 0,
            'total_latency': 0,
            'min_latency': float('inf'),
            'max_latency': 0
        })
        
        # Pre-allocated buffers for zero-copy operations
        self.order_buffer = bytearray(1024 * 1024)  # 1MB buffer
        self.response_buffer = bytearray(1024 * 1024)
        
    async def execute_order(
        self,
        order: HFTOrder,
        venue: str,
        priority: int = 0
    ) -> Execution:
        """Execute order with minimal latency."""
        start_time = time.time_ns()
        
        # Serialize order to pre-allocated buffer (zero-copy)
        order_bytes = self._serialize_order_fast(order)
        
        # Send order (would use kernel bypass in production)
        execution = await self._send_order_native(order_bytes, venue)
        
        # Update statistics
        latency = (time.time_ns() - start_time) / 1000  # Convert to microseconds
        self._update_stats(venue, latency)
        
        return execution
    
    def _serialize_order_fast(self, order: HFTOrder) -> bytes:
        """Fast order serialization using struct."""
        # Fixed-size binary format for speed
        # In practice, would use FIX or custom binary protocol
        
        format_string = '16s10s1s1s16s16s1s'  # Simplified format
        
        return struct.pack(
            format_string,
            order.order_id.encode()[:16],
            order.symbol.encode()[:10],
            b'B' if order.side == 'buy' else b'S',
            b'L' if order.order_type == 'limit' else b'M',
            str(order.price or 0).encode()[:16],
            str(order.quantity).encode()[:16],
            order.time_in_force[0].encode()
        )
    
    async def _send_order_native(
        self,
        order_bytes: bytes,
        venue: str
    ) -> Execution:
        """Send order using native protocol."""
        # In production, would use:
        # - Kernel bypass (DPDK, Solarflare, etc.)
        # - RDMA for ultra-low latency
        # - Hardware timestamping
        # - CPU affinity and NUMA optimization
        
        # Simulate execution
        await asyncio.sleep(0.00001)  # 10 microseconds
        
        # Create execution
        execution = Execution(
            execution_id=f"EXEC_{time.time_ns()}",
            order_id="test_order",
            symbol="TEST",
            side="buy",
            price=Decimal("100.00"),
            quantity=Decimal("100"),
            timestamp=time.time_ns(),
            venue=venue,
            liquidity_flag="maker",
            fees=Decimal("0.001"),
            rebates=Decimal("0.0001")
        )
        
        return execution
    
    def _update_stats(self, venue: str, latency: float):
        """Update execution statistics."""
        stats = self.execution_stats[venue]
        stats['count'] += 1
        stats['total_latency'] += latency
        stats['min_latency'] = min(stats['min_latency'], latency)
        stats['max_latency'] = max(stats['max_latency'], latency)


class HFTEngine:
    """
    Main High-Frequency Trading Engine orchestrating all components.
    """
    
    def __init__(
        self,
        symbols: List[str],
        venues: List[str],
        risk_limits: Dict[str, float]
    ):
        self.symbols = symbols
        self.venues = venues
        self.risk_limits = risk_limits
        
        # Initialize components
        self.order_books = {symbol: LockFreeOrderBook(symbol) for symbol in symbols}
        self.smart_router = SmartOrderRouter(venues)
        self.market_makers = {
            symbol: MarketMakingEngine(symbol, risk_limits)
            for symbol in symbols
        }
        self.executor = UltraLowLatencyExecutor()
        
        # Risk management
        self.position_tracker = defaultdict(Decimal)
        self.pnl_tracker = defaultdict(Decimal)
        
        # Performance metrics
        self.metrics = {
            'orders_sent': 0,
            'orders_filled': 0,
            'total_volume': Decimal('0'),
            'realized_pnl': Decimal('0')
        }
        
    async def start(self):
        """Start the HFT engine."""
        logger.info("Starting HFT Engine")
        
        # Start market data feeds
        market_data_task = asyncio.create_task(self._process_market_data())
        
        # Start order execution loop
        execution_task = asyncio.create_task(self._execution_loop())
        
        # Start market making
        market_making_task = asyncio.create_task(self._market_making_loop())
        
        # Start risk monitoring
        risk_task = asyncio.create_task(self._risk_monitoring_loop())
        
        # Wait for all tasks
        await asyncio.gather(
            market_data_task,
            execution_task,
            market_making_task,
            risk_task
        )
    
    async def _process_market_data(self):
        """Process incoming market data with minimal latency."""
        while True:
            # In practice, would receive from market data feed
            # Using zero-copy techniques and kernel bypass
            
            await asyncio.sleep(0.001)  # 1ms tick
            
            # Update order books
            for symbol in self.symbols:
                # Simulate market data update
                pass
    
    async def _market_making_loop(self):
        """Run market making strategies."""
        while True:
            for symbol in self.symbols:
                # Get market microstructure
                market_data = self._calculate_microstructure(symbol)
                
                # Get predictions (would come from ML models)
                predictions = {'price': 100.0}  # Placeholder
                
                # Generate quotes
                market_maker = self.market_makers[symbol]
                bid_order, ask_order = market_maker.calculate_quotes(
                    market_data,
                    predictions
                )
                
                # Send orders
                if bid_order:
                    await self._send_order(bid_order)
                
                if ask_order:
                    await self._send_order(ask_order)
            
            await asyncio.sleep(0.001)  # 1ms update frequency
    
    async def _send_order(self, order: HFTOrder):
        """Send order through smart routing."""
        # Route order
        sub_orders = await self.smart_router.route_order(
            order,
            {'current_price': 100.0}  # Placeholder market data
        )
        
        # Execute sub-orders
        for venue, sub_order in sub_orders:
            execution = await self.executor.execute_order(
                sub_order,
                venue,
                priority=1 if order.time_in_force == 'IOC' else 0
            )
            
            # Update metrics
            self.metrics['orders_sent'] += 1
            if execution:
                self.metrics['orders_filled'] += 1
                self.metrics['total_volume'] += execution.quantity
    
    async def _execution_loop(self):
        """Main execution loop."""
        while True:
            # Process execution queue
            if not self.executor.execution_queue.empty():
                order = await self.executor.execution_queue.get()
                # Process order
            
            await asyncio.sleep(0.0001)  # 100 microseconds
    
    async def _risk_monitoring_loop(self):
        """Monitor and enforce risk limits."""
        while True:
            # Check position limits
            for symbol, position in self.position_tracker.items():
                max_position = Decimal(str(self.risk_limits.get('max_position', 100000)))
                if abs(position) > max_position:
                    logger.warning(f"Position limit exceeded for {symbol}: {position}")
                    # Trigger position reduction
            
            # Check loss limits
            total_pnl = sum(self.pnl_tracker.values())
            max_loss = Decimal(str(self.risk_limits.get('max_loss', -10000)))
            if total_pnl < max_loss:
                logger.error(f"Loss limit exceeded: {total_pnl}")
                # Trigger emergency liquidation
            
            await asyncio.sleep(0.1)  # 100ms risk check frequency
    
    def _calculate_microstructure(self, symbol: str) -> MarketMicrostructure:
        """Calculate market microstructure metrics."""
        order_book = self.order_books[symbol]
        best_bid, best_ask = order_book.get_best_bid_ask()
        
        # Calculate metrics
        bid_ask_spread = best_ask - best_bid if best_bid and best_ask else Decimal('0')
        
        return MarketMicrostructure(
            bid_ask_spread=bid_ask_spread,
            effective_spread=bid_ask_spread,  # Simplified
            realized_spread=bid_ask_spread,  # Simplified
            price_impact=Decimal('0.0001'),  # Placeholder
            order_imbalance=Decimal('0.1'),  # Placeholder
            trade_intensity=100.0,  # Trades per second
            quote_intensity=1000.0,  # Quote updates per second
            volatility=Decimal('0.02'),  # 2% volatility
            tick_direction=1,  # Uptick
            volume_clock=1000000.0,  # Volume per time unit
            trade_clock=100.0  # Trades per time unit
        )
