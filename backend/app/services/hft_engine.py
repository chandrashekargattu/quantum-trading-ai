"""High-Frequency Trading (HFT) engine implementation."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from sortedcontainers import SortedDict
import logging
from dataclasses import dataclass, field
from enum import Enum
import numba
from decimal import Decimal
import aioredis

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.hft import HFTOrder, OrderBook, HFTStrategy
from app.models.trading import OrderSide, OrderType, OrderStatus
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.services.market_data import MarketDataService
from app.api.v1.websocket import manager

logger = logging.getLogger(__name__)


class HFTOrderType(Enum):
    """HFT-specific order types."""
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    PEG = "PEG"  # Pegged to market
    ICEBERG = "ICEBERG"  # Hidden quantity
    TWAP = "TWAP"  # Time-weighted average price
    VWAP = "VWAP"  # Volume-weighted average price


@dataclass
class HFTOrderData:
    """High-performance order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: int
    timestamp: int  # Nanoseconds
    order_type: HFTOrderType
    client_id: str
    priority: int = 0
    hidden_quantity: int = 0
    
    def __lt__(self, other):
        """Price-time priority for order matching."""
        if self.side == OrderSide.BUY:
            # Buy orders: higher price has priority
            if self.price != other.price:
                return self.price > other.price
        else:
            # Sell orders: lower price has priority
            if self.price != other.price:
                return self.price < other.price
        # Same price: earlier timestamp has priority
        return self.timestamp < other.timestamp


class LockFreeOrderBook:
    """Lock-free order book implementation for ultra-low latency."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = SortedDict()  # Price -> List[HFTOrderData]
        self.asks = SortedDict()  # Price -> List[HFTOrderData]
        self.order_map = {}  # order_id -> HFTOrderData
        self.last_update_time = time.time_ns()
        
    def add_order(self, order: HFTOrderData) -> bool:
        """Add order to book with O(log n) complexity."""
        try:
            price_key = float(order.price)
            
            if order.side == OrderSide.BUY:
                if price_key not in self.bids:
                    self.bids[price_key] = deque()
                self.bids[price_key].append(order)
            else:
                if price_key not in self.asks:
                    self.asks[price_key] = deque()
                self.asks[price_key].append(order)
            
            self.order_map[order.order_id] = order
            self.last_update_time = time.time_ns()
            return True
            
        except Exception as e:
            logger.error(f"Error adding order: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> Optional[HFTOrderData]:
        """Cancel order with O(1) average complexity."""
        if order_id not in self.order_map:
            return None
        
        order = self.order_map.pop(order_id)
        price_key = float(order.price)
        
        # Remove from price level
        if order.side == OrderSide.BUY:
            if price_key in self.bids:
                self.bids[price_key] = deque(
                    o for o in self.bids[price_key] if o.order_id != order_id
                )
                if not self.bids[price_key]:
                    del self.bids[price_key]
        else:
            if price_key in self.asks:
                self.asks[price_key] = deque(
                    o for o in self.asks[price_key] if o.order_id != order_id
                )
                if not self.asks[price_key]:
                    del self.asks[price_key]
        
        self.last_update_time = time.time_ns()
        return order
    
    def get_best_bid(self) -> Optional[Tuple[Decimal, int]]:
        """Get best bid price and quantity."""
        if not self.bids:
            return None
        
        best_price = self.bids.keys()[-1]  # Highest price
        total_quantity = sum(o.quantity for o in self.bids[best_price])
        return (Decimal(str(best_price)), total_quantity)
    
    def get_best_ask(self) -> Optional[Tuple[Decimal, int]]:
        """Get best ask price and quantity."""
        if not self.asks:
            return None
        
        best_price = self.asks.keys()[0]  # Lowest price
        total_quantity = sum(o.quantity for o in self.asks[best_price])
        return (Decimal(str(best_price)), total_quantity)
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Tuple[Decimal, int]]]:
        """Get market depth up to specified levels."""
        bid_depth = []
        ask_depth = []
        
        # Get bid levels
        for i, (price, orders) in enumerate(reversed(self.bids.items())):
            if i >= levels:
                break
            total_quantity = sum(o.quantity for o in orders)
            bid_depth.append((Decimal(str(price)), total_quantity))
        
        # Get ask levels
        for i, (price, orders) in enumerate(self.asks.items()):
            if i >= levels:
                break
            total_quantity = sum(o.quantity for o in orders)
            ask_depth.append((Decimal(str(price)), total_quantity))
        
        return {
            "bids": bid_depth,
            "asks": ask_depth
        }


class HFTEngine:
    """High-frequency trading engine with ultra-low latency."""
    
    def __init__(self):
        self.order_books: Dict[str, LockFreeOrderBook] = {}
        self.active_strategies: Dict[str, HFTStrategy] = {}
        self.position_tracker: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.latency_monitor = LatencyMonitor()
        self.risk_manager = HFTRiskManager()
        self.market_service = MarketDataService()
        self.redis_client = None
        self._running = False
        
    async def initialize(self):
        """Initialize HFT engine components."""
        # Initialize Redis for inter-process communication
        self.redis_client = await aioredis.create_redis_pool('redis://localhost')
        
        # Pre-allocate memory for hot path
        self._preallocate_memory()
        
        logger.info("HFT Engine initialized")
    
    def _preallocate_memory(self):
        """Pre-allocate memory to avoid garbage collection pauses."""
        # Pre-allocate order books for common symbols
        common_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        for symbol in common_symbols:
            self.order_books[symbol] = LockFreeOrderBook(symbol)
    
    async def start(self):
        """Start HFT engine main loop."""
        self._running = True
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._market_data_handler()),
            asyncio.create_task(self._order_matching_engine()),
            asyncio.create_task(self._strategy_executor()),
            asyncio.create_task(self._risk_monitor()),
            asyncio.create_task(self._latency_reporter())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"HFT Engine error: {e}")
        finally:
            self._running = False
    
    async def stop(self):
        """Gracefully stop HFT engine."""
        self._running = False
        
        # Cancel all pending orders
        for book in self.order_books.values():
            for order_id in list(book.order_map.keys()):
                book.cancel_order(order_id)
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        logger.info("HFT Engine stopped")
    
    @numba.jit(nopython=True)
    def _calculate_mid_price(self, bid: float, ask: float) -> float:
        """JIT-compiled mid price calculation."""
        return (bid + ask) / 2.0
    
    async def submit_order(
        self,
        order: HFTOrderData,
        db: AsyncSession
    ) -> Tuple[bool, Optional[str]]:
        """Submit HFT order with nanosecond timestamp."""
        start_time = time.time_ns()
        
        try:
            # Risk checks
            if not await self.risk_manager.check_order(order, self.position_tracker):
                return False, "Risk check failed"
            
            # Get or create order book
            if order.symbol not in self.order_books:
                self.order_books[order.symbol] = LockFreeOrderBook(order.symbol)
            
            book = self.order_books[order.symbol]
            
            # Add to order book
            if not book.add_order(order):
                return False, "Failed to add order to book"
            
            # Store in database (async, non-blocking)
            asyncio.create_task(self._persist_order(order, db))
            
            # Record latency
            latency = time.time_ns() - start_time
            self.latency_monitor.record("order_submit", latency)
            
            return True, order.order_id
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return False, str(e)
    
    async def _persist_order(self, order: HFTOrderData, db: AsyncSession):
        """Persist order to database asynchronously."""
        try:
            hft_order = HFTOrder(
                id=order.order_id,
                strategy_id=order.client_id,  # Assuming client_id is strategy_id
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=float(order.price),
                status="PENDING",
                timestamp_ns=order.timestamp
            )
            db.add(hft_order)
            await db.commit()
        except Exception as e:
            logger.error(f"Error persisting order: {e}")
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> Tuple[bool, Optional[str]]:
        """Cancel HFT order."""
        start_time = time.time_ns()
        
        try:
            if symbol not in self.order_books:
                return False, "Order book not found"
            
            book = self.order_books[symbol]
            cancelled_order = book.cancel_order(order_id)
            
            if not cancelled_order:
                return False, "Order not found"
            
            # Record latency
            latency = time.time_ns() - start_time
            self.latency_monitor.record("order_cancel", latency)
            
            return True, "Order cancelled"
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False, str(e)
    
    async def _market_data_handler(self):
        """Handle incoming market data with minimal latency."""
        while self._running:
            try:
                # Subscribe to market data feed
                async for update in self._subscribe_market_data():
                    if not self._running:
                        break
                    
                    start_time = time.time_ns()
                    
                    # Update order books
                    symbol = update["symbol"]
                    if symbol in self.order_books:
                        # Trigger strategy evaluation
                        await self._evaluate_strategies(symbol, update)
                    
                    # Record latency
                    latency = time.time_ns() - start_time
                    self.latency_monitor.record("market_data_processing", latency)
                    
            except Exception as e:
                logger.error(f"Market data handler error: {e}")
                await asyncio.sleep(0.1)
    
    async def _subscribe_market_data(self):
        """Subscribe to real-time market data feed."""
        # In production, this would connect to exchange/broker feed
        while self._running:
            # Simulate market data
            for symbol in self.order_books.keys():
                if np.random.random() < 0.1:  # 10% chance of update
                    yield {
                        "symbol": symbol,
                        "bid": float(Decimal(str(100 + np.random.randn()))),
                        "ask": float(Decimal(str(100.05 + np.random.randn()))),
                        "last": float(Decimal(str(100.02 + np.random.randn()))),
                        "volume": np.random.randint(1000, 10000)
                    }
            await asyncio.sleep(0.001)  # 1ms between checks
    
    async def _order_matching_engine(self):
        """Ultra-fast order matching engine."""
        while self._running:
            try:
                # Process each order book
                for symbol, book in self.order_books.items():
                    if not book.bids or not book.asks:
                        continue
                    
                    # Check for crossing orders
                    best_bid = book.get_best_bid()
                    best_ask = book.get_best_ask()
                    
                    if best_bid and best_ask and best_bid[0] >= best_ask[0]:
                        # Match orders
                        await self._match_orders(book)
                
                # Minimal sleep to prevent CPU spinning
                await asyncio.sleep(0.0001)  # 100 microseconds
                
            except Exception as e:
                logger.error(f"Order matching error: {e}")
    
    async def _match_orders(self, book: LockFreeOrderBook):
        """Match crossing orders in the book."""
        matches = []
        
        while book.bids and book.asks:
            best_bid_price = book.bids.keys()[-1]
            best_ask_price = book.asks.keys()[0]
            
            if best_bid_price < best_ask_price:
                break
            
            # Get orders at best prices
            bid_orders = book.bids[best_bid_price]
            ask_orders = book.asks[best_ask_price]
            
            if not bid_orders or not ask_orders:
                break
            
            # Match orders (price-time priority)
            bid_order = bid_orders[0]
            ask_order = ask_orders[0]
            
            # Determine match quantity
            match_quantity = min(bid_order.quantity, ask_order.quantity)
            match_price = ask_order.price  # Passive order price
            
            # Create match record
            matches.append({
                "buy_order_id": bid_order.order_id,
                "sell_order_id": ask_order.order_id,
                "symbol": book.symbol,
                "price": match_price,
                "quantity": match_quantity,
                "timestamp": time.time_ns()
            })
            
            # Update order quantities
            bid_order.quantity -= match_quantity
            ask_order.quantity -= match_quantity
            
            # Remove filled orders
            if bid_order.quantity == 0:
                bid_orders.popleft()
                book.order_map.pop(bid_order.order_id, None)
            
            if ask_order.quantity == 0:
                ask_orders.popleft()
                book.order_map.pop(ask_order.order_id, None)
            
            # Clean up empty price levels
            if not bid_orders:
                del book.bids[best_bid_price]
            if not ask_orders:
                del book.asks[best_ask_price]
        
        # Process matches
        for match in matches:
            await self._process_match(match)
    
    async def _process_match(self, match: Dict[str, Any]):
        """Process matched orders."""
        # Update position tracking
        symbol = match["symbol"]
        quantity = match["quantity"]
        
        # Update positions (simplified - in production, track by account)
        self.position_tracker[symbol]["long"] += quantity
        self.position_tracker[symbol]["short"] -= quantity
        
        # Send execution reports
        await self._send_execution_report(match)
        
        # Record match latency
        self.latency_monitor.record("order_match", 0)  # Already measured
    
    async def _send_execution_report(self, match: Dict[str, Any]):
        """Send execution reports to clients."""
        # In production, send via FIX or other protocol
        logger.info(f"Trade executed: {match}")
    
    async def _strategy_executor(self):
        """Execute registered HFT strategies."""
        while self._running:
            try:
                for strategy_id, strategy in self.active_strategies.items():
                    if strategy.is_active:
                        await self._run_strategy(strategy)
                
                await asyncio.sleep(0.001)  # 1ms between strategy runs
                
            except Exception as e:
                logger.error(f"Strategy execution error: {e}")
    
    async def _run_strategy(self, strategy: HFTStrategy):
        """Run individual HFT strategy."""
        # This would implement specific strategy logic
        # Examples: Market making, arbitrage, momentum trading
        pass
    
    async def _evaluate_strategies(self, symbol: str, market_update: Dict[str, Any]):
        """Evaluate strategies based on market update."""
        for strategy in self.active_strategies.values():
            if strategy.is_active and symbol in strategy.config.get("symbols", []):
                # Strategy-specific evaluation logic
                if strategy.strategy_type == "MARKET_MAKING":
                    await self._evaluate_market_making(strategy, symbol, market_update)
                elif strategy.strategy_type == "ARBITRAGE":
                    await self._evaluate_arbitrage(strategy, symbol, market_update)
    
    async def _evaluate_market_making(
        self,
        strategy: HFTStrategy,
        symbol: str,
        market_update: Dict[str, Any]
    ):
        """Evaluate market making strategy."""
        # Calculate spreads and update quotes
        mid_price = self._calculate_mid_price(
            market_update["bid"],
            market_update["ask"]
        )
        
        # Determine quote prices
        spread = strategy.config.get("spread", 0.01)
        bid_price = Decimal(str(mid_price * (1 - spread)))
        ask_price = Decimal(str(mid_price * (1 + spread)))
        
        # Submit/update orders
        # Implementation depends on strategy parameters
    
    async def _evaluate_arbitrage(
        self,
        strategy: HFTStrategy,
        symbol: str,
        market_update: Dict[str, Any]
    ):
        """Evaluate arbitrage opportunities."""
        # Check for price discrepancies across venues
        # Implementation depends on available markets
        pass
    
    async def _risk_monitor(self):
        """Monitor risk metrics in real-time."""
        while self._running:
            try:
                # Check position limits
                for symbol, positions in self.position_tracker.items():
                    net_position = positions["long"] - positions["short"]
                    
                    if abs(net_position) > self.risk_manager.position_limit:
                        logger.warning(f"Position limit breach: {symbol} = {net_position}")
                        # Trigger risk reduction
                        await self._reduce_position(symbol, net_position)
                
                # Check loss limits
                # Implementation depends on P&L tracking
                
                await asyncio.sleep(0.1)  # 100ms risk check interval
                
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
    
    async def _reduce_position(self, symbol: str, current_position: int):
        """Reduce position to comply with risk limits."""
        # Cancel pending orders
        if symbol in self.order_books:
            book = self.order_books[symbol]
            for order_id in list(book.order_map.keys()):
                await self.cancel_order(order_id, symbol)
        
        # Submit offsetting orders if needed
        # Implementation depends on strategy
    
    async def _latency_reporter(self):
        """Report latency metrics periodically."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Report every second
                
                stats = self.latency_monitor.get_stats()
                logger.info(f"Latency stats: {stats}")
                
                # Send to monitoring system
                if self.redis_client:
                    await self.redis_client.publish(
                        "hft:latency",
                        str(stats)
                    )
                
            except Exception as e:
                logger.error(f"Latency reporter error: {e}")
    
    def register_strategy(self, strategy: HFTStrategy):
        """Register HFT strategy."""
        self.active_strategies[str(strategy.id)] = strategy
        logger.info(f"Registered HFT strategy: {strategy.name}")
    
    def unregister_strategy(self, strategy_id: str):
        """Unregister HFT strategy."""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Unregistered HFT strategy: {strategy_id}")
    
    async def get_order_book_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current order book snapshot."""
        if symbol not in self.order_books:
            return None
        
        book = self.order_books[symbol]
        depth = book.get_market_depth(levels=10)
        
        return {
            "symbol": symbol,
            "bids": depth["bids"],
            "asks": depth["asks"],
            "timestamp": time.time_ns(),
            "last_update": book.last_update_time
        }


class LatencyMonitor:
    """Monitor and track latency metrics."""
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def record(self, operation: str, latency_ns: int):
        """Record latency measurement in nanoseconds."""
        self.metrics[operation].append(latency_ns)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics."""
        stats = {}
        
        for operation, latencies in self.metrics.items():
            if latencies:
                latency_array = np.array(latencies)
                stats[operation] = {
                    "mean": np.mean(latency_array),
                    "median": np.median(latency_array),
                    "p95": np.percentile(latency_array, 95),
                    "p99": np.percentile(latency_array, 99),
                    "max": np.max(latency_array),
                    "count": len(latencies)
                }
        
        return stats


class HFTRiskManager:
    """Risk management for HFT operations."""
    
    def __init__(self):
        self.position_limit = 10000
        self.order_rate_limit = 1000  # Orders per second
        self.loss_limit = 10000  # Daily loss limit
        self.order_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    async def check_order(
        self,
        order: HFTOrderData,
        position_tracker: Dict[str, Dict[str, int]]
    ) -> bool:
        """Check if order passes risk controls."""
        # Check rate limit
        current_time = time.time()
        self.order_counts[order.client_id].append(current_time)
        
        # Count orders in last second
        recent_orders = sum(
            1 for t in self.order_counts[order.client_id]
            if current_time - t < 1.0
        )
        
        if recent_orders > self.order_rate_limit:
            logger.warning(f"Order rate limit exceeded for {order.client_id}")
            return False
        
        # Check position limit
        current_position = position_tracker[order.symbol]["long"] - position_tracker[order.symbol]["short"]
        
        if order.side == OrderSide.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        if abs(new_position) > self.position_limit:
            logger.warning(f"Position limit would be exceeded: {new_position}")
            return False
        
        # Additional risk checks can be added here
        # - Loss limits
        # - Concentration limits
        # - Market impact estimates
        
        return True


# Global HFT engine instance
hft_engine = HFTEngine()
