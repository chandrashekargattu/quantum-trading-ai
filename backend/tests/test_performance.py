"""
Performance tests for the Quantum Trading AI application.

These tests measure response times, throughput, and resource usage
to ensure the application meets performance requirements.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import aiohttp
import psutil
import numpy as np
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.market_data import MarketDataService
from app.services.portfolio_service import PortfolioService
from app.services.trading_engine import TradingEngine
from app.services.backtesting_engine import BacktestingEngine
from app.services.hft_engine import HFTEngine
from app.quantum.portfolio_optimizer import QuantumPortfolioOptimizer
from app.ml.model_manager import ModelManager


class TestPerformance:
    """Test suite for performance benchmarking"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_api_response_time(self, client: AsyncClient, auth_headers: dict):
        """Test API endpoint response times"""
        endpoints = [
            ("/api/v1/auth/me", "GET", None),
            ("/api/v1/market/quote/AAPL", "GET", None),
            ("/api/v1/portfolios", "GET", None),
            ("/api/v1/trades", "GET", None),
        ]
        
        results = {}
        
        for endpoint, method, data in endpoints:
            times = []
            
            # Warm up
            await client.request(method, endpoint, headers=auth_headers, json=data)
            
            # Measure response times
            for _ in range(50):
                start = time.perf_counter()
                response = await client.request(method, endpoint, headers=auth_headers, json=data)
                end = time.perf_counter()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
            
            if times:
                results[endpoint] = {
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99),
                    "min": min(times),
                    "max": max(times)
                }
        
        # Assert performance requirements
        for endpoint, metrics in results.items():
            assert metrics["median"] < 100, f"{endpoint} median response time > 100ms"
            assert metrics["p95"] < 200, f"{endpoint} p95 response time > 200ms"
            assert metrics["p99"] < 500, f"{endpoint} p99 response time > 500ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_user_load(self, client: AsyncClient):
        """Test system performance under concurrent user load"""
        async def simulate_user_session():
            """Simulate a user session with multiple API calls"""
            # Register
            register_data = {
                "email": f"user{time.time()}@example.com",
                "password": "Test123!@#",
                "username": f"user{int(time.time() * 1000)}",
                "full_name": "Test User"
            }
            response = await client.post("/api/v1/auth/register", json=register_data)
            if response.status_code != 201:
                return None
            
            # Login
            login_data = {"username": register_data["email"], "password": register_data["password"]}
            response = await client.post("/api/v1/auth/login", data=login_data)
            if response.status_code != 200:
                return None
            
            token = response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Perform various operations
            operations = [
                client.get("/api/v1/auth/me", headers=headers),
                client.get("/api/v1/market/quote/AAPL", headers=headers),
                client.get("/api/v1/portfolios", headers=headers),
                client.get("/api/v1/market/movers", headers=headers),
            ]
            
            results = await asyncio.gather(*operations, return_exceptions=True)
            return results
        
        # Test with increasing concurrent users
        for num_users in [10, 25, 50]:
            start = time.perf_counter()
            
            tasks = [simulate_user_session() for _ in range(num_users)]
            results = await asyncio.gather(*tasks)
            
            end = time.perf_counter()
            
            # Calculate success rate
            successful = sum(1 for r in results if r is not None)
            success_rate = successful / num_users
            
            # Assert requirements
            assert success_rate > 0.95, f"Success rate < 95% with {num_users} users"
            assert (end - start) < num_users * 0.5, f"Total time too high for {num_users} users"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_market_data_throughput(self):
        """Test market data service throughput"""
        service = MarketDataService()
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"] * 20  # 100 requests
        
        start = time.perf_counter()
        
        # Batch fetch quotes
        tasks = [service.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end = time.perf_counter()
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        throughput = successful / (end - start)
        
        assert throughput > 50, "Market data throughput < 50 quotes/second"
        assert successful / len(symbols) > 0.95, "Market data success rate < 95%"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_portfolio_calculation_performance(self, db: AsyncSession):
        """Test portfolio calculation performance"""
        service = PortfolioService(db)
        
        # Create test portfolio with many positions
        positions = []
        for i in range(100):
            positions.append({
                "symbol": f"STOCK{i}",
                "quantity": 100 + i * 10,
                "average_cost": 50 + i * 2,
                "current_price": 55 + i * 1.5
            })
        
        times = []
        
        for _ in range(10):
            start = time.perf_counter()
            
            # Calculate portfolio metrics
            metrics = await service.calculate_portfolio_metrics(positions)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_time = statistics.mean(times)
        assert mean_time < 50, f"Portfolio calculation mean time {mean_time}ms > 50ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_order_execution_latency(self, db: AsyncSession):
        """Test order execution latency"""
        engine = TradingEngine(db)
        
        latencies = []
        
        for _ in range(100):
            order = {
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "order_type": "MARKET",
                "portfolio_id": "test-portfolio"
            }
            
            start = time.perf_counter_ns()
            result = await engine.execute_order(order)
            end = time.perf_counter_ns()
            
            if result:
                latencies.append((end - start) / 1_000_000)  # Convert to ms
        
        if latencies:
            mean_latency = statistics.mean(latencies)
            p99_latency = np.percentile(latencies, 99)
            
            assert mean_latency < 10, f"Mean order latency {mean_latency}ms > 10ms"
            assert p99_latency < 50, f"P99 order latency {p99_latency}ms > 50ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_hft_engine_performance(self):
        """Test HFT engine performance"""
        engine = HFTEngine()
        
        # Test order book operations
        operations_per_second = []
        
        for _ in range(5):
            count = 0
            start = time.perf_counter()
            
            while time.perf_counter() - start < 1.0:
                # Add order
                engine.add_order({
                    "id": f"order-{count}",
                    "symbol": "AAPL",
                    "side": "BUY" if count % 2 == 0 else "SELL",
                    "price": 100 + (count % 10) * 0.1,
                    "quantity": 100,
                    "timestamp": time.time_ns()
                })
                
                # Match orders
                engine.match_orders("AAPL")
                
                count += 1
            
            operations_per_second.append(count)
        
        mean_ops = statistics.mean(operations_per_second)
        assert mean_ops > 10000, f"HFT engine throughput {mean_ops} ops/sec < 10,000"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_backtesting_performance(self):
        """Test backtesting engine performance"""
        engine = BacktestingEngine()
        
        # Create test data
        historical_data = []
        for i in range(252 * 5):  # 5 years of daily data
            historical_data.append({
                "date": f"2019-01-01T00:00:00Z",
                "open": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "close": 100.5 + i * 0.1,
                "volume": 1000000 + i * 1000
            })
        
        strategy = {
            "name": "simple_ma",
            "parameters": {"fast_period": 50, "slow_period": 200}
        }
        
        start = time.perf_counter()
        
        result = await engine.run_backtest(
            strategy=strategy,
            symbols=["AAPL", "GOOGL", "MSFT"],
            historical_data=historical_data,
            initial_capital=100000
        )
        
        end = time.perf_counter()
        
        execution_time = end - start
        assert execution_time < 10, f"Backtest execution time {execution_time}s > 10s"
        assert result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ml_model_inference_speed(self):
        """Test ML model inference speed"""
        manager = ModelManager()
        
        # Prepare test data
        market_data = np.random.rand(100, 50).astype(np.float32)  # 100 samples, 50 features
        
        # Test transformer model
        transformer_times = []
        for _ in range(10):
            start = time.perf_counter()
            predictions = await manager.predict_transformer(market_data)
            end = time.perf_counter()
            transformer_times.append((end - start) * 1000)
        
        mean_transformer = statistics.mean(transformer_times)
        assert mean_transformer < 100, f"Transformer inference mean time {mean_transformer}ms > 100ms"
        
        # Test batch prediction
        batch_sizes = [10, 50, 100]
        for batch_size in batch_sizes:
            batch_data = market_data[:batch_size]
            
            start = time.perf_counter()
            predictions = await manager.predict_transformer(batch_data)
            end = time.perf_counter()
            
            time_per_sample = (end - start) * 1000 / batch_size
            assert time_per_sample < 10, f"Time per sample {time_per_sample}ms > 10ms for batch size {batch_size}"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_websocket_message_throughput(self):
        """Test WebSocket message throughput"""
        messages_sent = 0
        messages_received = 0
        
        async def websocket_client():
            nonlocal messages_received
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect('ws://localhost:8000/ws/market') as ws:
                    # Subscribe to updates
                    await ws.send_json({
                        "type": "subscribe",
                        "symbols": ["AAPL", "GOOGL", "MSFT"]
                    })
                    
                    # Receive messages for 5 seconds
                    start = time.perf_counter()
                    while time.perf_counter() - start < 5:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                messages_received += 1
                        except asyncio.TimeoutError:
                            continue
        
        # Simulate market data updates
        async def send_updates():
            nonlocal messages_sent
            
            start = time.perf_counter()
            while time.perf_counter() - start < 5:
                # Broadcast price update
                await asyncio.sleep(0.01)  # 100 updates per second
                messages_sent += 1
        
        # Run client and server simulation
        await asyncio.gather(
            websocket_client(),
            send_updates()
        )
        
        throughput = messages_received / 5
        assert throughput > 50, f"WebSocket throughput {throughput} msg/sec < 50"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_query_performance(self, db: AsyncSession):
        """Test database query performance"""
        from sqlalchemy import select, func
        from app.models import User, Portfolio, Trade
        
        # Test various query patterns
        queries = [
            # Simple select
            select(User).limit(100),
            
            # Join query
            select(Portfolio).join(User).limit(100),
            
            # Aggregation
            select(func.count(Trade.id)).group_by(Trade.symbol),
            
            # Complex filter
            select(Trade).where(
                Trade.created_at >= '2024-01-01',
                Trade.status == 'EXECUTED',
                Trade.quantity > 100
            ).limit(100)
        ]
        
        for query in queries:
            times = []
            
            for _ in range(20):
                start = time.perf_counter()
                result = await db.execute(query)
                _ = result.scalars().all()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)
            
            mean_time = statistics.mean(times)
            assert mean_time < 50, f"Query mean time {mean_time}ms > 50ms"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage(self, client: AsyncClient, auth_headers: dict):
        """Test memory usage under load"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        tasks = []
        for i in range(100):
            tasks.append(
                client.get(f"/api/v1/market/history/AAPL?period=1Y", headers=auth_headers)
            )
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Force garbage collection
        import gc
        gc.collect()
        await asyncio.sleep(1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500, f"Memory increase {memory_increase}MB > 500MB"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_performance(self):
        """Test cache performance"""
        from app.core.cache import cache_manager
        
        # Test cache operations
        operations = 10000
        
        # Write performance
        start = time.perf_counter()
        for i in range(operations):
            await cache_manager.set(f"key_{i}", {"value": i, "data": "x" * 100})
        write_time = time.perf_counter() - start
        
        write_ops_per_second = operations / write_time
        assert write_ops_per_second > 5000, f"Cache write throughput {write_ops_per_second} ops/sec < 5000"
        
        # Read performance
        start = time.perf_counter()
        for i in range(operations):
            await cache_manager.get(f"key_{i}")
        read_time = time.perf_counter() - start
        
        read_ops_per_second = operations / read_time
        assert read_ops_per_second > 10000, f"Cache read throughput {read_ops_per_second} ops/sec < 10000"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_quantum_algorithm_performance(self):
        """Test quantum algorithm performance"""
        optimizer = QuantumPortfolioOptimizer()
        
        # Test with different portfolio sizes
        portfolio_sizes = [4, 6, 8]  # Limited by quantum simulator
        
        for size in portfolio_sizes:
            returns = np.random.rand(size, 100) * 0.01
            
            start = time.perf_counter()
            result = await optimizer.optimize_portfolio(returns, target_return=0.10)
            end = time.perf_counter()
            
            execution_time = end - start
            assert execution_time < 30, f"Quantum optimization for {size} assets took {execution_time}s > 30s"
            assert result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_stress_test_endpoints(self, client: AsyncClient, auth_headers: dict):
        """Stress test critical endpoints"""
        critical_endpoints = [
            "/api/v1/trades",
            "/api/v1/market/quote/AAPL",
            "/api/v1/portfolios"
        ]
        
        async def hammer_endpoint(endpoint: str, duration: int = 10):
            """Send requests continuously for specified duration"""
            count = 0
            errors = 0
            start = time.perf_counter()
            
            while time.perf_counter() - start < duration:
                try:
                    response = await client.get(endpoint, headers=auth_headers)
                    if response.status_code != 200:
                        errors += 1
                    count += 1
                except Exception:
                    errors += 1
                    count += 1
            
            return count, errors
        
        results = {}
        for endpoint in critical_endpoints:
            count, errors = await hammer_endpoint(endpoint)
            error_rate = errors / count if count > 0 else 1
            
            results[endpoint] = {
                "requests": count,
                "errors": errors,
                "error_rate": error_rate,
                "rps": count / 10
            }
            
            assert error_rate < 0.01, f"{endpoint} error rate {error_rate:.2%} > 1%"
            assert results[endpoint]["rps"] > 50, f"{endpoint} RPS {results[endpoint]['rps']:.1f} < 50"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_portfolio_updates(self, db: AsyncSession, auth_headers: dict):
        """Test concurrent portfolio update performance"""
        # Create test portfolio
        portfolio_service = PortfolioService(db)
        portfolio = await portfolio_service.create_portfolio({
            "name": "Performance Test Portfolio",
            "initial_capital": 100000
        })
        
        async def update_portfolio():
            """Simulate portfolio update operation"""
            update_data = {
                "cash_balance": 95000 + np.random.randint(-1000, 1000)
            }
            return await portfolio_service.update_portfolio(
                portfolio.id,
                update_data
            )
        
        # Test concurrent updates
        concurrent_updates = 50
        start = time.perf_counter()
        
        tasks = [update_portfolio() for _ in range(concurrent_updates)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end = time.perf_counter()
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successful / concurrent_updates
        
        assert success_rate > 0.95, f"Concurrent update success rate {success_rate:.2%} < 95%"
        assert (end - start) < 5, f"Concurrent updates took {end - start:.1f}s > 5s"
