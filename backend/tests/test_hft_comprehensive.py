"""Comprehensive HFT engine tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.hft import HFTOrder, OrderBook, MarketMaker
from app.models.portfolio import Portfolio
from fastapi import status
import asyncio
import time


class TestHFTEngine:
    """Test HFT engine with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_order_book_initialization(self, client: AsyncClient, auth_headers: dict):
        """Test order book initialization."""
        response = await client.get(
            "/api/v1/hft/orderbook/AAPL",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bids" in data
        assert "asks" in data
        assert "spread" in data
        assert "mid_price" in data

    @pytest.mark.asyncio
    async def test_hft_order_placement_speed(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test HFT order placement latency."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "limit",
            "price": 149.99,
            "time_in_force": "IOC"  # Immediate or Cancel
        }
        
        start_time = time.perf_counter()
        response = await client.post(
            "/api/v1/hft/orders",
            json=order_data,
            headers=auth_headers
        )
        latency = (time.perf_counter() - start_time) * 1000  # milliseconds
        
        assert response.status_code == status.HTTP_201_CREATED
        assert latency < 10  # Should be under 10ms

    @pytest.mark.asyncio
    async def test_market_making_strategy(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test market making strategy initialization."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "spread_bps": 5,  # 5 basis points
            "position_limit": 10000,
            "inventory_target": 0,
            "max_order_size": 500
        }
        response = await client.post(
            "/api/v1/hft/strategies/market-making",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["strategy_type"] == "market_making"
        assert data["is_active"] == True

    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, client: AsyncClient, auth_headers: dict):
        """Test arbitrage opportunity detection."""
        response = await client.get(
            "/api/v1/hft/arbitrage/opportunities",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "opportunities" in data
        for opp in data["opportunities"]:
            assert "symbol" in opp
            assert "profit_bps" in opp
            assert "venues" in opp

    @pytest.mark.asyncio
    async def test_triangular_arbitrage(self, client: AsyncClient, auth_headers: dict):
        """Test triangular arbitrage in forex/crypto."""
        response = await client.get(
            "/api/v1/hft/arbitrage/triangular",
            params={"base_currency": "USD"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "triangles" in data

    @pytest.mark.asyncio
    async def test_statistical_arbitrage(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test statistical arbitrage pair trading."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "pair": ["AAPL", "MSFT"],
            "lookback_period": 60,
            "z_score_entry": 2.0,
            "z_score_exit": 0.5,
            "position_size": 10000
        }
        response = await client.post(
            "/api/v1/hft/strategies/stat-arb",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_order_book_imbalance(self, client: AsyncClient, auth_headers: dict):
        """Test order book imbalance calculation."""
        response = await client.get(
            "/api/v1/hft/orderbook/AAPL/imbalance",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "imbalance_ratio" in data
        assert "pressure" in data
        assert data["pressure"] in ["buy", "sell", "neutral"]

    @pytest.mark.asyncio
    async def test_micro_price_calculation(self, client: AsyncClient, auth_headers: dict):
        """Test micro price calculation from order book."""
        response = await client.get(
            "/api/v1/hft/orderbook/AAPL/microprice",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "microprice" in data
        assert "weighted_mid" in data

    @pytest.mark.asyncio
    async def test_latency_arbitrage(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test latency arbitrage detection."""
        response = await client.get(
            "/api/v1/hft/latency-arb/opportunities",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "opportunities" in data

    @pytest.mark.asyncio
    async def test_order_flow_toxicity(self, client: AsyncClient, auth_headers: dict):
        """Test order flow toxicity metrics."""
        response = await client.get(
            "/api/v1/hft/analytics/toxicity",
            params={"symbol": "AAPL", "window": 100},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "vpin" in data  # Volume-synchronized PIN
        assert "toxic_ratio" in data

    @pytest.mark.asyncio
    async def test_smart_order_routing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test smart order routing optimization."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 10000,  # Large order
            "side": "buy",
            "urgency": "high",
            "max_slippage_bps": 5
        }
        response = await client.post(
            "/api/v1/hft/orders/smart-route",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "route_plan" in data
        assert "expected_slippage" in data

    @pytest.mark.asyncio
    async def test_dark_pool_routing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test dark pool order routing."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 50000,
            "side": "buy",
            "venue_preference": "dark",
            "min_fill_size": 1000
        }
        response = await client.post(
            "/api/v1/hft/orders/dark-pool",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_quote_stuffing_detection(self, client: AsyncClient, auth_headers: dict):
        """Test quote stuffing detection."""
        response = await client.get(
            "/api/v1/hft/surveillance/quote-stuffing",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "alerts" in data

    @pytest.mark.asyncio
    async def test_spoofing_detection(self, client: AsyncClient, auth_headers: dict):
        """Test spoofing detection in order flow."""
        response = await client.get(
            "/api/v1/hft/surveillance/spoofing",
            params={"symbol": "AAPL", "lookback_seconds": 60},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_momentum_ignition_detection(self, client: AsyncClient, auth_headers: dict):
        """Test momentum ignition pattern detection."""
        response = await client.get(
            "/api/v1/hft/patterns/momentum-ignition",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detected_patterns" in data

    @pytest.mark.asyncio
    async def test_inventory_management(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test HFT inventory management."""
        response = await client.get(
            f"/api/v1/hft/inventory/{test_portfolio.id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "positions" in data
        assert "risk_metrics" in data
        assert "inventory_score" in data

    @pytest.mark.asyncio
    async def test_position_netting(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test automatic position netting."""
        # Create offsetting positions
        orders = [
            {"symbol": "AAPL", "quantity": 100, "side": "buy"},
            {"symbol": "AAPL", "quantity": 50, "side": "sell"},
            {"symbol": "AAPL", "quantity": 75, "side": "buy"},
            {"symbol": "AAPL", "quantity": 125, "side": "sell"}
        ]
        
        for order in orders:
            order["portfolio_id"] = str(test_portfolio.id)
            order["order_type"] = "market"
            await client.post("/api/v1/hft/orders", json=order, headers=auth_headers)
        
        # Check net position
        response = await client.get(
            f"/api/v1/hft/positions/{test_portfolio.id}/net",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["net_position"] == 0  # Should be flat

    @pytest.mark.asyncio
    async def test_tick_data_processing(self, client: AsyncClient, auth_headers: dict):
        """Test high-frequency tick data processing."""
        # Send batch of ticks
        tick_data = {
            "symbol": "AAPL",
            "ticks": [
                {"price": 150.00, "size": 100, "timestamp": time.time()},
                {"price": 150.01, "size": 200, "timestamp": time.time() + 0.001},
                {"price": 149.99, "size": 150, "timestamp": time.time() + 0.002}
            ]
        }
        response = await client.post(
            "/api/v1/hft/ticks/process",
            json=tick_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_order_book_reconstruction(self, client: AsyncClient, auth_headers: dict):
        """Test order book reconstruction from market data."""
        response = await client.post(
            "/api/v1/hft/orderbook/reconstruct",
            json={
                "symbol": "AAPL",
                "timestamp": datetime.now().isoformat(),
                "depth": 10
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test circuit breaker activation."""
        # Trigger circuit breaker with rapid losses
        for i in range(10):
            order_data = {
                "portfolio_id": str(test_portfolio.id),
                "symbol": "VOLATILE",
                "quantity": 1000,
                "side": "buy" if i % 2 == 0 else "sell",
                "order_type": "market",
                "simulated_loss": 1000  # Simulate loss
            }
            response = await client.post(
                "/api/v1/hft/orders",
                json=order_data,
                headers=auth_headers
            )
        
        # Check if circuit breaker activated
        response = await client.get(
            f"/api/v1/hft/circuit-breaker/{test_portfolio.id}/status",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_active"] == True

    @pytest.mark.asyncio
    async def test_kill_switch(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test emergency kill switch."""
        response = await client.post(
            f"/api/v1/hft/kill-switch/{test_portfolio.id}/activate",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Verify all strategies stopped
        strategies_response = await client.get(
            f"/api/v1/hft/strategies/{test_portfolio.id}",
            headers=auth_headers
        )
        strategies = strategies_response.json()
        assert all(not s["is_active"] for s in strategies)

    @pytest.mark.asyncio
    async def test_colocation_benefits(self, client: AsyncClient, auth_headers: dict):
        """Test colocation server benefits simulation."""
        response = await client.get(
            "/api/v1/hft/infrastructure/colocation-benefits",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "latency_reduction_ms" in data
        assert "fill_rate_improvement" in data

    @pytest.mark.asyncio
    async def test_order_types_performance(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test various HFT-specific order types."""
        order_types = [
            {"type": "peg_primary", "offset": -0.01},
            {"type": "peg_midpoint", "offset": 0},
            {"type": "hidden", "display_size": 0},
            {"type": "discretionary", "discretion_amount": 0.05},
            {"type": "minimum_quantity", "min_qty": 100}
        ]
        
        for order_type in order_types:
            order_data = {
                "portfolio_id": str(test_portfolio.id),
                "symbol": "AAPL",
                "quantity": 200,
                "side": "buy",
                "order_type": "limit",
                "price": 150.00,
                "special_type": order_type["type"],
                **{k: v for k, v in order_type.items() if k != "type"}
            }
            response = await client.post(
                "/api/v1/hft/orders",
                json=order_data,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_queue_position_estimation(self, client: AsyncClient, auth_headers: dict):
        """Test queue position estimation in order book."""
        order_data = {
            "symbol": "AAPL",
            "price": 150.00,
            "size": 100,
            "side": "buy"
        }
        response = await client.post(
            "/api/v1/hft/analytics/queue-position",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "estimated_position" in data
        assert "expected_fill_time" in data

    @pytest.mark.asyncio
    async def test_adaptive_market_making(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adaptive market making with volatility adjustments."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "base_spread_bps": 5,
            "volatility_multiplier": 2.0,
            "inventory_skew": 0.5,
            "adaptive": True
        }
        response = await client.post(
            "/api/v1/hft/strategies/adaptive-mm",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_market_impact_model(self, client: AsyncClient, auth_headers: dict):
        """Test market impact estimation."""
        impact_data = {
            "symbol": "AAPL",
            "order_size": 100000,
            "avg_daily_volume": 50000000,
            "volatility": 0.02,
            "urgency": "medium"
        }
        response = await client.post(
            "/api/v1/hft/analytics/market-impact",
            json=impact_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "temporary_impact_bps" in data
        assert "permanent_impact_bps" in data

    @pytest.mark.asyncio
    async def test_cross_venue_analytics(self, client: AsyncClient, auth_headers: dict):
        """Test cross-venue market analytics."""
        response = await client.get(
            "/api/v1/hft/analytics/cross-venue",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "venues" in data
        assert "best_bid_venue" in data
        assert "best_ask_venue" in data

    @pytest.mark.asyncio
    async def test_liquidity_detection(self, client: AsyncClient, auth_headers: dict):
        """Test hidden liquidity detection."""
        response = await client.get(
            "/api/v1/hft/liquidity/hidden",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "iceberg_probability" in data
        assert "hidden_volume_estimate" in data

    @pytest.mark.asyncio
    async def test_execution_quality_metrics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test execution quality metrics."""
        response = await client.get(
            f"/api/v1/hft/metrics/execution-quality/{test_portfolio.id}",
            params={"period": "1h"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "implementation_shortfall" in data
        assert "effective_spread" in data
        assert "price_improvement" in data
        assert "fill_rate" in data
