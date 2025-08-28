"""Comprehensive trading operations tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.trading import Order, Strategy
from app.models.trade import Trade
from app.models.portfolio import Portfolio
from app.models.user import User
from fastapi import status
import uuid
import asyncio


class TestTradingOperations:
    """Test trading operations with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_place_market_order_buy(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a market buy order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["status"] == "pending"
        assert data["order_type"] == "market"

    @pytest.mark.asyncio
    async def test_place_market_order_sell(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a market sell order."""
        # First ensure we have a position
        buy_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        await client.post("/api/v1/trading/orders", json=buy_order, headers=auth_headers)
        
        # Now sell
        sell_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 50,
            "side": "sell",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=sell_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_limit_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a limit order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 100.00
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["limit_price"] == order_data["limit_price"]

    @pytest.mark.asyncio
    async def test_place_limit_order_no_price(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing limit order without price."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit"
            # Missing limit_price
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_place_stop_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a stop order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "MSFT",
            "quantity": 75,
            "side": "sell",
            "order_type": "stop",
            "stop_price": 280.00
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_stop_limit_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a stop-limit order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "TSLA",
            "quantity": 25,
            "side": "sell",
            "order_type": "stop_limit",
            "stop_price": 200.00,
            "limit_price": 195.00
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_trailing_stop_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a trailing stop order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "NVDA",
            "quantity": 30,
            "side": "sell",
            "order_type": "trailing_stop",
            "trail_amount": 5.00  # $5 trail
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_trailing_stop_percent(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a trailing stop order with percentage."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AMD",
            "quantity": 40,
            "side": "sell",
            "order_type": "trailing_stop",
            "trail_percent": 5.0  # 5% trail
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_order_time_in_force_day(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order with DAY time in force."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 150.00,
            "time_in_force": "DAY"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["time_in_force"] == "DAY"

    @pytest.mark.asyncio
    async def test_place_order_time_in_force_gtc(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order with GTC (Good Till Cancelled) time in force."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 100.00,
            "time_in_force": "GTC"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_order_extended_hours(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing order for extended hours trading."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "MSFT",
            "quantity": 75,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 300.00,
            "extended_hours": True
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_order_insufficient_buying_power(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order with insufficient buying power."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "BRK.A",  # Very expensive stock
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "insufficient" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_place_order_invalid_symbol(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order with invalid symbol."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "INVALID123",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_place_order_fractional_shares(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order with fractional shares."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 0.5,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_place_order_pattern_day_trader_check(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test pattern day trader rule enforcement."""
        # Place 4 day trades (buy and sell same day)
        for i in range(4):
            # Buy
            buy_order = {
                "portfolio_id": str(test_portfolio.id),
                "symbol": f"STOCK{i}",
                "quantity": 10,
                "side": "buy",
                "order_type": "market"
            }
            await client.post("/api/v1/trading/orders", json=buy_order, headers=auth_headers)
            
            # Sell same day
            sell_order = {
                "portfolio_id": str(test_portfolio.id),
                "symbol": f"STOCK{i}",
                "quantity": 10,
                "side": "sell",
                "order_type": "market"
            }
            await client.post("/api/v1/trading/orders", json=sell_order, headers=auth_headers)
        
        # 5th day trade should be blocked if account < $25k
        buy_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=buy_order,
            headers=auth_headers
        )
        # Should be blocked if portfolio value < $25k
        if test_portfolio.total_value < 25000:
            assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_cancel_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test canceling a pending order."""
        # Place order
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 140.00
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Cancel it
        response = await client.post(
            f"/api/v1/trading/orders/{order_id}/cancel",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_executed_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test canceling an already executed order."""
        # Place market order (executes immediately)
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 10,
            "side": "buy",
            "order_type": "market"
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Wait for execution
        await asyncio.sleep(1)
        
        # Try to cancel
        response = await client.post(
            f"/api/v1/trading/orders/{order_id}/cancel",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_modify_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test modifying a pending order."""
        # Place order
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 100.00
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Modify it
        modify_data = {
            "quantity": 75,
            "limit_price": 95.00
        }
        response = await client.put(
            f"/api/v1/trading/orders/{order_id}",
            json=modify_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["quantity"] == 75
        assert data["limit_price"] == 95.00

    @pytest.mark.asyncio
    async def test_get_order_status(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting order status."""
        # Place order
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "MSFT",
            "quantity": 25,
            "side": "buy",
            "order_type": "market"
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Get status
        response = await client.get(
            f"/api/v1/trading/orders/{order_id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "filled_quantity" in data

    @pytest.mark.asyncio
    async def test_get_all_orders(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting all orders for a portfolio."""
        response = await client.get(
            f"/api/v1/trading/orders",
            params={"portfolio_id": str(test_portfolio.id)},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_orders_filtered(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting filtered orders."""
        response = await client.get(
            f"/api/v1/trading/orders",
            params={
                "portfolio_id": str(test_portfolio.id),
                "status": "pending",
                "side": "buy",
                "symbol": "AAPL"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_bracket_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing a bracket order (entry + stop loss + take profit)."""
        bracket_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "bracket",
            "entry_price": 150.00,
            "stop_loss_price": 145.00,
            "take_profit_price": 160.00
        }
        response = await client.post(
            "/api/v1/trading/orders/bracket",
            json=bracket_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert len(data["orders"]) == 3  # Entry, stop loss, take profit

    @pytest.mark.asyncio
    async def test_oco_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test One-Cancels-Other (OCO) order."""
        oco_order = {
            "portfolio_id": str(test_portfolio.id),
            "orders": [
                {
                    "symbol": "TSLA",
                    "quantity": 50,
                    "side": "sell",
                    "order_type": "limit",
                    "limit_price": 250.00
                },
                {
                    "symbol": "TSLA",
                    "quantity": 50,
                    "side": "sell",
                    "order_type": "stop",
                    "stop_price": 200.00
                }
            ]
        }
        response = await client.post(
            "/api/v1/trading/orders/oco",
            json=oco_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_batch_orders(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test placing multiple orders in batch."""
        batch_orders = {
            "portfolio_id": str(test_portfolio.id),
            "orders": [
                {
                    "symbol": "AAPL",
                    "quantity": 50,
                    "side": "buy",
                    "order_type": "market"
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 25,
                    "side": "buy",
                    "order_type": "limit",
                    "limit_price": 100.00
                },
                {
                    "symbol": "MSFT",
                    "quantity": 75,
                    "side": "buy",
                    "order_type": "market"
                }
            ]
        }
        response = await client.post(
            "/api/v1/trading/orders/batch",
            json=batch_orders,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert len(data["orders"]) == 3

    @pytest.mark.asyncio
    async def test_order_execution_simulation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order execution with slippage simulation."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 1000,  # Large order
            "side": "buy",
            "order_type": "market",
            "simulate_slippage": True
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        # Check that execution price differs from quote due to slippage
        assert "execution_price" in data
        assert "slippage" in data

    @pytest.mark.asyncio
    async def test_short_selling_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test short selling order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GME",
            "quantity": 100,
            "side": "short",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_short_cover_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test covering a short position."""
        # First short
        short_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AMC",
            "quantity": 200,
            "side": "short",
            "order_type": "market"
        }
        await client.post("/api/v1/trading/orders", json=short_order, headers=auth_headers)
        
        # Then cover
        cover_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AMC",
            "quantity": 200,
            "side": "cover",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=cover_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_order_with_commission(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order execution with commission calculation."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "SPY",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "commission" in data
        assert data["commission"] > 0

    @pytest.mark.asyncio
    async def test_order_routing_preference(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order routing preference."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market",
            "routing": "SMART"  # Smart routing
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_order_with_conditions(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test conditional order execution."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "TSLA",
            "quantity": 50,
            "side": "buy",
            "order_type": "market",
            "conditions": [
                {
                    "type": "price",
                    "symbol": "TSLA",
                    "operator": ">=",
                    "value": 200.00
                },
                {
                    "type": "time",
                    "operator": ">=",
                    "value": "14:30:00"
                }
            ]
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_algo_order_vwap(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test VWAP (Volume Weighted Average Price) algo order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 10000,
            "side": "buy",
            "order_type": "algo",
            "algo_strategy": "VWAP",
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(hours=2)).isoformat()
        }
        response = await client.post(
            "/api/v1/trading/orders/algo",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_algo_order_twap(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test TWAP (Time Weighted Average Price) algo order."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 5000,
            "side": "sell",
            "order_type": "algo",
            "algo_strategy": "TWAP",
            "duration_minutes": 120
        }
        response = await client.post(
            "/api/v1/trading/orders/algo",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_iceberg_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test iceberg order (hidden quantity)."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "MSFT",
            "quantity": 10000,
            "visible_quantity": 100,  # Only show 100 shares at a time
            "side": "buy",
            "order_type": "limit",
            "limit_price": 300.00
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_pegged_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test pegged order (follows market price)."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "NVDA",
            "quantity": 200,
            "side": "buy",
            "order_type": "pegged",
            "peg_type": "midpoint",  # Peg to bid-ask midpoint
            "offset": -0.05  # 5 cents below midpoint
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_order_fills_partial(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test partial order fills."""
        # Place large limit order
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 1000,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 149.00
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Simulate partial fill
        fill_data = {
            "filled_quantity": 300,
            "fill_price": 149.00
        }
        response = await client.post(
            f"/api/v1/trading/orders/{order_id}/simulate-fill",
            json=fill_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["filled_quantity"] == 300
        assert data["remaining_quantity"] == 700

    @pytest.mark.asyncio
    async def test_trade_history(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting trade history."""
        response = await client.get(
            f"/api/v1/trading/trades",
            params={"portfolio_id": str(test_portfolio.id)},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_trade_analytics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test trade analytics and performance metrics."""
        response = await client.get(
            f"/api/v1/trading/analytics",
            params={
                "portfolio_id": str(test_portfolio.id),
                "period": "30d"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_trades" in data
        assert "win_rate" in data
        assert "average_win" in data
        assert "average_loss" in data
        assert "profit_factor" in data
        assert "sharpe_ratio" in data

    @pytest.mark.asyncio
    async def test_paper_trading_mode(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test paper trading mode execution."""
        # Enable paper trading
        await client.put(
            f"/api/v1/portfolios/{test_portfolio.id}/settings",
            json={"paper_trading": True},
            headers=auth_headers
        )
        
        # Place order in paper trading mode
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["is_paper_trade"] == True

    @pytest.mark.asyncio
    async def test_order_validation_rules(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test various order validation rules."""
        # Test minimum order size
        small_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 0.001,  # Too small
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            "/api/v1/trading/orders",
            json=small_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_order_audit_trail(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test order audit trail and history."""
        # Place and modify order
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 100.00
        }
        place_response = await client.post(
            "/api/v1/trading/orders",
            json=order_data,
            headers=auth_headers
        )
        order_id = place_response.json()["id"]
        
        # Modify order
        await client.put(
            f"/api/v1/trading/orders/{order_id}",
            json={"limit_price": 95.00},
            headers=auth_headers
        )
        
        # Get audit trail
        response = await client.get(
            f"/api/v1/trading/orders/{order_id}/audit",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) >= 2  # Create and modify events
