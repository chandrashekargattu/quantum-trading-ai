"""Tests for trading endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.models.portfolio import Portfolio
from app.models.stock import Stock


class TestTrading:
    """Test trading endpoints."""
    
    @pytest.fixture
    async def test_stock(self, db_session: AsyncSession) -> Stock:
        """Create a test stock."""
        stock = Stock(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            current_price=150.0,
            previous_close=148.0,
            change_amount=2.0,
            change_percent=1.35,
            volume=1000000,
            is_active=True,
            is_optionable=True
        )
        db_session.add(stock)
        await db_session.commit()
        await db_session.refresh(stock)
        return stock
    
    @pytest.mark.asyncio
    async def test_place_market_order(
        self, 
        client: TestClient, 
        test_portfolio: Portfolio, 
        test_stock: Stock,
        auth_headers: dict
    ):
        """Test placing a market order."""
        order_data = {
            "symbol": test_stock.symbol,
            "asset_type": "stock",
            "side": "buy",
            "quantity": 10,
            "order_type": "market"
        }
        
        response = client.post(
            f"{settings.API_V1_STR}/trades/order",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "order_id" in data
        assert data["status"] in ["filled", "pending"]
    
    @pytest.mark.asyncio
    async def test_place_limit_order(
        self,
        client: TestClient,
        test_stock: Stock,
        auth_headers: dict
    ):
        """Test placing a limit order."""
        order_data = {
            "symbol": test_stock.symbol,
            "asset_type": "stock",
            "side": "buy",
            "quantity": 5,
            "order_type": "limit",
            "limit_price": 145.0,
            "time_in_force": "day"
        }
        
        response = client.post(
            f"{settings.API_V1_STR}/trades/order",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_place_order_insufficient_funds(
        self,
        client: TestClient,
        test_stock: Stock,
        auth_headers: dict
    ):
        """Test placing order with insufficient funds."""
        order_data = {
            "symbol": test_stock.symbol,
            "asset_type": "stock",
            "side": "buy",
            "quantity": 10000,  # Too many shares
            "order_type": "market"
        }
        
        response = client.post(
            f"{settings.API_V1_STR}/trades/order",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "Insufficient buying power" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_place_order_invalid_symbol(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test placing order with invalid symbol."""
        order_data = {
            "symbol": "INVALID",
            "asset_type": "stock",
            "side": "buy",
            "quantity": 1,
            "order_type": "market"
        }
        
        response = client.post(
            f"{settings.API_V1_STR}/trades/order",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_trades(self, client: TestClient, auth_headers: dict):
        """Test getting trade history."""
        response = client.get(
            f"{settings.API_V1_STR}/trades",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_trades_with_filters(
        self,
        client: TestClient,
        test_stock: Stock,
        auth_headers: dict
    ):
        """Test getting trades with filters."""
        response = client.get(
            f"{settings.API_V1_STR}/trades",
            params={
                "symbol": test_stock.symbol,
                "status": "filled",
                "limit": 10
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client: TestClient, test_stock: Stock, auth_headers: dict):
        """Test canceling an order."""
        # First place a limit order
        order_response = client.post(
            f"{settings.API_V1_STR}/trades/order",
            json={
                "symbol": test_stock.symbol,
                "asset_type": "stock",
                "side": "buy",
                "quantity": 1,
                "order_type": "limit",
                "limit_price": 140.0
            },
            headers=auth_headers
        )
        order_id = order_response.json()["order_id"]
        
        # Cancel it
        response = client.delete(
            f"{settings.API_V1_STR}/trades/order/{order_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "cancelled successfully" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, client: TestClient, auth_headers: dict):
        """Test canceling non-existent order."""
        fake_order_id = "fake-order-123"
        response = client.delete(
            f"{settings.API_V1_STR}/trades/order/{fake_order_id}",
            headers=auth_headers
        )
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_open_orders(self, client: TestClient, auth_headers: dict):
        """Test getting open orders."""
        response = client.get(
            f"{settings.API_V1_STR}/trades/orders/open",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_order_history(self, client: TestClient, auth_headers: dict):
        """Test getting order history."""
        response = client.get(
            f"{settings.API_V1_STR}/trades/orders/history",
            params={"limit": 20},
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
