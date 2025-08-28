"""Tests for portfolio management endpoints and services."""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.portfolio import Portfolio
from app.models.trade import Trade
from app.models.position import Position
from app.services.portfolio import PortfolioService
from app.schemas.portfolio import PortfolioCreate, PortfolioUpdate


class TestPortfolioEndpoints:
    """Test cases for portfolio endpoints."""

    @pytest.mark.asyncio
    async def test_create_portfolio(self, client: AsyncClient, auth_headers: dict):
        """Test creating a new portfolio."""
        portfolio_data = {
            "name": "Growth Portfolio",
            "description": "Long-term growth focused portfolio",
            "initial_balance": 50000.0
        }
        
        response = await client.post(
            "/api/v1/portfolios/",
            json=portfolio_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == portfolio_data["name"]
        assert data["description"] == portfolio_data["description"]
        assert data["initial_balance"] == portfolio_data["initial_balance"]
        assert data["current_balance"] == portfolio_data["initial_balance"]
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_portfolio_invalid_balance(self, client: AsyncClient, auth_headers: dict):
        """Test creating portfolio with invalid balance."""
        portfolio_data = {
            "name": "Invalid Portfolio",
            "description": "Test",
            "initial_balance": -1000.0
        }
        
        response = await client.post(
            "/api/v1/portfolios/",
            json=portfolio_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_portfolios(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test getting user portfolios."""
        response = await client.get("/api/v1/portfolios/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(p["id"] == str(test_portfolio.id) for p in data)

    @pytest.mark.asyncio
    async def test_get_portfolio_by_id(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test getting specific portfolio."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(test_portfolio.id)
        assert data["name"] == test_portfolio.name

    @pytest.mark.asyncio
    async def test_get_portfolio_unauthorized(self, client: AsyncClient, test_portfolio: Portfolio):
        """Test accessing portfolio without auth."""
        response = await client.get(f"/api/v1/portfolios/{test_portfolio.id}")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_update_portfolio(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test updating portfolio."""
        update_data = {
            "name": "Updated Portfolio",
            "description": "Updated description"
        }
        
        response = await client.put(
            f"/api/v1/portfolios/{test_portfolio.id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]

    @pytest.mark.asyncio
    async def test_delete_portfolio(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test deleting portfolio."""
        response = await client.delete(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify deletion
        get_response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=auth_headers
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_portfolio_performance(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test getting portfolio performance metrics."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/performance",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_return" in data
        assert "total_return_percentage" in data
        assert "daily_returns" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data

    @pytest.mark.asyncio
    async def test_get_portfolio_positions(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test getting portfolio positions."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_portfolio_transactions(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test getting portfolio transactions."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/transactions",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict):
        """Test portfolio rebalancing."""
        target_allocation = {
            "AAPL": 0.3,
            "GOOGL": 0.3,
            "MSFT": 0.2,
            "AMZN": 0.2
        }
        
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/rebalance",
            json={"target_allocation": target_allocation},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "trades" in data
        assert isinstance(data["trades"], list)


class TestPortfolioService:
    """Test cases for portfolio service."""

    @pytest.mark.asyncio
    async def test_calculate_portfolio_value(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test calculating total portfolio value."""
        service = PortfolioService(db)
        
        # Add some positions
        positions = [
            Position(
                portfolio_id=test_portfolio.id,
                symbol="AAPL",
                quantity=100,
                average_price=150.0,
                current_price=155.0
            ),
            Position(
                portfolio_id=test_portfolio.id,
                symbol="GOOGL",
                quantity=50,
                average_price=2800.0,
                current_price=2850.0
            )
        ]
        
        for pos in positions:
            db.add(pos)
        await db.commit()
        
        value = await service.calculate_portfolio_value(test_portfolio.id)
        expected_value = (100 * 155.0) + (50 * 2850.0) + test_portfolio.current_balance
        assert value == expected_value

    @pytest.mark.asyncio
    async def test_calculate_returns(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test calculating portfolio returns."""
        service = PortfolioService(db)
        
        # Simulate portfolio growth
        test_portfolio.current_balance = 12000.0  # 20% gain from 10000
        await db.commit()
        
        returns = await service.calculate_returns(test_portfolio.id)
        assert returns["total_return"] == 2000.0
        assert returns["total_return_percentage"] == 20.0

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test calculating risk metrics."""
        service = PortfolioService(db)
        
        # Add historical performance data
        daily_returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005]
        
        metrics = service.calculate_risk_metrics(daily_returns)
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics

    @pytest.mark.asyncio
    async def test_position_management(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test position creation and updates."""
        service = PortfolioService(db)
        
        # Create a new position
        position = await service.create_position(
            portfolio_id=test_portfolio.id,
            symbol="TSLA",
            quantity=50,
            price=700.0
        )
        
        assert position.symbol == "TSLA"
        assert position.quantity == 50
        assert position.average_price == 700.0
        
        # Update position (buy more)
        updated_position = await service.update_position(
            position_id=position.id,
            quantity=25,
            price=720.0,
            action="buy"
        )
        
        assert updated_position.quantity == 75
        assert updated_position.average_price == 706.67  # Weighted average

    @pytest.mark.asyncio
    async def test_trade_execution(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test trade execution and portfolio updates."""
        service = PortfolioService(db)
        
        # Execute a buy trade
        trade = await service.execute_trade(
            portfolio_id=test_portfolio.id,
            symbol="NVDA",
            side="buy",
            quantity=10,
            price=450.0
        )
        
        assert trade.symbol == "NVDA"
        assert trade.side == "buy"
        assert trade.quantity == 10
        assert trade.price == 450.0
        assert trade.total_amount == 4500.0
        
        # Check portfolio balance updated
        await db.refresh(test_portfolio)
        assert test_portfolio.current_balance == 5500.0  # 10000 - 4500

    @pytest.mark.asyncio
    async def test_portfolio_allocation(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test portfolio allocation calculations."""
        service = PortfolioService(db)
        
        # Add positions
        positions = [
            Position(
                portfolio_id=test_portfolio.id,
                symbol="AAPL",
                quantity=100,
                current_price=150.0  # $15,000
            ),
            Position(
                portfolio_id=test_portfolio.id,
                symbol="GOOGL",
                quantity=10,
                current_price=2800.0  # $28,000
            )
        ]
        
        for pos in positions:
            db.add(pos)
        await db.commit()
        
        allocation = await service.get_portfolio_allocation(test_portfolio.id)
        
        total_value = 15000 + 28000 + 10000  # positions + cash
        assert allocation["AAPL"] == pytest.approx(15000 / total_value, rel=1e-3)
        assert allocation["GOOGL"] == pytest.approx(28000 / total_value, rel=1e-3)
        assert allocation["CASH"] == pytest.approx(10000 / total_value, rel=1e-3)

    @pytest.mark.asyncio
    async def test_portfolio_constraints(self, db: AsyncSession, test_portfolio: Portfolio):
        """Test portfolio constraints and validations."""
        service = PortfolioService(db)
        
        # Test insufficient balance
        with pytest.raises(ValueError, match="Insufficient balance"):
            await service.execute_trade(
                portfolio_id=test_portfolio.id,
                symbol="AAPL",
                side="buy",
                quantity=100,
                price=150.0  # $15,000 > $10,000 balance
            )
        
        # Test selling non-existent position
        with pytest.raises(ValueError, match="No position found"):
            await service.execute_trade(
                portfolio_id=test_portfolio.id,
                symbol="TSLA",
                side="sell",
                quantity=10,
                price=700.0
            )