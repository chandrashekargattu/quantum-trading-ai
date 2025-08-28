"""Comprehensive portfolio management tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.portfolio import Portfolio
from app.models.position import Position, Transaction
from app.models.user import User
from fastapi import status
import uuid


class TestPortfolioManagement:
    """Test portfolio management with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_create_portfolio_success(self, client: AsyncClient, auth_headers: dict):
        """Test creating a new portfolio."""
        portfolio_data = {
            "name": "Growth Portfolio",
            "description": "High growth tech stocks",
            "initial_balance": 100000.0,
            "currency": "USD"
        }
        response = await client.post(
            "/api/v1/portfolios",
            json=portfolio_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == portfolio_data["name"]
        assert data["cash_balance"] == portfolio_data["initial_balance"]

    @pytest.mark.asyncio
    async def test_create_portfolio_duplicate_name(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating portfolio with duplicate name."""
        portfolio_data = {
            "name": test_portfolio.name,  # Duplicate name
            "description": "Another portfolio",
            "initial_balance": 50000.0
        }
        response = await client.post(
            "/api/v1/portfolios",
            json=portfolio_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_create_portfolio_negative_balance(self, client: AsyncClient, auth_headers: dict):
        """Test creating portfolio with negative initial balance."""
        portfolio_data = {
            "name": "Invalid Portfolio",
            "initial_balance": -1000.0
        }
        response = await client.post(
            "/api/v1/portfolios",
            json=portfolio_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_create_portfolio_zero_balance(self, client: AsyncClient, auth_headers: dict):
        """Test creating portfolio with zero initial balance."""
        portfolio_data = {
            "name": "Zero Balance Portfolio",
            "initial_balance": 0.0
        }
        response = await client.post(
            "/api/v1/portfolios",
            json=portfolio_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_create_portfolio_max_limit(self, client: AsyncClient, auth_headers: dict):
        """Test portfolio creation limit per user."""
        # Create multiple portfolios up to the limit
        for i in range(10):  # Assuming 10 is the limit
            portfolio_data = {
                "name": f"Portfolio {i}",
                "initial_balance": 10000.0
            }
            response = await client.post(
                "/api/v1/portfolios",
                json=portfolio_data,
                headers=auth_headers
            )
            if i < 10:
                assert response.status_code == status.HTTP_201_CREATED
            else:
                assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_portfolio_by_id(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting portfolio by ID."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(test_portfolio.id)

    @pytest.mark.asyncio
    async def test_get_portfolio_invalid_id(self, client: AsyncClient, auth_headers: dict):
        """Test getting portfolio with invalid ID."""
        invalid_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/v1/portfolios/{invalid_id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_portfolio_unauthorized(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test accessing another user's portfolio."""
        # Create another user's auth headers
        other_auth_headers = {"Authorization": "Bearer other_user_token"}
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=other_auth_headers
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_update_portfolio(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test updating portfolio details."""
        update_data = {
            "name": "Updated Portfolio Name",
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

    @pytest.mark.asyncio
    async def test_delete_portfolio_empty(self, client: AsyncClient, auth_headers: dict):
        """Test deleting an empty portfolio."""
        # Create a portfolio
        portfolio_data = {"name": "To Delete", "initial_balance": 0.0}
        create_response = await client.post(
            "/api/v1/portfolios",
            json=portfolio_data,
            headers=auth_headers
        )
        portfolio_id = create_response.json()["id"]
        
        # Delete it
        response = await client.delete(
            f"/api/v1/portfolios/{portfolio_id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_204_NO_CONTENT

    @pytest.mark.asyncio
    async def test_delete_portfolio_with_positions(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test deleting portfolio with open positions."""
        # Add a position first
        position_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy"
        }
        await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        
        # Try to delete
        response = await client.delete(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_add_position_buy(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding a buy position."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["symbol"] == position_data["symbol"]
        assert data["quantity"] == position_data["quantity"]

    @pytest.mark.asyncio
    async def test_add_position_insufficient_funds(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding position with insufficient funds."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 1000000,  # Very large quantity
            "side": "buy"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "insufficient funds" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_add_position_fractional_shares(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding fractional share position."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 0.5,
            "side": "buy"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_add_position_zero_quantity(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding position with zero quantity."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 0,
            "side": "buy"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_add_position_limit_order(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding position with limit order."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 50,
            "side": "buy",
            "order_type": "limit",
            "limit_price": 150.00
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_add_position_stop_loss(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test adding position with stop loss."""
        position_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "stop_loss": 140.00,
            "take_profit": 160.00
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["stop_loss"] == position_data["stop_loss"]
        assert data["take_profit"] == position_data["take_profit"]

    @pytest.mark.asyncio
    async def test_sell_position_full(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test selling entire position."""
        # First buy
        buy_data = {"symbol": "AAPL", "quantity": 100, "side": "buy"}
        buy_response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=buy_data,
            headers=auth_headers
        )
        position_id = buy_response.json()["id"]
        
        # Then sell
        sell_data = {"quantity": 100, "side": "sell"}
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions/{position_id}/sell",
            json=sell_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_sell_position_partial(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test partial position sell."""
        # First buy
        buy_data = {"symbol": "AAPL", "quantity": 100, "side": "buy"}
        buy_response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=buy_data,
            headers=auth_headers
        )
        position_id = buy_response.json()["id"]
        
        # Sell partial
        sell_data = {"quantity": 30, "side": "sell"}
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions/{position_id}/sell",
            json=sell_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["quantity"] == 70  # Remaining quantity

    @pytest.mark.asyncio
    async def test_sell_position_more_than_owned(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test selling more shares than owned."""
        # First buy
        buy_data = {"symbol": "AAPL", "quantity": 50, "side": "buy"}
        buy_response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=buy_data,
            headers=auth_headers
        )
        position_id = buy_response.json()["id"]
        
        # Try to sell more
        sell_data = {"quantity": 100, "side": "sell"}
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions/{position_id}/sell",
            json=sell_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_short_selling(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test short selling functionality."""
        position_data = {
            "symbol": "TSLA",
            "quantity": 50,
            "side": "short",
            "margin_requirement": 0.5
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            json=position_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["side"] == "short"

    @pytest.mark.asyncio
    async def test_portfolio_performance_metrics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio performance calculation."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/performance",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_return" in data
        assert "annualized_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "win_rate" in data

    @pytest.mark.asyncio
    async def test_portfolio_performance_date_range(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio performance for specific date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/performance",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_portfolio_risk_metrics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio risk metrics calculation."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "var_95" in data  # Value at Risk
        assert "cvar_95" in data  # Conditional VaR
        assert "beta" in data
        assert "standard_deviation" in data
        assert "downside_deviation" in data

    @pytest.mark.asyncio
    async def test_portfolio_correlation_analysis(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio correlation with market."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/correlation",
            params={"benchmark": "SPY"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "correlation" in data
        assert -1 <= data["correlation"] <= 1

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio rebalancing functionality."""
        rebalance_data = {
            "target_allocations": {
                "AAPL": 0.3,
                "GOOGL": 0.3,
                "MSFT": 0.2,
                "CASH": 0.2
            },
            "threshold": 0.05  # 5% threshold
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/rebalance",
            json=rebalance_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "trades" in data
        assert "estimated_cost" in data

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_invalid_allocations(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test rebalancing with invalid allocations (not summing to 1)."""
        rebalance_data = {
            "target_allocations": {
                "AAPL": 0.5,
                "GOOGL": 0.6  # Sum > 1
            }
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/rebalance",
            json=rebalance_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_transaction_history(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test getting transaction history."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/transactions",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_transaction_history_filtered(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test filtered transaction history."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/transactions",
            params={
                "transaction_type": "buy",
                "symbol": "AAPL",
                "start_date": (datetime.now() - timedelta(days=30)).isoformat()
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_portfolio_cash_operations_deposit(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test cash deposit to portfolio."""
        deposit_data = {
            "amount": 10000.0,
            "operation": "deposit"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/cash",
            json=deposit_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cash_balance"] > test_portfolio.cash_balance

    @pytest.mark.asyncio
    async def test_portfolio_cash_operations_withdraw(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test cash withdrawal from portfolio."""
        withdraw_data = {
            "amount": 5000.0,
            "operation": "withdraw"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/cash",
            json=withdraw_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_portfolio_cash_withdraw_insufficient(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test withdrawal with insufficient cash balance."""
        withdraw_data = {
            "amount": 1000000.0,  # More than available
            "operation": "withdraw"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/cash",
            json=withdraw_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_portfolio_dividend_tracking(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test dividend tracking and reinvestment."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/dividends",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_dividends" in data
        assert "dividend_yield" in data
        assert "dividend_history" in data

    @pytest.mark.asyncio
    async def test_portfolio_tax_reporting(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test tax reporting functionality."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/tax-report",
            params={"tax_year": 2024},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "realized_gains" in data
        assert "unrealized_gains" in data
        assert "dividend_income" in data
        assert "tax_lots" in data

    @pytest.mark.asyncio
    async def test_portfolio_export_csv(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test exporting portfolio data to CSV."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/export",
            params={"format": "csv"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/csv"

    @pytest.mark.asyncio
    async def test_portfolio_import_transactions(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test importing transactions from file."""
        import_data = {
            "format": "csv",
            "data": "Date,Symbol,Side,Quantity,Price\n2024-01-01,AAPL,buy,100,150.00"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/import",
            json=import_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "imported_count" in data

    @pytest.mark.asyncio
    async def test_portfolio_benchmark_comparison(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio comparison with benchmark."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/benchmark",
            params={"benchmark": "SPY", "period": "1Y"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "portfolio_return" in data
        assert "benchmark_return" in data
        assert "alpha" in data
        assert "tracking_error" in data

    @pytest.mark.asyncio
    async def test_portfolio_sector_allocation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio sector allocation analysis."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/sectors",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert sum(data.values()) <= 1.0  # Sum of allocations <= 100%

    @pytest.mark.asyncio
    async def test_portfolio_geographic_allocation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio geographic allocation."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/geographic",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_portfolio_concentration_risk(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio concentration risk analysis."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/concentration",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "top_holdings_weight" in data
        assert "herfindahl_index" in data
        assert "concentration_warnings" in data

    @pytest.mark.asyncio
    async def test_portfolio_liquidity_analysis(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio liquidity analysis."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/liquidity",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "liquid_assets_ratio" in data
        assert "days_to_liquidate" in data
        assert "illiquid_positions" in data

    @pytest.mark.asyncio
    async def test_portfolio_stress_testing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio stress testing scenarios."""
        stress_scenarios = {
            "market_crash": {"SPY": -0.20, "VIX": 0.50},
            "interest_rate_hike": {"TLT": -0.10, "BANK": -0.15},
            "tech_bubble": {"QQQ": -0.30}
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/stress-test",
            json=stress_scenarios,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "scenario_results" in data
        assert "worst_case_loss" in data

    @pytest.mark.asyncio
    async def test_portfolio_options_exposure(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio options exposure analysis."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/options-exposure",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "delta_exposure" in data
        assert "gamma_exposure" in data
        assert "theta_decay" in data

    @pytest.mark.asyncio
    async def test_portfolio_margin_requirements(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio margin requirements calculation."""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/margin",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "initial_margin" in data
        assert "maintenance_margin" in data
        assert "available_margin" in data
        assert "margin_call_price" in data

    @pytest.mark.asyncio
    async def test_portfolio_position_sizing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test position sizing recommendations."""
        sizing_data = {
            "symbol": "AAPL",
            "risk_percent": 2.0,  # 2% portfolio risk
            "stop_loss_percent": 5.0  # 5% stop loss
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/position-size",
            json=sizing_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "recommended_shares" in data
        assert "position_value" in data
        assert "max_loss" in data

    @pytest.mark.asyncio
    async def test_portfolio_clone(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test cloning a portfolio."""
        clone_data = {
            "new_name": "Cloned Portfolio",
            "include_positions": True,
            "include_history": False
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/clone",
            json=clone_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == clone_data["new_name"]

    @pytest.mark.asyncio
    async def test_portfolio_alerts(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio alert management."""
        alert_data = {
            "type": "price",
            "condition": "above",
            "threshold": 100000.0,
            "notification_method": "email"
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/alerts",
            json=alert_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_portfolio_auto_rebalancing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test automatic rebalancing setup."""
        auto_rebalance_data = {
            "enabled": True,
            "frequency": "monthly",
            "threshold": 0.05,
            "target_allocations": {
                "AAPL": 0.3,
                "GOOGL": 0.3,
                "MSFT": 0.2,
                "CASH": 0.2
            }
        }
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/auto-rebalance",
            json=auto_rebalance_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
