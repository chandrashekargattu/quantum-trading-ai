"""Comprehensive options trading tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.options import Option, OptionOrder, OptionStrategy
from app.models.portfolio import Portfolio
from app.models.user import User
from fastapi import status
import math


class TestOptionsTrading:
    """Test options trading with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_get_options_chain(self, client: AsyncClient, auth_headers: dict):
        """Test getting options chain for a symbol."""
        response = await client.get(
            "/api/v1/options/chain/AAPL",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expirations" in data
        assert "calls" in data
        assert "puts" in data
        assert len(data["expirations"]) > 0

    @pytest.mark.asyncio
    async def test_get_options_chain_specific_expiry(self, client: AsyncClient, auth_headers: dict):
        """Test getting options chain for specific expiration."""
        expiry = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        response = await client.get(
            f"/api/v1/options/chain/AAPL",
            params={"expiration": expiry},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["expiration"] == expiry

    @pytest.mark.asyncio
    async def test_get_options_chain_strike_range(self, client: AsyncClient, auth_headers: dict):
        """Test getting options chain with strike price range."""
        response = await client.get(
            "/api/v1/options/chain/AAPL",
            params={
                "min_strike": 140,
                "max_strike": 160,
                "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify all strikes are within range
        for option in data["calls"] + data["puts"]:
            assert 140 <= option["strike"] <= 160

    @pytest.mark.asyncio
    async def test_calculate_option_greeks(self, client: AsyncClient, auth_headers: dict):
        """Test calculating option Greeks."""
        greek_data = {
            "underlying_price": 150.00,
            "strike_price": 155.00,
            "time_to_expiry": 30,  # days
            "volatility": 0.25,
            "risk_free_rate": 0.05,
            "option_type": "call"
        }
        response = await client.post(
            "/api/v1/options/greeks",
            json=greek_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "delta" in data
        assert "gamma" in data
        assert "theta" in data
        assert "vega" in data
        assert "rho" in data
        # Validate Greek ranges
        assert -1 <= data["delta"] <= 1
        assert data["gamma"] >= 0
        assert data["theta"] <= 0  # Time decay is negative

    @pytest.mark.asyncio
    async def test_calculate_put_option_greeks(self, client: AsyncClient, auth_headers: dict):
        """Test calculating put option Greeks."""
        greek_data = {
            "underlying_price": 150.00,
            "strike_price": 145.00,
            "time_to_expiry": 30,
            "volatility": 0.25,
            "risk_free_rate": 0.05,
            "option_type": "put"
        }
        response = await client.post(
            "/api/v1/options/greeks",
            json=greek_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["delta"] < 0  # Put delta is negative

    @pytest.mark.asyncio
    async def test_implied_volatility_calculation(self, client: AsyncClient, auth_headers: dict):
        """Test implied volatility calculation."""
        iv_data = {
            "option_price": 5.50,
            "underlying_price": 150.00,
            "strike_price": 155.00,
            "time_to_expiry": 30,
            "risk_free_rate": 0.05,
            "option_type": "call"
        }
        response = await client.post(
            "/api/v1/options/implied-volatility",
            json=iv_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "implied_volatility" in data
        assert 0 < data["implied_volatility"] < 2  # Reasonable IV range

    @pytest.mark.asyncio
    async def test_buy_call_option(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test buying a call option."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 150.00,
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "quantity": 1,  # 1 contract = 100 shares
            "order_type": "market",
            "action": "buy_to_open"
        }
        response = await client.post(
            "/api/v1/options/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["option_type"] == "call"
        assert data["action"] == "buy_to_open"

    @pytest.mark.asyncio
    async def test_buy_put_option(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test buying a put option."""
        order_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "put",
            "strike": 145.00,
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "quantity": 2,
            "order_type": "limit",
            "limit_price": 3.50,
            "action": "buy_to_open"
        }
        response = await client.post(
            "/api/v1/options/orders",
            json=order_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_sell_call_option_covered(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test selling covered call option."""
        # First buy 100 shares of stock
        stock_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "quantity": 100,
            "side": "buy",
            "order_type": "market"
        }
        await client.post("/api/v1/trading/orders", json=stock_order, headers=auth_headers)
        
        # Now sell covered call
        option_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 160.00,
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "quantity": 1,
            "order_type": "market",
            "action": "sell_to_open",
            "covered": True
        }
        response = await client.post(
            "/api/v1/options/orders",
            json=option_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_sell_naked_call_insufficient_margin(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test selling naked call with insufficient margin."""
        option_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "TSLA",
            "option_type": "call",
            "strike": 250.00,
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "quantity": 10,  # Large quantity
            "order_type": "market",
            "action": "sell_to_open",
            "covered": False
        }
        response = await client.post(
            "/api/v1/options/orders",
            json=option_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "margin" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_close_option_position(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test closing an option position."""
        # First buy option
        buy_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "GOOGL",
            "option_type": "call",
            "strike": 100.00,
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "quantity": 1,
            "order_type": "market",
            "action": "buy_to_open"
        }
        buy_response = await client.post(
            "/api/v1/options/orders",
            json=buy_order,
            headers=auth_headers
        )
        position_id = buy_response.json()["position_id"]
        
        # Now close it
        close_order = {
            "portfolio_id": str(test_portfolio.id),
            "position_id": position_id,
            "quantity": 1,
            "order_type": "market",
            "action": "sell_to_close"
        }
        response = await client.post(
            "/api/v1/options/orders",
            json=close_order,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_option_exercise(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test exercising an option."""
        # Buy an ITM call option
        buy_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 140.00,  # Assuming AAPL is at 150
            "expiration": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "quantity": 1,
            "order_type": "market",
            "action": "buy_to_open"
        }
        buy_response = await client.post(
            "/api/v1/options/orders",
            json=buy_order,
            headers=auth_headers
        )
        position_id = buy_response.json()["position_id"]
        
        # Exercise the option
        response = await client.post(
            f"/api/v1/options/positions/{position_id}/exercise",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["shares_received"] == 100

    @pytest.mark.asyncio
    async def test_option_assignment(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test option assignment notification."""
        # Simulate being assigned on a short option
        assignment_data = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "put",
            "strike": 150.00,
            "quantity": 1,
            "assignment_price": 150.00
        }
        response = await client.post(
            "/api/v1/options/assignment",
            json=assignment_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_bull_call_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a bull call spread strategy."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "bull_call_spread",
            "symbol": "MSFT",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "legs": [
                {
                    "action": "buy",
                    "option_type": "call",
                    "strike": 300.00,
                    "quantity": 1
                },
                {
                    "action": "sell",
                    "option_type": "call",
                    "strike": 310.00,
                    "quantity": 1
                }
            ]
        }
        response = await client.post(
            "/api/v1/options/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["strategy_type"] == "bull_call_spread"
        assert len(data["legs"]) == 2

    @pytest.mark.asyncio
    async def test_bear_put_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a bear put spread strategy."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "bear_put_spread",
            "symbol": "SPY",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "legs": [
                {
                    "action": "buy",
                    "option_type": "put",
                    "strike": 420.00,
                    "quantity": 1
                },
                {
                    "action": "sell",
                    "option_type": "put",
                    "strike": 410.00,
                    "quantity": 1
                }
            ]
        }
        response = await client.post(
            "/api/v1/options/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_iron_condor(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating an iron condor strategy."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "iron_condor",
            "symbol": "GOOGL",
            "expiration": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
            "legs": [
                {
                    "action": "sell",
                    "option_type": "put",
                    "strike": 90.00,
                    "quantity": 1
                },
                {
                    "action": "buy",
                    "option_type": "put",
                    "strike": 85.00,
                    "quantity": 1
                },
                {
                    "action": "sell",
                    "option_type": "call",
                    "strike": 110.00,
                    "quantity": 1
                },
                {
                    "action": "buy",
                    "option_type": "call",
                    "strike": 115.00,
                    "quantity": 1
                }
            ]
        }
        response = await client.post(
            "/api/v1/options/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert len(data["legs"]) == 4

    @pytest.mark.asyncio
    async def test_straddle(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a straddle strategy."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "straddle",
            "symbol": "TSLA",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "strike": 200.00,
            "quantity": 1
        }
        response = await client.post(
            "/api/v1/options/strategies/straddle",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert len(data["legs"]) == 2  # Call and put

    @pytest.mark.asyncio
    async def test_strangle(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a strangle strategy."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "strangle",
            "symbol": "NVDA",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "call_strike": 510.00,
            "put_strike": 490.00,
            "quantity": 1
        }
        response = await client.post(
            "/api/v1/options/strategies/strangle",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_butterfly_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a butterfly spread."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "butterfly",
            "symbol": "AAPL",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "strikes": [145.00, 150.00, 155.00],
            "option_type": "call"
        }
        response = await client.post(
            "/api/v1/options/strategies/butterfly",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_calendar_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a calendar spread."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "calendar",
            "symbol": "META",
            "strike": 300.00,
            "option_type": "call",
            "short_expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "long_expiration": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
        }
        response = await client.post(
            "/api/v1/options/strategies/calendar",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_diagonal_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a diagonal spread."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "diagonal",
            "symbol": "AMZN",
            "option_type": "call",
            "short_strike": 110.00,
            "short_expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "long_strike": 105.00,
            "long_expiration": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
        }
        response = await client.post(
            "/api/v1/options/strategies/diagonal",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_ratio_spread(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test creating a ratio spread."""
        strategy_data = {
            "portfolio_id": str(test_portfolio.id),
            "strategy_type": "ratio_spread",
            "symbol": "SPY",
            "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "legs": [
                {
                    "action": "buy",
                    "option_type": "call",
                    "strike": 420.00,
                    "quantity": 1
                },
                {
                    "action": "sell",
                    "option_type": "call",
                    "strike": 430.00,
                    "quantity": 2  # Ratio 1:2
                }
            ]
        }
        response = await client.post(
            "/api/v1/options/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_option_volatility_smile(self, client: AsyncClient, auth_headers: dict):
        """Test getting volatility smile data."""
        response = await client.get(
            "/api/v1/options/volatility/smile",
            params={
                "symbol": "AAPL",
                "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "strikes" in data
        assert "implied_volatilities" in data
        assert len(data["strikes"]) == len(data["implied_volatilities"])

    @pytest.mark.asyncio
    async def test_option_volatility_surface(self, client: AsyncClient, auth_headers: dict):
        """Test getting full volatility surface."""
        response = await client.get(
            "/api/v1/options/volatility/surface",
            params={"symbol": "SPY"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expirations" in data
        assert "strikes" in data
        assert "surface" in data

    @pytest.mark.asyncio
    async def test_option_skew_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test option skew analysis."""
        response = await client.get(
            "/api/v1/options/analysis/skew",
            params={"symbol": "QQQ"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "put_call_skew" in data
        assert "risk_reversal" in data

    @pytest.mark.asyncio
    async def test_option_term_structure(self, client: AsyncClient, auth_headers: dict):
        """Test option term structure analysis."""
        response = await client.get(
            "/api/v1/options/analysis/term-structure",
            params={"symbol": "VIX"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expirations" in data
        assert "at_the_money_iv" in data

    @pytest.mark.asyncio
    async def test_option_open_interest(self, client: AsyncClient, auth_headers: dict):
        """Test option open interest analysis."""
        response = await client.get(
            "/api/v1/options/analysis/open-interest",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "put_call_ratio" in data
        assert "max_pain" in data
        assert "high_oi_strikes" in data

    @pytest.mark.asyncio
    async def test_option_flow_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test unusual options activity analysis."""
        response = await client.get(
            "/api/v1/options/analysis/flow",
            params={
                "min_premium": 100000,
                "min_volume_oi_ratio": 5
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_option_strategy_scanner(self, client: AsyncClient, auth_headers: dict):
        """Test option strategy scanner."""
        scanner_params = {
            "strategy_types": ["covered_call", "cash_secured_put"],
            "min_return": 0.02,  # 2% minimum return
            "max_days": 45,
            "symbols": ["AAPL", "MSFT", "GOOGL"]
        }
        response = await client.post(
            "/api/v1/options/scanner",
            json=scanner_params,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "opportunities" in data

    @pytest.mark.asyncio
    async def test_option_position_greeks_aggregation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test aggregated Greeks for all option positions."""
        response = await client.get(
            f"/api/v1/options/portfolio/{test_portfolio.id}/greeks",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_delta" in data
        assert "total_gamma" in data
        assert "total_theta" in data
        assert "total_vega" in data

    @pytest.mark.asyncio
    async def test_option_pnl_analysis(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test option P&L analysis."""
        response = await client.get(
            f"/api/v1/options/portfolio/{test_portfolio.id}/pnl",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "realized_pnl" in data
        assert "unrealized_pnl" in data
        assert "positions" in data

    @pytest.mark.asyncio
    async def test_option_expiration_management(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test option expiration management."""
        response = await client.get(
            f"/api/v1/options/portfolio/{test_portfolio.id}/expirations",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "upcoming_expirations" in data
        assert "expiring_positions" in data

    @pytest.mark.asyncio
    async def test_option_rolling(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test rolling an option position."""
        # First create a position
        buy_order = {
            "portfolio_id": str(test_portfolio.id),
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 150.00,
            "expiration": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "quantity": 1,
            "order_type": "market",
            "action": "buy_to_open"
        }
        buy_response = await client.post(
            "/api/v1/options/orders",
            json=buy_order,
            headers=auth_headers
        )
        position_id = buy_response.json()["position_id"]
        
        # Roll to next month
        roll_data = {
            "new_expiration": (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d"),
            "new_strike": 155.00
        }
        response = await client.post(
            f"/api/v1/options/positions/{position_id}/roll",
            json=roll_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_option_early_assignment_risk(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test early assignment risk analysis."""
        response = await client.get(
            f"/api/v1/options/portfolio/{test_portfolio.id}/assignment-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "high_risk_positions" in data
        assert "risk_factors" in data

    @pytest.mark.asyncio
    async def test_option_margin_requirements(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test option margin requirement calculation."""
        margin_data = {
            "portfolio_id": str(test_portfolio.id),
            "positions": [
                {
                    "symbol": "SPY",
                    "option_type": "put",
                    "strike": 400.00,
                    "expiration": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "quantity": -5,  # Short 5 puts
                    "action": "sell_to_open"
                }
            ]
        }
        response = await client.post(
            "/api/v1/options/margin/calculate",
            json=margin_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "initial_margin" in data
        assert "maintenance_margin" in data

    @pytest.mark.asyncio
    async def test_option_hedging_suggestions(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test option hedging suggestions."""
        response = await client.get(
            f"/api/v1/options/portfolio/{test_portfolio.id}/hedging",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "suggestions" in data
        assert "current_risk" in data

    @pytest.mark.asyncio
    async def test_option_backtesting(self, client: AsyncClient, auth_headers: dict):
        """Test option strategy backtesting."""
        backtest_data = {
            "strategy": "covered_call",
            "symbol": "AAPL",
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "parameters": {
                "days_to_expiration": 30,
                "delta_target": 0.30
            }
        }
        response = await client.post(
            "/api/v1/options/backtest",
            json=backtest_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "win_rate" in data

    @pytest.mark.asyncio
    async def test_option_pricing_models(self, client: AsyncClient, auth_headers: dict):
        """Test different option pricing models."""
        models = ["black-scholes", "binomial", "monte-carlo"]
        option_params = {
            "underlying_price": 150.00,
            "strike_price": 155.00,
            "time_to_expiry": 0.25,  # 3 months
            "volatility": 0.30,
            "risk_free_rate": 0.05,
            "option_type": "call"
        }
        
        for model in models:
            response = await client.post(
                f"/api/v1/options/price/{model}",
                json=option_params,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "price" in data
            assert data["price"] > 0
