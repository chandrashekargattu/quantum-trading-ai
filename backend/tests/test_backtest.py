"""Tests for backtesting functionality."""
import pytest
from datetime import datetime, timedelta
from typing import List, Dict
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy import Strategy
from app.models.backtest import BacktestResult
from app.services.backtest import BacktestService
from app.strategies.base import BaseStrategy
from app.schemas.backtest import BacktestRequest, BacktestConfig


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str = "Mock Strategy"):
        self.name = name
        self.positions = {}
        
    def on_data(self, data: Dict) -> List[Dict]:
        """Simple buy/sell logic for testing."""
        trades = []
        
        # Buy if price < 100, sell if price > 110
        if data["close"] < 100 and "position" not in self.positions:
            trades.append({
                "symbol": data["symbol"],
                "side": "buy",
                "quantity": 100,
                "price": data["close"]
            })
            self.positions["position"] = True
        elif data["close"] > 110 and "position" in self.positions:
            trades.append({
                "symbol": data["symbol"],
                "side": "sell",
                "quantity": 100,
                "price": data["close"]
            })
            del self.positions["position"]
            
        return trades


class TestBacktestEndpoints:
    """Test cases for backtesting endpoints."""

    @pytest.mark.asyncio
    async def test_run_backtest(self, client: AsyncClient, test_strategy: Strategy, auth_headers: dict):
        """Test running a backtest."""
        backtest_request = {
            "strategy_id": str(test_strategy.id),
            "symbol": "AAPL",
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_capital": 10000.0,
            "commission": 0.001
        }
        
        response = await client.post(
            "/api/v1/backtest/run",
            json=backtest_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] in ["running", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_get_backtest_results(self, client: AsyncClient, test_backtest_result: BacktestResult, auth_headers: dict):
        """Test getting backtest results."""
        response = await client.get(
            f"/api/v1/backtest/results/{test_backtest_result.id}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(test_backtest_result.id)
        assert "metrics" in data
        assert "trades" in data
        assert "equity_curve" in data

    @pytest.mark.asyncio
    async def test_list_backtests(self, client: AsyncClient, auth_headers: dict):
        """Test listing user backtests."""
        response = await client.get("/api/v1/backtest/", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_cancel_backtest(self, client: AsyncClient, test_backtest_result: BacktestResult, auth_headers: dict):
        """Test canceling a running backtest."""
        response = await client.post(
            f"/api/v1/backtest/{test_backtest_result.id}/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_backtest_comparison(self, client: AsyncClient, auth_headers: dict):
        """Test comparing multiple backtests."""
        backtest_ids = ["id1", "id2", "id3"]  # Mock IDs
        
        response = await client.post(
            "/api/v1/backtest/compare",
            json={"backtest_ids": backtest_ids},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "comparison" in data
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_optimize_strategy(self, client: AsyncClient, test_strategy: Strategy, auth_headers: dict):
        """Test strategy parameter optimization."""
        optimization_request = {
            "strategy_id": str(test_strategy.id),
            "symbol": "AAPL",
            "start_date": (datetime.now() - timedelta(days=90)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "parameters": {
                "sma_short": {"min": 10, "max": 30, "step": 5},
                "sma_long": {"min": 40, "max": 60, "step": 5}
            },
            "optimization_metric": "sharpe_ratio"
        }
        
        response = await client.post(
            "/api/v1/backtest/optimize",
            json=optimization_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "best_parameters" in data
        assert "results" in data


class TestBacktestService:
    """Test cases for backtest service."""

    @pytest.mark.asyncio
    async def test_execute_backtest(self, db: AsyncSession):
        """Test executing a backtest."""
        service = BacktestService(db)
        strategy = MockStrategy()
        
        # Mock historical data
        historical_data = [
            {"symbol": "AAPL", "close": 95.0, "volume": 1000000, "timestamp": datetime.now() - timedelta(days=10)},
            {"symbol": "AAPL", "close": 98.0, "volume": 1100000, "timestamp": datetime.now() - timedelta(days=9)},
            {"symbol": "AAPL", "close": 102.0, "volume": 1200000, "timestamp": datetime.now() - timedelta(days=8)},
            {"symbol": "AAPL", "close": 108.0, "volume": 1300000, "timestamp": datetime.now() - timedelta(days=7)},
            {"symbol": "AAPL", "close": 112.0, "volume": 1400000, "timestamp": datetime.now() - timedelta(days=6)},
            {"symbol": "AAPL", "close": 109.0, "volume": 1500000, "timestamp": datetime.now() - timedelta(days=5)},
        ]
        
        result = await service.execute_backtest(
            strategy=strategy,
            historical_data=historical_data,
            initial_capital=10000.0,
            commission=0.001
        )
        
        assert result is not None
        assert len(result.trades) == 2  # One buy, one sell
        assert result.metrics["total_trades"] == 2

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, db: AsyncSession):
        """Test backtest metrics calculation."""
        service = BacktestService(db)
        
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 100.0, "timestamp": datetime.now() - timedelta(days=10)},
            {"symbol": "AAPL", "side": "sell", "quantity": 100, "price": 110.0, "timestamp": datetime.now() - timedelta(days=5)},
            {"symbol": "AAPL", "side": "buy", "quantity": 50, "price": 105.0, "timestamp": datetime.now() - timedelta(days=4)},
            {"symbol": "AAPL", "side": "sell", "quantity": 50, "price": 108.0, "timestamp": datetime.now() - timedelta(days=1)},
        ]
        
        equity_curve = [10000, 10100, 10300, 10250, 10400, 10350]
        
        metrics = service.calculate_metrics(trades, equity_curve, 10000.0)
        
        assert metrics["total_trades"] == 4
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 100.0
        assert metrics["total_return"] > 0
        assert metrics["sharpe_ratio"] is not None
        assert metrics["max_drawdown"] < 0

    @pytest.mark.asyncio
    async def test_equity_curve_calculation(self, db: AsyncSession):
        """Test equity curve calculation."""
        service = BacktestService(db)
        
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 100.0, "timestamp": datetime.now() - timedelta(days=10), "commission": 1.0},
            {"symbol": "AAPL", "side": "sell", "quantity": 100, "price": 110.0, "timestamp": datetime.now() - timedelta(days=5), "commission": 1.1},
        ]
        
        prices = {
            datetime.now() - timedelta(days=10): 100.0,
            datetime.now() - timedelta(days=9): 102.0,
            datetime.now() - timedelta(days=8): 105.0,
            datetime.now() - timedelta(days=7): 103.0,
            datetime.now() - timedelta(days=6): 108.0,
            datetime.now() - timedelta(days=5): 110.0,
        }
        
        equity_curve = service.calculate_equity_curve(
            trades=trades,
            prices=prices,
            initial_capital=10000.0
        )
        
        assert len(equity_curve) == len(prices)
        assert equity_curve[0] == 10000.0  # Initial capital
        assert equity_curve[-1] > 10000.0  # Profit from trade

    @pytest.mark.asyncio
    async def test_risk_metrics(self, db: AsyncSession):
        """Test risk metrics calculation."""
        service = BacktestService(db)
        
        returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, 0.03, -0.015, 0.01]
        
        volatility = service.calculate_volatility(returns)
        sharpe = service.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        sortino = service.calculate_sortino_ratio(returns, risk_free_rate=0.02)
        max_dd = service.calculate_max_drawdown(returns)
        
        assert volatility > 0
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert max_dd < 0

    @pytest.mark.asyncio
    async def test_trade_analysis(self, db: AsyncSession):
        """Test trade analysis functionality."""
        service = BacktestService(db)
        
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 100.0, "timestamp": datetime.now() - timedelta(days=10)},
            {"symbol": "AAPL", "side": "sell", "quantity": 100, "price": 110.0, "timestamp": datetime.now() - timedelta(days=8)},
            {"symbol": "GOOGL", "side": "buy", "quantity": 10, "price": 2800.0, "timestamp": datetime.now() - timedelta(days=7)},
            {"symbol": "GOOGL", "side": "sell", "quantity": 10, "price": 2750.0, "timestamp": datetime.now() - timedelta(days=5)},
        ]
        
        analysis = service.analyze_trades(trades)
        
        assert analysis["total_trades"] == 4
        assert analysis["unique_symbols"] == 2
        assert analysis["avg_trade_duration"] > 0
        assert "profit_by_symbol" in analysis
        assert analysis["profit_by_symbol"]["AAPL"] > 0
        assert analysis["profit_by_symbol"]["GOOGL"] < 0

    @pytest.mark.asyncio
    async def test_parameter_optimization(self, db: AsyncSession):
        """Test strategy parameter optimization."""
        service = BacktestService(db)
        
        parameter_ranges = {
            "fast_ma": range(10, 30, 5),
            "slow_ma": range(40, 60, 5)
        }
        
        # Mock optimization results
        results = []
        for fast in parameter_ranges["fast_ma"]:
            for slow in parameter_ranges["slow_ma"]:
                results.append({
                    "parameters": {"fast_ma": fast, "slow_ma": slow},
                    "sharpe_ratio": (slow - fast) * 0.01  # Simple mock metric
                })
        
        best_params = service.find_optimal_parameters(
            results=results,
            metric="sharpe_ratio"
        )
        
        assert best_params["fast_ma"] == 10
        assert best_params["slow_ma"] == 55

    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, db: AsyncSession):
        """Test walk-forward analysis."""
        service = BacktestService(db)
        strategy = MockStrategy()
        
        # Mock data for multiple periods
        historical_data = []
        for i in range(100):
            historical_data.append({
                "symbol": "AAPL",
                "close": 100 + (i % 20),  # Oscillating price
                "volume": 1000000,
                "timestamp": datetime.now() - timedelta(days=100-i)
            })
        
        results = await service.walk_forward_analysis(
            strategy=strategy,
            historical_data=historical_data,
            window_size=30,
            step_size=10,
            initial_capital=10000.0
        )
        
        assert len(results) > 0
        assert all("in_sample" in r for r in results)
        assert all("out_sample" in r for r in results)