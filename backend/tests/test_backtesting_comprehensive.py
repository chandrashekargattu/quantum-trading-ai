"""Comprehensive backtesting tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.backtest import Backtest, BacktestResult
from app.models.trading import Strategy
from app.models.portfolio import Portfolio
from fastapi import status
import json


class TestBacktestingEngine:
    """Test backtesting engine with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_create_simple_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a simple buy-and-hold strategy."""
        strategy_data = {
            "name": "Buy and Hold SPY",
            "strategy_type": "buy_and_hold",
            "symbols": ["SPY"],
            "allocation": {"SPY": 1.0}
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == strategy_data["name"]
        assert data["strategy_type"] == strategy_data["strategy_type"]

    @pytest.mark.asyncio
    async def test_create_momentum_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a momentum trading strategy."""
        strategy_data = {
            "name": "Momentum Strategy",
            "strategy_type": "momentum",
            "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            "parameters": {
                "lookback_period": 20,
                "holding_period": 5,
                "rebalance_frequency": "weekly",
                "num_positions": 3
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_create_mean_reversion_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a mean reversion strategy."""
        strategy_data = {
            "name": "Mean Reversion",
            "strategy_type": "mean_reversion",
            "symbols": ["SPY", "QQQ"],
            "parameters": {
                "lookback_period": 20,
                "entry_z_score": -2.0,
                "exit_z_score": 0.0,
                "stop_loss_z_score": -3.0
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_create_pairs_trading_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a pairs trading strategy."""
        strategy_data = {
            "name": "Pairs Trading",
            "strategy_type": "pairs_trading",
            "pairs": [["AAPL", "MSFT"], ["JPM", "BAC"]],
            "parameters": {
                "lookback_period": 60,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "cointegration_pvalue": 0.05
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_create_technical_indicator_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a technical indicator based strategy."""
        strategy_data = {
            "name": "RSI + MACD Strategy",
            "strategy_type": "technical",
            "symbols": ["SPY"],
            "rules": {
                "entry": [
                    {"indicator": "RSI", "operator": "<", "value": 30},
                    {"indicator": "MACD", "condition": "bullish_crossover"}
                ],
                "exit": [
                    {"indicator": "RSI", "operator": ">", "value": 70},
                    {"indicator": "MACD", "condition": "bearish_crossover"}
                ]
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_create_portfolio_optimization_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test creating a portfolio optimization strategy."""
        strategy_data = {
            "name": "Markowitz Optimization",
            "strategy_type": "portfolio_optimization",
            "symbols": ["SPY", "TLT", "GLD", "IEF", "VNQ"],
            "parameters": {
                "optimization_method": "mean_variance",
                "target_return": 0.10,
                "rebalance_frequency": "monthly",
                "constraints": {
                    "min_weight": 0.05,
                    "max_weight": 0.40
                }
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies",
            json=strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, client: AsyncClient, auth_headers: dict):
        """Test running a basic backtest."""
        # First create a strategy
        strategy_response = await client.post(
            "/api/v1/backtest/strategies",
            json={
                "name": "Test Strategy",
                "strategy_type": "buy_and_hold",
                "symbols": ["SPY"]
            },
            headers=auth_headers
        )
        strategy_id = strategy_response.json()["id"]
        
        # Run backtest
        backtest_data = {
            "strategy_id": strategy_id,
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "initial_capital": 100000.0,
            "commission": 0.001,  # 0.1%
            "slippage": 0.0005   # 0.05%
        }
        response = await client.post(
            "/api/v1/backtest/run",
            json=backtest_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "backtest_id" in data
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_run_backtest_with_transaction_costs(self, client: AsyncClient, auth_headers: dict):
        """Test backtest with realistic transaction costs."""
        backtest_data = {
            "strategy_id": "test_strategy_id",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000.0,
            "transaction_costs": {
                "commission_per_share": 0.005,
                "commission_minimum": 1.0,
                "commission_maximum": 5.0,
                "slippage_model": "square_root",
                "slippage_constant": 0.1
            }
        }
        response = await client.post(
            "/api/v1/backtest/run",
            json=backtest_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_run_backtest_with_position_limits(self, client: AsyncClient, auth_headers: dict):
        """Test backtest with position size limits."""
        backtest_data = {
            "strategy_id": "test_strategy_id",
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000.0,
            "position_limits": {
                "max_position_size": 0.20,  # 20% per position
                "max_sector_exposure": 0.40,  # 40% per sector
                "max_leverage": 1.0,
                "min_position_size": 0.01
            }
        }
        response = await client.post(
            "/api/v1/backtest/run",
            json=backtest_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_get_backtest_results(self, client: AsyncClient, auth_headers: dict):
        """Test getting backtest results."""
        # Assume backtest_id from previous test
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "metrics" in data
        assert "equity_curve" in data
        assert "trades" in data

    @pytest.mark.asyncio
    async def test_backtest_performance_metrics(self, client: AsyncClient, auth_headers: dict):
        """Test comprehensive performance metrics calculation."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/metrics",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check all required metrics
        required_metrics = [
            "total_return", "annualized_return", "volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "max_drawdown_duration", "win_rate",
            "profit_factor", "expectancy", "kelly_criterion",
            "var_95", "cvar_95", "beta", "alpha", "treynor_ratio",
            "information_ratio", "capture_ratio_up", "capture_ratio_down"
        ]
        for metric in required_metrics:
            assert metric in data

    @pytest.mark.asyncio
    async def test_backtest_drawdown_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test drawdown analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/drawdown-analysis",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "max_drawdown" in data
        assert "drawdown_periods" in data
        assert "recovery_times" in data
        assert "underwater_curve" in data

    @pytest.mark.asyncio
    async def test_backtest_trade_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test detailed trade analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/trade-analysis",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_trades" in data
        assert "winning_trades" in data
        assert "losing_trades" in data
        assert "avg_win" in data
        assert "avg_loss" in data
        assert "largest_win" in data
        assert "largest_loss" in data
        assert "avg_holding_period" in data

    @pytest.mark.asyncio
    async def test_backtest_monte_carlo_simulation(self, client: AsyncClient, auth_headers: dict):
        """Test Monte Carlo simulation on backtest results."""
        backtest_id = "test_backtest_id"
        simulation_params = {
            "num_simulations": 1000,
            "num_periods": 252,  # 1 year
            "confidence_levels": [0.05, 0.25, 0.50, 0.75, 0.95]
        }
        response = await client.post(
            f"/api/v1/backtest/results/{backtest_id}/monte-carlo",
            json=simulation_params,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "percentiles" in data
        assert "expected_return" in data
        assert "expected_volatility" in data

    @pytest.mark.asyncio
    async def test_backtest_walk_forward_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test walk-forward analysis."""
        analysis_params = {
            "strategy_id": "test_strategy_id",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "in_sample_periods": 252,  # 1 year
            "out_sample_periods": 63,   # 3 months
            "optimization_metric": "sharpe_ratio"
        }
        response = await client.post(
            "/api/v1/backtest/walk-forward",
            json=analysis_params,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "periods" in data
        assert "aggregate_performance" in data

    @pytest.mark.asyncio
    async def test_backtest_parameter_optimization(self, client: AsyncClient, auth_headers: dict):
        """Test strategy parameter optimization."""
        optimization_params = {
            "strategy_id": "test_strategy_id",
            "parameters_to_optimize": {
                "lookback_period": {"min": 10, "max": 50, "step": 5},
                "entry_threshold": {"min": 1.5, "max": 3.0, "step": 0.25}
            },
            "optimization_metric": "sharpe_ratio",
            "method": "grid_search"
        }
        response = await client.post(
            "/api/v1/backtest/optimize",
            json=optimization_params,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "optimal_parameters" in data
        assert "performance_surface" in data

    @pytest.mark.asyncio
    async def test_backtest_genetic_optimization(self, client: AsyncClient, auth_headers: dict):
        """Test genetic algorithm optimization."""
        optimization_params = {
            "strategy_id": "test_strategy_id",
            "parameters_to_optimize": {
                "param1": {"min": 0, "max": 100},
                "param2": {"min": 0.1, "max": 1.0},
                "param3": {"min": 1, "max": 50}
            },
            "optimization_metric": "total_return",
            "method": "genetic",
            "genetic_params": {
                "population_size": 100,
                "generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            }
        }
        response = await client.post(
            "/api/v1/backtest/optimize",
            json=optimization_params,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_overfitting_detection(self, client: AsyncClient, auth_headers: dict):
        """Test overfitting detection methods."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/overfitting-analysis",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "deflated_sharpe_ratio" in data
        assert "probabilistic_sharpe_ratio" in data
        assert "overfitting_probability" in data

    @pytest.mark.asyncio
    async def test_backtest_regime_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test market regime analysis during backtest."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/regime-analysis",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "regimes" in data
        assert "performance_by_regime" in data

    @pytest.mark.asyncio
    async def test_backtest_factor_attribution(self, client: AsyncClient, auth_headers: dict):
        """Test factor attribution analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/factor-attribution",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "factor_exposures" in data
        assert "factor_returns" in data
        assert "specific_returns" in data

    @pytest.mark.asyncio
    async def test_backtest_stress_testing(self, client: AsyncClient, auth_headers: dict):
        """Test stress testing on historical scenarios."""
        backtest_id = "test_backtest_id"
        stress_scenarios = {
            "scenarios": [
                {"name": "2008 Financial Crisis", "start": "2008-09-01", "end": "2009-03-31"},
                {"name": "COVID-19 Crash", "start": "2020-02-20", "end": "2020-03-23"},
                {"name": "Dot-com Bubble", "start": "2000-03-01", "end": "2002-10-01"}
            ]
        }
        response = await client.post(
            f"/api/v1/backtest/results/{backtest_id}/stress-test",
            json=stress_scenarios,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "scenario_results" in data

    @pytest.mark.asyncio
    async def test_backtest_benchmark_comparison(self, client: AsyncClient, auth_headers: dict):
        """Test comparison with multiple benchmarks."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/benchmark-comparison",
            params={"benchmarks": ["SPY", "AGG", "60/40"]},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "strategy_performance" in data
        assert "benchmark_performance" in data
        assert "relative_metrics" in data

    @pytest.mark.asyncio
    async def test_backtest_rolling_window_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test rolling window performance analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/rolling-analysis",
            params={"window_size": 252, "step": 21},  # 1 year window, monthly step
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "rolling_returns" in data
        assert "rolling_sharpe" in data
        assert "rolling_volatility" in data

    @pytest.mark.asyncio
    async def test_backtest_transaction_cost_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test detailed transaction cost analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/transaction-costs",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_commission" in data
        assert "total_slippage" in data
        assert "cost_per_trade" in data
        assert "turnover_rate" in data

    @pytest.mark.asyncio
    async def test_backtest_position_sizing_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test position sizing effectiveness."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/position-sizing",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "avg_position_size" in data
        assert "position_size_distribution" in data
        assert "kelly_fraction_used" in data

    @pytest.mark.asyncio
    async def test_backtest_multi_asset_correlation(self, client: AsyncClient, auth_headers: dict):
        """Test multi-asset correlation analysis."""
        backtest_id = "test_backtest_id"
        response = await client.get(
            f"/api/v1/backtest/results/{backtest_id}/correlation-analysis",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "correlation_matrix" in data
        assert "rolling_correlations" in data

    @pytest.mark.asyncio
    async def test_backtest_export_results(self, client: AsyncClient, auth_headers: dict):
        """Test exporting backtest results in various formats."""
        backtest_id = "test_backtest_id"
        formats = ["json", "csv", "excel", "pdf"]
        
        for fmt in formats:
            response = await client.get(
                f"/api/v1/backtest/results/{backtest_id}/export",
                params={"format": fmt},
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_backtest_live_trading_comparison(self, client: AsyncClient, auth_headers: dict):
        """Test comparison between backtest and live trading results."""
        comparison_data = {
            "backtest_id": "test_backtest_id",
            "live_portfolio_id": "test_portfolio_id",
            "comparison_period": {
                "start": "2023-01-01",
                "end": "2023-12-31"
            }
        }
        response = await client.post(
            "/api/v1/backtest/live-comparison",
            json=comparison_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "backtest_metrics" in data
        assert "live_metrics" in data
        assert "divergence_analysis" in data

    @pytest.mark.asyncio
    async def test_backtest_machine_learning_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test backtesting ML-based strategies."""
        ml_strategy_data = {
            "name": "ML Prediction Strategy",
            "strategy_type": "machine_learning",
            "model_type": "random_forest",
            "features": ["returns_5d", "rsi", "volume_ratio", "volatility"],
            "target": "returns_1d",
            "training_window": 252,
            "retraining_frequency": 21
        }
        response = await client.post(
            "/api/v1/backtest/strategies/ml",
            json=ml_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_reinforcement_learning(self, client: AsyncClient, auth_headers: dict):
        """Test RL-based trading strategy backtest."""
        rl_strategy_data = {
            "name": "RL Trading Agent",
            "strategy_type": "reinforcement_learning",
            "algorithm": "PPO",
            "state_features": ["price", "volume", "technical_indicators"],
            "action_space": "discrete",  # buy, hold, sell
            "reward_function": "sharpe_ratio",
            "training_episodes": 1000
        }
        response = await client.post(
            "/api/v1/backtest/strategies/rl",
            json=rl_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_high_frequency_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test high-frequency trading strategy backtest."""
        hft_strategy_data = {
            "name": "HFT Market Making",
            "strategy_type": "hft",
            "data_frequency": "tick",
            "symbols": ["AAPL"],
            "parameters": {
                "spread_bps": 2,
                "inventory_limit": 1000,
                "order_size": 100
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies/hft",
            json=hft_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_options_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test options strategy backtest."""
        options_strategy_data = {
            "name": "Iron Condor Weekly",
            "strategy_type": "options",
            "underlying": "SPY",
            "strategy_name": "iron_condor",
            "parameters": {
                "days_to_expiration": 7,
                "delta_short": 0.15,
                "wing_width": 5
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies/options",
            json=options_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_multi_timeframe_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test multi-timeframe strategy backtest."""
        mtf_strategy_data = {
            "name": "Multi-Timeframe Momentum",
            "strategy_type": "multi_timeframe",
            "timeframes": ["1h", "4h", "1d"],
            "rules": {
                "1d": {"indicator": "trend", "direction": "up"},
                "4h": {"indicator": "momentum", "threshold": 0.7},
                "1h": {"indicator": "rsi", "oversold": 30}
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies/multi-timeframe",
            json=mtf_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_seasonality_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test seasonality-based strategy backtest."""
        seasonality_strategy_data = {
            "name": "Monthly Seasonality",
            "strategy_type": "seasonality",
            "patterns": [
                {"month": 1, "action": "buy", "symbols": ["SPY"]},
                {"month": 5, "action": "sell", "symbols": ["SPY"]},
                {"month": 10, "action": "buy", "symbols": ["SPY"]}
            ]
        }
        response = await client.post(
            "/api/v1/backtest/strategies/seasonality",
            json=seasonality_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_event_driven_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test event-driven strategy backtest."""
        event_strategy_data = {
            "name": "Earnings Momentum",
            "strategy_type": "event_driven",
            "event_type": "earnings",
            "rules": {
                "pre_event": {"days_before": 5, "action": "buy"},
                "post_event": {"days_after": 2, "action": "sell"},
                "filters": {"earnings_surprise": ">0.05"}
            }
        }
        response = await client.post(
            "/api/v1/backtest/strategies/event-driven",
            json=event_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_risk_parity_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test risk parity portfolio strategy."""
        risk_parity_data = {
            "name": "Risk Parity Portfolio",
            "strategy_type": "risk_parity",
            "assets": ["SPY", "TLT", "GLD", "VNQ"],
            "target_risk_contribution": 0.25,
            "rebalance_frequency": "monthly",
            "leverage": 1.5
        }
        response = await client.post(
            "/api/v1/backtest/strategies/risk-parity",
            json=risk_parity_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_custom_strategy_code(self, client: AsyncClient, auth_headers: dict):
        """Test custom strategy with user-provided code."""
        custom_strategy_data = {
            "name": "Custom Strategy",
            "strategy_type": "custom",
            "code": '''
def generate_signals(data):
    signals = pd.Series(index=data.index, data=0)
    sma_short = data['close'].rolling(window=10).mean()
    sma_long = data['close'].rolling(window=30).mean()
    signals[sma_short > sma_long] = 1
    signals[sma_short <= sma_long] = -1
    return signals
            ''',
            "symbols": ["AAPL", "GOOGL"]
        }
        response = await client.post(
            "/api/v1/backtest/strategies/custom",
            json=custom_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_portfolio_rebalancing_strategies(self, client: AsyncClient, auth_headers: dict):
        """Test various portfolio rebalancing strategies."""
        rebalancing_strategies = [
            {"method": "calendar", "frequency": "quarterly"},
            {"method": "threshold", "deviation": 0.05},
            {"method": "volatility_targeting", "target_vol": 0.15},
            {"method": "cppi", "floor": 0.80, "multiplier": 3}
        ]
        
        for strategy in rebalancing_strategies:
            response = await client.post(
                "/api/v1/backtest/strategies/rebalancing",
                json={
                    "name": f"Rebalancing {strategy['method']}",
                    "assets": ["SPY", "AGG"],
                    "target_weights": {"SPY": 0.6, "AGG": 0.4},
                    **strategy
                },
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_backtest_factor_investing_strategy(self, client: AsyncClient, auth_headers: dict):
        """Test factor-based investing strategy."""
        factor_strategy_data = {
            "name": "Multi-Factor Strategy",
            "strategy_type": "factor",
            "factors": {
                "value": {"metric": "pe_ratio", "weight": 0.3},
                "momentum": {"lookback": 252, "weight": 0.3},
                "quality": {"metric": "roe", "weight": 0.2},
                "low_volatility": {"window": 60, "weight": 0.2}
            },
            "universe": "SP500",
            "num_holdings": 50,
            "rebalance_frequency": "monthly"
        }
        response = await client.post(
            "/api/v1/backtest/strategies/factor",
            json=factor_strategy_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
