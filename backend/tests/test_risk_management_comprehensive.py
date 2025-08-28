"""Comprehensive risk management tests covering all edge cases."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.risk import RiskMetrics, RiskLimit, StressTestScenario, RiskModel
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.services.risk_management import RiskManagementService


class TestRiskCalculations:
    """Test risk calculation algorithms."""
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_var(self, db: AsyncSession, test_portfolio):
        """Test Value at Risk calculation."""
        risk_service = RiskManagementService()
        
        # Create test positions
        positions = [
            Position(
                portfolio_id=test_portfolio.id,
                symbol="AAPL",
                quantity=100,
                current_price=150.0,
                avg_cost=140.0
            ),
            Position(
                portfolio_id=test_portfolio.id,
                symbol="GOOGL",
                quantity=50,
                current_price=2800.0,
                avg_cost=2750.0
            )
        ]
        
        for position in positions:
            db.add(position)
        await db.commit()
        
        # Calculate VaR
        var_95 = await risk_service.calculate_var(
            portfolio_id=test_portfolio.id,
            confidence_level=0.95,
            time_horizon=1,
            db=db
        )
        
        assert var_95 is not None
        assert var_95 > 0
        assert var_95 < test_portfolio.total_value * 0.1  # Reasonable VaR
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_cvar(self, db: AsyncSession, test_portfolio):
        """Test Conditional Value at Risk calculation."""
        risk_service = RiskManagementService()
        
        cvar_95 = await risk_service.calculate_cvar(
            portfolio_id=test_portfolio.id,
            confidence_level=0.95,
            time_horizon=1,
            db=db
        )
        
        var_95 = await risk_service.calculate_var(
            portfolio_id=test_portfolio.id,
            confidence_level=0.95,
            time_horizon=1,
            db=db
        )
        
        assert cvar_95 is not None
        assert cvar_95 >= var_95  # CVaR should be >= VaR
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, db: AsyncSession, test_portfolio):
        """Test Sharpe ratio calculation."""
        risk_service = RiskManagementService()
        
        sharpe = await risk_service.calculate_sharpe_ratio(
            portfolio_id=test_portfolio.id,
            risk_free_rate=0.02,
            db=db
        )
        
        assert sharpe is not None
        assert -5 < sharpe < 5  # Reasonable Sharpe ratio range
    
    @pytest.mark.asyncio
    async def test_calculate_beta(self, db: AsyncSession, test_portfolio):
        """Test portfolio beta calculation."""
        risk_service = RiskManagementService()
        
        beta = await risk_service.calculate_portfolio_beta(
            portfolio_id=test_portfolio.id,
            benchmark_symbol="SPY",
            db=db
        )
        
        assert beta is not None
        assert 0 < beta < 3  # Reasonable beta range
    
    @pytest.mark.asyncio
    async def test_calculate_maximum_drawdown(self, db: AsyncSession, test_portfolio):
        """Test maximum drawdown calculation."""
        risk_service = RiskManagementService()
        
        max_dd = await risk_service.calculate_maximum_drawdown(
            portfolio_id=test_portfolio.id,
            lookback_days=30,
            db=db
        )
        
        assert max_dd is not None
        assert 0 <= max_dd <= 1  # Drawdown is a percentage


class TestRiskLimits:
    """Test risk limit management."""
    
    @pytest.mark.asyncio
    async def test_create_risk_limit(self, db: AsyncSession, test_portfolio):
        """Test creating risk limits."""
        risk_limit = RiskLimit(
            portfolio_id=test_portfolio.id,
            limit_type="position_size",
            max_value=10000.0,
            current_value=5000.0,
            is_active=True
        )
        
        db.add(risk_limit)
        await db.commit()
        
        assert risk_limit.id is not None
        assert risk_limit.utilization_percentage == 50.0
    
    @pytest.mark.asyncio
    async def test_check_risk_limits(self, db: AsyncSession, test_portfolio):
        """Test checking risk limit violations."""
        risk_service = RiskManagementService()
        
        # Create risk limits
        limits = [
            RiskLimit(
                portfolio_id=test_portfolio.id,
                limit_type="max_position_size",
                max_value=10000.0,
                current_value=12000.0,  # Violated
                is_active=True
            ),
            RiskLimit(
                portfolio_id=test_portfolio.id,
                limit_type="max_leverage",
                max_value=2.0,
                current_value=1.5,  # Not violated
                is_active=True
            )
        ]
        
        for limit in limits:
            db.add(limit)
        await db.commit()
        
        violations = await risk_service.check_risk_limits(
            portfolio_id=test_portfolio.id,
            db=db
        )
        
        assert len(violations) == 1
        assert violations[0].limit_type == "max_position_size"
    
    @pytest.mark.asyncio
    async def test_update_risk_limit_values(self, db: AsyncSession, test_portfolio):
        """Test updating risk limit current values."""
        risk_service = RiskManagementService()
        
        # Create risk limit
        risk_limit = RiskLimit(
            portfolio_id=test_portfolio.id,
            limit_type="concentration",
            max_value=0.3,  # 30% max concentration
            current_value=0.25,
            is_active=True
        )
        
        db.add(risk_limit)
        await db.commit()
        
        # Update current value
        await risk_service.update_risk_limit_value(
            limit_id=risk_limit.id,
            new_value=0.35,  # Now violating
            db=db
        )
        
        await db.refresh(risk_limit)
        assert risk_limit.current_value == 0.35
        assert risk_limit.is_breached is True


class TestStressTesting:
    """Test stress testing functionality."""
    
    @pytest.mark.asyncio
    async def test_create_stress_scenario(self, db: AsyncSession):
        """Test creating stress test scenarios."""
        scenario = StressTestScenario(
            name="Market Crash 2008",
            description="Simulates 2008 financial crisis conditions",
            scenario_type="historical",
            market_shocks={
                "SPY": -0.50,
                "VIX": 2.5,
                "DXY": 0.15
            },
            is_active=True
        )
        
        db.add(scenario)
        await db.commit()
        
        assert scenario.id is not None
        assert scenario.market_shocks["SPY"] == -0.50
    
    @pytest.mark.asyncio
    async def test_run_stress_test(self, db: AsyncSession, test_portfolio):
        """Test running stress test on portfolio."""
        risk_service = RiskManagementService()
        
        # Create scenario
        scenario = StressTestScenario(
            name="Tech Sector Crash",
            description="Tech stocks drop 30%",
            scenario_type="hypothetical",
            market_shocks={
                "TECH": -0.30,
                "NASDAQ": -0.25
            }
        )
        
        db.add(scenario)
        await db.commit()
        
        # Run stress test
        results = await risk_service.run_stress_test(
            portfolio_id=test_portfolio.id,
            scenario_id=scenario.id,
            db=db
        )
        
        assert results is not None
        assert "portfolio_impact" in results
        assert "position_impacts" in results
        assert results["portfolio_impact"] < 0  # Negative impact expected
    
    @pytest.mark.asyncio
    async def test_monte_carlo_stress_test(self, db: AsyncSession, test_portfolio):
        """Test Monte Carlo simulation for stress testing."""
        risk_service = RiskManagementService()
        
        results = await risk_service.run_monte_carlo_simulation(
            portfolio_id=test_portfolio.id,
            num_simulations=1000,
            time_horizon=30,
            db=db
        )
        
        assert results is not None
        assert "simulations" in results
        assert len(results["simulations"]) == 1000
        assert "percentiles" in results
        assert results["percentiles"]["p5"] < results["percentiles"]["p50"]
        assert results["percentiles"]["p50"] < results["percentiles"]["p95"]


class TestRiskModels:
    """Test risk model implementations."""
    
    @pytest.mark.asyncio
    async def test_create_risk_model(self, db: AsyncSession):
        """Test creating risk models."""
        model = RiskModel(
            name="Factor Risk Model",
            model_type="factor",
            parameters={
                "factors": ["market", "size", "value", "momentum"],
                "lookback_period": 252
            },
            is_active=True
        )
        
        db.add(model)
        await db.commit()
        
        assert model.id is not None
        assert len(model.parameters["factors"]) == 4
    
    @pytest.mark.asyncio
    async def test_garch_volatility_model(self, db: AsyncSession, test_portfolio):
        """Test GARCH volatility forecasting."""
        risk_service = RiskManagementService()
        
        volatility_forecast = await risk_service.forecast_volatility_garch(
            portfolio_id=test_portfolio.id,
            forecast_horizon=5,
            db=db
        )
        
        assert volatility_forecast is not None
        assert len(volatility_forecast) == 5
        assert all(v > 0 for v in volatility_forecast)
    
    @pytest.mark.asyncio
    async def test_copula_correlation_model(self, db: AsyncSession, test_portfolio):
        """Test copula-based correlation modeling."""
        risk_service = RiskManagementService()
        
        correlation_matrix = await risk_service.calculate_copula_correlations(
            portfolio_id=test_portfolio.id,
            copula_type="gaussian",
            db=db
        )
        
        assert correlation_matrix is not None
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Symmetric
        assert np.all(np.diag(correlation_matrix) == 1)  # Diagonal = 1


class TestRiskAlerts:
    """Test risk alerting system."""
    
    @pytest.mark.asyncio
    async def test_risk_breach_alert(self, db: AsyncSession, test_portfolio):
        """Test alerts on risk limit breaches."""
        risk_service = RiskManagementService()
        
        # Create breached limit
        risk_limit = RiskLimit(
            portfolio_id=test_portfolio.id,
            limit_type="var_limit",
            max_value=50000.0,
            current_value=60000.0,
            is_active=True
        )
        
        db.add(risk_limit)
        await db.commit()
        
        # Check for alerts
        alerts = await risk_service.check_risk_alerts(
            portfolio_id=test_portfolio.id,
            db=db
        )
        
        assert len(alerts) > 0
        assert any(alert["type"] == "risk_limit_breach" for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_drawdown_alert(self, db: AsyncSession, test_portfolio):
        """Test drawdown alerts."""
        risk_service = RiskManagementService()
        
        # Simulate drawdown
        with patch.object(risk_service, 'calculate_maximum_drawdown', return_value=0.25):
            alerts = await risk_service.check_drawdown_alerts(
                portfolio_id=test_portfolio.id,
                threshold=0.20,
                db=db
            )
        
        assert len(alerts) > 0
        assert alerts[0]["severity"] == "high"


class TestRiskReporting:
    """Test risk reporting functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_risk_report(self, db: AsyncSession, test_portfolio):
        """Test comprehensive risk report generation."""
        risk_service = RiskManagementService()
        
        report = await risk_service.generate_risk_report(
            portfolio_id=test_portfolio.id,
            include_stress_tests=True,
            db=db
        )
        
        assert report is not None
        assert "var" in report
        assert "cvar" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "risk_limits" in report
        assert "recommendations" in report
    
    @pytest.mark.asyncio
    async def test_historical_risk_metrics(self, db: AsyncSession, test_portfolio):
        """Test retrieving historical risk metrics."""
        risk_service = RiskManagementService()
        
        # Create historical metrics
        for i in range(5):
            metric = RiskMetrics(
                portfolio_id=test_portfolio.id,
                calculated_at=datetime.utcnow() - timedelta(days=i),
                var_95=50000 + i * 1000,
                cvar_95=55000 + i * 1000,
                sharpe_ratio=1.5 - i * 0.1,
                portfolio_beta=1.1
            )
            db.add(metric)
        
        await db.commit()
        
        # Get historical data
        history = await risk_service.get_risk_metrics_history(
            portfolio_id=test_portfolio.id,
            days=7,
            db=db
        )
        
        assert len(history) == 5
        assert history[0].var_95 < history[-1].var_95  # Increasing VaR


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_risk(self, db: AsyncSession, test_portfolio):
        """Test risk calculations on empty portfolio."""
        risk_service = RiskManagementService()
        
        var = await risk_service.calculate_var(
            portfolio_id=test_portfolio.id,
            confidence_level=0.95,
            time_horizon=1,
            db=db
        )
        
        assert var == 0
    
    @pytest.mark.asyncio
    async def test_single_position_risk(self, db: AsyncSession, test_portfolio):
        """Test risk calculations with single position."""
        risk_service = RiskManagementService()
        
        # Add single position
        position = Position(
            portfolio_id=test_portfolio.id,
            symbol="AAPL",
            quantity=100,
            current_price=150.0,
            avg_cost=140.0
        )
        db.add(position)
        await db.commit()
        
        var = await risk_service.calculate_var(
            portfolio_id=test_portfolio.id,
            confidence_level=0.95,
            time_horizon=1,
            db=db
        )
        
        assert var > 0
        assert var < position.market_value * 0.2
    
    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, db: AsyncSession, test_portfolio):
        """Test risk calculations under extreme conditions."""
        risk_service = RiskManagementService()
        
        # Create extreme scenario
        scenario = StressTestScenario(
            name="Black Swan Event",
            description="Extreme market crash",
            scenario_type="hypothetical",
            market_shocks={
                "MARKET": -0.90,  # 90% crash
                "VIX": 5.0  # 500% VIX increase
            }
        )
        
        db.add(scenario)
        await db.commit()
        
        results = await risk_service.run_stress_test(
            portfolio_id=test_portfolio.id,
            scenario_id=scenario.id,
            db=db
        )
        
        assert results["portfolio_impact"] <= -0.70  # At least 70% loss
    
    @pytest.mark.asyncio
    async def test_risk_calculation_with_options(self, db: AsyncSession, test_portfolio):
        """Test risk calculations including options positions."""
        risk_service = RiskManagementService()
        
        # Add option position
        option_position = Position(
            portfolio_id=test_portfolio.id,
            symbol="AAPL_CALL_150",
            asset_type="option",
            quantity=10,
            current_price=5.0,
            avg_cost=4.0,
            option_delta=0.6,
            option_gamma=0.02,
            option_vega=0.15
        )
        
        db.add(option_position)
        await db.commit()
        
        # Calculate Greeks-adjusted risk
        risk_metrics = await risk_service.calculate_options_risk(
            portfolio_id=test_portfolio.id,
            db=db
        )
        
        assert "delta_exposure" in risk_metrics
        assert "gamma_risk" in risk_metrics
        assert "vega_risk" in risk_metrics