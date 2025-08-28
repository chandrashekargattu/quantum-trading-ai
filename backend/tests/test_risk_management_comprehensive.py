"""Comprehensive risk management tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.risk import RiskMetrics, RiskLimits, RiskAlert
from app.models.portfolio import Portfolio
from fastapi import status


class TestRiskManagement:
    """Test risk management with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_calculate_portfolio_var(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test Value at Risk (VaR) calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/var",
            params={
                "confidence_level": 0.95,
                "time_horizon": 1,
                "method": "historical"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "var_95" in data
        assert "var_99" in data
        assert data["var_95"] < 0  # VaR should be negative

    @pytest.mark.asyncio
    async def test_calculate_cvar(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test Conditional Value at Risk (CVaR) calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/cvar",
            params={"confidence_level": 0.95},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "cvar_95" in data
        assert data["cvar_95"] <= data["var_95"]  # CVaR should be worse than VaR

    @pytest.mark.asyncio
    async def test_parametric_var(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test parametric VaR calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/var",
            params={
                "method": "parametric",
                "confidence_level": 0.95
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_monte_carlo_var(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test Monte Carlo VaR calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/var",
            params={
                "method": "monte_carlo",
                "simulations": 10000,
                "confidence_level": 0.95
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_portfolio_beta(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test portfolio beta calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/beta",
            params={"benchmark": "SPY"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "beta" in data
        assert "r_squared" in data

    @pytest.mark.asyncio
    async def test_tracking_error(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test tracking error calculation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/tracking-error",
            params={"benchmark": "SPY", "period": 252},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "tracking_error" in data
        assert "information_ratio" in data

    @pytest.mark.asyncio
    async def test_downside_risk_metrics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test downside risk metrics."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/downside-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "downside_deviation" in data
        assert "sortino_ratio" in data
        assert "omega_ratio" in data
        assert "gain_loss_ratio" in data

    @pytest.mark.asyncio
    async def test_tail_risk_analysis(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test tail risk analysis using Extreme Value Theory."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/tail-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expected_shortfall" in data
        assert "tail_index" in data
        assert "extreme_quantiles" in data

    @pytest.mark.asyncio
    async def test_stress_testing_scenarios(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test predefined stress testing scenarios."""
        scenarios = [
            {"name": "Market Crash", "equity_shock": -0.30, "vol_shock": 2.0},
            {"name": "Interest Rate Spike", "rate_shock": 0.02, "duration": 5},
            {"name": "Currency Crisis", "fx_shock": -0.20}
        ]
        
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/stress-test",
            json={"scenarios": scenarios},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "scenario_results" in data
        assert len(data["scenario_results"]) == len(scenarios)

    @pytest.mark.asyncio
    async def test_historical_stress_testing(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test historical scenario stress testing."""
        historical_scenarios = [
            {"name": "Black Monday 1987", "date": "1987-10-19"},
            {"name": "Asian Crisis 1997", "start": "1997-07-01", "end": "1997-12-31"},
            {"name": "Lehman Collapse 2008", "date": "2008-09-15"},
            {"name": "Flash Crash 2010", "date": "2010-05-06"},
            {"name": "COVID-19 2020", "start": "2020-02-20", "end": "2020-03-23"}
        ]
        
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/historical-stress",
            json={"scenarios": historical_scenarios},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_correlation_breakdown_detection(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test correlation breakdown detection in crisis."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/correlation-breakdown",
            params={"crisis_threshold": -0.10},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "normal_correlation" in data
        assert "crisis_correlation" in data
        assert "breakdown_probability" in data

    @pytest.mark.asyncio
    async def test_liquidity_risk_assessment(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test liquidity risk assessment."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/liquidity-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "liquidity_score" in data
        assert "days_to_liquidate" in data
        assert "market_impact_cost" in data
        assert "illiquid_positions" in data

    @pytest.mark.asyncio
    async def test_concentration_risk(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test concentration risk metrics."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/concentration",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "herfindahl_index" in data
        assert "top_5_concentration" in data
        assert "sector_concentration" in data
        assert "single_stock_limits_breach" in data

    @pytest.mark.asyncio
    async def test_counterparty_risk(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test counterparty risk assessment."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/counterparty-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_exposure" in data
        assert "counterparty_limits" in data
        assert "collateral_requirements" in data

    @pytest.mark.asyncio
    async def test_margin_risk_calculation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test margin risk and requirements."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/margin-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "current_margin" in data
        assert "margin_requirement" in data
        assert "excess_margin" in data
        assert "margin_call_price" in data

    @pytest.mark.asyncio
    async def test_leverage_monitoring(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test leverage monitoring and limits."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/leverage",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "gross_leverage" in data
        assert "net_leverage" in data
        assert "leverage_limit" in data
        assert "leverage_utilization" in data

    @pytest.mark.asyncio
    async def test_greeks_risk_aggregation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test options Greeks risk aggregation."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/greeks",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "portfolio_delta" in data
        assert "portfolio_gamma" in data
        assert "portfolio_vega" in data
        assert "portfolio_theta" in data

    @pytest.mark.asyncio
    async def test_scenario_analysis_custom(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test custom scenario analysis."""
        custom_scenario = {
            "name": "Custom Scenario",
            "shocks": {
                "SPY": -0.15,
                "TLT": 0.05,
                "GLD": 0.10,
                "VIX": 1.5
            },
            "correlations": {
                "breakdown_factor": 1.5
            }
        }
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/scenario",
            json=custom_scenario,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_risk_decomposition(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk decomposition by various factors."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/decomposition",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "asset_contributions" in data
        assert "factor_contributions" in data
        assert "systematic_risk" in data
        assert "idiosyncratic_risk" in data

    @pytest.mark.asyncio
    async def test_risk_limits_setup(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test setting up risk limits."""
        risk_limits = {
            "var_limit": -0.02,  # 2% daily VaR limit
            "leverage_limit": 2.0,
            "concentration_limit": 0.10,  # 10% single position
            "sector_limit": 0.30,  # 30% sector
            "loss_limit_daily": -0.03,
            "loss_limit_monthly": -0.10
        }
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/limits",
            json=risk_limits,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_risk_limit_breach_detection(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk limit breach detection."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/limit-breaches",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "active_breaches" in data
        assert "breach_history" in data

    @pytest.mark.asyncio
    async def test_risk_alerts_configuration(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk alert configuration."""
        alert_config = {
            "alerts": [
                {
                    "type": "var_breach",
                    "threshold": -0.015,
                    "notification": "email"
                },
                {
                    "type": "drawdown",
                    "threshold": -0.05,
                    "notification": "sms"
                },
                {
                    "type": "correlation_spike",
                    "threshold": 0.8,
                    "notification": "webhook"
                }
            ]
        }
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/alerts",
            json=alert_config,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test real-time risk monitoring."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/real-time",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "current_var" in data
        assert "current_exposure" in data
        assert "intraday_pnl" in data
        assert "risk_utilization" in data

    @pytest.mark.asyncio
    async def test_risk_attribution(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk attribution analysis."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/attribution",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "var_attribution" in data
        assert "pnl_attribution" in data
        assert "factor_attribution" in data

    @pytest.mark.asyncio
    async def test_regime_dependent_risk(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test regime-dependent risk models."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/regime-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "current_regime" in data
        assert "regime_probabilities" in data
        assert "regime_specific_risks" in data

    @pytest.mark.asyncio
    async def test_copula_risk_modeling(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test copula-based risk modeling."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/copula-risk",
            params={"copula_type": "gaussian"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "joint_risk_measures" in data
        assert "tail_dependence" in data

    @pytest.mark.asyncio
    async def test_dynamic_hedging_recommendations(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test dynamic hedging recommendations."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/hedging",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "recommended_hedges" in data
        assert "hedge_effectiveness" in data
        assert "cost_benefit_analysis" in data

    @pytest.mark.asyncio
    async def test_risk_budgeting(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk budgeting allocation."""
        risk_budget = {
            "total_risk_budget": 0.15,  # 15% annual volatility
            "allocations": {
                "equities": 0.60,
                "fixed_income": 0.20,
                "alternatives": 0.20
            }
        }
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/risk-budget",
            json=risk_budget,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_risk_parity_optimization(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk parity portfolio optimization."""
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/risk-parity",
            json={"target_volatility": 0.10},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "optimal_weights" in data
        assert "risk_contributions" in data

    @pytest.mark.asyncio
    async def test_black_swan_detection(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test black swan event detection."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/black-swan",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "anomaly_score" in data
        assert "outlier_events" in data
        assert "tail_risk_indicators" in data

    @pytest.mark.asyncio
    async def test_systemic_risk_exposure(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test systemic risk exposure measurement."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/systemic-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "marginal_expected_shortfall" in data
        assert "systemic_risk_contribution" in data
        assert "network_centrality" in data

    @pytest.mark.asyncio
    async def test_climate_risk_assessment(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test climate risk assessment."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/climate-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "carbon_footprint" in data
        assert "transition_risk_score" in data
        assert "physical_risk_score" in data

    @pytest.mark.asyncio
    async def test_operational_risk_metrics(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test operational risk metrics."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/operational-risk",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "trading_errors" in data
        assert "system_downtime" in data
        assert "compliance_breaches" in data

    @pytest.mark.asyncio
    async def test_model_risk_validation(self, client: AsyncClient, auth_headers: dict):
        """Test model risk validation."""
        model_validation_data = {
            "model_type": "var_calculation",
            "backtesting_period": 252,
            "confidence_levels": [0.95, 0.99]
        }
        response = await client.post(
            "/api/v1/risk/model-validation",
            json=model_validation_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "kupiec_test" in data
        assert "christoffersen_test" in data
        assert "breach_frequency" in data

    @pytest.mark.asyncio
    async def test_risk_report_generation(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test comprehensive risk report generation."""
        response = await client.post(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/report",
            json={
                "report_type": "comprehensive",
                "format": "pdf",
                "include_sections": [
                    "executive_summary",
                    "var_analysis",
                    "stress_testing",
                    "concentration_risk",
                    "recommendations"
                ]
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_risk_dashboard_data(self, client: AsyncClient, auth_headers: dict, test_portfolio):
        """Test risk dashboard data endpoint."""
        response = await client.get(
            f"/api/v1/risk/portfolio/{test_portfolio.id}/dashboard",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "summary_metrics" in data
        assert "risk_charts" in data
        assert "alerts" in data
        assert "recommendations" in data
