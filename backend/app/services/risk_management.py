"""Risk management service implementation."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import logging

from app.models.risk import (
    RiskMetrics, RiskLimit, StressTestScenario, 
    RiskModel, AuditLog
)
from app.models.portfolio import Portfolio, PortfolioPerformance
from app.models.position import Position
from app.models.stock import Stock, PriceHistory
from app.services.market_data import MarketDataService
from app.api.v1.websocket import manager

logger = logging.getLogger(__name__)


class RiskManagementService:
    """Service for portfolio risk management."""
    
    def __init__(self):
        self.market_service = MarketDataService()
    
    async def calculate_var(
        self,
        portfolio_id: str,
        confidence_level: float,
        time_horizon: int,
        db: AsyncSession,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk (VaR)."""
        # Get portfolio positions
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True
                )
            )
        )
        positions = positions_result.scalars().all()
        
        if not positions:
            return 0.0
        
        # Get historical returns for each position
        returns_data = []
        position_weights = []
        total_value = sum(pos.market_value or 0 for pos in positions)
        
        if total_value == 0:
            return 0.0
        
        for position in positions:
            # Get price history
            history_result = await db.execute(
                select(PriceHistory).join(Stock).where(
                    and_(
                        Stock.symbol == position.symbol,
                        PriceHistory.date >= datetime.utcnow() - timedelta(days=252)
                    )
                ).order_by(PriceHistory.date.asc())
            )
            price_history = history_result.scalars().all()
            
            if len(price_history) > 1:
                prices = [h.close_price for h in price_history]
                returns = np.diff(np.log(prices))
                returns_data.append(returns)
                position_weights.append((position.market_value or 0) / total_value)
        
        if not returns_data:
            return 0.0
        
        # Calculate portfolio returns
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[:min_length] for r in returns_data])
        portfolio_returns = np.dot(position_weights, returns_matrix)
        
        # Calculate VaR based on method
        if method == "historical":
            var_percentile = (1 - confidence_level) * 100
            var_return = np.percentile(portfolio_returns, var_percentile)
            var = -var_return * total_value * np.sqrt(time_horizon)
        
        elif method == "parametric":
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var_return = mean_return + z_score * std_return
            var = -var_return * total_value * np.sqrt(time_horizon)
        
        elif method == "monte_carlo":
            # Monte Carlo simulation
            num_simulations = 10000
            simulated_returns = np.random.normal(
                np.mean(portfolio_returns),
                np.std(portfolio_returns),
                num_simulations
            )
            var_percentile = (1 - confidence_level) * 100
            var_return = np.percentile(simulated_returns, var_percentile)
            var = -var_return * total_value * np.sqrt(time_horizon)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return max(0, var)  # VaR should be positive
    
    async def calculate_cvar(
        self,
        portfolio_id: str,
        confidence_level: float,
        time_horizon: int,
        db: AsyncSession
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        # First calculate VaR
        var = await self.calculate_var(
            portfolio_id, confidence_level, time_horizon, db
        )
        
        # Get portfolio positions and returns
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True
                )
            )
        )
        positions = positions_result.scalars().all()
        
        if not positions:
            return 0.0
        
        # Similar process to VaR but calculate expected shortfall
        returns_data = []
        position_weights = []
        total_value = sum(pos.market_value or 0 for pos in positions)
        
        if total_value == 0:
            return 0.0
        
        for position in positions:
            history_result = await db.execute(
                select(PriceHistory).join(Stock).where(
                    and_(
                        Stock.symbol == position.symbol,
                        PriceHistory.date >= datetime.utcnow() - timedelta(days=252)
                    )
                ).order_by(PriceHistory.date.asc())
            )
            price_history = history_result.scalars().all()
            
            if len(price_history) > 1:
                prices = [h.close_price for h in price_history]
                returns = np.diff(np.log(prices))
                returns_data.append(returns)
                position_weights.append((position.market_value or 0) / total_value)
        
        if not returns_data:
            return 0.0
        
        # Calculate portfolio returns
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[:min_length] for r in returns_data])
        portfolio_returns = np.dot(position_weights, returns_matrix)
        
        # Calculate CVaR (expected shortfall)
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(portfolio_returns, var_percentile)
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail_returns) > 0:
            cvar_return = np.mean(tail_returns)
            cvar = -cvar_return * total_value * np.sqrt(time_horizon)
        else:
            cvar = var  # If no tail returns, use VaR
        
        return max(0, cvar)
    
    async def calculate_sharpe_ratio(
        self,
        portfolio_id: str,
        risk_free_rate: float,
        db: AsyncSession,
        lookback_days: int = 252
    ) -> float:
        """Calculate Sharpe ratio."""
        # Get portfolio performance history
        perf_result = await db.execute(
            select(PortfolioPerformance).where(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date >= date.today() - timedelta(days=lookback_days)
                )
            ).order_by(PortfolioPerformance.date.asc())
        )
        performance_history = perf_result.scalars().all()
        
        if len(performance_history) < 2:
            return 0.0
        
        # Calculate daily returns
        values = [p.total_value for p in performance_history]
        returns = np.diff(values) / values[:-1]
        
        # Annualize returns and volatility
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
    
    async def calculate_portfolio_beta(
        self,
        portfolio_id: str,
        benchmark_symbol: str,
        db: AsyncSession,
        lookback_days: int = 252
    ) -> float:
        """Calculate portfolio beta against benchmark."""
        # Get portfolio returns
        perf_result = await db.execute(
            select(PortfolioPerformance).where(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date >= date.today() - timedelta(days=lookback_days)
                )
            ).order_by(PortfolioPerformance.date.asc())
        )
        performance_history = perf_result.scalars().all()
        
        if len(performance_history) < 2:
            return 1.0
        
        # Get benchmark returns
        benchmark_data = await self.market_service.fetch_historical_data(
            symbol=benchmark_symbol,
            start_date=date.today() - timedelta(days=lookback_days),
            end_date=date.today()
        )
        
        if not benchmark_data or len(benchmark_data) < 2:
            return 1.0
        
        # Calculate returns
        portfolio_values = [p.total_value for p in performance_history]
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        benchmark_prices = [d['close'] for d in benchmark_data]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
        
        # Align returns
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # Calculate beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        beta = covariance / benchmark_variance
        return beta
    
    async def calculate_maximum_drawdown(
        self,
        portfolio_id: str,
        lookback_days: int,
        db: AsyncSession
    ) -> float:
        """Calculate maximum drawdown."""
        # Get portfolio performance history
        perf_result = await db.execute(
            select(PortfolioPerformance).where(
                and_(
                    PortfolioPerformance.portfolio_id == portfolio_id,
                    PortfolioPerformance.date >= date.today() - timedelta(days=lookback_days)
                )
            ).order_by(PortfolioPerformance.date.asc())
        )
        performance_history = perf_result.scalars().all()
        
        if len(performance_history) < 2:
            return 0.0
        
        # Calculate drawdown
        values = np.array([p.total_value for p in performance_history])
        cumulative_returns = values / values[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        return abs(max_drawdown)  # Return as positive percentage
    
    async def check_risk_limits(
        self,
        portfolio_id: str,
        db: AsyncSession
    ) -> List[RiskLimit]:
        """Check for risk limit violations."""
        # Get active risk limits
        limits_result = await db.execute(
            select(RiskLimit).where(
                and_(
                    RiskLimit.portfolio_id == portfolio_id,
                    RiskLimit.is_active == True
                )
            )
        )
        risk_limits = limits_result.scalars().all()
        
        violations = []
        
        for limit in risk_limits:
            # Update current values based on limit type
            if limit.limit_type == "var_limit":
                current_var = await self.calculate_var(
                    portfolio_id, 0.95, 1, db
                )
                limit.current_value = current_var
            
            elif limit.limit_type == "concentration":
                # Calculate position concentration
                positions_result = await db.execute(
                    select(Position).where(
                        and_(
                            Position.portfolio_id == portfolio_id,
                            Position.is_open == True
                        )
                    )
                )
                positions = positions_result.scalars().all()
                
                if positions:
                    total_value = sum(pos.market_value or 0 for pos in positions)
                    if total_value > 0:
                        max_concentration = max(
                            (pos.market_value or 0) / total_value 
                            for pos in positions
                        )
                        limit.current_value = max_concentration
            
            # Check if limit is breached
            if limit.current_value > limit.max_value:
                limit.is_breached = True
                violations.append(limit)
            else:
                limit.is_breached = False
        
        await db.commit()
        
        return violations
    
    async def run_stress_test(
        self,
        portfolio_id: str,
        scenario_id: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Run stress test on portfolio."""
        # Get scenario
        scenario_result = await db.execute(
            select(StressTestScenario).where(
                StressTestScenario.id == scenario_id
            )
        )
        scenario = scenario_result.scalar_one_or_none()
        
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        # Get portfolio positions
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True
                )
            )
        )
        positions = positions_result.scalars().all()
        
        # Apply shocks
        portfolio_impact = 0.0
        position_impacts = []
        
        for position in positions:
            position_value = position.market_value or 0
            shock = 0.0
            
            # Apply market shocks based on asset characteristics
            if position.symbol in scenario.market_shocks:
                shock = scenario.market_shocks[position.symbol]
            elif position.asset_type == "stock":
                # Apply sector or market-wide shock
                shock = scenario.market_shocks.get("MARKET", 0)
            
            impact = position_value * shock
            portfolio_impact += impact
            
            position_impacts.append({
                "symbol": position.symbol,
                "current_value": position_value,
                "shock": shock,
                "impact": impact,
                "stressed_value": position_value + impact
            })
        
        # Create stress test result
        result = {
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "portfolio_impact": portfolio_impact,
            "portfolio_impact_percent": portfolio_impact / sum(p.market_value or 0 for p in positions) if positions else 0,
            "position_impacts": position_impacts,
            "timestamp": datetime.utcnow()
        }
        
        # Store result in RiskMetrics
        risk_metrics = RiskMetrics(
            portfolio_id=portfolio_id,
            calculated_at=datetime.utcnow(),
            stress_test_results=result
        )
        db.add(risk_metrics)
        await db.commit()
        
        return result
    
    async def run_monte_carlo_simulation(
        self,
        portfolio_id: str,
        num_simulations: int,
        time_horizon: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio."""
        # Get portfolio positions and historical data
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True
                )
            )
        )
        positions = positions_result.scalars().all()
        
        if not positions:
            return {"simulations": [], "percentiles": {}}
        
        # Get historical returns for correlation matrix
        returns_data = []
        position_weights = []
        total_value = sum(pos.market_value or 0 for pos in positions)
        
        for position in positions:
            history_result = await db.execute(
                select(PriceHistory).join(Stock).where(
                    and_(
                        Stock.symbol == position.symbol,
                        PriceHistory.date >= datetime.utcnow() - timedelta(days=252)
                    )
                ).order_by(PriceHistory.date.asc())
            )
            price_history = history_result.scalars().all()
            
            if len(price_history) > 1:
                prices = [h.close_price for h in price_history]
                returns = np.diff(np.log(prices))
                returns_data.append(returns)
                position_weights.append((position.market_value or 0) / total_value)
        
        if not returns_data:
            return {"simulations": [], "percentiles": {}}
        
        # Calculate correlation matrix
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[:min_length] for r in returns_data]).T
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Run simulations
        simulated_values = []
        
        for _ in range(num_simulations):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, time_horizon
            )
            
            # Calculate portfolio value path
            portfolio_returns = np.dot(random_returns, position_weights)
            cumulative_return = np.prod(1 + portfolio_returns)
            final_value = total_value * cumulative_return
            
            simulated_values.append(final_value)
        
        # Calculate percentiles
        percentiles = {
            "p5": np.percentile(simulated_values, 5),
            "p25": np.percentile(simulated_values, 25),
            "p50": np.percentile(simulated_values, 50),
            "p75": np.percentile(simulated_values, 75),
            "p95": np.percentile(simulated_values, 95)
        }
        
        return {
            "simulations": simulated_values,
            "percentiles": percentiles,
            "mean": np.mean(simulated_values),
            "std": np.std(simulated_values),
            "min": np.min(simulated_values),
            "max": np.max(simulated_values)
        }
    
    async def generate_risk_report(
        self,
        portfolio_id: str,
        include_stress_tests: bool,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        # Calculate all risk metrics
        var_95 = await self.calculate_var(portfolio_id, 0.95, 1, db)
        cvar_95 = await self.calculate_cvar(portfolio_id, 0.95, 1, db)
        sharpe_ratio = await self.calculate_sharpe_ratio(portfolio_id, 0.02, db)
        portfolio_beta = await self.calculate_portfolio_beta(portfolio_id, "SPY", db)
        max_drawdown = await self.calculate_maximum_drawdown(portfolio_id, 30, db)
        
        # Check risk limits
        risk_limit_violations = await self.check_risk_limits(portfolio_id, db)
        
        report = {
            "portfolio_id": portfolio_id,
            "generated_at": datetime.utcnow(),
            "var": var_95,
            "cvar": cvar_95,
            "sharpe_ratio": sharpe_ratio,
            "portfolio_beta": portfolio_beta,
            "max_drawdown": max_drawdown,
            "risk_limits": {
                "total": await db.execute(
                    select(func.count(RiskLimit.id)).where(
                        and_(
                            RiskLimit.portfolio_id == portfolio_id,
                            RiskLimit.is_active == True
                        )
                    )
                ).then(lambda r: r.scalar()),
                "breached": len(risk_limit_violations),
                "violations": [
                    {
                        "limit_type": v.limit_type,
                        "max_value": v.max_value,
                        "current_value": v.current_value,
                        "breach_amount": v.current_value - v.max_value
                    }
                    for v in risk_limit_violations
                ]
            },
            "recommendations": self._generate_recommendations(
                var_95, cvar_95, sharpe_ratio, max_drawdown, risk_limit_violations
            )
        }
        
        # Add stress test results if requested
        if include_stress_tests:
            # Run predefined stress scenarios
            stress_results = []
            scenarios_result = await db.execute(
                select(StressTestScenario).where(
                    StressTestScenario.is_active == True
                ).limit(3)
            )
            scenarios = scenarios_result.scalars().all()
            
            for scenario in scenarios:
                try:
                    result = await self.run_stress_test(
                        portfolio_id, str(scenario.id), db
                    )
                    stress_results.append(result)
                except Exception as e:
                    logger.error(f"Error running stress test: {e}")
            
            report["stress_tests"] = stress_results
        
        # Store report
        risk_metrics = RiskMetrics(
            portfolio_id=portfolio_id,
            calculated_at=datetime.utcnow(),
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            portfolio_beta=portfolio_beta,
            max_drawdown=max_drawdown,
            risk_metrics=report
        )
        db.add(risk_metrics)
        await db.commit()
        
        return report
    
    def _generate_recommendations(
        self,
        var: float,
        cvar: float,
        sharpe_ratio: float,
        max_drawdown: float,
        violations: List[RiskLimit]
    ) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # VaR/CVaR recommendations
        if cvar > var * 1.5:
            recommendations.append(
                "High tail risk detected. Consider hedging strategies or "
                "reducing position sizes in volatile assets."
            )
        
        # Sharpe ratio recommendations
        if sharpe_ratio < 0.5:
            recommendations.append(
                "Low risk-adjusted returns. Review portfolio composition "
                "and consider rebalancing to improve efficiency."
            )
        elif sharpe_ratio < 0:
            recommendations.append(
                "Negative risk-adjusted returns. Urgent portfolio review needed."
            )
        
        # Drawdown recommendations
        if max_drawdown > 0.2:
            recommendations.append(
                f"Maximum drawdown of {max_drawdown:.1%} exceeds 20%. "
                "Consider implementing stop-loss strategies."
            )
        
        # Risk limit recommendations
        if violations:
            recommendations.append(
                f"{len(violations)} risk limits breached. Immediate action required "
                "to bring portfolio within compliance."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "Portfolio risk metrics are within acceptable ranges. "
                "Continue monitoring for changes."
            )
        
        return recommendations
    
    async def check_risk_alerts(
        self,
        portfolio_id: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Check for risk-based alerts."""
        alerts = []
        
        # Check risk limit breaches
        violations = await self.check_risk_limits(portfolio_id, db)
        for violation in violations:
            alerts.append({
                "type": "risk_limit_breach",
                "severity": "high",
                "limit_type": violation.limit_type,
                "current_value": violation.current_value,
                "limit_value": violation.max_value,
                "message": f"Risk limit breached: {violation.limit_type}"
            })
        
        # Check drawdown
        max_drawdown = await self.calculate_maximum_drawdown(portfolio_id, 30, db)
        if max_drawdown > 0.15:
            alerts.append({
                "type": "drawdown_alert",
                "severity": "high" if max_drawdown > 0.25 else "medium",
                "drawdown": max_drawdown,
                "message": f"Portfolio drawdown: {max_drawdown:.1%}"
            })
        
        # Check VaR spike
        current_var = await self.calculate_var(portfolio_id, 0.95, 1, db)
        
        # Get historical VaR
        hist_result = await db.execute(
            select(RiskMetrics).where(
                RiskMetrics.portfolio_id == portfolio_id
            ).order_by(RiskMetrics.calculated_at.desc()).limit(10)
        )
        historical = hist_result.scalars().all()
        
        if historical:
            avg_var = np.mean([h.var_95 for h in historical if h.var_95])
            if current_var > avg_var * 1.5:
                alerts.append({
                    "type": "var_spike",
                    "severity": "medium",
                    "current_var": current_var,
                    "average_var": avg_var,
                    "message": "Value at Risk has increased significantly"
                })
        
        return alerts
    
    async def check_drawdown_alerts(
        self,
        portfolio_id: str,
        threshold: float,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Check for drawdown alerts."""
        drawdown = await self.calculate_maximum_drawdown(portfolio_id, 30, db)
        
        if drawdown > threshold:
            return [{
                "type": "drawdown_alert",
                "severity": "high",
                "drawdown": drawdown,
                "threshold": threshold,
                "message": f"Portfolio drawdown {drawdown:.1%} exceeds threshold {threshold:.1%}"
            }]
        
        return []
    
    async def update_risk_limit_value(
        self,
        limit_id: str,
        new_value: float,
        db: AsyncSession
    ):
        """Update current value of risk limit."""
        result = await db.execute(
            select(RiskLimit).where(RiskLimit.id == limit_id)
        )
        risk_limit = result.scalar_one_or_none()
        
        if risk_limit:
            risk_limit.current_value = new_value
            risk_limit.is_breached = new_value > risk_limit.max_value
            risk_limit.updated_at = datetime.utcnow()
            await db.commit()
    
    async def forecast_volatility_garch(
        self,
        portfolio_id: str,
        forecast_horizon: int,
        db: AsyncSession
    ) -> List[float]:
        """Forecast volatility using GARCH model."""
        # Get portfolio returns
        perf_result = await db.execute(
            select(PortfolioPerformance).where(
                PortfolioPerformance.portfolio_id == portfolio_id
            ).order_by(PortfolioPerformance.date.asc()).limit(252)
        )
        performance_history = perf_result.scalars().all()
        
        if len(performance_history) < 30:
            # Not enough data for GARCH
            return [0.2] * forecast_horizon  # Default volatility
        
        # Calculate returns
        values = [p.total_value for p in performance_history]
        returns = np.diff(np.log(values))
        
        # Simple GARCH(1,1) implementation
        # In production, use arch package
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        variance = np.var(returns)
        forecasts = []
        
        for h in range(forecast_horizon):
            # Forecast variance
            variance = omega + alpha * returns[-1]**2 + beta * variance
            volatility = np.sqrt(variance * 252)  # Annualized
            forecasts.append(volatility)
        
        return forecasts
    
    async def calculate_copula_correlations(
        self,
        portfolio_id: str,
        copula_type: str,
        db: AsyncSession
    ) -> np.ndarray:
        """Calculate correlations using copula models."""
        # Get position returns
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True
                )
            )
        )
        positions = positions_result.scalars().all()
        
        if len(positions) < 2:
            return np.array([[1.0]])
        
        # Get returns data
        returns_data = []
        
        for position in positions:
            history_result = await db.execute(
                select(PriceHistory).join(Stock).where(
                    Stock.symbol == position.symbol
                ).order_by(PriceHistory.date.asc()).limit(252)
            )
            price_history = history_result.scalars().all()
            
            if len(price_history) > 1:
                prices = [h.close_price for h in price_history]
                returns = np.diff(np.log(prices))
                returns_data.append(returns)
        
        if len(returns_data) < 2:
            return np.eye(len(positions))
        
        # Align returns
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r[:min_length] for r in returns_data]).T
        
        # For now, return standard correlation
        # In production, implement proper copula estimation
        correlation_matrix = np.corrcoef(returns_matrix.T)
        
        return correlation_matrix
    
    async def calculate_options_risk(
        self,
        portfolio_id: str,
        db: AsyncSession
    ) -> Dict[str, float]:
        """Calculate risk metrics for options positions."""
        # Get options positions
        positions_result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == portfolio_id,
                    Position.asset_type == "option",
                    Position.is_open == True
                )
            )
        )
        options_positions = positions_result.scalars().all()
        
        # Aggregate Greeks
        total_delta = sum(pos.option_delta * pos.quantity for pos in options_positions if pos.option_delta)
        total_gamma = sum(pos.option_gamma * pos.quantity for pos in options_positions if pos.option_gamma)
        total_vega = sum(pos.option_vega * pos.quantity for pos in options_positions if pos.option_vega)
        total_theta = sum(pos.option_theta * pos.quantity for pos in options_positions if pos.option_theta)
        
        # Calculate risk exposures
        underlying_price_move = 0.01  # 1% move
        volatility_move = 0.01  # 1% vol move
        
        return {
            "delta_exposure": total_delta,
            "gamma_risk": total_gamma * (underlying_price_move ** 2) / 2,
            "vega_risk": total_vega * volatility_move,
            "theta_decay": total_theta,
            "total_options_value": sum(pos.market_value or 0 for pos in options_positions)
        }
    
    async def get_risk_metrics_history(
        self,
        portfolio_id: str,
        days: int,
        db: AsyncSession
    ) -> List[RiskMetrics]:
        """Get historical risk metrics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(RiskMetrics).where(
                and_(
                    RiskMetrics.portfolio_id == portfolio_id,
                    RiskMetrics.calculated_at >= cutoff_date
                )
            ).order_by(RiskMetrics.calculated_at.desc())
        )
        
        return result.scalars().all()
