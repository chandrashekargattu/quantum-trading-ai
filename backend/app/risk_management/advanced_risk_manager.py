"""
Advanced Risk Management with Tail Risk Modeling

Implements sophisticated risk management techniques used by top quant firms:
- Extreme Value Theory (EVT) for tail risk
- Copula models for dependency structures
- Regime-switching models
- Dynamic hedging strategies
- Stress testing and scenario analysis
- Real-time risk monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r, norm, t
import torch
import torch.nn as nn
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from copulas.multivariate import GaussianMultivariate, VineMultivariate
from copulas.univariate import GaussianKDE
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    
    # Standard risk metrics
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    
    # Tail risk metrics
    expected_tail_loss: float
    tail_risk_contribution: Dict[str, float]
    extreme_downside_risk: float
    
    # Greeks and sensitivities
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    
    # Portfolio metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Stress test results
    stress_scenarios: Dict[str, float]
    
    # Regime-specific risks
    regime_risks: Dict[str, float]
    
    # Correlation and dependency
    correlation_risk: float
    tail_dependence: float
    
    # Liquidity risk
    liquidity_score: float
    market_impact: float
    
    # Model risk
    model_confidence: float
    parameter_uncertainty: float


@dataclass
class TailRiskEvent:
    """Represents a tail risk event."""
    
    event_type: str  # 'market_crash', 'flash_crash', 'liquidity_crisis', etc.
    probability: float
    impact: float  # Expected loss
    duration: int  # Expected duration in days
    affected_assets: List[str]
    hedging_strategy: Dict[str, Any]
    confidence: float


class ExtremeValueAnalyzer:
    """
    Analyzes extreme market events using Extreme Value Theory (EVT).
    Models tail distributions more accurately than normal distributions.
    """
    
    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile
        self.threshold = None
        self.shape_param = None  # Xi parameter
        self.scale_param = None  # Beta parameter
        self.tail_index = None
        
    def fit_tail_distribution(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit Generalized Pareto Distribution to tail returns."""
        # Set threshold
        self.threshold = np.quantile(returns, self.threshold_quantile)
        
        # Get exceedances
        exceedances = returns[returns > self.threshold] - self.threshold
        
        if len(exceedances) < 10:
            logger.warning("Insufficient tail observations for EVT")
            return {}
        
        # Fit GPD using MLE
        shape, loc, scale = genextreme.fit(exceedances)
        
        self.shape_param = shape
        self.scale_param = scale
        
        # Calculate tail index (alpha = 1/xi for xi > 0)
        if shape > 0:
            self.tail_index = 1 / shape
        else:
            self.tail_index = float('inf')
        
        return {
            'shape': shape,
            'scale': scale,
            'threshold': self.threshold,
            'tail_index': self.tail_index,
            'n_exceedances': len(exceedances)
        }
    
    def calculate_extreme_var(
        self,
        confidence_level: float = 0.99,
        horizon: int = 1
    ) -> float:
        """Calculate VaR using EVT for extreme confidence levels."""
        if self.shape_param is None:
            raise ValueError("Model not fitted. Call fit_tail_distribution first.")
        
        # Probability of exceeding threshold
        p_exceed = 1 - self.threshold_quantile
        
        # EVT VaR formula
        if self.shape_param != 0:
            var = self.threshold + (self.scale_param / self.shape_param) * (
                ((1 - confidence_level) / p_exceed) ** (-self.shape_param) - 1
            )
        else:
            # Exponential tail
            var = self.threshold - self.scale_param * np.log(
                (1 - confidence_level) / p_exceed
            )
        
        # Scale for time horizon
        var *= np.sqrt(horizon)
        
        return var
    
    def calculate_expected_shortfall(
        self,
        confidence_level: float = 0.99
    ) -> float:
        """Calculate Expected Shortfall (CVaR) using EVT."""
        var = self.calculate_extreme_var(confidence_level)
        
        if self.shape_param < 1:
            # ES formula for GPD
            es = var + (self.scale_param - self.shape_param * self.threshold) / (
                1 - self.shape_param
            )
        else:
            # Heavy tail - ES may not exist
            es = float('inf')
        
        return es
    
    def estimate_tail_probability(self, loss_threshold: float) -> float:
        """Estimate probability of extreme loss."""
        if loss_threshold <= self.threshold:
            # Use empirical probability
            return 1 - self.threshold_quantile
        
        # Use GPD tail estimate
        if self.shape_param != 0:
            prob = (1 - self.threshold_quantile) * (
                1 + self.shape_param * (loss_threshold - self.threshold) / self.scale_param
            ) ** (-1 / self.shape_param)
        else:
            prob = (1 - self.threshold_quantile) * np.exp(
                -(loss_threshold - self.threshold) / self.scale_param
            )
        
        return max(0, min(1, prob))


class CopulaRiskModeler:
    """
    Models complex dependencies between assets using copulas.
    Captures tail dependence and non-linear correlations.
    """
    
    def __init__(self, copula_type: str = 'gaussian'):
        self.copula_type = copula_type
        self.copula_model = None
        self.marginal_models = {}
        
    def fit(self, returns_data: pd.DataFrame):
        """Fit copula model to multivariate returns."""
        # Fit marginal distributions
        for column in returns_data.columns:
            # Fit flexible marginal using KDE
            marginal = GaussianKDE()
            marginal.fit(returns_data[column].values)
            self.marginal_models[column] = marginal
        
        # Transform to uniform margins
        uniform_data = pd.DataFrame()
        for column in returns_data.columns:
            uniform_data[column] = self.marginal_models[column].cdf(
                returns_data[column].values
            )
        
        # Fit copula
        if self.copula_type == 'gaussian':
            self.copula_model = GaussianMultivariate()
        elif self.copula_type == 'vine':
            self.copula_model = VineMultivariate('regular')
        else:
            raise ValueError(f"Unknown copula type: {self.copula_type}")
        
        self.copula_model.fit(uniform_data)
    
    def simulate_scenarios(
        self,
        n_scenarios: int = 1000,
        horizon: int = 1
    ) -> pd.DataFrame:
        """Simulate scenarios preserving dependency structure."""
        # Generate from copula
        uniform_samples = self.copula_model.sample(n_scenarios * horizon)
        
        # Transform back to returns
        scenarios = pd.DataFrame()
        for i, column in enumerate(self.marginal_models.keys()):
            marginal = self.marginal_models[column]
            scenarios[column] = marginal.percent_point(uniform_samples.iloc[:, i])
        
        # Aggregate over horizon
        if horizon > 1:
            # Reshape and sum
            n_assets = len(self.marginal_models)
            reshaped = scenarios.values.reshape(n_scenarios, horizon, n_assets)
            scenarios = pd.DataFrame(
                reshaped.sum(axis=1),
                columns=scenarios.columns
            )
        
        return scenarios
    
    def calculate_tail_dependence(self) -> Dict[str, float]:
        """Calculate tail dependence coefficients."""
        # For Gaussian copula, calculate from correlation
        if hasattr(self.copula_model, 'covariance'):
            corr_matrix = self.copula_model.covariance
            
            # Lower tail dependence for Gaussian is 0 unless perfect correlation
            # Upper tail dependence formula
            tail_deps = {}
            
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    rho = corr_matrix[i, j]
                    
                    # Gaussian copula tail dependence
                    if abs(rho) < 1:
                        lambda_lower = 0
                        lambda_upper = 0
                    else:
                        lambda_lower = 1 if rho == 1 else 0
                        lambda_upper = 1 if rho == 1 else 0
                    
                    pair_name = f"{i}_{j}"
                    tail_deps[pair_name] = {
                        'lower': lambda_lower,
                        'upper': lambda_upper,
                        'correlation': rho
                    }
            
            return tail_deps
        
        return {}


class RegimeSwitchingRiskModel:
    """
    Models risk under different market regimes (bull, bear, high volatility, etc).
    Uses Hidden Markov Models and regime-switching models.
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.current_regime = None
        self.regime_probabilities = None
        
        # Regime characteristics
        self.regime_stats = {}
        
    def fit(self, returns_data: pd.Series):
        """Fit regime-switching model."""
        # Prepare data
        data = returns_data.to_frame('returns')
        data['const'] = 1
        
        # Fit Markov Switching model
        self.model = MarkovRegression(
            data['returns'],
            k_regimes=self.n_regimes,
            exog=data['const'],
            switching_variance=True
        )
        
        try:
            self.model = self.model.fit(disp=False)
            
            # Extract regime statistics
            for i in range(self.n_regimes):
                self.regime_stats[i] = {
                    'mean': float(self.model.params[f'regime{i}[const]']),
                    'volatility': float(np.sqrt(self.model.params[f'sigma2.regime{i}'])),
                    'persistence': float(self.model.transition_probabilities[i, i])
                }
            
            # Get current regime probabilities
            self.regime_probabilities = self.model.smoothed_marginal_probabilities
            self.current_regime = np.argmax(self.regime_probabilities.iloc[-1])
            
        except Exception as e:
            logger.error(f"Failed to fit regime-switching model: {e}")
            # Fallback to simple volatility-based regimes
            self._fit_simple_regimes(returns_data)
    
    def _fit_simple_regimes(self, returns_data: pd.Series):
        """Simple regime identification based on volatility."""
        rolling_vol = returns_data.rolling(window=20).std()
        
        # Define regimes by volatility terciles
        vol_33 = rolling_vol.quantile(0.33)
        vol_67 = rolling_vol.quantile(0.67)
        
        for i in range(self.n_regimes):
            if i == 0:  # Low volatility
                mask = rolling_vol <= vol_33
                regime_returns = returns_data[mask]
            elif i == 1:  # Medium volatility
                mask = (rolling_vol > vol_33) & (rolling_vol <= vol_67)
                regime_returns = returns_data[mask]
            else:  # High volatility
                mask = rolling_vol > vol_67
                regime_returns = returns_data[mask]
            
            self.regime_stats[i] = {
                'mean': float(regime_returns.mean()),
                'volatility': float(regime_returns.std()),
                'persistence': 0.8  # Assumed
            }
    
    def calculate_regime_specific_risk(
        self,
        portfolio_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict[int, Dict[str, float]]:
        """Calculate risk metrics for each regime."""
        regime_risks = {}
        
        for regime in range(self.n_regimes):
            # Get returns for this regime
            if self.regime_probabilities is not None:
                regime_probs = self.regime_probabilities.iloc[:, regime]
                # Weight returns by regime probability
                regime_weights = regime_probs / regime_probs.sum()
                regime_returns = portfolio_returns * regime_weights
            else:
                # Simple assignment
                regime_returns = portfolio_returns
            
            # Calculate risk metrics
            var = np.quantile(regime_returns, 1 - confidence_level)
            cvar = regime_returns[regime_returns <= var].mean()
            
            regime_risks[regime] = {
                'var': var,
                'cvar': cvar,
                'volatility': self.regime_stats[regime]['volatility'],
                'expected_return': self.regime_stats[regime]['mean'],
                'probability': float(regime_probs.iloc[-1]) if self.regime_probabilities is not None else 1/self.n_regimes
            }
        
        return regime_risks
    
    def predict_regime_transition(self, horizon: int = 5) -> np.ndarray:
        """Predict regime probabilities over horizon."""
        if self.model is None:
            # Simple persistence model
            current_probs = np.zeros(self.n_regimes)
            current_probs[self.current_regime or 0] = 1.0
            
            predictions = [current_probs]
            for _ in range(horizon):
                # Simple transition
                next_probs = current_probs * 0.8 + (1 - 0.8) / self.n_regimes
                predictions.append(next_probs)
                current_probs = next_probs
            
            return np.array(predictions)
        
        # Use fitted model transition matrix
        transition_matrix = self.model.transition_probabilities
        current_probs = self.regime_probabilities.iloc[-1].values
        
        predictions = [current_probs]
        for _ in range(horizon):
            current_probs = current_probs @ transition_matrix
            predictions.append(current_probs)
        
        return np.array(predictions)


class DynamicHedgingEngine:
    """
    Implements dynamic hedging strategies to manage tail risk.
    Uses options, futures, and other derivatives.
    """
    
    def __init__(self, hedging_budget: float = 0.02):  # 2% of portfolio
        self.hedging_budget = hedging_budget
        self.active_hedges = []
        self.hedging_performance = []
        
    def calculate_optimal_hedge(
        self,
        portfolio_value: float,
        risk_metrics: RiskMetrics,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate optimal hedging strategy."""
        hedges = []
        
        # Determine hedging need based on risk metrics
        tail_risk_score = self._calculate_tail_risk_score(risk_metrics)
        
        if tail_risk_score > 0.7:  # High tail risk
            # Protective puts
            put_hedge = self._design_put_hedge(
                portfolio_value,
                risk_metrics,
                market_data
            )
            hedges.append(put_hedge)
        
        if risk_metrics.correlation_risk > 0.8:  # High correlation risk
            # Correlation hedge using VIX
            vix_hedge = self._design_vix_hedge(
                portfolio_value,
                risk_metrics,
                market_data
            )
            hedges.append(vix_hedge)
        
        if market_data.get('regime') == 'high_volatility':
            # Volatility hedge
            vol_hedge = self._design_volatility_hedge(
                portfolio_value,
                risk_metrics,
                market_data
            )
            hedges.append(vol_hedge)
        
        # Optimize hedge allocation within budget
        optimized_hedges = self._optimize_hedge_allocation(
            hedges,
            portfolio_value * self.hedging_budget
        )
        
        return optimized_hedges
    
    def _calculate_tail_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """Calculate composite tail risk score."""
        # Combine multiple tail risk indicators
        var_ratio = abs(risk_metrics.cvar_99 / risk_metrics.var_99)
        tail_contribution = np.mean(list(risk_metrics.tail_risk_contribution.values()))
        extreme_risk = risk_metrics.extreme_downside_risk
        
        # Weighted score
        score = (
            0.3 * min(var_ratio / 2, 1) +  # CVaR/VaR ratio
            0.3 * tail_contribution +
            0.4 * extreme_risk
        )
        
        return min(1.0, score)
    
    def _design_put_hedge(
        self,
        portfolio_value: float,
        risk_metrics: RiskMetrics,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design protective put strategy."""
        current_price = market_data.get('index_price', 4000)
        volatility = market_data.get('implied_volatility', 0.2)
        
        # Strike selection: 5-10% OTM based on tail risk
        moneyness = 0.95 - 0.05 * risk_metrics.extreme_downside_risk
        strike = current_price * moneyness
        
        # Calculate option premium (simplified Black-Scholes)
        from scipy.stats import norm
        
        time_to_expiry = 30 / 365  # 30 days
        risk_free_rate = 0.05
        
        d1 = (np.log(current_price / strike) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
              volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        put_price = (strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    current_price * norm.cdf(-d1))
        
        # Number of contracts
        hedge_notional = portfolio_value * 0.01  # 1% of portfolio
        n_contracts = int(hedge_notional / (put_price * 100))  # 100 multiplier
        
        return {
            'type': 'protective_put',
            'instrument': 'SPX_PUT',
            'strike': strike,
            'expiry': 30,
            'contracts': n_contracts,
            'premium': put_price * n_contracts * 100,
            'hedge_ratio': 0.5,  # Delta hedge ratio
            'expected_payout': self._calculate_expected_payout(
                current_price, strike, volatility
            )
        }
    
    def _design_vix_hedge(
        self,
        portfolio_value: float,
        risk_metrics: RiskMetrics,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design VIX-based correlation hedge."""
        current_vix = market_data.get('vix', 15)
        
        # VIX futures or calls
        if current_vix < 20:  # Low VIX - buy calls
            strike = current_vix * 1.2
            instrument = 'VIX_CALL'
        else:  # Already elevated - use futures
            strike = current_vix
            instrument = 'VIX_FUTURE'
        
        # Size based on correlation risk
        hedge_notional = portfolio_value * 0.005 * risk_metrics.correlation_risk
        vix_multiplier = 1000
        n_contracts = int(hedge_notional / (current_vix * vix_multiplier))
        
        return {
            'type': 'vix_hedge',
            'instrument': instrument,
            'strike': strike,
            'current_level': current_vix,
            'contracts': n_contracts,
            'notional': n_contracts * current_vix * vix_multiplier,
            'correlation_beta': -0.3  # Typical SPX-VIX correlation
        }
    
    def _design_volatility_hedge(
        self,
        portfolio_value: float,
        risk_metrics: RiskMetrics,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design volatility hedge using options strategies."""
        # Implement straddle or strangle
        current_price = market_data.get('index_price', 4000)
        volatility = market_data.get('implied_volatility', 0.2)
        
        # Long strangle (OTM put + OTM call)
        put_strike = current_price * 0.95
        call_strike = current_price * 1.05
        
        return {
            'type': 'long_strangle',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiry': 30,
            'contracts': int(portfolio_value * 0.001 / current_price),
            'vega_exposure': 100,  # Vega per contract
            'gamma_exposure': 0.05  # Gamma per contract
        }
    
    def _optimize_hedge_allocation(
        self,
        hedges: List[Dict[str, Any]],
        budget: float
    ) -> List[Dict[str, Any]]:
        """Optimize hedge allocation within budget constraint."""
        if not hedges:
            return []
        
        # Simple optimization - allocate proportionally to expected benefit
        total_cost = sum(h.get('premium', h.get('notional', 0)) for h in hedges)
        
        if total_cost <= budget:
            return hedges
        
        # Scale down proportionally
        scale_factor = budget / total_cost
        
        optimized = []
        for hedge in hedges:
            scaled_hedge = hedge.copy()
            if 'contracts' in scaled_hedge:
                scaled_hedge['contracts'] = int(
                    scaled_hedge['contracts'] * scale_factor
                )
            if 'premium' in scaled_hedge:
                scaled_hedge['premium'] *= scale_factor
            if 'notional' in scaled_hedge:
                scaled_hedge['notional'] *= scale_factor
            
            if scaled_hedge.get('contracts', 1) > 0:
                optimized.append(scaled_hedge)
        
        return optimized
    
    def _calculate_expected_payout(
        self,
        spot: float,
        strike: float,
        volatility: float
    ) -> float:
        """Calculate expected payout for option hedge."""
        # Simplified - would use proper option pricing model
        moneyness = strike / spot
        
        if moneyness < 1:  # Put option
            # Probability of ending ITM
            prob_itm = norm.cdf(
                (np.log(moneyness) + 0.5 * volatility**2 * 30/365) /
                (volatility * np.sqrt(30/365))
            )
            expected_payout = prob_itm * (1 - moneyness) * spot
        else:
            expected_payout = 0
        
        return expected_payout


class StressTestingEngine:
    """
    Comprehensive stress testing and scenario analysis.
    """
    
    def __init__(self):
        self.historical_scenarios = self._load_historical_scenarios()
        self.hypothetical_scenarios = self._define_hypothetical_scenarios()
        
    def _load_historical_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Load historical stress scenarios."""
        return {
            'black_monday_1987': {
                'equity': -0.22,
                'volatility': 2.5,
                'correlation': 0.9,
                'duration': 1
            },
            'asian_crisis_1997': {
                'equity': -0.35,
                'volatility': 1.8,
                'correlation': 0.85,
                'duration': 90
            },
            'dotcom_crash_2000': {
                'equity': -0.45,
                'volatility': 1.5,
                'correlation': 0.7,
                'duration': 365
            },
            'financial_crisis_2008': {
                'equity': -0.55,
                'volatility': 3.0,
                'correlation': 0.95,
                'duration': 180
            },
            'covid_crash_2020': {
                'equity': -0.35,
                'volatility': 4.0,
                'correlation': 0.8,
                'duration': 30
            }
        }
    
    def _define_hypothetical_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Define hypothetical stress scenarios."""
        return {
            'inflation_shock': {
                'rates': 0.02,  # 200bps increase
                'equity': -0.15,
                'real_assets': 0.1,
                'duration': 180
            },
            'geopolitical_crisis': {
                'equity': -0.25,
                'commodities': 0.3,
                'safe_haven': 0.1,
                'duration': 60
            },
            'tech_bubble_burst': {
                'tech_equity': -0.6,
                'other_equity': -0.2,
                'volatility': 2.0,
                'duration': 365
            },
            'liquidity_crisis': {
                'equity': -0.3,
                'credit_spread': 0.05,
                'funding_cost': 0.03,
                'duration': 90
            },
            'climate_event': {
                'equity': -0.2,
                'insurance': -0.4,
                'commodities': 0.2,
                'duration': 30
            }
        }
    
    def run_stress_tests(
        self,
        portfolio: Dict[str, float],
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Run comprehensive stress tests."""
        results = {}
        
        # Historical scenarios
        for scenario_name, scenario in self.historical_scenarios.items():
            impact = self._calculate_scenario_impact(portfolio, scenario)
            results[f"hist_{scenario_name}"] = impact
        
        # Hypothetical scenarios
        for scenario_name, scenario in self.hypothetical_scenarios.items():
            impact = self._calculate_scenario_impact(portfolio, scenario)
            results[f"hypo_{scenario_name}"] = impact
        
        # Custom scenarios
        if custom_scenarios:
            for scenario_name, scenario in custom_scenarios.items():
                impact = self._calculate_scenario_impact(portfolio, scenario)
                results[f"custom_{scenario_name}"] = impact
        
        # Reverse stress test - find scenario that causes maximum loss
        reverse_scenario = self._reverse_stress_test(portfolio)
        results['reverse_stress'] = reverse_scenario
        
        return results
    
    def _calculate_scenario_impact(
        self,
        portfolio: Dict[str, float],
        scenario: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate impact of scenario on portfolio."""
        total_impact = 0
        component_impacts = {}
        
        # Map portfolio components to scenario factors
        factor_mapping = {
            'stocks': 'equity',
            'bonds': 'rates',
            'commodities': 'commodities',
            'options': 'volatility'
        }
        
        for asset_type, weight in portfolio.items():
            factor = factor_mapping.get(asset_type, 'equity')
            
            if factor in scenario:
                impact = weight * scenario[factor]
                component_impacts[asset_type] = impact
                total_impact += impact
        
        return {
            'total_impact': total_impact,
            'component_impacts': component_impacts,
            'var_impact': total_impact * 1.5,  # Amplified VaR impact
            'duration': scenario.get('duration', 30)
        }
    
    def _reverse_stress_test(
        self,
        portfolio: Dict[str, float],
        target_loss: float = -0.25
    ) -> Dict[str, float]:
        """Find scenario that produces target loss."""
        # Optimization to find worst-case scenario
        def objective(scenario_params):
            scenario = {
                'equity': scenario_params[0],
                'rates': scenario_params[1],
                'volatility': scenario_params[2],
                'correlation': scenario_params[3]
            }
            impact = self._calculate_scenario_impact(portfolio, scenario)
            return abs(impact['total_impact'] - target_loss)
        
        # Constraints
        bounds = [
            (-0.5, 0),  # Equity: -50% to 0%
            (-0.05, 0.05),  # Rates: -5% to +5%
            (1.0, 5.0),  # Volatility multiplier
            (0.5, 1.0)  # Correlation
        ]
        
        result = minimize(
            objective,
            x0=[-0.25, 0.02, 2.0, 0.9],
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return {
            'equity_shock': result.x[0],
            'rates_shock': result.x[1],
            'volatility_shock': result.x[2],
            'correlation_shock': result.x[3],
            'probability': self._estimate_scenario_probability(result.x)
        }
    
    def _estimate_scenario_probability(self, scenario_params: np.ndarray) -> float:
        """Estimate probability of scenario occurring."""
        # Based on historical frequency and extremeness
        equity_shock = abs(scenario_params[0])
        
        if equity_shock < 0.1:
            return 0.1  # 10% for mild scenario
        elif equity_shock < 0.2:
            return 0.05  # 5% for moderate
        elif equity_shock < 0.3:
            return 0.01  # 1% for severe
        else:
            return 0.001  # 0.1% for extreme


class AdvancedRiskManager:
    """
    Main risk management system integrating all components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.evt_analyzer = ExtremeValueAnalyzer()
        self.copula_modeler = CopulaRiskModeler()
        self.regime_model = RegimeSwitchingRiskModel()
        self.hedging_engine = DynamicHedgingEngine(
            config.get('hedging_budget', 0.02)
        )
        self.stress_tester = StressTestingEngine()
        
        # Risk limits
        self.risk_limits = config.get('risk_limits', {
            'max_var_95': 0.02,  # 2% VaR limit
            'max_var_99': 0.05,  # 5% VaR limit
            'max_leverage': 2.0,
            'max_concentration': 0.2,
            'max_correlation': 0.8
        })
        
        # Real-time monitoring
        self.risk_breaches = []
        self.risk_history = []
        
    async def calculate_portfolio_risk(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame,
        lookback_days: int = 252
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Get portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio,
            market_data
        )
        
        # Fit risk models
        self.evt_analyzer.fit_tail_distribution(portfolio_returns.values)
        self.copula_modeler.fit(market_data)
        self.regime_model.fit(portfolio_returns)
        
        # Calculate standard risk metrics
        var_95 = np.quantile(portfolio_returns, 0.05)
        var_99 = np.quantile(portfolio_returns, 0.01)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Calculate tail risk metrics
        extreme_var = self.evt_analyzer.calculate_extreme_var(0.999)  # 99.9% VaR
        expected_tail_loss = self.evt_analyzer.calculate_expected_shortfall(0.99)
        
        # Calculate Greeks (placeholder - would use option pricing models)
        greeks = self._calculate_portfolio_greeks(portfolio, market_data)
        
        # Performance metrics
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        sortino_ratio = portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Stress test results
        stress_results = self.stress_tester.run_stress_tests(portfolio)
        
        # Regime-specific risks
        regime_risks = self.regime_model.calculate_regime_specific_risk(
            portfolio_returns
        )
        
        # Dependency measures
        tail_deps = self.copula_modeler.calculate_tail_dependence()
        avg_tail_dep = np.mean([
            v['upper'] for v in tail_deps.values()
        ]) if tail_deps else 0
        
        return RiskMetrics(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            expected_tail_loss=float(expected_tail_loss),
            tail_risk_contribution=self._calculate_tail_contributions(
                portfolio,
                market_data
            ),
            extreme_downside_risk=float(extreme_var),
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            vega=greeks['vega'],
            theta=greeks['theta'],
            rho=greeks['rho'],
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(portfolio_returns.mean() / abs(max_drawdown) * 252),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration=int(self._calculate_drawdown_duration(drawdown)),
            stress_scenarios={
                k: v['total_impact'] for k, v in stress_results.items()
            },
            regime_risks={
                str(k): v['var'] for k, v in regime_risks.items()
            },
            correlation_risk=float(market_data.corr().values.mean()),
            tail_dependence=avg_tail_dep,
            liquidity_score=self._calculate_liquidity_score(portfolio, market_data),
            market_impact=self._estimate_market_impact(portfolio),
            model_confidence=0.85,  # Placeholder
            parameter_uncertainty=0.1  # Placeholder
        )
    
    def _calculate_portfolio_returns(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns from component returns."""
        portfolio_returns = pd.Series(0, index=market_data.index)
        
        for asset, weight in portfolio.items():
            if asset in market_data.columns:
                portfolio_returns += weight * market_data[asset].pct_change()
        
        return portfolio_returns.dropna()
    
    def _calculate_portfolio_greeks(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio Greeks (simplified)."""
        # Placeholder - would use proper option pricing
        return {
            'delta': 0.5,
            'gamma': 0.1,
            'vega': 0.2,
            'theta': -0.05,
            'rho': 0.3
        }
    
    def _calculate_tail_contributions(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate each asset's contribution to tail risk."""
        contributions = {}
        
        # Use Conditional Value at Risk decomposition
        portfolio_returns = self._calculate_portfolio_returns(portfolio, market_data)
        var_threshold = np.quantile(portfolio_returns, 0.05)
        
        for asset, weight in portfolio.items():
            if asset in market_data.columns:
                asset_returns = market_data[asset].pct_change().dropna()
                
                # Calculate contribution during tail events
                tail_mask = portfolio_returns <= var_threshold
                tail_contribution = (
                    weight * asset_returns[tail_mask].mean() /
                    portfolio_returns[tail_mask].mean()
                )
                
                contributions[asset] = float(tail_contribution)
        
        return contributions
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown < 0
        
        # Find consecutive periods of drawdown
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_liquidity_score(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame
    ) -> float:
        """Calculate portfolio liquidity score."""
        # Simplified - based on asset class
        liquidity_scores = {
            'stocks': 0.9,
            'bonds': 0.7,
            'options': 0.5,
            'alternatives': 0.3
        }
        
        total_score = sum(
            weight * liquidity_scores.get(asset, 0.5)
            for asset, weight in portfolio.items()
        )
        
        return total_score
    
    def _estimate_market_impact(
        self,
        portfolio: Dict[str, float]
    ) -> float:
        """Estimate market impact of liquidating portfolio."""
        # Square-root market impact model
        total_value = sum(portfolio.values())
        
        # Simplified impact = sqrt(size) * volatility * constant
        impact = np.sqrt(total_value / 1000000) * 0.02 * 0.1
        
        return min(impact, 0.05)  # Cap at 5%
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[Dict[str, Any]]:
        """Check if any risk limits are breached."""
        breaches = []
        
        # VaR limits
        if abs(risk_metrics.var_95) > self.risk_limits['max_var_95']:
            breaches.append({
                'type': 'var_95_breach',
                'limit': self.risk_limits['max_var_95'],
                'current': risk_metrics.var_95,
                'severity': 'high'
            })
        
        if abs(risk_metrics.var_99) > self.risk_limits['max_var_99']:
            breaches.append({
                'type': 'var_99_breach',
                'limit': self.risk_limits['max_var_99'],
                'current': risk_metrics.var_99,
                'severity': 'critical'
            })
        
        # Correlation limit
        if risk_metrics.correlation_risk > self.risk_limits['max_correlation']:
            breaches.append({
                'type': 'correlation_breach',
                'limit': self.risk_limits['max_correlation'],
                'current': risk_metrics.correlation_risk,
                'severity': 'medium'
            })
        
        return breaches
    
    def generate_risk_report(
        self,
        risk_metrics: RiskMetrics,
        portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': portfolio,
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'cvar_95': risk_metrics.cvar_95,
                'cvar_99': risk_metrics.cvar_99,
                'expected_tail_loss': risk_metrics.expected_tail_loss,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio
            },
            'tail_risk_analysis': {
                'extreme_downside_risk': risk_metrics.extreme_downside_risk,
                'tail_dependence': risk_metrics.tail_dependence,
                'tail_contributions': risk_metrics.tail_risk_contribution
            },
            'stress_test_results': risk_metrics.stress_scenarios,
            'regime_analysis': {
                'current_regime': self.regime_model.current_regime,
                'regime_risks': risk_metrics.regime_risks
            },
            'risk_breaches': self.check_risk_limits(risk_metrics),
            'recommended_hedges': self.hedging_engine.calculate_optimal_hedge(
                sum(portfolio.values()),
                risk_metrics,
                {'regime': 'normal'}  # Placeholder
            )
        }
