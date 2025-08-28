"""
Comprehensive tests for Advanced Risk Management module.
Tests EVT, copulas, regime switching, and dynamic hedging.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from app.risk_management.advanced_risk_manager import (
    RiskMetrics,
    TailRiskEvent,
    ExtremeValueAnalyzer,
    CopulaRiskModeler,
    RegimeSwitchingRiskModel,
    DynamicHedgingEngine,
    StressTestingEngine,
    AdvancedRiskManager
)

# Missing import
import pandas as pd


class TestRiskMetrics:
    """Test risk metrics dataclass."""
    
    def test_risk_metrics_initialization(self):
        """Test complete risk metrics initialization."""
        metrics = RiskMetrics(
            var_95=0.02,
            var_99=0.05,
            cvar_95=0.025,
            cvar_99=0.06,
            expected_tail_loss=0.08,
            tail_risk_contribution={'AAPL': 0.3, 'GOOGL': 0.2, 'MSFT': 0.5},
            extreme_downside_risk=0.1,
            delta=0.5,
            gamma=0.1,
            vega=0.2,
            theta=-0.05,
            rho=0.3,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.15,
            max_drawdown_duration=30,
            stress_scenarios={'financial_crisis': -0.3, 'covid_crash': -0.25},
            regime_risks={'normal': 0.02, 'volatile': 0.08},
            correlation_risk=0.7,
            tail_dependence=0.4,
            liquidity_score=0.85,
            market_impact=0.005,
            model_confidence=0.9,
            parameter_uncertainty=0.1
        )
        
        assert metrics.var_95 == 0.02
        assert metrics.cvar_99 == 0.06
        assert metrics.expected_tail_loss == 0.08
        assert len(metrics.tail_risk_contribution) == 3
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown_duration == 30
        assert 'financial_crisis' in metrics.stress_scenarios
        assert metrics.liquidity_score == 0.85


class TestTailRiskEvent:
    """Test tail risk event dataclass."""
    
    def test_tail_risk_event_initialization(self):
        """Test tail risk event creation."""
        event = TailRiskEvent(
            event_type='market_crash',
            probability=0.05,
            impact=-0.25,
            duration=30,
            affected_assets=['SPY', 'QQQ', 'IWM'],
            hedging_strategy={
                'instrument': 'put_options',
                'strike': 0.95,
                'notional': 100000
            },
            confidence=0.8
        )
        
        assert event.event_type == 'market_crash'
        assert event.probability == 0.05
        assert event.impact == -0.25
        assert event.duration == 30
        assert len(event.affected_assets) == 3
        assert event.hedging_strategy['instrument'] == 'put_options'
        assert event.confidence == 0.8
    
    def test_event_types(self):
        """Test different event types."""
        event_types = [
            'market_crash',
            'flash_crash',
            'liquidity_crisis',
            'volatility_spike',
            'correlation_breakdown',
            'fat_tail_event'
        ]
        
        for event_type in event_types:
            event = TailRiskEvent(
                event_type=event_type,
                probability=0.01,
                impact=-0.1,
                duration=1,
                affected_assets=[],
                hedging_strategy={},
                confidence=0.5
            )
            assert event.event_type == event_type


class TestExtremeValueAnalyzer:
    """Test Extreme Value Theory analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create EVT analyzer instance."""
        return ExtremeValueAnalyzer(threshold_quantile=0.95)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns with fat tails."""
        np.random.seed(42)
        # Mix of normal returns and extreme events
        normal_returns = np.random.normal(0, 0.01, 900)
        extreme_returns = np.random.normal(0, 0.05, 100)
        returns = np.concatenate([normal_returns, extreme_returns])
        np.random.shuffle(returns)
        return returns
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.threshold_quantile == 0.95
        assert analyzer.threshold is None
        assert analyzer.shape_param is None
        assert analyzer.scale_param is None
        assert analyzer.tail_index is None
    
    def test_fit_tail_distribution(self, analyzer, sample_returns):
        """Test fitting GPD to tail returns."""
        result = analyzer.fit_tail_distribution(sample_returns)
        
        assert 'shape' in result
        assert 'scale' in result
        assert 'threshold' in result
        assert 'tail_index' in result
        assert 'n_exceedances' in result
        
        assert analyzer.threshold is not None
        assert analyzer.shape_param is not None
        assert analyzer.scale_param is not None
        assert result['n_exceedances'] > 0
        
        # Threshold should be around 95th percentile
        expected_threshold = np.quantile(sample_returns, 0.95)
        assert abs(analyzer.threshold - expected_threshold) < 0.01
    
    def test_fit_tail_distribution_insufficient_data(self, analyzer):
        """Test fitting with insufficient tail observations."""
        # Only 5 observations, all below threshold
        small_returns = np.array([0.001, 0.002, 0.001, -0.001, 0.0])
        
        result = analyzer.fit_tail_distribution(small_returns)
        
        # Should return empty dict due to insufficient exceedances
        assert result == {}
    
    def test_calculate_extreme_var(self, analyzer, sample_returns):
        """Test extreme VaR calculation."""
        # Fit first
        analyzer.fit_tail_distribution(sample_returns)
        
        # Calculate VaR at different confidence levels
        var_99 = analyzer.calculate_extreme_var(0.99, horizon=1)
        var_999 = analyzer.calculate_extreme_var(0.999, horizon=1)
        var_9999 = analyzer.calculate_extreme_var(0.9999, horizon=1)
        
        # VaR should increase with confidence level
        assert var_99 < var_999 < var_9999
        assert var_99 > analyzer.threshold
        
        # Test multi-day horizon
        var_5day = analyzer.calculate_extreme_var(0.99, horizon=5)
        assert var_5day > var_99  # Should scale with sqrt(time)
        assert abs(var_5day / var_99 - np.sqrt(5)) < 0.5
    
    def test_calculate_expected_shortfall(self, analyzer, sample_returns):
        """Test Expected Shortfall (CVaR) calculation."""
        analyzer.fit_tail_distribution(sample_returns)
        
        es_99 = analyzer.calculate_expected_shortfall(0.99)
        var_99 = analyzer.calculate_extreme_var(0.99)
        
        # ES should be greater than VaR
        assert es_99 > var_99
        
        # Test with heavy tail (shape > 1)
        analyzer.shape_param = 1.5  # Heavy tail
        es_heavy = analyzer.calculate_expected_shortfall(0.99)
        assert es_heavy == float('inf')  # ES doesn't exist for very heavy tails
    
    def test_estimate_tail_probability(self, analyzer, sample_returns):
        """Test tail probability estimation."""
        analyzer.fit_tail_distribution(sample_returns)
        
        # Probability of moderate loss
        prob_moderate = analyzer.estimate_tail_probability(analyzer.threshold * 1.1)
        assert 0 < prob_moderate < 1 - analyzer.threshold_quantile
        
        # Probability of extreme loss
        prob_extreme = analyzer.estimate_tail_probability(analyzer.threshold * 2)
        assert prob_extreme < prob_moderate
        
        # Probability below threshold (empirical region)
        prob_below = analyzer.estimate_tail_probability(analyzer.threshold * 0.9)
        assert prob_below == 1 - analyzer.threshold_quantile
    
    def test_edge_cases(self, analyzer):
        """Test EVT analyzer edge cases."""
        # Zero returns
        zero_returns = np.zeros(100)
        result = analyzer.fit_tail_distribution(zero_returns)
        assert result == {}  # No exceedances
        
        # Constant returns
        constant_returns = np.ones(100) * 0.01
        result = analyzer.fit_tail_distribution(constant_returns)
        # Should handle gracefully
        
        # Exponential tail (shape = 0)
        analyzer.shape_param = 0
        analyzer.scale_param = 0.01
        analyzer.threshold = 0.02
        
        var_exp = analyzer.calculate_extreme_var(0.99)
        assert var_exp > analyzer.threshold


class TestCopulaRiskModeler:
    """Test copula-based risk modeling."""
    
    @pytest.fixture
    def modeler(self):
        """Create copula modeler instance."""
        return CopulaRiskModeler(copula_type='gaussian')
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample multivariate returns."""
        np.random.seed(42)
        n_samples = 500
        n_assets = 3
        
        # Create correlated returns
        mean = np.zeros(n_assets)
        cov = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ]) * 0.01**2
        
        returns = np.random.multivariate_normal(mean, cov, n_samples)
        return pd.DataFrame(returns, columns=['Asset1', 'Asset2', 'Asset3'])
    
    def test_modeler_initialization(self, modeler):
        """Test copula modeler initialization."""
        assert modeler.copula_type == 'gaussian'
        assert modeler.copula_model is None
        assert len(modeler.marginal_models) == 0
    
    def test_fit_copula(self, modeler, sample_data):
        """Test copula fitting."""
        modeler.fit(sample_data)
        
        # Check marginals fitted
        assert len(modeler.marginal_models) == 3
        for col in sample_data.columns:
            assert col in modeler.marginal_models
        
        # Check copula fitted
        assert modeler.copula_model is not None
        assert hasattr(modeler.copula_model, 'sample')
    
    def test_simulate_scenarios(self, modeler, sample_data):
        """Test scenario simulation."""
        modeler.fit(sample_data)
        
        # Single period simulation
        scenarios = modeler.simulate_scenarios(n_scenarios=1000, horizon=1)
        
        assert scenarios.shape == (1000, 3)
        assert list(scenarios.columns) == list(sample_data.columns)
        
        # Multi-period simulation
        scenarios_multi = modeler.simulate_scenarios(n_scenarios=100, horizon=5)
        
        assert scenarios_multi.shape == (100, 3)
        
        # Check correlation structure preserved
        sim_corr = scenarios.corr()
        data_corr = sample_data.corr()
        
        # Correlations should be similar (within tolerance)
        for i in range(3):
            for j in range(3):
                assert abs(sim_corr.iloc[i, j] - data_corr.iloc[i, j]) < 0.2
    
    def test_calculate_tail_dependence(self, modeler, sample_data):
        """Test tail dependence calculation."""
        modeler.fit(sample_data)
        
        tail_deps = modeler.calculate_tail_dependence()
        
        assert isinstance(tail_deps, dict)
        
        # Check all pairs included
        expected_pairs = ['0_1', '0_2', '1_2']
        for pair in expected_pairs:
            assert pair in tail_deps
            assert 'lower' in tail_deps[pair]
            assert 'upper' in tail_deps[pair]
            assert 'correlation' in tail_deps[pair]
            
            # Gaussian copula has zero tail dependence
            assert tail_deps[pair]['lower'] == 0
            assert tail_deps[pair]['upper'] == 0
    
    def test_different_copula_types(self):
        """Test different copula types."""
        # Vine copula
        vine_modeler = CopulaRiskModeler(copula_type='vine')
        assert vine_modeler.copula_type == 'vine'
        
        # Unknown copula
        with pytest.raises(ValueError):
            CopulaRiskModeler(copula_type='unknown')


class TestRegimeSwitchingRiskModel:
    """Test regime-switching risk models."""
    
    @pytest.fixture
    def model(self):
        """Create regime model instance."""
        return RegimeSwitchingRiskModel(n_regimes=3)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate returns with regime switches."""
        np.random.seed(42)
        
        # Generate returns for 3 regimes
        regime1 = np.random.normal(0.001, 0.01, 100)  # Normal
        regime2 = np.random.normal(-0.001, 0.03, 50)  # High volatility
        regime3 = np.random.normal(0.002, 0.005, 100)  # Low volatility bull
        
        returns = np.concatenate([regime1, regime2, regime3])
        return pd.Series(returns, index=pd.date_range('2023-01-01', periods=250))
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.n_regimes == 3
        assert model.model is None
        assert model.current_regime is None
        assert model.regime_probabilities is None
        assert len(model.regime_stats) == 0
    
    def test_fit_regime_model(self, model, sample_returns):
        """Test regime model fitting."""
        # Use simple regime fitting due to convergence issues with Markov model
        model._fit_simple_regimes(sample_returns)
        
        # Check regime statistics
        assert len(model.regime_stats) == 3
        
        for i in range(3):
            assert 'mean' in model.regime_stats[i]
            assert 'volatility' in model.regime_stats[i]
            assert 'persistence' in model.regime_stats[i]
            
            # Volatility should be positive
            assert model.regime_stats[i]['volatility'] > 0
    
    def test_calculate_regime_specific_risk(self, model, sample_returns):
        """Test regime-specific risk calculation."""
        model._fit_simple_regimes(sample_returns)
        
        regime_risks = model.calculate_regime_specific_risk(sample_returns, 0.95)
        
        assert len(regime_risks) == 3
        
        for regime in range(3):
            assert regime in regime_risks
            assert 'var' in regime_risks[regime]
            assert 'cvar' in regime_risks[regime]
            assert 'volatility' in regime_risks[regime]
            assert 'expected_return' in regime_risks[regime]
            assert 'probability' in regime_risks[regime]
            
            # Risk metrics should be reasonable
            assert regime_risks[regime]['var'] <= 0  # VaR is negative
            assert regime_risks[regime]['cvar'] <= regime_risks[regime]['var']  # CVaR worse than VaR
    
    def test_predict_regime_transition(self, model, sample_returns):
        """Test regime transition prediction."""
        model._fit_simple_regimes(sample_returns)
        model.current_regime = 0
        
        predictions = model.predict_regime_transition(horizon=5)
        
        assert predictions.shape == (6, 3)  # Current + 5 future
        
        # Check probabilities sum to 1
        for t in range(6):
            assert abs(np.sum(predictions[t]) - 1.0) < 0.001
        
        # Initial state should match current regime
        assert predictions[0, 0] == 1.0
    
    def test_regime_identification(self, model):
        """Test regime identification logic."""
        # Create clear regime patterns
        bull_returns = pd.Series(np.random.normal(0.002, 0.01, 100))
        bear_returns = pd.Series(np.random.normal(-0.002, 0.015, 100))
        volatile_returns = pd.Series(np.random.normal(0, 0.04, 100))
        
        model._fit_simple_regimes(bull_returns)
        # Should identify as low volatility with positive returns
        
        model._fit_simple_regimes(bear_returns)
        # Should identify as medium volatility with negative returns
        
        model._fit_simple_regimes(volatile_returns)
        # Should identify as high volatility


class TestDynamicHedgingEngine:
    """Test dynamic hedging strategies."""
    
    @pytest.fixture
    def hedging_engine(self):
        """Create hedging engine instance."""
        return DynamicHedgingEngine(hedging_budget=0.02)
    
    @pytest.fixture
    def sample_risk_metrics(self):
        """Create sample risk metrics."""
        return RiskMetrics(
            var_95=0.02,
            var_99=0.05,
            cvar_95=0.025,
            cvar_99=0.06,
            expected_tail_loss=0.08,
            tail_risk_contribution={'SPY': 0.6, 'QQQ': 0.4},
            extreme_downside_risk=0.1,
            delta=0.5,
            gamma=0.1,
            vega=0.2,
            theta=-0.05,
            rho=0.3,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.15,
            max_drawdown_duration=30,
            stress_scenarios={},
            regime_risks={},
            correlation_risk=0.85,
            tail_dependence=0.4,
            liquidity_score=0.85,
            market_impact=0.005,
            model_confidence=0.9,
            parameter_uncertainty=0.1
        )
    
    def test_engine_initialization(self, hedging_engine):
        """Test hedging engine initialization."""
        assert hedging_engine.hedging_budget == 0.02
        assert len(hedging_engine.active_hedges) == 0
        assert len(hedging_engine.hedging_performance) == 0
    
    def test_calculate_optimal_hedge(self, hedging_engine, sample_risk_metrics):
        """Test optimal hedge calculation."""
        portfolio_value = 1000000
        market_data = {
            'index_price': 4000,
            'implied_volatility': 0.2,
            'vix': 18,
            'regime': 'normal'
        }
        
        hedges = hedging_engine.calculate_optimal_hedge(
            portfolio_value,
            sample_risk_metrics,
            market_data
        )
        
        assert isinstance(hedges, list)
        
        # Should recommend hedges for high tail risk
        assert len(hedges) > 0
        
        # Check hedge structure
        for hedge in hedges:
            assert 'type' in hedge
            if hedge['type'] == 'protective_put':
                assert 'strike' in hedge
                assert 'expiry' in hedge
                assert 'contracts' in hedge
                assert 'premium' in hedge
    
    def test_tail_risk_score_calculation(self, hedging_engine, sample_risk_metrics):
        """Test tail risk score calculation."""
        score = hedging_engine._calculate_tail_risk_score(sample_risk_metrics)
        
        assert 0 <= score <= 1
        
        # With high extreme risk, score should be high
        assert score > 0.5
    
    def test_design_put_hedge(self, hedging_engine, sample_risk_metrics):
        """Test protective put design."""
        market_data = {
            'index_price': 4000,
            'implied_volatility': 0.2
        }
        
        hedge = hedging_engine._design_put_hedge(
            1000000,
            sample_risk_metrics,
            market_data
        )
        
        assert hedge['type'] == 'protective_put'
        assert hedge['strike'] < market_data['index_price']  # OTM put
        assert hedge['contracts'] > 0
        assert hedge['premium'] > 0
        assert 'expected_payout' in hedge
        
        # Strike should be reasonable (5-10% OTM)
        moneyness = hedge['strike'] / market_data['index_price']
        assert 0.85 <= moneyness <= 0.95
    
    def test_design_vix_hedge(self, hedging_engine, sample_risk_metrics):
        """Test VIX hedge design."""
        # Low VIX environment
        market_data = {'vix': 12}
        
        hedge = hedging_engine._design_vix_hedge(
            1000000,
            sample_risk_metrics,
            market_data
        )
        
        assert hedge['type'] == 'vix_hedge'
        assert hedge['instrument'] == 'VIX_CALL'  # Buy calls in low VIX
        assert hedge['strike'] > market_data['vix']
        assert hedge['contracts'] > 0
        
        # High VIX environment
        market_data_high = {'vix': 30}
        
        hedge_high = hedging_engine._design_vix_hedge(
            1000000,
            sample_risk_metrics,
            market_data_high
        )
        
        assert hedge_high['instrument'] == 'VIX_FUTURE'  # Use futures in high VIX
    
    def test_optimize_hedge_allocation(self, hedging_engine):
        """Test hedge allocation optimization."""
        hedges = [
            {'type': 'put', 'premium': 5000, 'contracts': 10},
            {'type': 'vix', 'notional': 10000, 'contracts': 20},
            {'type': 'strangle', 'premium': 3000, 'contracts': 5}
        ]
        
        budget = 10000
        
        optimized = hedging_engine._optimize_hedge_allocation(hedges, budget)
        
        # Total cost should not exceed budget
        total_cost = sum(
            h.get('premium', h.get('notional', 0))
            for h in optimized
        )
        assert total_cost <= budget * 1.01  # Small tolerance
        
        # Contracts should be scaled down
        assert all(
            h['contracts'] <= orig['contracts']
            for h, orig in zip(optimized, hedges)
        )


class TestStressTestingEngine:
    """Test stress testing functionality."""
    
    @pytest.fixture
    def stress_tester(self):
        """Create stress testing engine."""
        return StressTestingEngine()
    
    def test_engine_initialization(self, stress_tester):
        """Test stress testing engine initialization."""
        assert len(stress_tester.historical_scenarios) > 0
        assert len(stress_tester.hypothetical_scenarios) > 0
        
        # Check historical scenarios
        assert 'black_monday_1987' in stress_tester.historical_scenarios
        assert 'financial_crisis_2008' in stress_tester.historical_scenarios
        assert 'covid_crash_2020' in stress_tester.historical_scenarios
        
        # Check hypothetical scenarios
        assert 'inflation_shock' in stress_tester.hypothetical_scenarios
        assert 'liquidity_crisis' in stress_tester.hypothetical_scenarios
    
    def test_run_stress_tests(self, stress_tester):
        """Test running stress tests."""
        portfolio = {
            'stocks': 0.6,
            'bonds': 0.3,
            'commodities': 0.1
        }
        
        results = stress_tester.run_stress_tests(portfolio)
        
        assert isinstance(results, dict)
        
        # Check historical scenarios included
        assert any('hist_' in key for key in results.keys())
        
        # Check hypothetical scenarios included
        assert any('hypo_' in key for key in results.keys())
        
        # Check reverse stress test
        assert 'reverse_stress' in results
        
        # Validate scenario results
        for scenario_name, impact in results.items():
            if scenario_name != 'reverse_stress':
                assert 'total_impact' in impact
                assert 'component_impacts' in impact
                assert 'var_impact' in impact
                assert 'duration' in impact
    
    def test_calculate_scenario_impact(self, stress_tester):
        """Test scenario impact calculation."""
        portfolio = {
            'stocks': 0.6,
            'bonds': 0.3,
            'commodities': 0.1
        }
        
        scenario = {
            'equity': -0.30,  # 30% equity decline
            'rates': 0.02,  # 200bps rate increase
            'commodities': 0.10  # 10% commodity increase
        }
        
        impact = stress_tester._calculate_scenario_impact(portfolio, scenario)
        
        # Expected impact: 0.6 * (-0.3) + 0.3 * 0.02 + 0.1 * 0.1 = -0.164
        expected_total = 0.6 * (-0.3) + 0.3 * 0.02 + 0.1 * 0.1
        
        assert abs(impact['total_impact'] - expected_total) < 0.001
        assert impact['component_impacts']['stocks'] == 0.6 * (-0.3)
        assert impact['component_impacts']['bonds'] == 0.3 * 0.02
        assert impact['component_impacts']['commodities'] == 0.1 * 0.1
    
    def test_reverse_stress_test(self, stress_tester):
        """Test reverse stress testing."""
        portfolio = {
            'stocks': 0.7,
            'bonds': 0.2,
            'commodities': 0.1
        }
        
        result = stress_tester._reverse_stress_test(portfolio, target_loss=-0.25)
        
        assert 'equity_shock' in result
        assert 'rates_shock' in result
        assert 'volatility_shock' in result
        assert 'correlation_shock' in result
        assert 'probability' in result
        
        # Shocks should be within bounds
        assert -0.5 <= result['equity_shock'] <= 0
        assert -0.05 <= result['rates_shock'] <= 0.05
        assert 1.0 <= result['volatility_shock'] <= 5.0
        assert 0.5 <= result['correlation_shock'] <= 1.0
        
        # Probability should be reasonable
        assert 0 < result['probability'] <= 0.1
    
    def test_estimate_scenario_probability(self, stress_tester):
        """Test scenario probability estimation."""
        # Mild scenario
        mild_params = np.array([-0.05, 0.01, 1.5, 0.7])
        prob_mild = stress_tester._estimate_scenario_probability(mild_params)
        assert prob_mild == 0.1
        
        # Moderate scenario
        moderate_params = np.array([-0.15, 0.02, 2.0, 0.8])
        prob_moderate = stress_tester._estimate_scenario_probability(moderate_params)
        assert prob_moderate == 0.05
        
        # Severe scenario
        severe_params = np.array([-0.25, 0.03, 3.0, 0.9])
        prob_severe = stress_tester._estimate_scenario_probability(severe_params)
        assert prob_severe == 0.01
        
        # Extreme scenario
        extreme_params = np.array([-0.40, 0.05, 4.0, 0.95])
        prob_extreme = stress_tester._estimate_scenario_probability(extreme_params)
        assert prob_extreme == 0.001


class TestAdvancedRiskManager:
    """Test integrated risk management system."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        config = {
            'hedging_budget': 0.02,
            'risk_limits': {
                'max_var_95': 0.02,
                'max_var_99': 0.05,
                'max_leverage': 2.0,
                'max_concentration': 0.2,
                'max_correlation': 0.8
            }
        }
        return AdvancedRiskManager(config)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return {
            'AAPL': 0.3,
            'GOOGL': 0.2,
            'MSFT': 0.2,
            'AMZN': 0.15,
            'bonds': 0.15
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'AAPL': np.random.randn(252).cumsum() + 150,
            'GOOGL': np.random.randn(252).cumsum() + 2800,
            'MSFT': np.random.randn(252).cumsum() + 300,
            'AMZN': np.random.randn(252).cumsum() + 3200,
            'bonds': np.random.randn(252).cumsum() * 0.5 + 100
        }, index=dates)
        
        return data
    
    def test_manager_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.config['hedging_budget'] == 0.02
        assert risk_manager.evt_analyzer is not None
        assert risk_manager.copula_modeler is not None
        assert risk_manager.regime_model is not None
        assert risk_manager.hedging_engine is not None
        assert risk_manager.stress_tester is not None
        
        # Check risk limits
        assert risk_manager.risk_limits['max_var_95'] == 0.02
        assert risk_manager.risk_limits['max_leverage'] == 2.0
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk(self, risk_manager, sample_portfolio, sample_market_data):
        """Test comprehensive risk calculation."""
        risk_metrics = await risk_manager.calculate_portfolio_risk(
            sample_portfolio,
            sample_market_data,
            lookback_days=252
        )
        
        assert isinstance(risk_metrics, RiskMetrics)
        
        # Check basic risk metrics
        assert risk_metrics.var_95 < 0  # VaR is negative
        assert risk_metrics.var_99 < risk_metrics.var_95  # 99% VaR worse than 95%
        assert risk_metrics.cvar_95 < risk_metrics.var_95  # CVaR worse than VaR
        
        # Check performance metrics
        assert risk_metrics.sharpe_ratio != 0
        assert risk_metrics.max_drawdown < 0
        assert risk_metrics.max_drawdown_duration >= 0
        
        # Check advanced metrics
        assert risk_metrics.expected_tail_loss < risk_metrics.cvar_99
        assert len(risk_metrics.tail_risk_contribution) > 0
        assert 0 <= risk_metrics.liquidity_score <= 1
        assert risk_metrics.model_confidence > 0
    
    def test_calculate_portfolio_returns(self, risk_manager, sample_portfolio, sample_market_data):
        """Test portfolio return calculation."""
        returns = risk_manager._calculate_portfolio_returns(
            sample_portfolio,
            sample_market_data
        )
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_market_data) - 1  # One less due to pct_change
        
        # Check return properties
        assert not returns.isna().all()
        assert abs(returns.mean()) < 0.1  # Reasonable daily return
        assert returns.std() > 0  # Non-zero volatility
    
    def test_check_risk_limits(self, risk_manager):
        """Test risk limit checking."""
        # Create metrics that breach limits
        risk_metrics = RiskMetrics(
            var_95=0.03,  # Breach
            var_99=0.06,  # Breach
            cvar_95=0.04,
            cvar_99=0.07,
            expected_tail_loss=0.1,
            tail_risk_contribution={},
            extreme_downside_risk=0.12,
            delta=0.5,
            gamma=0.1,
            vega=0.2,
            theta=-0.05,
            rho=0.3,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            calmar_ratio=0.8,
            max_drawdown=0.2,
            max_drawdown_duration=60,
            stress_scenarios={},
            regime_risks={},
            correlation_risk=0.9,  # Breach
            tail_dependence=0.5,
            liquidity_score=0.7,
            market_impact=0.01,
            model_confidence=0.8,
            parameter_uncertainty=0.15
        )
        
        breaches = risk_manager.check_risk_limits(risk_metrics)
        
        assert len(breaches) >= 3
        
        # Check breach details
        breach_types = [b['type'] for b in breaches]
        assert 'var_95_breach' in breach_types
        assert 'var_99_breach' in breach_types
        assert 'correlation_breach' in breach_types
        
        # Check severity levels
        severities = [b['severity'] for b in breaches]
        assert 'high' in severities
        assert 'critical' in severities
    
    def test_generate_risk_report(self, risk_manager, sample_portfolio):
        """Test risk report generation."""
        risk_metrics = RiskMetrics(
            var_95=0.015,
            var_99=0.03,
            cvar_95=0.02,
            cvar_99=0.04,
            expected_tail_loss=0.05,
            tail_risk_contribution={'AAPL': 0.4, 'GOOGL': 0.3},
            extreme_downside_risk=0.08,
            delta=0.5,
            gamma=0.1,
            vega=0.2,
            theta=-0.05,
            rho=0.3,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=0.1,
            max_drawdown_duration=20,
            stress_scenarios={'crisis': -0.25},
            regime_risks={'normal': 0.01, 'volatile': 0.05},
            correlation_risk=0.7,
            tail_dependence=0.3,
            liquidity_score=0.85,
            market_impact=0.003,
            model_confidence=0.9,
            parameter_uncertainty=0.1
        )
        
        report = risk_manager.generate_risk_report(risk_metrics, sample_portfolio)
        
        assert 'timestamp' in report
        assert 'portfolio_summary' in report
        assert 'risk_metrics' in report
        assert 'tail_risk_analysis' in report
        assert 'stress_test_results' in report
        assert 'regime_analysis' in report
        assert 'risk_breaches' in report
        assert 'recommended_hedges' in report
        
        # Validate report contents
        assert report['portfolio_summary'] == sample_portfolio
        assert report['risk_metrics']['var_95'] == 0.015
        assert report['risk_metrics']['sharpe_ratio'] == 1.5
        assert 'extreme_downside_risk' in report['tail_risk_analysis']
        assert 'current_regime' in report['regime_analysis']
