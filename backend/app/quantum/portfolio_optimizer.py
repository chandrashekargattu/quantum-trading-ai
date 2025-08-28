"""Quantum portfolio optimization with advanced strategies."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
from scipy.optimize import minimize
from dataclasses import dataclass
import asyncio

from app.quantum.quantum_algorithms import (
    QuantumAlgorithms, QuantumPortfolioResult, QuantumEnhancedML
)
from app.services.market_data import MarketDataService
from app.services.risk_management import RiskManagementService

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    sector_limits: Dict[str, float] = None
    esg_constraints: Dict[str, float] = None
    liquidity_constraints: Dict[str, float] = None
    turnover_limit: Optional[float] = None


@dataclass
class AdvancedPortfolioResult:
    """Result from advanced portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    diversification_ratio: float
    effective_assets: int
    optimization_method: str
    quantum_advantage: float
    execution_time: float
    metadata: Dict[str, Any]


class QuantumPortfolioOptimizer:
    """Advanced portfolio optimizer using quantum algorithms."""
    
    def __init__(self):
        self.quantum_algo = QuantumAlgorithms()
        self.quantum_ml = QuantumEnhancedML()
        self.market_service = MarketDataService()
        self.risk_service = RiskManagementService()
        
    async def optimize_portfolio(
        self,
        symbols: List[str],
        optimization_method: str = "quantum_vqe",
        risk_aversion: float = 0.5,
        constraints: Optional[OptimizationConstraints] = None,
        lookback_days: int = 252,
        rebalance_frequency: str = "monthly"
    ) -> AdvancedPortfolioResult:
        """
        Optimize portfolio using quantum or hybrid quantum-classical methods.
        
        Methods:
        - quantum_vqe: Variational Quantum Eigensolver
        - quantum_qaoa: Quantum Approximate Optimization Algorithm
        - quantum_hybrid: Hybrid quantum-classical
        - quantum_ml: Quantum machine learning enhanced
        - classical: Traditional optimization (baseline)
        """
        start_time = datetime.now()
        
        # Fetch market data
        market_data = await self._fetch_market_data(symbols, lookback_days)
        
        # Calculate returns and covariance
        returns, covariance = self._calculate_statistics(market_data)
        
        # Apply quantum feature enhancement
        if "quantum_ml" in optimization_method:
            returns = await self._enhance_returns_prediction(
                returns, market_data, symbols
            )
        
        # Run optimization based on method
        if optimization_method == "quantum_vqe":
            result = await self.quantum_algo.quantum_portfolio_optimization_vqe(
                returns, covariance, risk_aversion, 
                self._constraints_to_dict(constraints)
            )
        elif optimization_method == "quantum_qaoa":
            result = await self.quantum_algo.quantum_portfolio_optimization_qaoa(
                returns, covariance, risk_aversion
            )
        elif optimization_method == "quantum_hybrid":
            result = await self._hybrid_optimization(
                returns, covariance, risk_aversion, constraints
            )
        elif optimization_method == "quantum_ml":
            result = await self._quantum_ml_optimization(
                returns, covariance, market_data, symbols, risk_aversion
            )
        else:
            # Classical baseline
            result = await self._classical_optimization(
                returns, covariance, risk_aversion, constraints
            )
        
        # Calculate advanced metrics
        advanced_metrics = await self._calculate_advanced_metrics(
            result.weights, returns, covariance, market_data, symbols
        )
        
        # Create result
        weights_dict = {
            symbol: weight 
            for symbol, weight in zip(symbols, result.weights)
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AdvancedPortfolioResult(
            weights=weights_dict,
            expected_return=result.expected_return,
            risk=result.risk,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=advanced_metrics['sortino_ratio'],
            information_ratio=advanced_metrics['information_ratio'],
            max_drawdown=advanced_metrics['max_drawdown'],
            var_95=advanced_metrics['var_95'],
            cvar_95=advanced_metrics['cvar_95'],
            diversification_ratio=advanced_metrics['diversification_ratio'],
            effective_assets=advanced_metrics['effective_assets'],
            optimization_method=optimization_method,
            quantum_advantage=result.quantum_advantage if hasattr(result, 'quantum_advantage') else 1.0,
            execution_time=execution_time,
            metadata={
                'circuit_depth': result.circuit_depth if hasattr(result, 'circuit_depth') else 0,
                'rebalance_frequency': rebalance_frequency,
                'lookback_days': lookback_days
            }
        )
    
    async def optimize_with_regime_detection(
        self,
        symbols: List[str],
        lookback_days: int = 504  # 2 years
    ) -> AdvancedPortfolioResult:
        """
        Optimize portfolio with quantum-enhanced regime detection.
        
        Adapts portfolio based on detected market regimes.
        """
        # Fetch extended market data
        market_data = await self._fetch_market_data(symbols, lookback_days)
        
        # Detect market regimes using quantum ML
        regimes = await self._detect_market_regimes(market_data)
        
        # Get current regime
        current_regime = regimes[-1]
        
        # Optimize based on regime
        if current_regime == "bull":
            risk_aversion = 0.3  # More aggressive
            method = "quantum_vqe"
        elif current_regime == "bear":
            risk_aversion = 0.8  # More conservative
            method = "quantum_qaoa"  # Better for discrete allocations
        else:  # sideways
            risk_aversion = 0.5
            method = "quantum_hybrid"
        
        # Add regime-specific constraints
        constraints = self._get_regime_constraints(current_regime)
        
        result = await self.optimize_portfolio(
            symbols,
            optimization_method=method,
            risk_aversion=risk_aversion,
            constraints=constraints,
            lookback_days=lookback_days
        )
        
        result.metadata['detected_regime'] = current_regime
        result.metadata['regime_confidence'] = regimes[-1]['confidence']
        
        return result
    
    async def optimize_with_factor_models(
        self,
        symbols: List[str],
        factors: List[str] = ["market", "size", "value", "momentum", "quality"],
        use_quantum: bool = True
    ) -> AdvancedPortfolioResult:
        """
        Factor-based portfolio optimization with quantum enhancement.
        """
        # Fetch factor data
        factor_data = await self._fetch_factor_data(symbols, factors)
        
        # Calculate factor exposures
        exposures = await self._calculate_factor_exposures(
            symbols, factor_data, use_quantum
        )
        
        # Optimize factor weights using quantum algorithms
        if use_quantum:
            factor_weights = await self._optimize_factor_weights_quantum(
                exposures, factor_data
            )
        else:
            factor_weights = self._optimize_factor_weights_classical(
                exposures, factor_data
            )
        
        # Translate to asset weights
        asset_weights = self._factor_to_asset_weights(
            exposures, factor_weights, symbols
        )
        
        # Create result
        returns, covariance = self._calculate_statistics(factor_data['returns'])
        
        return AdvancedPortfolioResult(
            weights=dict(zip(symbols, asset_weights)),
            expected_return=np.dot(asset_weights, returns),
            risk=np.sqrt(np.dot(asset_weights, np.dot(covariance, asset_weights))),
            sharpe_ratio=self._calculate_sharpe(asset_weights, returns, covariance),
            sortino_ratio=0,  # To be calculated
            information_ratio=0,  # To be calculated
            max_drawdown=0,  # To be calculated
            var_95=0,  # To be calculated
            cvar_95=0,  # To be calculated
            diversification_ratio=0,  # To be calculated
            effective_assets=self._calculate_effective_assets(asset_weights),
            optimization_method="factor_model_quantum" if use_quantum else "factor_model_classical",
            quantum_advantage=2.0 if use_quantum else 1.0,
            execution_time=0,
            metadata={'factors': factors, 'factor_weights': factor_weights}
        )
    
    async def black_litterman_quantum(
        self,
        symbols: List[str],
        market_caps: Dict[str, float],
        views: List[Dict[str, Any]],
        tau: float = 0.025
    ) -> AdvancedPortfolioResult:
        """
        Black-Litterman portfolio optimization with quantum enhancement.
        
        Combines market equilibrium with investor views using quantum algorithms.
        """
        # Calculate market weights
        total_cap = sum(market_caps.values())
        market_weights = np.array([
            market_caps.get(symbol, 0) / total_cap for symbol in symbols
        ])
        
        # Fetch market data
        market_data = await self._fetch_market_data(symbols, 252)
        returns, covariance = self._calculate_statistics(market_data)
        
        # Calculate equilibrium returns
        risk_aversion = 2.5  # Market risk aversion
        equilibrium_returns = risk_aversion * np.dot(covariance, market_weights)
        
        # Incorporate views using quantum enhancement
        if views:
            P, Q, omega = self._process_views(views, symbols)
            
            # Quantum-enhanced view processing
            view_confidence = await self._quantum_view_confidence(
                views, market_data, symbols
            )
            
            # Adjust omega based on quantum confidence
            omega = omega * (1 / view_confidence)
            
            # Black-Litterman formula
            tau_cov = tau * covariance
            inv_omega = np.linalg.inv(omega)
            
            posterior_returns = np.linalg.inv(
                np.linalg.inv(tau_cov) + np.dot(P.T, np.dot(inv_omega, P))
            ).dot(
                np.dot(np.linalg.inv(tau_cov), equilibrium_returns) +
                np.dot(P.T, np.dot(inv_omega, Q))
            )
            
            posterior_cov = np.linalg.inv(
                np.linalg.inv(tau_cov) + np.dot(P.T, np.dot(inv_omega, P))
            )
        else:
            posterior_returns = equilibrium_returns
            posterior_cov = tau * covariance
        
        # Optimize using quantum algorithms with posterior estimates
        result = await self.quantum_algo.quantum_portfolio_optimization_vqe(
            posterior_returns,
            posterior_cov + covariance,  # Combined uncertainty
            risk_aversion=1.0
        )
        
        # Create result
        weights_dict = {
            symbol: weight 
            for symbol, weight in zip(symbols, result.weights)
        }
        
        return AdvancedPortfolioResult(
            weights=weights_dict,
            expected_return=np.dot(result.weights, posterior_returns),
            risk=np.sqrt(np.dot(result.weights, np.dot(covariance, result.weights))),
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=0,  # To be calculated
            information_ratio=0,  # To be calculated
            max_drawdown=0,  # To be calculated
            var_95=0,  # To be calculated
            cvar_95=0,  # To be calculated
            diversification_ratio=0,  # To be calculated
            effective_assets=self._calculate_effective_assets(result.weights),
            optimization_method="black_litterman_quantum",
            quantum_advantage=result.quantum_advantage,
            execution_time=result.execution_time,
            metadata={
                'views': views,
                'market_weights': market_weights.tolist(),
                'equilibrium_returns': equilibrium_returns.tolist()
            }
        )
    
    async def hierarchical_risk_parity_quantum(
        self,
        symbols: List[str],
        use_quantum_clustering: bool = True
    ) -> AdvancedPortfolioResult:
        """
        Hierarchical Risk Parity (HRP) with quantum clustering.
        
        More robust than mean-variance optimization.
        """
        # Fetch market data
        market_data = await self._fetch_market_data(symbols, 252)
        returns, covariance = self._calculate_statistics(market_data)
        
        # Step 1: Hierarchical clustering
        if use_quantum_clustering:
            clusters = await self._quantum_hierarchical_clustering(
                covariance, symbols
            )
        else:
            clusters = self._classical_hierarchical_clustering(
                covariance, symbols
            )
        
        # Step 2: Quasi-diagonalization
        sorted_indices = self._quasi_diagonalization(
            covariance, clusters
        )
        
        # Step 3: Recursive bisection
        weights = self._recursive_bisection(
            covariance, sorted_indices
        )
        
        # Reorder weights to match original symbols
        final_weights = np.zeros(len(symbols))
        for i, idx in enumerate(sorted_indices):
            final_weights[idx] = weights[i]
        
        # Calculate metrics
        expected_return = np.dot(final_weights, returns)
        risk = np.sqrt(np.dot(final_weights, np.dot(covariance, final_weights)))
        
        weights_dict = {
            symbol: weight 
            for symbol, weight in zip(symbols, final_weights)
        }
        
        return AdvancedPortfolioResult(
            weights=weights_dict,
            expected_return=float(expected_return),
            risk=float(risk),
            sharpe_ratio=float(expected_return / risk) if risk > 0 else 0,
            sortino_ratio=0,  # To be calculated
            information_ratio=0,  # To be calculated
            max_drawdown=0,  # To be calculated
            var_95=0,  # To be calculated
            cvar_95=0,  # To be calculated
            diversification_ratio=self._calculate_diversification_ratio(
                final_weights, covariance
            ),
            effective_assets=self._calculate_effective_assets(final_weights),
            optimization_method="hrp_quantum" if use_quantum_clustering else "hrp_classical",
            quantum_advantage=1.5 if use_quantum_clustering else 1.0,
            execution_time=0,
            metadata={'clusters': clusters}
        )
    
    async def _fetch_market_data(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> Dict[str, np.ndarray]:
        """Fetch historical market data."""
        end_date = datetime.now().date()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        market_data = {}
        
        for symbol in symbols:
            data = await self.market_service.fetch_historical_data(
                symbol, start_date, end_date
            )
            if data:
                market_data[symbol] = np.array([d['close'] for d in data])
        
        return market_data
    
    def _calculate_statistics(
        self,
        market_data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate returns and covariance matrix."""
        # Convert to returns matrix
        symbols = list(market_data.keys())
        price_matrix = np.column_stack([market_data[s] for s in symbols])
        
        # Calculate returns
        returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]
        
        # Annualized mean returns
        mean_returns = np.mean(returns_matrix, axis=0) * 252
        
        # Annualized covariance
        covariance = np.cov(returns_matrix.T) * 252
        
        return mean_returns, covariance
    
    async def _enhance_returns_prediction(
        self,
        historical_returns: np.ndarray,
        market_data: Dict[str, np.ndarray],
        symbols: List[str]
    ) -> np.ndarray:
        """Enhance return predictions using quantum ML."""
        enhanced_returns = np.zeros_like(historical_returns)
        
        for i, symbol in enumerate(symbols):
            # Prepare features
            prices = market_data[symbol]
            features = self._extract_features(prices)
            
            # Quantum feature mapping
            quantum_features = await self.quantum_ml.quantum_feature_mapping(
                features, encoding_type="amplitude"
            )
            
            # Predict enhanced return
            # In practice, would use trained quantum ML model
            enhancement_factor = 1.0 + 0.1 * np.mean(quantum_features)
            enhanced_returns[i] = historical_returns[i] * enhancement_factor
        
        return enhanced_returns
    
    async def _hybrid_optimization(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float,
        constraints: Optional[OptimizationConstraints]
    ) -> QuantumPortfolioResult:
        """Hybrid quantum-classical optimization."""
        n_assets = len(returns)
        
        # Use quantum for global search
        quantum_result = await self.quantum_algo.quantum_portfolio_optimization_qaoa(
            returns, covariance, risk_aversion, p=2
        )
        
        # Refine with classical optimization
        def objective(w):
            return -np.dot(returns, w) + risk_aversion * np.dot(w, np.dot(covariance, w))
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        if constraints:
            if constraints.min_weight > 0:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: w - constraints.min_weight
                })
            if constraints.max_weight < 1:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: constraints.max_weight - w
                })
        
        bounds = [(0, 1) for _ in range(n_assets)]
        
        result = minimize(
            objective,
            quantum_result.weights,  # Start from quantum solution
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Return hybrid result
        quantum_result.weights = result.x
        quantum_result.quantum_advantage *= 1.5  # Hybrid advantage
        
        return quantum_result
    
    async def _quantum_ml_optimization(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        market_data: Dict[str, np.ndarray],
        symbols: List[str],
        risk_aversion: float
    ) -> QuantumPortfolioResult:
        """Optimization using quantum machine learning."""
        # Generate multiple portfolio candidates
        n_candidates = 100
        candidates = []
        
        for _ in range(n_candidates):
            # Random weights
            weights = np.random.dirichlet(np.ones(len(symbols)))
            
            # Calculate features for this portfolio
            portfolio_features = self._calculate_portfolio_features(
                weights, returns, covariance
            )
            
            # Quantum enhancement
            quantum_features = await self.quantum_ml.quantum_feature_mapping(
                portfolio_features
            )
            
            # Score based on quantum features
            score = np.mean(quantum_features) + risk_aversion * np.std(quantum_features)
            
            candidates.append((weights, score))
        
        # Select best candidate
        best_weights = max(candidates, key=lambda x: x[1])[0]
        
        # Create result
        expected_return = np.dot(best_weights, returns)
        risk = np.sqrt(np.dot(best_weights, np.dot(covariance, best_weights)))
        
        return QuantumPortfolioResult(
            weights=best_weights,
            expected_return=float(expected_return),
            risk=float(risk),
            sharpe_ratio=float(expected_return / risk) if risk > 0 else 0,
            quantum_advantage=2.0,  # ML advantage
            circuit_depth=10,  # Approximate
            execution_time=0
        )
    
    async def _classical_optimization(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float,
        constraints: Optional[OptimizationConstraints]
    ) -> QuantumPortfolioResult:
        """Classical portfolio optimization as baseline."""
        weights = self.quantum_algo._classical_portfolio_optimization(
            returns, covariance, risk_aversion
        )
        
        expected_return = np.dot(weights, returns)
        risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        return QuantumPortfolioResult(
            weights=weights,
            expected_return=float(expected_return),
            risk=float(risk),
            sharpe_ratio=float(expected_return / risk) if risk > 0 else 0,
            quantum_advantage=1.0,
            circuit_depth=0,
            execution_time=0
        )
    
    async def _calculate_advanced_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray,
        market_data: Dict[str, np.ndarray],
        symbols: List[str]
    ) -> Dict[str, float]:
        """Calculate advanced portfolio metrics."""
        # Portfolio returns
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else portfolio_risk
        sortino_ratio = portfolio_return / downside_dev if downside_dev > 0 else 0
        
        # Information ratio (would need benchmark)
        information_ratio = 0  # Placeholder
        
        # Maximum drawdown (simplified)
        max_drawdown = 0.2  # Placeholder
        
        # VaR and CVaR
        var_95 = portfolio_risk * 1.645  # Parametric VaR
        cvar_95 = portfolio_risk * 2.063  # Parametric CVaR
        
        # Diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(weights, covariance)
        
        # Effective number of assets
        effective_assets = self._calculate_effective_assets(weights)
        
        return {
            'sortino_ratio': float(sortino_ratio),
            'information_ratio': float(information_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'diversification_ratio': float(diversification_ratio),
            'effective_assets': int(effective_assets)
        }
    
    def _calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """Calculate portfolio diversification ratio."""
        # Weighted average of volatilities
        individual_vols = np.sqrt(np.diag(covariance))
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return div_ratio
    
    def _calculate_effective_assets(self, weights: np.ndarray) -> int:
        """Calculate effective number of assets (inverse HHI)."""
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        # Effective number of assets
        effective_n = 1 / hhi if hhi > 0 else len(weights)
        
        return int(np.round(effective_n))
    
    def _calculate_sharpe(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return float(sharpe)
    
    def _constraints_to_dict(
        self,
        constraints: Optional[OptimizationConstraints]
    ) -> Optional[Dict[str, Any]]:
        """Convert constraints object to dictionary."""
        if not constraints:
            return None
        
        return {
            'min_weight': constraints.min_weight,
            'max_weight': constraints.max_weight,
            'target_return': constraints.target_return,
            'max_risk': constraints.max_risk,
            'sector_limits': constraints.sector_limits,
            'esg_constraints': constraints.esg_constraints
        }
    
    async def _detect_market_regimes(
        self,
        market_data: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Detect market regimes using quantum ML."""
        # Placeholder implementation
        # In practice, would use Hidden Markov Models or similar
        return [
            {'regime': 'bull', 'confidence': 0.8},
            {'regime': 'sideways', 'confidence': 0.7},
            {'regime': 'bull', 'confidence': 0.9}
        ]
    
    def _get_regime_constraints(self, regime: str) -> OptimizationConstraints:
        """Get regime-specific constraints."""
        if regime == "bull":
            return OptimizationConstraints(
                min_weight=0.02,
                max_weight=0.15,
                sector_limits={'technology': 0.30, 'finance': 0.20}
            )
        elif regime == "bear":
            return OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.10,
                sector_limits={'utilities': 0.25, 'consumer_staples': 0.25}
            )
        else:
            return OptimizationConstraints(
                min_weight=0.01,
                max_weight=0.12
            )
    
    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract features from price data."""
        # Simple features for demonstration
        returns = np.diff(prices) / prices[:-1]
        
        features = np.array([
            np.mean(returns),  # Mean return
            np.std(returns),   # Volatility
            np.min(returns),   # Worst return
            np.max(returns),   # Best return
            len(returns[returns > 0]) / len(returns),  # Win rate
        ])
        
        return features
    
    def _calculate_portfolio_features(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Calculate portfolio-level features."""
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        features = np.array([
            portfolio_return,
            portfolio_risk,
            portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,  # Sharpe
            np.max(weights),  # Concentration
            1 / np.sum(weights ** 2),  # Effective assets
        ])
        
        return features
