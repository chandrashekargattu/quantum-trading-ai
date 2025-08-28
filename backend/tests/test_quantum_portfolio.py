"""
Comprehensive tests for Quantum Portfolio Optimization module.
Tests VQE, QAOA, and quantum risk analysis with edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
import torch

from app.quantum.quantum_portfolio import (
    QuantumPortfolioOptimizer,
    QuantumPortfolioResult
)


class TestQuantumPortfolioOptimizer:
    """Test quantum portfolio optimization algorithms."""
    
    @pytest.fixture
    def optimizer(self):
        """Create quantum optimizer instance."""
        return QuantumPortfolioOptimizer(n_qubits=4, shots=1024)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        np.random.seed(42)
        n_assets = 4
        n_samples = 100
        
        returns = np.random.normal(0.001, 0.02, (n_samples, n_assets))
        covariance = np.cov(returns.T)
        
        return {
            'returns': np.mean(returns, axis=0),
            'covariance': covariance,
            'n_assets': n_assets
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with various parameters."""
        # Default initialization
        opt1 = QuantumPortfolioOptimizer()
        assert opt1.n_qubits == 4
        assert opt1.shots == 1024
        
        # Custom parameters
        opt2 = QuantumPortfolioOptimizer(n_qubits=8, shots=2048)
        assert opt2.n_qubits == 8
        assert opt2.shots == 2048
        
        # Edge case: minimum qubits
        opt3 = QuantumPortfolioOptimizer(n_qubits=1)
        assert opt3.n_qubits == 1
    
    @patch('app.quantum.quantum_portfolio.VQE')
    @patch('app.quantum.quantum_portfolio.COBYLA')
    def test_optimize_portfolio_vqe_success(self, mock_cobyla, mock_vqe, optimizer, sample_data):
        """Test successful VQE portfolio optimization."""
        # Mock VQE result
        mock_result = MagicMock()
        mock_result.optimal_point = np.array([0.25, 0.25, 0.25, 0.25])
        mock_result.eigenvalue = MagicMock(real=-0.5)
        mock_result.optimizer_result.nfev = 100
        
        mock_vqe_instance = MagicMock()
        mock_vqe_instance.compute_minimum_eigenvalue.return_value = mock_result
        mock_vqe.return_value = mock_vqe_instance
        
        # Run optimization
        result = optimizer.optimize_portfolio_vqe(
            returns=sample_data['returns'],
            covariance=sample_data['covariance'],
            risk_aversion=0.5
        )
        
        # Assertions
        assert isinstance(result, QuantumPortfolioResult)
        assert len(result.optimal_weights) == sample_data['n_assets']
        assert np.isclose(np.sum(result.optimal_weights), 1.0, rtol=1e-5)
        assert result.expected_return > 0
        assert result.risk > 0
        assert result.sharpe_ratio > 0
        assert result.quantum_advantage > 1.0
        assert result.convergence_iterations == 100
    
    def test_optimize_portfolio_vqe_edge_cases(self, optimizer):
        """Test VQE optimization with edge cases."""
        # Single asset (degenerate case)
        returns_single = np.array([0.01])
        cov_single = np.array([[0.0001]])
        
        result = optimizer.optimize_portfolio_vqe(returns_single, cov_single)
        assert np.isclose(result.optimal_weights[0], 1.0)
        
        # Zero returns
        returns_zero = np.zeros(4)
        cov_normal = np.eye(4) * 0.01
        
        result = optimizer.optimize_portfolio_vqe(returns_zero, cov_normal)
        assert result.expected_return == 0
        
        # High risk aversion
        returns_normal = np.array([0.01, 0.02, 0.015, 0.025])
        result_high_risk = optimizer.optimize_portfolio_vqe(
            returns_normal, cov_normal, risk_aversion=10.0
        )
        # Should prefer lower variance assets
        assert result_high_risk.risk < 0.1
    
    @patch('app.quantum.quantum_portfolio.QAOA')
    @patch('app.quantum.quantum_portfolio.MinimumEigenOptimizer')
    def test_optimize_portfolio_qaoa(self, mock_optimizer, mock_qaoa, optimizer, sample_data):
        """Test QAOA portfolio optimization."""
        # Mock QAOA result
        mock_result = MagicMock()
        mock_result.x = np.array([1, 0, 1, 0])  # Binary selection
        mock_result.fval = 0.1
        
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.solve.return_value = mock_result
        mock_optimizer.return_value = mock_optimizer_instance
        
        # Run optimization
        result = optimizer.optimize_portfolio_qaoa(
            returns=sample_data['returns'],
            covariance=sample_data['covariance'],
            target_return=0.01,
            p=3
        )
        
        # Assertions
        assert isinstance(result, QuantumPortfolioResult)
        assert result.circuit_depth == 6  # p * 2
        assert result.eigenvalue == 0.1
    
    def test_optimize_portfolio_qaoa_constraints(self, optimizer, sample_data):
        """Test QAOA with various constraints."""
        # Impossible target return
        with pytest.raises(Exception):
            optimizer.optimize_portfolio_qaoa(
                returns=sample_data['returns'],
                covariance=sample_data['covariance'],
                target_return=1.0  # Impossible high return
            )
        
        # Different p values
        for p in [1, 2, 5]:
            result = optimizer.optimize_portfolio_qaoa(
                returns=sample_data['returns'],
                covariance=sample_data['covariance'],
                target_return=0.001,
                p=p
            )
            assert result.circuit_depth == p * 2
    
    @patch('qml.device')
    def test_quantum_risk_analysis(self, mock_device, optimizer):
        """Test quantum risk analysis functionality."""
        # Setup
        portfolio_weights = np.array([0.3, 0.3, 0.2, 0.2])
        n_scenarios = 100
        market_scenarios = np.random.normal(0, 0.02, (n_scenarios, 4))
        
        # Mock quantum device
        mock_dev = MagicMock()
        mock_device.return_value = mock_dev
        
        # Run risk analysis
        risk_metrics = optimizer.quantum_risk_analysis(
            portfolio_weights=portfolio_weights,
            market_scenarios=market_scenarios,
            confidence_level=0.95
        )
        
        # Assertions
        assert 'quantum_var' in risk_metrics
        assert 'quantum_cvar' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'tail_risk_probability' in risk_metrics
        assert risk_metrics['tail_risk_probability'] == 0.05  # 1 - confidence_level
        assert risk_metrics['risk_quantum_advantage'] > 1.0
    
    def test_quantum_correlation_analysis(self, optimizer):
        """Test quantum correlation analysis."""
        # Generate correlated asset returns
        n_samples = 100
        n_assets = 4
        
        # Create correlation structure
        correlation_matrix = np.array([
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0]
        ])
        
        # Generate correlated returns
        mean = np.zeros(n_assets)
        returns = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
        
        # Analyze correlations
        quantum_kernel = optimizer.quantum_correlation_analysis(returns)
        
        # Assertions
        assert quantum_kernel.shape == (n_samples, n_samples)
        assert np.all(np.diag(quantum_kernel) >= 0)  # Diagonal elements non-negative
        assert np.allclose(quantum_kernel, quantum_kernel.T)  # Symmetric
    
    def test_build_vqe_circuit(self, optimizer):
        """Test VQE circuit construction."""
        from qiskit import QuantumRegister, ClassicalRegister
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        returns = np.array([0.01, 0.02, 0.015, 0.025])
        covariance = np.eye(4) * 0.01
        
        circuit = optimizer._build_vqe_circuit(qr, cr, returns, covariance)
        
        # Assertions
        assert circuit.num_qubits == 4
        assert circuit.num_parameters > 0  # Has parameterized gates
        assert any(gate.name == 'h' for gate, _, _ in circuit.data)  # Has Hadamard
        assert any(gate.name == 'ry' for gate, _, _ in circuit.data)  # Has RY rotation
        assert any(gate.name == 'measure' for gate, _, _ in circuit.data)  # Has measurement
    
    def test_extract_portfolio_weights(self, optimizer):
        """Test portfolio weight extraction from quantum state."""
        # Test various parameter configurations
        test_cases = [
            (np.array([0.5, 0.5, 0.5, 0.5]), 4),  # Equal weights
            (np.array([1.0, 0.0, 0.0, 0.0]), 4),  # Single asset
            (np.array([0.6, 0.8, 0.0, 0.0]), 4),  # Partial allocation
        ]
        
        for params, n_assets in test_cases:
            weights = optimizer._extract_portfolio_weights(params)
            
            # Assertions
            assert len(weights) == n_assets
            assert np.isclose(np.sum(weights), 1.0, rtol=1e-5)  # Sum to 1
            assert np.all(weights >= 0)  # Non-negative
    
    def test_calculate_quantum_advantage(self, optimizer):
        """Test quantum advantage calculation."""
        # Test different problem sizes
        problem_sizes = [4, 10, 20, 50, 100]
        
        for size in problem_sizes:
            advantage = optimizer._calculate_quantum_advantage(size)
            
            # Quantum advantage should increase with problem size
            assert advantage > 1.0
            assert advantage < size ** 2  # Reasonable upper bound
            
        # Verify monotonic increase
        advantages = [optimizer._calculate_quantum_advantage(n) for n in problem_sizes]
        assert all(advantages[i] < advantages[i+1] for i in range(len(advantages)-1))
    
    def test_quantum_max_drawdown(self, optimizer):
        """Test quantum-enhanced maximum drawdown calculation."""
        # Generate sample returns with known drawdown
        returns = np.array([0.01, 0.02, -0.05, -0.03, 0.02, 0.01, -0.02])
        
        max_dd = optimizer._quantum_max_drawdown(returns)
        
        # Calculate expected drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        expected_dd = np.min((cumulative - running_max) / running_max)
        
        assert np.isclose(max_dd, expected_dd, rtol=1e-5)
        assert max_dd < 0  # Drawdown is negative
    
    def test_error_handling(self, optimizer):
        """Test error handling and edge cases."""
        # Empty data
        with pytest.raises(Exception):
            optimizer.optimize_portfolio_vqe(
                returns=np.array([]),
                covariance=np.array([[]])
            )
        
        # Mismatched dimensions
        with pytest.raises(Exception):
            optimizer.optimize_portfolio_vqe(
                returns=np.array([0.01, 0.02]),
                covariance=np.array([[0.01, 0], [0, 0.01], [0, 0, 0.01]])
            )
        
        # Negative covariance (invalid)
        with pytest.raises(Exception):
            optimizer.optimize_portfolio_vqe(
                returns=np.array([0.01, 0.02]),
                covariance=np.array([[-0.01, 0], [0, 0.01]])
            )
    
    @pytest.mark.parametrize("n_assets,risk_aversion", [
        (2, 0.1),
        (4, 0.5),
        (8, 1.0),
        (3, 5.0),
    ])
    def test_parametrized_optimization(self, optimizer, n_assets, risk_aversion):
        """Test optimization with various parameter combinations."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.005, n_assets)
        covariance = np.random.rand(n_assets, n_assets)
        covariance = (covariance + covariance.T) / 2  # Make symmetric
        covariance += np.eye(n_assets) * 0.01  # Ensure positive definite
        
        result = optimizer.optimize_portfolio_vqe(
            returns=returns,
            covariance=covariance,
            risk_aversion=risk_aversion
        )
        
        # General assertions
        assert len(result.optimal_weights) == n_assets
        assert np.isclose(np.sum(result.optimal_weights), 1.0, rtol=1e-3)
        assert result.risk > 0
        
        # Risk aversion should affect risk level
        if risk_aversion > 1.0:
            assert result.risk < 0.05  # Lower risk for high aversion


class TestQuantumIntegration:
    """Test integration with other system components."""
    
    @pytest.mark.asyncio
    async def test_quantum_with_risk_management(self):
        """Test quantum optimizer integration with risk management."""
        from app.risk_management import AdvancedRiskManager
        
        # Setup
        optimizer = QuantumPortfolioOptimizer()
        risk_manager = AdvancedRiskManager({'hedging_budget': 0.02})
        
        # Generate portfolio
        returns = np.array([0.01, 0.02, 0.015, 0.025])
        covariance = np.eye(4) * 0.01
        
        quantum_result = optimizer.optimize_portfolio_vqe(returns, covariance)
        
        # Convert to portfolio dict
        portfolio = {
            f'asset_{i}': float(quantum_result.optimal_weights[i])
            for i in range(len(quantum_result.optimal_weights))
        }
        
        # Risk analysis should work with quantum-optimized portfolio
        market_data = pd.DataFrame(
            np.random.normal(0, 0.02, (100, 4)),
            columns=[f'asset_{i}' for i in range(4)]
        )
        
        risk_metrics = await risk_manager.calculate_portfolio_risk(
            portfolio, market_data
        )
        
        assert risk_metrics.sharpe_ratio > 0
        assert risk_metrics.quantum_advantage == quantum_result.quantum_advantage
