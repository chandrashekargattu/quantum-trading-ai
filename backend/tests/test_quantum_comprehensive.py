"""Comprehensive quantum algorithm tests."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.quantum.portfolio_optimizer import QuantumPortfolioOptimizer
from app.quantum.quantum_algorithms import QuantumAlgorithms
from app.quantum.quantum_ml import QuantumMLPredictor


class TestQuantumPortfolioOptimization:
    """Test quantum portfolio optimization algorithms."""
    
    @pytest.mark.asyncio
    async def test_vqe_portfolio_optimization(self):
        """Test Variational Quantum Eigensolver for portfolio optimization."""
        optimizer = QuantumPortfolioOptimizer()
        
        # Test portfolio data
        returns = np.array([0.05, 0.08, 0.12, 0.03])
        covariance = np.array([
            [0.01, 0.002, 0.001, 0.0005],
            [0.002, 0.015, 0.003, 0.001],
            [0.001, 0.003, 0.02, 0.002],
            [0.0005, 0.001, 0.002, 0.008]
        ])
        
        # Mock quantum backend
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            weights = await optimizer.optimize_portfolio_vqe(
                returns=returns,
                covariance=covariance,
                risk_aversion=0.5
            )
        
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 0.01  # Weights sum to 1
        assert all(0 <= w <= 1 for w in weights)  # Valid weight range
    
    @pytest.mark.asyncio
    async def test_qaoa_combinatorial_optimization(self):
        """Test Quantum Approximate Optimization Algorithm."""
        optimizer = QuantumPortfolioOptimizer()
        
        # Asset selection problem
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        expected_returns = [0.08, 0.10, 0.09, 0.11, 0.15]
        constraints = {
            "max_assets": 3,
            "min_return": 0.09
        }
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            selected = await optimizer.select_assets_qaoa(
                assets=assets,
                returns=expected_returns,
                constraints=constraints
            )
        
        assert len(selected) <= 3
        assert all(asset in assets for asset in selected)
    
    @pytest.mark.asyncio
    async def test_quantum_monte_carlo(self):
        """Test Quantum Monte Carlo for option pricing."""
        algorithms = QuantumAlgorithms()
        
        # Option parameters
        spot_price = 100
        strike_price = 105
        time_to_maturity = 0.25
        volatility = 0.2
        risk_free_rate = 0.05
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            option_price = await algorithms.quantum_option_pricing(
                spot=spot_price,
                strike=strike_price,
                maturity=time_to_maturity,
                volatility=volatility,
                rate=risk_free_rate
            )
        
        assert option_price > 0
        assert option_price < spot_price  # Call option OTM
    
    @pytest.mark.asyncio
    async def test_amplitude_estimation(self):
        """Test quantum amplitude estimation for risk calculation."""
        algorithms = QuantumAlgorithms()
        
        # Risk calculation parameters
        portfolio_value = 1000000
        confidence_level = 0.95
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            var_estimate = await algorithms.estimate_var_quantum(
                portfolio_value=portfolio_value,
                confidence_level=confidence_level,
                num_qubits=5
            )
        
        assert var_estimate > 0
        assert var_estimate < portfolio_value * 0.1  # Reasonable VaR


class TestQuantumMachineLearning:
    """Test quantum machine learning algorithms."""
    
    @pytest.mark.asyncio
    async def test_quantum_svm_classification(self):
        """Test Quantum Support Vector Machine for market prediction."""
        qml = QuantumMLPredictor()
        
        # Training data
        X_train = np.random.randn(50, 4)  # 50 samples, 4 features
        y_train = np.random.randint(0, 2, 50)  # Binary classification
        
        with patch('pennylane.device') as mock_device:
            mock_device.return_value = MagicMock()
            
            await qml.train_quantum_svm(X_train, y_train)
            
            # Test prediction
            X_test = np.random.randn(10, 4)
            predictions = await qml.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    @pytest.mark.asyncio
    async def test_quantum_neural_network(self):
        """Test Variational Quantum Circuit as neural network."""
        qml = QuantumMLPredictor()
        
        # Time series data
        historical_prices = np.random.randn(100) * 10 + 100
        
        with patch('pennylane.device') as mock_device:
            mock_device.return_value = MagicMock()
            
            # Train QNN
            await qml.train_quantum_nn(
                data=historical_prices,
                lookback=10,
                epochs=5
            )
            
            # Predict next price
            next_price = await qml.predict_next_price(
                recent_prices=historical_prices[-10:]
            )
        
        assert isinstance(next_price, float)
        assert 50 < next_price < 150  # Reasonable range
    
    @pytest.mark.asyncio
    async def test_quantum_feature_map(self):
        """Test quantum feature mapping for data encoding."""
        qml = QuantumMLPredictor()
        
        # Classical features
        features = np.array([0.1, 0.5, -0.3, 0.8])
        
        with patch('pennylane.device') as mock_device:
            mock_device.return_value = MagicMock()
            
            quantum_features = await qml.encode_features_quantum(
                features=features,
                encoding_type="amplitude"
            )
        
        assert quantum_features is not None
        assert len(quantum_features.shape) == 1


class TestQuantumRiskAnalysis:
    """Test quantum algorithms for risk analysis."""
    
    @pytest.mark.asyncio
    async def test_quantum_correlation_analysis(self):
        """Test quantum algorithm for correlation matrix estimation."""
        algorithms = QuantumAlgorithms()
        
        # Asset returns data
        returns_data = np.random.randn(100, 5)  # 100 days, 5 assets
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            correlation_matrix = await algorithms.quantum_correlation_estimation(
                returns=returns_data,
                num_qubits=3
            )
        
        assert correlation_matrix.shape == (5, 5)
        assert np.allclose(correlation_matrix, correlation_matrix.T)
        assert np.all(np.diag(correlation_matrix) == 1)
    
    @pytest.mark.asyncio
    async def test_quantum_scenario_generation(self):
        """Test quantum algorithm for stress scenario generation."""
        algorithms = QuantumAlgorithms()
        
        # Market conditions
        current_conditions = {
            "volatility": 0.15,
            "correlation": 0.3,
            "trend": 0.02
        }
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            scenarios = await algorithms.generate_quantum_scenarios(
                current_conditions=current_conditions,
                num_scenarios=10,
                severity="extreme"
            )
        
        assert len(scenarios) == 10
        assert all("market_shock" in s for s in scenarios)


class TestHybridQuantumClassical:
    """Test hybrid quantum-classical algorithms."""
    
    @pytest.mark.asyncio
    async def test_hybrid_optimization(self):
        """Test hybrid quantum-classical optimization."""
        optimizer = QuantumPortfolioOptimizer()
        
        # Large portfolio problem
        num_assets = 20
        returns = np.random.randn(num_assets) * 0.1 + 0.05
        covariance = np.random.randn(num_assets, num_assets) * 0.01
        covariance = covariance @ covariance.T  # Make positive definite
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            # Hybrid approach: classical preprocessing + quantum optimization
            weights = await optimizer.hybrid_portfolio_optimization(
                returns=returns,
                covariance=covariance,
                classical_preprocessing=True,
                quantum_iterations=10
            )
        
        assert len(weights) == num_assets
        assert abs(sum(weights) - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_quantum_classical_ensemble(self):
        """Test ensemble of quantum and classical models."""
        qml = QuantumMLPredictor()
        
        # Market data
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 3, 100)  # 3-class problem
        
        with patch('pennylane.device') as mock_device:
            mock_device.return_value = MagicMock()
            
            # Train ensemble
            await qml.train_hybrid_ensemble(
                X=features,
                y=labels,
                quantum_models=2,
                classical_models=3
            )
            
            # Ensemble prediction
            predictions = await qml.ensemble_predict(features[:10])
        
        assert len(predictions) == 10
        assert all(0 <= p <= 2 for p in predictions)


class TestQuantumErrorMitigation:
    """Test quantum error mitigation techniques."""
    
    @pytest.mark.asyncio
    async def test_noise_mitigation(self):
        """Test noise mitigation in quantum calculations."""
        algorithms = QuantumAlgorithms()
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            # Run calculation with error mitigation
            result_mitigated = await algorithms.calculate_with_mitigation(
                circuit_type="portfolio_optimization",
                error_mitigation="zero_noise_extrapolation",
                shots=1000
            )
            
            # Run without mitigation
            result_raw = await algorithms.calculate_with_mitigation(
                circuit_type="portfolio_optimization",
                error_mitigation=None,
                shots=1000
            )
        
        assert result_mitigated is not None
        assert result_raw is not None
    
    @pytest.mark.asyncio
    async def test_circuit_optimization(self):
        """Test quantum circuit optimization techniques."""
        optimizer = QuantumPortfolioOptimizer()
        
        with patch('qiskit.transpile') as mock_transpile:
            mock_transpile.return_value = MagicMock()
            
            # Create and optimize circuit
            optimized_circuit = await optimizer.optimize_quantum_circuit(
                circuit_type="vqe",
                optimization_level=3,
                target_backend="ibmq_qasm_simulator"
            )
        
        assert optimized_circuit is not None


class TestQuantumEdgeCases:
    """Test edge cases in quantum algorithms."""
    
    @pytest.mark.asyncio
    async def test_small_portfolio_quantum(self):
        """Test quantum optimization with very small portfolio."""
        optimizer = QuantumPortfolioOptimizer()
        
        # 2-asset portfolio
        returns = np.array([0.05, 0.08])
        covariance = np.array([[0.01, 0.002], [0.002, 0.015]])
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            weights = await optimizer.optimize_portfolio_vqe(
                returns=returns,
                covariance=covariance,
                risk_aversion=0.5
            )
        
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_quantum_with_constraints(self):
        """Test quantum optimization with complex constraints."""
        optimizer = QuantumPortfolioOptimizer()
        
        constraints = {
            "max_weight": 0.3,
            "min_weight": 0.05,
            "sector_limits": {"tech": 0.4, "finance": 0.3},
            "esg_minimum": 0.5
        }
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            result = await optimizer.constrained_optimization(
                num_assets=10,
                constraints=constraints,
                quantum_approach="vqe"
            )
        
        assert result["feasible"] is True
        assert all(0.05 <= w <= 0.3 for w in result["weights"])
    
    @pytest.mark.asyncio
    async def test_quantum_circuit_depth_limit(self):
        """Test handling of circuit depth limitations."""
        algorithms = QuantumAlgorithms()
        
        # Large problem requiring deep circuit
        problem_size = 100
        
        with patch('qiskit_aer.Aer.get_backend') as mock_backend:
            mock_backend.return_value = MagicMock()
            
            # Should automatically decompose or approximate
            result = await algorithms.solve_large_problem(
                problem_size=problem_size,
                max_circuit_depth=100,
                decomposition_strategy="block"
            )
        
        assert result is not None
        assert result["approximation_error"] < 0.1
