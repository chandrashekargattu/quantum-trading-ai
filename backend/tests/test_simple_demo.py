"""
Simple demo test to verify test framework functionality.
This doesn't require heavy dependencies like qiskit or tensorflow.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from decimal import Decimal


class TestSimpleMathOperations:
    """Test basic math operations to verify pytest works."""
    
    def test_addition(self):
        """Test simple addition."""
        assert 2 + 2 == 4
        assert 10 + 5 == 15
    
    def test_numpy_operations(self):
        """Test numpy array operations."""
        arr = np.array([1, 2, 3, 4, 5])
        assert np.sum(arr) == 15
        assert np.mean(arr) == 3.0
        assert len(arr) == 5
    
    def test_decimal_precision(self):
        """Test decimal operations for financial calculations."""
        price1 = Decimal('100.50')
        price2 = Decimal('50.25')
        total = price1 + price2
        
        assert total == Decimal('150.75')
        assert str(total) == '150.75'
    
    @pytest.mark.parametrize("input_val,expected", [
        (0, 0),
        (1, 1),
        (100, 10000),
        (-5, 25),
    ])
    def test_parametrized_square(self, input_val, expected):
        """Test parametrized squaring function."""
        assert input_val ** 2 == expected


class TestMockingDemo:
    """Demonstrate mocking capabilities."""
    
    def test_mock_api_call(self):
        """Test mocking an API call."""
        # Create a mock API client
        mock_client = Mock()
        mock_client.get_price.return_value = {'symbol': 'AAPL', 'price': 150.0}
        
        # Use the mock
        result = mock_client.get_price('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 150.0
        mock_client.get_price.assert_called_once_with('AAPL')
    
    def test_mock_with_side_effect(self):
        """Test mock with side effects."""
        mock_calculator = Mock()
        mock_calculator.calculate_return.side_effect = [0.05, 0.10, -0.02]
        
        # First call
        assert mock_calculator.calculate_return() == 0.05
        # Second call
        assert mock_calculator.calculate_return() == 0.10
        # Third call
        assert mock_calculator.calculate_return() == -0.02


class TestExceptionHandling:
    """Test exception handling."""
    
    def test_division_by_zero(self):
        """Test that division by zero raises exception."""
        with pytest.raises(ZeroDivisionError):
            result = 10 / 0
    
    def test_custom_exception(self):
        """Test custom exception with message."""
        with pytest.raises(ValueError, match="Invalid price"):
            if -100 < 0:
                raise ValueError("Invalid price: cannot be negative")


class TestAsyncOperations:
    """Test async functionality."""
    
    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test async function execution."""
        async def fetch_data():
            return {'status': 'success', 'data': [1, 2, 3]}
        
        result = await fetch_data()
        assert result['status'] == 'success'
        assert len(result['data']) == 3


class TestDataStructures:
    """Test various data structures."""
    
    def test_portfolio_dict(self):
        """Test portfolio dictionary operations."""
        portfolio = {
            'AAPL': 100,
            'GOOGL': 50,
            'MSFT': 75
        }
        
        # Test total shares
        total_shares = sum(portfolio.values())
        assert total_shares == 225
        
        # Test portfolio update
        portfolio['TSLA'] = 25
        assert len(portfolio) == 4
        assert portfolio['TSLA'] == 25
    
    def test_price_history_list(self):
        """Test price history operations."""
        prices = [100, 102, 98, 105, 103]
        
        # Calculate returns
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                  for i in range(1, len(prices))]
        
        assert len(returns) == 4
        assert returns[0] == 0.02  # 2% return
        assert returns[1] < 0  # Negative return


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio for testing."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


class TestFinancialMetrics:
    """Test financial calculations."""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.05, 0.10, -0.02, 0.08, 0.03])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive Sharpe ratio
    
    def test_portfolio_weights(self):
        """Test portfolio weight normalization."""
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        assert np.isclose(np.sum(weights), 1.0)
        
        # Test weight adjustment
        raw_weights = np.array([30, 30, 20, 20])
        normalized = raw_weights / np.sum(raw_weights)
        assert np.isclose(np.sum(normalized), 1.0)


# Performance benchmark example
@pytest.mark.benchmark
class TestPerformance:
    """Test performance benchmarks."""
    
    def test_numpy_vs_list_performance(self):
        """Compare numpy vs list performance."""
        size = 10000
        
        # List operations
        list_data = list(range(size))
        list_sum = sum(list_data)
        
        # Numpy operations
        np_data = np.arange(size)
        np_sum = np.sum(np_data)
        
        assert list_sum == np_sum
        # In real benchmarks, numpy would be measured to be faster


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
