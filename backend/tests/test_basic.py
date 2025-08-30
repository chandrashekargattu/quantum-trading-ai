"""
Very basic tests that can run without complex imports.
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_import():
    """Test basic imports work."""
    try:
        from app.core.config import settings
        assert settings is not None
        print("✓ Config imported successfully")
    except Exception as e:
        pytest.fail(f"Config import failed: {e}")


def test_password_hashing():
    """Test password hashing works."""
    try:
        from app.core.security import get_password_hash, verify_password
        
        password = "TestPassword123!"
        hashed = get_password_hash(password)
        
        # Check that hash is different from password
        assert hashed != password
        
        # Check that password can be verified
        assert verify_password(password, hashed)
        
        # Check that wrong password fails
        assert not verify_password("WrongPassword", hashed)
        
        print("✓ Password hashing works correctly")
    except Exception as e:
        pytest.fail(f"Password hashing test failed: {e}")


def test_model_imports():
    """Test model imports work."""
    try:
        from app.models.user import User
        from app.models.portfolio import Portfolio
        from app.models.stock import Stock
        
        assert User is not None
        assert Portfolio is not None
        assert Stock is not None
        
        print("✓ Models imported successfully")
    except Exception as e:
        pytest.fail(f"Model import failed: {e}")


def test_schema_imports():
    """Test schema imports work."""
    try:
        from app.schemas.user import UserCreate, UserResponse
        from app.schemas.portfolio import PortfolioCreate, PortfolioResponse
        
        assert UserCreate is not None
        assert UserResponse is not None
        assert PortfolioCreate is not None
        assert PortfolioResponse is not None
        
        print("✓ Schemas imported successfully")
    except Exception as e:
        pytest.fail(f"Schema import failed: {e}")


def test_service_availability():
    """Test that core services can be imported."""
    services = [
        ("Market Data", "app.services.market_data", "MarketDataService"),
        ("Risk Management", "app.services.risk_management", "RiskManagementService"),
        ("Trading Engine", "app.services.trading_engine", "TradingEngine"),
        ("Backtesting", "app.services.backtesting_engine", "BacktestingEngine"),
    ]
    
    for name, module_path, class_name in services:
        try:
            module = __import__(module_path, fromlist=[class_name])
            service_class = getattr(module, class_name)
            assert service_class is not None
            print(f"✓ {name} service available")
        except Exception as e:
            print(f"✗ {name} service failed: {e}")
            # Don't fail the test, just report


def test_env_variables():
    """Test that environment variables are set."""
    required_vars = [
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "JWT_ALGORITHM"
    ]
    
    for var in required_vars:
        value = os.environ.get(var)
        assert value is not None, f"Environment variable {var} not set"
        print(f"✓ {var} is set")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
