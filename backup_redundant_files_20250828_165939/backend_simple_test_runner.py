#!/usr/bin/env python3
"""
Simple test runner to demonstrate core functionality without complex dependencies.
"""

import os
import sys
import asyncio
from datetime import datetime

# Set up environment
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_ALGORITHM'] = 'HS256'

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test results
passed = 0
failed = 0

def test_result(name, passed_test):
    global passed, failed
    if passed_test:
        print(f"✅ {name}")
        passed += 1
    else:
        print(f"❌ {name}")
        failed += 1

print("🧪 Running Quantum Trading AI Test Suite")
print("=" * 50)
print()

# Test 1: Password Hashing
print("📋 Testing Authentication...")
try:
    from app.core.security import get_password_hash, verify_password
    
    password = "TestPassword123!"
    hashed = get_password_hash(password)
    
    test_result("Password hashing", hashed != password)
    test_result("Password verification", verify_password(password, hashed))
    test_result("Wrong password rejection", not verify_password("wrong", hashed))
except Exception as e:
    print(f"❌ Auth tests failed: {e}")
    failed += 3

# Test 2: JWT Token Creation
print("\n📋 Testing JWT Tokens...")
try:
    from app.core.security import create_access_token, decode_access_token
    
    user_id = "test-user-123"
    token = create_access_token(user_id)
    
    test_result("Token creation", token is not None and len(token) > 0)
    
    payload = decode_access_token(token)
    test_result("Token decoding", payload is not None)
    test_result("Token payload correct", payload.get("sub") == user_id)
except Exception as e:
    print(f"❌ JWT tests failed: {e}")
    failed += 3

# Test 3: Model Imports
print("\n📋 Testing Models...")
try:
    from app.models.user import User
    from app.models.portfolio import Portfolio
    from app.models.stock import Stock
    from app.models.trade import Trade, TradeType
    
    test_result("User model", User is not None)
    test_result("Portfolio model", Portfolio is not None)
    test_result("Stock model", Stock is not None)
    test_result("Trade model", Trade is not None)
    test_result("TradeType enum", TradeType.BUY == "buy")
except Exception as e:
    print(f"❌ Model tests failed: {e}")
    failed += 5

# Test 4: Schema Validation
print("\n📋 Testing Schemas...")
try:
    from app.schemas.user import UserCreate
    from app.schemas.stock import StockCreate
    
    # Test user schema
    user_data = UserCreate(
        email="test@example.com",
        username="testuser",
        password="TestPassword123!"
    )
    test_result("UserCreate schema", user_data.email == "test@example.com")
    
    # Test stock schema
    stock_data = StockCreate(
        symbol="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ"
    )
    test_result("StockCreate schema", stock_data.symbol == "AAPL")
except Exception as e:
    print(f"❌ Schema tests failed: {e}")
    failed += 2

# Test 5: Database Connection
print("\n📋 Testing Database...")
async def test_database():
    try:
        from app.db.database import init_db, get_db
        from app.models.user import User
        from sqlalchemy import select
        import uuid
        
        # Initialize database
        await init_db()
        test_result("Database initialization", True)
        
        # Test database session
        async for db in get_db():
            # Create a test user
            test_user = User(
                id=uuid.uuid4(),
                email="dbtest@example.com",
                username="dbtest",
                hashed_password="hashed",
                is_active=True
            )
            db.add(test_user)
            await db.commit()
            
            # Query the user
            result = await db.execute(
                select(User).where(User.email == "dbtest@example.com")
            )
            found_user = result.scalar_one_or_none()
            
            test_result("User creation", found_user is not None)
            test_result("User query", found_user.username == "dbtest")
            
            # Cleanup
            await db.delete(test_user)
            await db.commit()
            break
            
    except Exception as e:
        print(f"❌ Database tests failed: {e}")
        global failed
        failed += 3

# Run async database test
asyncio.run(test_database())

# Test 6: Service Functionality
print("\n📋 Testing Core Services...")
try:
    from app.services.risk_calculator import calculate_position_size
    
    # Test position sizing
    account_value = 100000
    risk_percent = 0.02
    stop_loss_percent = 0.05
    
    position_size = calculate_position_size(account_value, risk_percent, stop_loss_percent)
    expected_size = (account_value * risk_percent) / stop_loss_percent
    
    test_result("Position size calculation", abs(position_size - expected_size) < 0.01)
except Exception as e:
    print(f"❌ Service tests failed: {e}")
    failed += 1

# Test 7: Configuration
print("\n📋 Testing Configuration...")
try:
    from app.core.config import settings
    
    test_result("JWT algorithm", settings.ALGORITHM == "HS256")
    test_result("API version", settings.API_V1_STR == "/api/v1")
    test_result("Project name", settings.PROJECT_NAME is not None)
except Exception as e:
    print(f"❌ Config tests failed: {e}")
    failed += 3

# Test 8: Trading Logic
print("\n📋 Testing Trading Logic...")
try:
    # Test option pricing
    from app.services.options_pricing import OptionsPricingService
    
    pricing = OptionsPricingService()
    
    # Test Black-Scholes pricing
    price = pricing.calculate_black_scholes_price(
        underlying_price=100,
        strike_price=100,
        time_to_expiry=0.25,  # 3 months
        volatility=0.3,       # 30% vol
        risk_free_rate=0.05,  # 5% rate
        option_type="call"
    )
    
    test_result("Option pricing", price > 0 and price < 20)
    
    # Test Greeks calculation
    greeks = pricing.calculate_greeks(
        underlying_price=100,
        strike_price=100,
        time_to_expiry=0.25,
        volatility=0.3,
        option_type="call"
    )
    
    test_result("Delta calculation", 0 < greeks['delta'] < 1)
    test_result("Gamma calculation", greeks['gamma'] > 0)
    test_result("Theta calculation", greeks['theta'] < 0)  # Time decay
    test_result("Vega calculation", greeks['vega'] > 0)
    
except Exception as e:
    print(f"❌ Trading logic tests failed: {e}")
    failed += 5

# Summary
print("\n" + "=" * 50)
print(f"📊 Test Summary:")
print(f"   ✅ Passed: {passed}")
print(f"   ❌ Failed: {failed}")
print(f"   📈 Total: {passed + failed}")
print(f"   🎯 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
print("=" * 50)

if failed == 0:
    print("\n🎉 All tests passed! The core functionality is working correctly.")
else:
    print(f"\n⚠️  {failed} tests failed. See above for details.")

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
