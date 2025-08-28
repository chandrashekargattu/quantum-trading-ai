#!/usr/bin/env python3
"""
Simple test runner to demonstrate the test framework works.
Run this from the project root.
"""

import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run tests
from tests.test_simple_demo import *

def run_simple_tests():
    """Run the simple demo tests manually."""
    print("🧪 Running Simple Demo Tests")
    print("=" * 50)
    
    # Test classes
    test_classes = [
        TestSimpleMathOperations,
        TestMockingDemo,
        TestExceptionHandling,
        TestDataStructures,
        TestFinancialMetrics
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📦 {test_class.__name__}")
        print("-" * 40)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                instance = test_class()
                method = getattr(instance, method_name)
                
                # Handle async tests
                if hasattr(method, '__wrapped__'):
                    # Skip async tests in simple runner
                    print(f"  ⏭️  {method_name} (async - skipped)")
                    continue
                    
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 50)
    
    # Demo parametrized test manually
    print("\n🔬 Parametrized Test Demo")
    print("-" * 40)
    test_obj = TestSimpleMathOperations()
    test_cases = [(0, 0), (1, 1), (100, 10000), (-5, 25)]
    
    for input_val, expected in test_cases:
        try:
            test_obj.test_parametrized_square(input_val, expected)
            print(f"  ✅ Square of {input_val} = {expected}")
        except Exception as e:
            print(f"  ❌ Square of {input_val} failed: {e}")
    
    # Demo exception handling
    print("\n⚠️  Exception Handling Demo")
    print("-" * 40)
    exception_test = TestExceptionHandling()
    
    try:
        exception_test.test_division_by_zero()
        print("  ❌ Division by zero test failed - no exception raised")
    except:
        print("  ✅ Division by zero correctly raises exception")
    
    try:
        exception_test.test_custom_exception()
        print("  ❌ Custom exception test failed - no exception raised")
    except:
        print("  ✅ Custom exception correctly raised")
    
    # Demo financial calculations
    print("\n💰 Financial Calculations Demo")
    print("-" * 40)
    
    returns = np.array([0.05, 0.10, -0.02, 0.08, 0.03])
    sharpe = calculate_sharpe_ratio(returns)
    print(f"  📈 Sharpe Ratio: {sharpe:.4f}")
    
    portfolio = {'AAPL': 100, 'GOOGL': 50, 'MSFT': 75}
    total_shares = sum(portfolio.values())
    print(f"  📊 Total Portfolio Shares: {total_shares}")
    
    prices = [100, 102, 98, 105, 103]
    price_changes = [f"{((prices[i]/prices[i-1])-1)*100:.1f}%" for i in range(1, len(prices))]
    print(f"  📉 Price Changes: {' → '.join(price_changes)}")
    
    print("\n✨ Demo test run complete!")


if __name__ == "__main__":
    run_simple_tests()
