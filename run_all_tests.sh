#!/bin/bash

echo "🧪 Running All Tests for Quantum Trading AI"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run tests and check result
run_tests() {
    local test_name=$1
    local test_command=$2
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ $test_name passed!${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ $test_name failed!${NC}"
        echo ""
        return 1
    fi
}

# Track overall success
all_passed=true

# Frontend Tests
echo "📱 FRONTEND TESTS"
echo "=================="
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Run frontend tests
if ! run_tests "Frontend Unit Tests" "npm test -- --passWithNoTests"; then
    all_passed=false
fi

cd ..

# Backend Tests
echo "🔧 BACKEND TESTS"
echo "================"
cd backend

# Create test requirements file if it doesn't exist
if [ ! -f "requirements-test.txt" ]; then
    echo "pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
httpx>=0.24.0
aiosqlite>=0.19.0" > requirements-test.txt
fi

# Install test dependencies
echo "Installing backend test dependencies..."
pip install -r requirements-test.txt

# Run backend tests
if ! run_tests "Authentication Tests" "python -m pytest tests/test_auth.py -v"; then
    all_passed=false
fi

if ! run_tests "Market Data Tests" "python -m pytest tests/test_market_data.py -v --ignore=tests/test_auth.py"; then
    all_passed=false
fi

if ! run_tests "Portfolio Tests" "python -m pytest tests/test_portfolios.py -v --ignore=tests/test_auth.py --ignore=tests/test_market_data.py"; then
    all_passed=false
fi

if ! run_tests "Backtesting Tests" "python -m pytest tests/test_backtest.py -v --ignore=tests/test_auth.py --ignore=tests/test_market_data.py --ignore=tests/test_portfolios.py"; then
    all_passed=false
fi

# Generate coverage report
echo -e "${YELLOW}Generating coverage report...${NC}"
python -m pytest --cov=app --cov-report=html --cov-report=term

cd ..

# Summary
echo ""
echo "📊 TEST SUMMARY"
echo "==============="

if $all_passed; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    echo ""
    echo "📈 Next Steps:"
    echo "  - Check coverage report: backend/htmlcov/index.html"
    echo "  - Run specific test suites with:"
    echo "    - Frontend: cd frontend && npm test"
    echo "    - Backend: cd backend && python -m pytest"
else
    echo -e "${RED}❌ Some tests failed. Please check the output above.${NC}"
    exit 1
fi
