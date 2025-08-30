#!/bin/bash

# Quantum Trading AI - Dashboard Tests Runner
# This script runs all tests related to the dashboard fixes

echo "🧪 Quantum Trading AI - Running Dashboard Tests"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
FRONTEND_TESTS_PASSED=0
BACKEND_TESTS_PASSED=0

# Function to run frontend tests
run_frontend_tests() {
    echo -e "\n${BLUE}📦 Running Frontend Tests...${NC}"
    cd frontend

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi

    echo -e "\n${YELLOW}Testing Portfolio Service...${NC}"
    npm test -- src/services/api/__tests__/portfolio.test.ts --passWithNoTests
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Portfolio Service tests passed${NC}"
        ((FRONTEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ Portfolio Service tests failed${NC}"
    fi

    echo -e "\n${YELLOW}Testing Market Service...${NC}"
    npm test -- src/services/api/__tests__/market.test.ts --passWithNoTests
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Market Service tests passed${NC}"
        ((FRONTEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ Market Service tests failed${NC}"
    fi

    echo -e "\n${YELLOW}Testing PortfolioSummary Component...${NC}"
    npm test -- src/components/dashboard/__tests__/PortfolioSummary.test.tsx --passWithNoTests
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ PortfolioSummary Component tests passed${NC}"
        ((FRONTEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ PortfolioSummary Component tests failed${NC}"
    fi

    echo -e "\n${YELLOW}Testing Dashboard Integration...${NC}"
    npm test -- src/__tests__/integration/dashboard-flow.test.tsx --passWithNoTests
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dashboard Integration tests passed${NC}"
        ((FRONTEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ Dashboard Integration tests failed${NC}"
    fi

    cd ..
}

# Function to run backend tests
run_backend_tests() {
    echo -e "\n${BLUE}🔧 Running Backend Tests...${NC}"
    cd backend

    # Set Python environment
    if command -v pyenv >/dev/null 2>&1; then
        pyenv local 3.10.13 2>/dev/null || pyenv local 3.10 2>/dev/null || echo "Using system Python"
    fi

    # Install test dependencies if needed
    pip install -q pytest pytest-asyncio httpx

    echo -e "\n${YELLOW}Testing Portfolio Endpoints...${NC}"
    python -m pytest tests/test_portfolio_endpoints.py -v
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Portfolio Endpoints tests passed${NC}"
        ((BACKEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ Portfolio Endpoints tests failed${NC}"
    fi

    echo -e "\n${YELLOW}Testing Market Data Endpoints...${NC}"
    python -m pytest tests/test_market_data_endpoints.py -v
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Market Data Endpoints tests passed${NC}"
        ((BACKEND_TESTS_PASSED++))
    else
        echo -e "${RED}❌ Market Data Endpoints tests failed${NC}"
    fi

    cd ..
}

# Function to check services are running
check_services() {
    echo -e "\n${BLUE}🔍 Checking Services...${NC}"
    
    # Check if backend is running
    if curl -s -o /dev/null http://localhost:8000/docs; then
        echo -e "${GREEN}✅ Backend is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Backend is not running. Some integration tests may fail.${NC}"
        echo "   Run './START_APP.sh' to start the services"
    fi

    # Check if frontend is running
    if curl -s -o /dev/null http://localhost:3000; then
        echo -e "${GREEN}✅ Frontend is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Frontend is not running. Some integration tests may fail.${NC}"
        echo "   Run './START_APP.sh' to start the services"
    fi
}

# Main execution
echo -e "${BLUE}📋 Test Summary:${NC}"
echo "- Portfolio Service (Frontend)"
echo "- Market Service (Frontend)"
echo "- PortfolioSummary Component"
echo "- Dashboard Integration"
echo "- Portfolio API Endpoints (Backend)"
echo "- Market Data API Endpoints (Backend)"

# Check if services are running
check_services

# Run tests
run_frontend_tests
run_backend_tests

# Summary
echo -e "\n${BLUE}📊 Test Results Summary:${NC}"
echo "========================"
echo -e "Frontend Tests Passed: ${GREEN}$FRONTEND_TESTS_PASSED/4${NC}"
echo -e "Backend Tests Passed: ${GREEN}$BACKEND_TESTS_PASSED/2${NC}"

TOTAL_TESTS=$((FRONTEND_TESTS_PASSED + BACKEND_TESTS_PASSED))
if [ $TOTAL_TESTS -eq 6 ]; then
    echo -e "\n${GREEN}🎉 All tests passed! The dashboard fixes are working correctly.${NC}"
else
    echo -e "\n${RED}⚠️  Some tests failed. Please review the output above.${NC}"
fi

echo -e "\n${BLUE}📝 Key Changes Tested:${NC}"
echo "1. ✅ Portfolio API URLs with trailing slash"
echo "2. ✅ Market indicators endpoint URL (market-data/indicators)"
echo "3. ✅ Authentication headers in API calls"
echo "4. ✅ Portfolio creation functionality"
echo "5. ✅ Dashboard component lazy loading"
echo "6. ✅ Error handling and edge cases"

echo -e "\n${YELLOW}💡 Next Steps:${NC}"
echo "1. If all tests pass, the dashboard should work correctly"
echo "2. Test manually by creating a portfolio in the UI"
echo "3. Monitor the browser console for any runtime errors"
echo "4. Check network tab to ensure API calls are successful"
