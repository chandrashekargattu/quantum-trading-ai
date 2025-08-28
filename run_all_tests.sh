#!/bin/bash

# Run all tests for Quantum Trading AI
# This script runs the complete test suite including unit, integration, E2E, performance, and security tests

set -e  # Exit on any error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Quantum Trading AI - Complete Test Suite              ║"
echo "║                     600+ Test Cases                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run tests and capture results
run_test_suite() {
    local suite_name=$1
    local command=$2
    local directory=$3
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running ${suite_name}...${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    cd "$directory"
    
    if eval "$command"; then
        echo -e "${GREEN}✅ ${suite_name} PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}❌ ${suite_name} FAILED${NC}"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
}

# Start time
START_TIME=$(date +%s)

# 1. Backend Unit Tests
echo -e "\n${YELLOW}🔧 BACKEND TESTS${NC}"
echo "=================="

cd backend

# Authentication tests (18 tests)
run_test_suite "Authentication Tests" "pytest tests/test_auth.py -v --tb=short" "."

# Market Data tests (50+ tests)
run_test_suite "Market Data Tests" "pytest tests/test_market_data_comprehensive.py -v --tb=short" "."

# Portfolio Management tests (50+ tests)
run_test_suite "Portfolio Management Tests" "pytest tests/test_portfolio_comprehensive.py -v --tb=short" "."

# Trading Strategy tests (40+ tests)
run_test_suite "Trading Strategy Tests" "pytest tests/test_trading_comprehensive.py -v --tb=short" "."

# Options Trading tests (40+ tests)
run_test_suite "Options Trading Tests" "pytest tests/test_options_comprehensive.py -v --tb=short" "."

# HFT Engine tests (30+ tests)
run_test_suite "HFT Engine Tests" "pytest tests/test_hft_comprehensive.py -v --tb=short" "."

# Backtesting tests (40+ tests)
run_test_suite "Backtesting Tests" "pytest tests/test_backtesting_comprehensive.py -v --tb=short" "."

# Risk Management tests (30+ tests)
run_test_suite "Risk Management Tests" "pytest tests/test_risk_management_comprehensive.py -v --tb=short" "."

# Quantum Algorithm tests (20+ tests)
run_test_suite "Quantum Algorithm Tests" "pytest tests/test_quantum_comprehensive.py -v --tb=short" "."

# WebSocket tests (20+ tests)
run_test_suite "WebSocket Tests" "pytest tests/test_websocket_comprehensive.py -v --tb=short" "."

# Security tests (30+ tests)
run_test_suite "Backend Security Tests" "pytest tests/test_security.py -v --tb=short" "."

# Performance tests (15+ tests)
run_test_suite "Backend Performance Tests" "pytest tests/test_performance.py -v --tb=short -m performance" "."

# 2. Frontend Unit Tests
echo -e "\n\n${YELLOW}🎨 FRONTEND TESTS${NC}"
echo "==================="

cd ../frontend

# Component tests (80+ tests)
run_test_suite "Component Tests" "npm test -- src/components/__tests__ --coverage --watchAll=false" "."

# Store tests (40+ tests)
run_test_suite "Store Tests" "npm test -- src/store/__tests__ --coverage --watchAll=false" "."

# Auth tests (24+ tests)
run_test_suite "Auth Flow Tests" "npm test -- src/app/auth --coverage --watchAll=false" "."

# Integration tests (60+ tests)
run_test_suite "Frontend Integration Tests" "npm test -- src/__tests__/integration --coverage --watchAll=false" "."

# Security tests (15+ tests)
run_test_suite "Frontend Security Tests" "npm test -- tests/security --coverage --watchAll=false" "."

# Performance tests (10+ tests)
run_test_suite "Frontend Performance Tests" "npm test -- tests/performance --coverage --watchAll=false" "."

# 3. End-to-End Tests
echo -e "\n\n${YELLOW}🌐 END-TO-END TESTS${NC}"
echo "====================="

# Check if Playwright is installed
if ! npx playwright --version > /dev/null 2>&1; then
    echo -e "${YELLOW}Installing Playwright browsers...${NC}"
    npx playwright install
fi

# Run E2E tests (40+ tests)
run_test_suite "E2E Authentication Tests" "npx playwright test e2e/auth.spec.ts" "."
run_test_suite "E2E Trading Tests" "npx playwright test e2e/trading.spec.ts" "."
run_test_suite "E2E Portfolio Tests" "npx playwright test e2e/portfolio.spec.ts" "."
run_test_suite "E2E Alerts Tests" "npx playwright test e2e/alerts.spec.ts" "."
run_test_suite "E2E Backtesting Tests" "npx playwright test e2e/backtesting.spec.ts" "."
run_test_suite "E2E Market Data Tests" "npx playwright test e2e/market-data.spec.ts" "."

# 4. Generate Coverage Report
echo -e "\n\n${YELLOW}📊 COVERAGE REPORT${NC}"
echo "===================="

cd ../backend
echo -e "\n${BLUE}Backend Coverage:${NC}"
pytest --cov=app --cov-report=term-missing --cov-report=html tests/

cd ../frontend
echo -e "\n${BLUE}Frontend Coverage:${NC}"
npm test -- --coverage --watchAll=false --coverageReporters=text

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Summary
echo -e "\n\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}                         TEST SUMMARY                          ${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Total Test Suites Run: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
echo -e "Duration: ${DURATION} seconds"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

# Count approximate total tests
APPROX_TESTS=$((18 + 50 + 50 + 40 + 40 + 30 + 40 + 30 + 20 + 20 + 30 + 15 + 80 + 40 + 24 + 60 + 15 + 10 + 40))
echo -e "\nApproximate Total Individual Tests: ${APPROX_TESTS}+"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL TEST SUITES PASSED! 🎉${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some test suites failed. Please check the logs above.${NC}"
    exit 1
fi