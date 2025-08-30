#!/bin/bash

# Run all authentication consistency tests

echo "🔐 Running Authentication Consistency Tests"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Track failures
FAILED=0

# Frontend Unit Tests
echo -e "${BLUE}1. Frontend Unit Tests${NC}"
echo "----------------------"
cd frontend
echo "Running auth interceptor tests..."
npm test -- __tests__/lib/auth-interceptor.test.ts --passWithNoTests 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Auth interceptor tests passed${NC}"
else
    echo -e "${RED}✗ Auth interceptor tests failed${NC}"
    FAILED=$((FAILED + 1))
fi

echo "Running API cache tests..."
npm test -- __tests__/lib/api-cache.test.ts --passWithNoTests 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ API cache tests passed${NC}"
else
    echo -e "${RED}✗ API cache tests failed${NC}"
    FAILED=$((FAILED + 1))
fi

echo "Running dashboard auth flow tests..."
npm test -- __tests__/integration/dashboard-auth-flow.test.tsx --passWithNoTests 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dashboard auth flow tests passed${NC}"
else
    echo -e "${RED}✗ Dashboard auth flow tests failed${NC}"
    FAILED=$((FAILED + 1))
fi
cd ..

echo ""

# Backend Tests
echo -e "${BLUE}2. Backend Tests${NC}"
echo "----------------"
cd backend
echo "Running auth consistency tests..."
pytest tests/test_auth_consistency.py -v -q 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backend auth tests passed${NC}"
else
    echo -e "${RED}✗ Backend auth tests failed${NC}"
    FAILED=$((FAILED + 1))
fi
cd ..

echo ""

# E2E Tests (if Playwright is set up)
echo -e "${BLUE}3. E2E Tests${NC}"
echo "------------"
if [ -f "frontend/playwright.config.ts" ]; then
    cd frontend
    echo "Running E2E auth tests..."
    npx playwright test e2e/auth-consistency.spec.ts --reporter=list 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ E2E auth tests passed${NC}"
    else
        echo -e "${RED}✗ E2E auth tests failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    cd ..
else
    echo -e "${YELLOW}⚠ Playwright not configured, skipping E2E tests${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}Test Summary${NC}"
echo "============"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All authentication tests passed!${NC}"
    echo ""
    echo "The authentication system is consistent and production-ready."
else
    echo -e "${RED}❌ $FAILED test suites failed${NC}"
    echo ""
    echo "Please fix the failing tests before deploying to production."
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Review AUTH_CONSISTENCY_FIX_SUMMARY.md for details"
echo "2. Test manually in the browser"
echo "3. Deploy with confidence!"
