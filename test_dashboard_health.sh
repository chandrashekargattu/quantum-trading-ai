#!/bin/bash

# Dashboard Health Check Script

echo "🏥 Dashboard Health Check"
echo "========================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check frontend is running
echo -n "1. Frontend Server: "
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    exit 1
fi

# Check backend is running
echo -n "2. Backend Server: "
if curl -s http://localhost:8000/docs > /dev/null; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
    exit 1
fi

# Check dashboard loads
echo -n "3. Dashboard Page: "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/dashboard)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ Loading (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${RED}✗ Error (HTTP $HTTP_CODE)${NC}"
fi

# Check API authentication
echo -n "4. API Authentication: "
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=testuser&password=TestPass123" | \
    grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ ! -z "$TOKEN" ]; then
    echo -e "${GREEN}✓ Working${NC}"
else
    echo -e "${YELLOW}⚠ No test user found${NC}"
fi

# Check market indicators API
echo -n "5. Market Indicators API: "
if [ ! -z "$TOKEN" ]; then
    INDICATORS=$(curl -s -H "Authorization: Bearer $TOKEN" \
        http://localhost:8000/api/v1/market-data/indicators | head -c 50)
    if [[ "$INDICATORS" == *"symbol"* ]]; then
        echo -e "${GREEN}✓ Working${NC}"
    else
        echo -e "${RED}✗ Not working${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipped (no auth)${NC}"
fi

# Check portfolios API
echo -n "6. Portfolios API: "
if [ ! -z "$TOKEN" ]; then
    PORTFOLIOS=$(curl -s -H "Authorization: Bearer $TOKEN" \
        http://localhost:8000/api/v1/portfolios/)
    if [[ "$PORTFOLIOS" == "[]" ]] || [[ "$PORTFOLIOS" == *"id"* ]]; then
        echo -e "${GREEN}✓ Working${NC}"
    else
        echo -e "${RED}✗ Not working${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipped (no auth)${NC}"
fi

# Check for frontend errors
echo -n "7. Frontend Compilation: "
ERRORS=$(tail -50 frontend.log | grep -i "error" | grep -v "Compiled" | wc -l)
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓ No errors${NC}"
else
    echo -e "${RED}✗ Found $ERRORS errors${NC}"
fi

# Performance check
echo -n "8. Dashboard Performance: "
LOAD_TIME=$(curl -s -o /dev/null -w "%{time_total}" http://localhost:3000/dashboard)
if (( $(echo "$LOAD_TIME < 0.5" | bc -l) )); then
    echo -e "${GREEN}✓ Fast (${LOAD_TIME}s)${NC}"
else
    echo -e "${YELLOW}⚠ Slow (${LOAD_TIME}s)${NC}"
fi

echo ""
echo "✅ Dashboard health check complete!"
