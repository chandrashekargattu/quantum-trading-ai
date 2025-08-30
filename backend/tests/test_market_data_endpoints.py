import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.market_data import MarketIndicator
from datetime import datetime
import uuid


@pytest.mark.asyncio
class TestMarketDataEndpoints:
    """Test market data API endpoints with correct URL paths"""

    async def test_get_market_indicators(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/indicators"""
        response = await client.get(
            "/api/v1/market-data/indicators",
            headers=test_user_token_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Check if standard indicators are present
        if len(data) > 0:
            indicator = data[0]
            assert "symbol" in indicator
            assert "name" in indicator
            assert "value" in indicator
            assert "change_amount" in indicator
            assert "change_percent" in indicator

    async def test_get_stock_data(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/stocks/{symbol}"""
        symbol = "RELIANCE"
        response = await client.get(
            f"/api/v1/market-data/stocks/{symbol}",
            headers=test_user_token_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["symbol"] == symbol
            assert "name" in data
            assert "price" in data
            assert "change" in data
            assert "change_percent" in data
            assert "volume" in data
        else:
            # API might be rate limited or symbol not found
            assert response.status_code in [404, 429, 503]

    async def test_get_batch_quotes(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test POST /api/v1/market-data/quotes"""
        symbols = ["INFY", "TCS", "WIPRO"]
        response = await client.post(
            "/api/v1/market-data/quotes",
            json={"symbols": symbols},
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            quote = data[0]
            assert "symbol" in quote
            assert "price" in quote
            assert "bid" in quote
            assert "ask" in quote
            assert "volume" in quote

    async def test_search_symbols(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/search"""
        query = "Tata"
        response = await client.get(
            f"/api/v1/market-data/search?q={query}",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            result = data[0]
            assert "symbol" in result
            assert "name" in result
            # Name should contain the search query (case insensitive)
            assert query.lower() in result["name"].lower()

    async def test_get_market_overview(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/overview"""
        response = await client.get(
            "/api/v1/market-data/overview",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "indices" in data
        assert "sectors" in data
        assert "top_gainers" in data
        assert "top_losers" in data
        assert "most_active" in data
        
        # Verify structure of indices
        if len(data["indices"]) > 0:
            index = data["indices"][0]
            assert "symbol" in index
            assert "name" in index
            assert "value" in index
            assert "change" in index
            assert "change_percent" in index

    async def test_get_historical_data(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/historical"""
        params = {
            "symbol": "NIFTY50",
            "interval": "1d",
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-04T00:00:00"
        }
        
        response = await client.get(
            "/api/v1/market-data/historical",
            params=params,
            headers=test_user_token_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "timestamps" in data
            assert "open" in data
            assert "high" in data
            assert "low" in data
            assert "close" in data
            assert "volume" in data
            
            # All arrays should have the same length
            length = len(data["timestamps"])
            assert len(data["open"]) == length
            assert len(data["high"]) == length
            assert len(data["low"]) == length
            assert len(data["close"]) == length
            assert len(data["volume"]) == length
        else:
            # Historical data might not be available
            assert response.status_code in [404, 503]

    async def test_invalid_symbol(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test with invalid symbol"""
        response = await client.get(
            "/api/v1/market-data/stocks/INVALID_SYMBOL_12345",
            headers=test_user_token_headers
        )
        
        assert response.status_code in [404, 400]

    async def test_market_data_without_auth(self, client: AsyncClient):
        """Test that market data requires authentication"""
        response = await client.get("/api/v1/market-data/indicators")
        assert response.status_code == 401

    async def test_option_chain_endpoint(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/options/chain/{symbol}"""
        symbol = "NIFTY"
        response = await client.get(
            f"/api/v1/options/chain/{symbol}",
            headers=test_user_token_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "calls" in data
            assert "puts" in data
            assert "expirations" in data
            assert "strikes" in data
            
            # Check option structure
            if len(data["calls"]) > 0:
                call_option = data["calls"][0]
                assert "symbol" in call_option
                assert "strike" in call_option
                assert "expiration" in call_option
                assert "type" in call_option
                assert call_option["type"] == "CALL"
                assert "bid" in call_option
                assert "ask" in call_option
                assert "implied_volatility" in call_option
        else:
            # Option chain might not be available
            assert response.status_code in [404, 503]

    async def test_market_depth(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/depth/{symbol}"""
        symbol = "RELIANCE"
        response = await client.get(
            f"/api/v1/market-data/depth/{symbol}",
            headers=test_user_token_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "total_bid_volume" in data
            assert "total_ask_volume" in data
            assert "imbalance" in data
            assert "levels" in data
            assert "timestamp" in data
        else:
            # Market depth might not be available
            assert response.status_code in [404, 503]

    async def test_order_book(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/market-data/orderbook/{symbol}"""
        symbol = "TCS"
        response = await client.get(
            f"/api/v1/market-data/orderbook/{symbol}?depth=5",
            headers=test_user_token_headers
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "bids" in data
            assert "asks" in data
            assert "spread" in data
            assert "spread_percent" in data
            assert "timestamp" in data
            
            # Check bid/ask structure
            if len(data["bids"]) > 0:
                bid = data["bids"][0]
                assert "price" in bid
                assert "quantity" in bid
                assert "orders" in bid
        else:
            # Order book might not be available
            assert response.status_code in [404, 503]

    async def test_rate_limiting(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test rate limiting on market data endpoints"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = await client.get(
                "/api/v1/market-data/indicators",
                headers=test_user_token_headers
            )
            responses.append(response.status_code)
        
        # Should not all be rate limited in test environment
        assert not all(status == 429 for status in responses)
        # But should all be either success or rate limited
        assert all(status in [200, 429] for status in responses)


# Test fixtures
@pytest.fixture
async def test_market_indicators(db: AsyncSession) -> list[MarketIndicator]:
    """Create test market indicators"""
    indicators = [
        MarketIndicator(
            id=str(uuid.uuid4()),
            symbol="^NSEI",
            name="NIFTY 50",
            value=21000.0,
            change_amount=150.0,
            change_percent=0.72,
            last_updated=datetime.utcnow(),
            created_at=datetime.utcnow()
        ),
        MarketIndicator(
            id=str(uuid.uuid4()),
            symbol="^BSESN",
            name="SENSEX",
            value=70000.0,
            change_amount=500.0,
            change_percent=0.72,
            last_updated=datetime.utcnow(),
            created_at=datetime.utcnow()
        ),
        MarketIndicator(
            id=str(uuid.uuid4()),
            symbol="^VIX",
            name="India VIX",
            value=12.5,
            change_amount=-0.5,
            change_percent=-3.85,
            last_updated=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
    ]
    
    for indicator in indicators:
        db.add(indicator)
    
    await db.commit()
    return indicators
