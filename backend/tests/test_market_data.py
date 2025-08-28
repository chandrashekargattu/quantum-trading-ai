"""Tests for market data endpoints and services."""
import pytest
from datetime import datetime, timedelta
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.stock import Stock
from app.models.market_data import MarketData, OHLCV
from app.services.market_data import MarketDataService
from app.schemas.market_data import MarketDataResponse, StockQuote


class TestMarketDataEndpoints:
    """Test cases for market data endpoints."""

    @pytest.mark.asyncio
    async def test_get_stock_quote(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test getting stock quote."""
        response = await client.get(
            f"/api/v1/market/quote/{test_stock.symbol}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["symbol"] == test_stock.symbol
        assert "price" in data
        assert "volume" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_stock_quote_invalid_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test getting quote for invalid symbol."""
        response = await client.get(
            "/api/v1/market/quote/INVALID",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_stock_quote_no_auth(self, client: AsyncClient, test_stock: Stock):
        """Test getting stock quote without authentication."""
        response = await client.get(f"/api/v1/market/quote/{test_stock.symbol}")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_historical_data(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test getting historical market data."""
        params = {
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "interval": "1d"
        }
        
        response = await client.get(
            f"/api/v1/market/historical/{test_stock.symbol}",
            params=params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        if data:  # If data is available
            assert "open" in data[0]
            assert "high" in data[0]
            assert "low" in data[0]
            assert "close" in data[0]
            assert "volume" in data[0]
            assert "timestamp" in data[0]

    @pytest.mark.asyncio
    async def test_get_historical_data_invalid_dates(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test historical data with invalid date range."""
        params = {
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "interval": "1d"
        }
        
        response = await client.get(
            f"/api/v1/market/historical/{test_stock.symbol}",
            params=params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_option_chain(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test getting option chain data."""
        response = await client.get(
            f"/api/v1/market/options/{test_stock.symbol}",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "symbol" in data
        assert "calls" in data
        assert "puts" in data
        assert isinstance(data["calls"], list)
        assert isinstance(data["puts"], list)

    @pytest.mark.asyncio
    async def test_get_option_chain_with_expiration(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test getting option chain for specific expiration."""
        expiration = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        response = await client.get(
            f"/api/v1/market/options/{test_stock.symbol}",
            params={"expiration": expiration},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "expiration" in data
        assert data["expiration"] == expiration

    @pytest.mark.asyncio
    async def test_search_stocks(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test stock search functionality."""
        response = await client.get(
            "/api/v1/market/search",
            params={"query": test_stock.name[:5]},  # Search with partial name
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert any(stock["symbol"] == test_stock.symbol for stock in data)

    @pytest.mark.asyncio
    async def test_search_stocks_empty_query(self, client: AsyncClient, auth_headers: dict):
        """Test search with empty query."""
        response = await client.get(
            "/api/v1/market/search",
            params={"query": ""},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_market_indicators(self, client: AsyncClient, auth_headers: dict):
        """Test getting market indicators."""
        response = await client.get(
            "/api/v1/market/indicators",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "indices" in data
        assert "vix" in data
        assert "sentiment" in data

    @pytest.mark.asyncio
    async def test_get_technical_indicators(self, client: AsyncClient, test_stock: Stock, auth_headers: dict):
        """Test getting technical indicators."""
        params = {
            "indicators": ["SMA", "RSI", "MACD"],
            "period": 20
        }
        
        response = await client.get(
            f"/api/v1/market/technical/{test_stock.symbol}",
            params=params,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "SMA" in data
        assert "RSI" in data
        assert "MACD" in data

    @pytest.mark.asyncio
    async def test_websocket_market_stream(self, client: AsyncClient, test_stock: Stock):
        """Test WebSocket market data streaming."""
        # This is a simplified test - real WebSocket testing would use websocket-client
        with pytest.raises(Exception):  # WebSocket upgrade expected to fail in test client
            await client.websocket_connect(f"/api/v1/market/stream/{test_stock.symbol}")


class TestMarketDataService:
    """Test cases for market data service."""

    @pytest.mark.asyncio
    async def test_fetch_stock_quote(self, db: AsyncSession, test_stock: Stock):
        """Test fetching stock quote from service."""
        service = MarketDataService(db)
        quote = await service.get_stock_quote(test_stock.symbol)
        
        assert quote is not None
        assert quote.symbol == test_stock.symbol
        assert quote.price > 0
        assert quote.volume >= 0

    @pytest.mark.asyncio
    async def test_fetch_historical_data(self, db: AsyncSession, test_stock: Stock):
        """Test fetching historical data from service."""
        service = MarketDataService(db)
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        data = await service.get_historical_data(
            test_stock.symbol,
            start_date,
            end_date,
            "1d"
        )
        
        assert isinstance(data, list)
        # Note: External API might not return data for all dates
        if data:
            assert all(isinstance(item, dict) for item in data)
            assert all("close" in item for item in data)

    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, db: AsyncSession, test_stock: Stock):
        """Test technical indicator calculations."""
        service = MarketDataService(db)
        
        # Create mock price data
        prices = [100 + i * 0.5 for i in range(50)]
        
        # Test SMA
        sma = service.calculate_sma(prices, 20)
        assert len(sma) == len(prices) - 19
        assert all(isinstance(val, float) for val in sma)
        
        # Test RSI
        rsi = service.calculate_rsi(prices, 14)
        assert len(rsi) == len(prices) - 14
        assert all(0 <= val <= 100 for val in rsi)
        
        # Test MACD
        macd_line, signal_line, histogram = service.calculate_macd(prices)
        assert len(macd_line) > 0
        assert len(signal_line) > 0
        assert len(histogram) > 0

    @pytest.mark.asyncio
    async def test_store_market_data(self, db: AsyncSession, test_stock: Stock):
        """Test storing market data in database."""
        service = MarketDataService(db)
        
        market_data = MarketData(
            symbol=test_stock.symbol,
            price=150.25,
            volume=1000000,
            bid=150.20,
            ask=150.30,
            high=151.00,
            low=149.50,
            open=149.75,
            close=150.25,
            timestamp=datetime.utcnow()
        )
        
        stored_data = await service.store_market_data(market_data)
        assert stored_data.id is not None
        assert stored_data.symbol == test_stock.symbol
        assert stored_data.price == 150.25

    @pytest.mark.asyncio
    async def test_get_latest_market_data(self, db: AsyncSession, test_stock: Stock):
        """Test retrieving latest market data."""
        service = MarketDataService(db)
        
        # Store some test data
        for i in range(5):
            data = MarketData(
                symbol=test_stock.symbol,
                price=150 + i,
                volume=1000000,
                timestamp=datetime.utcnow() - timedelta(minutes=5-i)
            )
            await service.store_market_data(data)
        
        # Get latest
        latest = await service.get_latest_market_data(test_stock.symbol)
        assert latest is not None
        assert latest.price == 154  # Should be the last one stored

    @pytest.mark.asyncio
    async def test_data_validation(self, db: AsyncSession):
        """Test market data validation."""
        service = MarketDataService(db)
        
        # Test invalid symbol
        with pytest.raises(ValueError):
            await service.get_stock_quote("")
        
        # Test invalid date range
        with pytest.raises(ValueError):
            await service.get_historical_data(
                "AAPL",
                datetime.now(),
                datetime.now() - timedelta(days=1),
                "1d"
            )
        
        # Test invalid interval
        with pytest.raises(ValueError):
            await service.get_historical_data(
                "AAPL",
                datetime.now() - timedelta(days=1),
                datetime.now(),
                "invalid"
            )