"""Comprehensive market data tests covering all edge cases."""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.stock import Stock, PriceHistory
from app.models.user import User
from fastapi import status
import asyncio
import json


class TestMarketDataEndpoints:
    """Test market data endpoints with comprehensive edge cases."""

    @pytest.mark.asyncio
    async def test_get_stock_quote_valid_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test getting stock quote with valid symbol."""
        response = await client.get("/api/v1/market/quote/AAPL", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "symbol" in data
        assert "price" in data
        assert "volume" in data

    @pytest.mark.asyncio
    async def test_get_stock_quote_invalid_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test getting stock quote with invalid symbol."""
        response = await client.get("/api/v1/market/quote/INVALID123", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_stock_quote_empty_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test getting stock quote with empty symbol."""
        response = await client.get("/api/v1/market/quote/", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_stock_quote_special_characters(self, client: AsyncClient, auth_headers: dict):
        """Test getting stock quote with special characters in symbol."""
        response = await client.get("/api/v1/market/quote/@#$%", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_multiple_quotes_batch(self, client: AsyncClient, auth_headers: dict):
        """Test getting multiple stock quotes in batch."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        response = await client.post(
            "/api/v1/market/quotes/batch",
            json={"symbols": symbols},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == len(symbols)

    @pytest.mark.asyncio
    async def test_get_quotes_batch_limit_exceeded(self, client: AsyncClient, auth_headers: dict):
        """Test batch quotes with too many symbols."""
        symbols = [f"SYM{i}" for i in range(101)]  # Assuming 100 is the limit
        response = await client.post(
            "/api/v1/market/quotes/batch",
            json={"symbols": symbols},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_historical_data_valid_range(self, client: AsyncClient, auth_headers: dict):
        """Test getting historical data with valid date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        response = await client.get(
            f"/api/v1/market/history/AAPL",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": "1d"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_get_historical_data_invalid_date_range(self, client: AsyncClient, auth_headers: dict):
        """Test historical data with end date before start date."""
        start_date = datetime.now()
        end_date = start_date - timedelta(days=30)
        
        response = await client.get(
            f"/api/v1/market/history/AAPL",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": "1d"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_historical_data_future_dates(self, client: AsyncClient, auth_headers: dict):
        """Test historical data with future dates."""
        start_date = datetime.now() + timedelta(days=1)
        end_date = start_date + timedelta(days=30)
        
        response = await client.get(
            f"/api/v1/market/history/AAPL",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": "1d"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_historical_data_large_range(self, client: AsyncClient, auth_headers: dict):
        """Test historical data with very large date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)  # 10 years
        
        response = await client.get(
            f"/api/v1/market/history/AAPL",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "interval": "1d"
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_get_options_chain_valid_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test getting options chain for valid symbol."""
        response = await client.get("/api/v1/market/options/AAPL", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "calls" in data
        assert "puts" in data
        assert "expirations" in data

    @pytest.mark.asyncio
    async def test_get_options_chain_specific_expiry(self, client: AsyncClient, auth_headers: dict):
        """Test getting options chain for specific expiry date."""
        expiry_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        response = await client.get(
            f"/api/v1/market/options/AAPL",
            params={"expiry": expiry_date},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_get_options_chain_invalid_expiry(self, client: AsyncClient, auth_headers: dict):
        """Test options chain with invalid expiry date format."""
        response = await client.get(
            f"/api/v1/market/options/AAPL",
            params={"expiry": "invalid-date"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, client: AsyncClient, auth_headers: dict):
        """Test technical indicators calculation."""
        response = await client.post(
            "/api/v1/market/indicators",
            json={
                "symbol": "AAPL",
                "indicators": ["SMA", "RSI", "MACD", "BB"],
                "period": 14
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "SMA" in data
        assert "RSI" in data
        assert "MACD" in data

    @pytest.mark.asyncio
    async def test_technical_indicators_invalid_period(self, client: AsyncClient, auth_headers: dict):
        """Test technical indicators with invalid period."""
        response = await client.post(
            "/api/v1/market/indicators",
            json={
                "symbol": "AAPL",
                "indicators": ["SMA"],
                "period": -1
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_market_news_sentiment(self, client: AsyncClient, auth_headers: dict):
        """Test market news sentiment analysis."""
        response = await client.get(
            "/api/v1/market/news/sentiment",
            params={"symbol": "AAPL", "days": 7},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "overall_sentiment" in data
        assert "news_count" in data
        assert "sentiment_scores" in data

    @pytest.mark.asyncio
    async def test_market_news_empty_symbol(self, client: AsyncClient, auth_headers: dict):
        """Test news sentiment with empty symbol."""
        response = await client.get(
            "/api/v1/market/news/sentiment",
            params={"symbol": "", "days": 7},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_intraday_data_valid_interval(self, client: AsyncClient, auth_headers: dict):
        """Test intraday data with valid intervals."""
        intervals = ["1m", "5m", "15m", "30m", "1h"]
        for interval in intervals:
            response = await client.get(
                f"/api/v1/market/intraday/AAPL",
                params={"interval": interval},
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_intraday_data_invalid_interval(self, client: AsyncClient, auth_headers: dict):
        """Test intraday data with invalid interval."""
        response = await client.get(
            f"/api/v1/market/intraday/AAPL",
            params={"interval": "2m"},  # Assuming only specific intervals are allowed
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_market_hours_check(self, client: AsyncClient, auth_headers: dict):
        """Test market hours status check."""
        response = await client.get("/api/v1/market/hours", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "is_open" in data
        assert "next_open" in data
        assert "next_close" in data

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client: AsyncClient, auth_headers: dict):
        """Test rate limiting on market data endpoints."""
        # Make many requests quickly
        tasks = []
        for _ in range(150):  # Assuming rate limit is 100/minute
            tasks.append(client.get("/api/v1/market/quote/AAPL", headers=auth_headers))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that some requests were rate limited
        rate_limited = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)
        assert rate_limited > 0

    @pytest.mark.asyncio
    async def test_cache_headers(self, client: AsyncClient, auth_headers: dict):
        """Test caching headers on market data."""
        response = await client.get("/api/v1/market/quote/AAPL", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        assert "Cache-Control" in response.headers
        assert "ETag" in response.headers

    @pytest.mark.asyncio
    async def test_conditional_request_etag(self, client: AsyncClient, auth_headers: dict):
        """Test conditional requests with ETag."""
        # First request
        response1 = await client.get("/api/v1/market/quote/AAPL", headers=auth_headers)
        etag = response1.headers.get("ETag")
        
        # Second request with If-None-Match
        headers = {**auth_headers, "If-None-Match": etag}
        response2 = await client.get("/api/v1/market/quote/AAPL", headers=headers)
        assert response2.status_code == status.HTTP_304_NOT_MODIFIED

    @pytest.mark.asyncio
    async def test_market_movers(self, client: AsyncClient, auth_headers: dict):
        """Test getting market movers (top gainers/losers)."""
        response = await client.get("/api/v1/market/movers", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "gainers" in data
        assert "losers" in data
        assert "most_active" in data

    @pytest.mark.asyncio
    async def test_sector_performance(self, client: AsyncClient, auth_headers: dict):
        """Test sector performance data."""
        response = await client.get("/api/v1/market/sectors", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert "Technology" in data
        assert "Healthcare" in data

    @pytest.mark.asyncio
    async def test_market_indices(self, client: AsyncClient, auth_headers: dict):
        """Test major market indices data."""
        response = await client.get("/api/v1/market/indices", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert any(idx["symbol"] == "SPY" for idx in data)
        assert any(idx["symbol"] == "DIA" for idx in data)

    @pytest.mark.asyncio
    async def test_earnings_calendar(self, client: AsyncClient, auth_headers: dict):
        """Test earnings calendar data."""
        response = await client.get(
            "/api/v1/market/earnings",
            params={"days": 7},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_dividends_calendar(self, client: AsyncClient, auth_headers: dict):
        """Test dividends calendar data."""
        response = await client.get(
            "/api/v1/market/dividends",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_economic_indicators(self, client: AsyncClient, auth_headers: dict):
        """Test economic indicators data."""
        response = await client.get("/api/v1/market/economic", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "gdp" in data
        assert "inflation" in data
        assert "unemployment" in data

    @pytest.mark.asyncio
    async def test_options_greeks_calculation(self, client: AsyncClient, auth_headers: dict):
        """Test options Greeks calculation."""
        response = await client.post(
            "/api/v1/market/options/greeks",
            json={
                "symbol": "AAPL",
                "strike": 150,
                "expiry": (datetime.now() + timedelta(days=30)).isoformat(),
                "option_type": "call",
                "risk_free_rate": 0.05
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "delta" in data
        assert "gamma" in data
        assert "theta" in data
        assert "vega" in data
        assert "rho" in data

    @pytest.mark.asyncio
    async def test_implied_volatility_surface(self, client: AsyncClient, auth_headers: dict):
        """Test implied volatility surface data."""
        response = await client.get(
            "/api/v1/market/options/iv-surface",
            params={"symbol": "AAPL"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "surface" in data
        assert "strikes" in data
        assert "expirations" in data

    @pytest.mark.asyncio
    async def test_market_breadth_indicators(self, client: AsyncClient, auth_headers: dict):
        """Test market breadth indicators."""
        response = await client.get("/api/v1/market/breadth", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "advance_decline_ratio" in data
        assert "new_highs_lows" in data
        assert "mcclellan_oscillator" in data

    @pytest.mark.asyncio
    async def test_correlation_matrix(self, client: AsyncClient, auth_headers: dict):
        """Test correlation matrix for multiple symbols."""
        response = await client.post(
            "/api/v1/market/correlation",
            json={
                "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "period": 30
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "matrix" in data
        assert len(data["matrix"]) == 4

    @pytest.mark.asyncio
    async def test_volatility_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test volatility analysis for a symbol."""
        response = await client.get(
            "/api/v1/market/volatility/AAPL",
            params={"period": 30},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "historical_volatility" in data
        assert "implied_volatility" in data
        assert "volatility_percentile" in data

    @pytest.mark.asyncio
    async def test_market_depth(self, client: AsyncClient, auth_headers: dict):
        """Test market depth (order book) data."""
        response = await client.get(
            "/api/v1/market/depth/AAPL",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bids" in data
        assert "asks" in data
        assert "spread" in data

    @pytest.mark.asyncio
    async def test_tick_data_stream(self, client: AsyncClient, auth_headers: dict):
        """Test tick data streaming setup."""
        response = await client.post(
            "/api/v1/market/stream/subscribe",
            json={"symbols": ["AAPL", "GOOGL"], "data_types": ["trades", "quotes"]},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "subscription_id" in data

    @pytest.mark.asyncio
    async def test_alternative_data_satellite(self, client: AsyncClient, auth_headers: dict):
        """Test satellite data for retail traffic analysis."""
        response = await client.get(
            "/api/v1/market/alt-data/satellite",
            params={"company": "WMT", "metric": "parking_lot_traffic"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_social_sentiment_analysis(self, client: AsyncClient, auth_headers: dict):
        """Test social media sentiment analysis."""
        response = await client.get(
            "/api/v1/market/alt-data/social",
            params={"symbol": "TSLA", "platform": "twitter"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sentiment_score" in data
        assert "mention_count" in data
        assert "influencer_sentiment" in data

    @pytest.mark.asyncio
    async def test_web_scraping_metrics(self, client: AsyncClient, auth_headers: dict):
        """Test web scraping for alternative metrics."""
        response = await client.get(
            "/api/v1/market/alt-data/web",
            params={"company": "AMZN", "metric": "job_postings"},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_crypto_market_data(self, client: AsyncClient, auth_headers: dict):
        """Test cryptocurrency market data."""
        response = await client.get(
            "/api/v1/market/crypto/BTC-USD",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "price" in data
        assert "volume_24h" in data
        assert "market_cap" in data

    @pytest.mark.asyncio
    async def test_forex_rates(self, client: AsyncClient, auth_headers: dict):
        """Test foreign exchange rates."""
        response = await client.get(
            "/api/v1/market/forex/EUR-USD",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "rate" in data
        assert "bid" in data
        assert "ask" in data

    @pytest.mark.asyncio
    async def test_commodities_data(self, client: AsyncClient, auth_headers: dict):
        """Test commodities market data."""
        response = await client.get(
            "/api/v1/market/commodities/GOLD",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_market_scanner(self, client: AsyncClient, auth_headers: dict):
        """Test market scanner with custom filters."""
        response = await client.post(
            "/api/v1/market/scanner",
            json={
                "filters": {
                    "market_cap": {"min": 1000000000, "max": 10000000000},
                    "pe_ratio": {"max": 25},
                    "volume": {"min": 1000000},
                    "price_change_pct": {"min": 2}
                },
                "sort_by": "volume",
                "limit": 50
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 50

    @pytest.mark.asyncio
    async def test_etf_holdings(self, client: AsyncClient, auth_headers: dict):
        """Test ETF holdings data."""
        response = await client.get(
            "/api/v1/market/etf/SPY/holdings",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "holdings" in data
        assert "total_assets" in data

    @pytest.mark.asyncio
    async def test_insider_trading(self, client: AsyncClient, auth_headers: dict):
        """Test insider trading data."""
        response = await client.get(
            "/api/v1/market/insider/AAPL",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_short_interest(self, client: AsyncClient, auth_headers: dict):
        """Test short interest data."""
        response = await client.get(
            "/api/v1/market/short-interest/GME",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "short_ratio" in data
        assert "short_percent_float" in data

    @pytest.mark.asyncio
    async def test_options_flow(self, client: AsyncClient, auth_headers: dict):
        """Test unusual options activity."""
        response = await client.get(
            "/api/v1/market/options/flow",
            params={"min_premium": 100000},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_market_data_export(self, client: AsyncClient, auth_headers: dict):
        """Test exporting market data to various formats."""
        formats = ["csv", "json", "excel"]
        for fmt in formats:
            response = await client.post(
                "/api/v1/market/export",
                json={
                    "symbols": ["AAPL", "GOOGL"],
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "format": fmt
                },
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_data_quality_check(self, client: AsyncClient, auth_headers: dict):
        """Test data quality validation."""
        response = await client.get(
            "/api/v1/market/data-quality/AAPL",
            params={"date": datetime.now().isoformat()},
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "completeness" in data
        assert "accuracy_score" in data
        assert "anomalies" in data
