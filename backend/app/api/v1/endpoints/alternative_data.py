"""Alternative data analysis endpoints."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta

from app.db.database import get_db
from app.core.security import get_current_active_user
from app.models import User
from app.alternative_data.alternative_data_processor import AlternativeDataAggregator

router = APIRouter()

# Initialize aggregator (in production, use dependency injection)
aggregator = None


def get_aggregator():
    """Get or create alternative data aggregator."""
    global aggregator
    if aggregator is None:
        # In production, get API keys from secure storage
        api_keys = {
            'twitter_bearer_token': 'your_token',
            'reddit_client_id': 'your_id',
            'reddit_client_secret': 'your_secret'
        }
        aggregator = AlternativeDataAggregator(api_keys)
    return aggregator


@router.post("/analyze-satellite-image")
async def analyze_satellite_image(
    file: UploadFile = File(...),
    location_lat: float = 0.0,
    location_lon: float = 0.0,
    image_type: str = "parking_lot",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Analyze satellite imagery for trading signals.
    
    - **file**: Satellite image file
    - **location_lat**: Latitude of the location
    - **location_lon**: Longitude of the location
    - **image_type**: Type of analysis (parking_lot, port_activity, etc.)
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        agg = get_aggregator()
        
        # Analyze image
        signal = await agg.satellite_analyzer.analyze_satellite_image(
            temp_path,
            {'lat': location_lat, 'lon': location_lon},
            image_type
        )
        
        return {
            "signal_type": signal.signal_type,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "affected_symbols": signal.affected_symbols,
            "metadata": signal.metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze satellite image: {str(e)}"
        )


@router.get("/social-sentiment/{symbol}")
async def get_social_sentiment(
    symbol: str,
    lookback_hours: int = 24,
    include_reddit: bool = True,
    include_twitter: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get social media sentiment analysis for a symbol.
    
    - **symbol**: Stock symbol to analyze
    - **lookback_hours**: Hours of historical data to analyze
    - **include_reddit**: Include Reddit sentiment
    - **include_twitter**: Include Twitter sentiment
    """
    try:
        agg = get_aggregator()
        results = {}
        
        if include_twitter:
            twitter_signals = await agg.social_analyzer.analyze_twitter_sentiment(
                [symbol], lookback_hours
            )
            if symbol in twitter_signals:
                signal = twitter_signals[symbol]
                results["twitter"] = {
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "metadata": signal.metadata
                }
        
        if include_reddit:
            reddit_signals = await agg.social_analyzer.analyze_reddit_sentiment(
                ['wallstreetbets', 'stocks', 'investing']
            )
            if symbol in reddit_signals:
                signal = reddit_signals[symbol]
                results["reddit"] = {
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "metadata": signal.metadata
                }
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze social sentiment: {str(e)}"
        )


@router.get("/news-analysis/{symbol}")
async def analyze_news(
    symbol: str,
    lookback_hours: int = 24,
    include_sec_filings: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Analyze news and SEC filings for a symbol.
    
    - **symbol**: Stock symbol to analyze
    - **lookback_hours**: Hours of historical news to analyze
    - **include_sec_filings**: Include SEC filing analysis
    """
    try:
        agg = get_aggregator()
        
        # Get news signals
        news_signals = await agg.news_scraper.scrape_financial_news(
            [symbol], lookback_hours
        )
        
        results = {
            "news": []
        }
        
        if symbol in news_signals:
            for signal in news_signals[symbol]:
                results["news"].append({
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "headline": signal.metadata.get("headline", ""),
                    "source": signal.metadata.get("source", ""),
                    "timestamp": signal.timestamp.isoformat()
                })
        
        # Get SEC filings if requested
        if include_sec_filings:
            sec_signals = await agg.news_scraper.analyze_sec_filings(
                symbol, ['8-K', '10-Q', '10-K']
            )
            results["sec_filings"] = []
            
            for signal in sec_signals:
                results["sec_filings"].append({
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "filing_type": signal.metadata.get("filing_type", ""),
                    "filing_url": signal.metadata.get("filing_url", "")
                })
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze news: {str(e)}"
        )


@router.get("/composite-signals")
async def get_composite_signals(
    symbols: List[str],
    lookback_hours: int = 24,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get composite alternative data signals for multiple symbols.
    
    - **symbols**: List of stock symbols to analyze
    - **lookback_hours**: Hours of historical data to analyze
    """
    try:
        agg = get_aggregator()
        
        # Get composite signals
        composite_signals = await agg.get_composite_signals(
            symbols, lookback_hours
        )
        
        results = {}
        for symbol, signal in composite_signals.items():
            results[symbol] = {
                "signal_type": signal["signal_type"],
                "strength": signal["strength"],
                "confidence": signal["confidence"],
                "sources": signal["sources"],
                "signal_count": signal["signal_count"],
                "latest_timestamp": signal["latest_timestamp"].isoformat()
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get composite signals: {str(e)}"
        )


@router.get("/supply-chain-disruptions")
async def get_supply_chain_disruptions(
    companies: List[str],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Monitor supply chain disruptions for companies.
    
    - **companies**: List of company names to monitor
    """
    try:
        agg = get_aggregator()
        
        # Get supply chain data
        disruptions = await agg.news_scraper.scrape_supply_chain_data(companies)
        
        results = {}
        for company, signal in disruptions.items():
            results[company] = {
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "disruption_type": signal.metadata.get("disruption_type", ""),
                "severity": signal.metadata.get("severity", ""),
                "affected_regions": signal.metadata.get("affected_regions", [])
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supply chain data: {str(e)}"
        )
