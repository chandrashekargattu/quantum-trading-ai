"""Alternative data processing for advanced trading signals."""

from .alternative_data_processor import (
    AlternativeDataAggregator,
    AlternativeDataSignal,
    SatelliteImageryAnalyzer,
    SocialMediaSentimentAnalyzer,
    NewsAndWebScraper
)

__all__ = [
    "AlternativeDataAggregator",
    "AlternativeDataSignal",
    "SatelliteImageryAnalyzer",
    "SocialMediaSentimentAnalyzer",
    "NewsAndWebScraper"
]
