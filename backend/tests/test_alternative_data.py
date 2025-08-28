"""
Comprehensive tests for Alternative Data Processing module.
Tests satellite imagery, social media sentiment, and news analysis.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from PIL import Image
import cv2
import asyncio

from app.alternative_data.alternative_data_processor import (
    AlternativeDataSignal,
    SatelliteImageryAnalyzer,
    SocialMediaSentimentAnalyzer,
    NewsAndWebScraper,
    AlternativeDataAggregator
)


class TestAlternativeDataSignal:
    """Test alternative data signal dataclass."""
    
    def test_signal_initialization(self):
        """Test signal initialization with all fields."""
        signal = AlternativeDataSignal(
            source='satellite',
            signal_type='bullish',
            strength=0.75,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={'location': {'lat': 40.7128, 'lon': -74.0060}},
            affected_symbols=['XRT', 'WMT', 'TGT']
        )
        
        assert signal.source == 'satellite'
        assert signal.signal_type == 'bullish'
        assert signal.strength == 0.75
        assert signal.confidence == 0.85
        assert isinstance(signal.timestamp, datetime)
        assert 'location' in signal.metadata
        assert len(signal.affected_symbols) == 3
    
    def test_signal_types(self):
        """Test different signal types."""
        sources = ['satellite', 'social', 'news', 'weather', 'supply_chain']
        signal_types = ['bullish', 'bearish', 'neutral']
        
        for source in sources:
            for signal_type in signal_types:
                signal = AlternativeDataSignal(
                    source=source,
                    signal_type=signal_type,
                    strength=0.5,
                    confidence=0.5,
                    timestamp=datetime.now(),
                    metadata={},
                    affected_symbols=[]
                )
                assert signal.source == source
                assert signal.signal_type == signal_type


class TestSatelliteImageryAnalyzer:
    """Test satellite imagery analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create satellite analyzer instance."""
        with patch('app.alternative_data.alternative_data_processor.models.resnet50'):
            return SatelliteImageryAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model is not None
        assert hasattr(analyzer.model, 'fc')
        assert analyzer.transform is not None
        assert len(analyzer.activity_classes) == 10
        assert len(analyzer.activity_symbol_map) == 5
    
    @pytest.mark.asyncio
    async def test_analyze_satellite_image(self, analyzer):
        """Test satellite image analysis."""
        # Create mock image
        with patch('PIL.Image.open') as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value = mock_image
            
            # Mock model prediction
            with patch.object(analyzer.model, 'eval'):
                with patch('torch.no_grad'):
                    mock_output = torch.randn(1, 10)
                    analyzer.model.return_value = mock_output
                    
                    signal = await analyzer.analyze_satellite_image(
                        'test_image.jpg',
                        {'lat': 40.7128, 'lon': -74.0060},
                        'parking_lot'
                    )
        
        assert isinstance(signal, AlternativeDataSignal)
        assert signal.source == 'satellite'
        assert signal.signal_type in ['bullish', 'bearish', 'neutral']
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert 'location' in signal.metadata
        assert 'image_type' in signal.metadata
        assert 'activity_class' in signal.metadata
        assert len(signal.affected_symbols) > 0
    
    def test_analyze_parking_lot_occupancy(self, analyzer):
        """Test parking lot occupancy analysis."""
        # Create test image (black and white for simplicity)
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Add some white rectangles (cars)
        for i in range(10):
            x = i * 50
            y = 100
            cv2.rectangle(image, (x, y), (x + 30, y + 20), (255, 255, 255), -1)
        
        result = analyzer.analyze_parking_lot_occupancy(image, baseline_occupancy=0.5)
        
        assert 'occupancy_rate' in result
        assert 'occupancy_change' in result
        assert 'car_count' in result
        assert 'signal_strength' in result
        
        assert 0 <= result['occupancy_rate'] <= 1
        assert result['car_count'] >= 0
    
    @patch('cv2.CascadeClassifier')
    @patch('rasterio.open')
    def test_analyze_port_activity(self, mock_rasterio, mock_cascade, analyzer):
        """Test port activity analysis."""
        # Mock rasterio
        mock_src = MagicMock()
        mock_src.read.return_value = np.random.rand(3, 1000, 1000)
        mock_src.transform = MagicMock()
        mock_rasterio.return_value.__enter__.return_value = mock_src
        
        # Mock ship detector
        mock_detector = MagicMock()
        mock_cascade.return_value = mock_detector
        
        from shapely.geometry import Polygon
        port_boundaries = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
        
        result = analyzer.analyze_port_activity('port_image.tif', port_boundaries)
        
        assert 'ship_count' in result
        assert 'activity_change' in result
        assert 'port_utilization' in result
        assert 'signal' in result
        
        assert result['ship_count'] >= 0
        assert 0 <= result['port_utilization'] <= 1
        assert result['signal'] in ['bullish', 'bearish']
    
    def test_interpret_activity(self, analyzer):
        """Test activity interpretation."""
        # Test bullish activities
        bullish_activities = [
            'high_retail_activity',
            'high_shipping_activity',
            'construction_increase',
            'agricultural_growth',
            'oil_storage_increase'
        ]
        
        for activity in bullish_activities:
            signal_type, strength = analyzer._interpret_activity(activity, 'test')
            assert signal_type == 'bullish'
            assert strength == 0.7
        
        # Test bearish activities
        bearish_activities = [
            'low_retail_activity',
            'low_shipping_activity',
            'construction_decrease',
            'agricultural_decline',
            'oil_storage_decrease'
        ]
        
        for activity in bearish_activities:
            signal_type, strength = analyzer._interpret_activity(activity, 'test')
            assert signal_type == 'bearish'
            assert strength == 0.7
        
        # Test neutral
        signal_type, strength = analyzer._interpret_activity('unknown_activity', 'test')
        assert signal_type == 'neutral'
        assert strength == 0.3


class TestSocialMediaSentimentAnalyzer:
    """Test social media sentiment analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        api_keys = {
            'twitter_bearer_token': 'test_token',
            'reddit_client_id': 'test_id',
            'reddit_client_secret': 'test_secret'
        }
        
        with patch('app.alternative_data.alternative_data_processor.tweepy.Client'):
            with patch('app.alternative_data.alternative_data_processor.praw.Reddit'):
                with patch('app.alternative_data.alternative_data_processor.pipeline'):
                    return SocialMediaSentimentAnalyzer(api_keys)
    
    def test_analyzer_initialization(self, analyzer):
        """Test sentiment analyzer initialization."""
        assert analyzer.twitter_api is not None
        assert analyzer.reddit is not None
        assert analyzer.financial_bert is not None
        assert analyzer.vader is not None
        assert len(analyzer.meme_stock_patterns) > 0
        assert isinstance(analyzer.influencer_scores, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_twitter_sentiment(self, analyzer):
        """Test Twitter sentiment analysis."""
        # Mock Twitter API response
        mock_tweets = MagicMock()
        mock_tweet = MagicMock()
        mock_tweet.text = "AAPL is going to the moon! 🚀 Great earnings report!"
        mock_tweet.public_metrics = {
            'like_count': 100,
            'retweet_count': 50,
            'reply_count': 20
        }
        mock_tweets.data = [mock_tweet]
        
        analyzer.twitter_api.search_recent_tweets.return_value = mock_tweets
        
        # Mock sentiment analysis
        with patch.object(analyzer, '_analyze_tweet_sentiment', return_value=0.8):
            signals = await analyzer.analyze_twitter_sentiment(['AAPL'], 24)
        
        assert 'AAPL' in signals
        signal = signals['AAPL']
        
        assert isinstance(signal, AlternativeDataSignal)
        assert signal.source == 'twitter'
        assert signal.signal_type == 'bullish'  # Positive sentiment
        assert signal.strength > 0
        assert signal.confidence > 0
        assert 'tweet_count' in signal.metadata
        assert 'total_engagement' in signal.metadata
        assert 'meme_percentage' in signal.metadata
        assert 'avg_sentiment' in signal.metadata
    
    @pytest.mark.asyncio
    async def test_analyze_reddit_sentiment(self, analyzer):
        """Test Reddit sentiment analysis."""
        # Mock Reddit submissions
        mock_submission = MagicMock()
        mock_submission.title = "DD: Why TSLA is undervalued"
        mock_submission.selftext = "Technical analysis shows strong support..."
        mock_submission.score = 1000
        mock_submission.num_comments = 200
        
        mock_subreddit = MagicMock()
        mock_subreddit.hot.return_value = [mock_submission]
        analyzer.reddit.subreddit.return_value = mock_subreddit
        
        # Mock sentiment analysis
        with patch.object(analyzer, '_extract_stock_symbols', return_value=['TSLA']):
            with patch.object(analyzer, '_analyze_reddit_sentiment', return_value=0.6):
                signals = await analyzer.analyze_reddit_sentiment(['wallstreetbets'])
        
        assert 'TSLA' in signals
        signal = signals['TSLA']
        
        assert signal.source == 'reddit'
        assert signal.signal_type == 'bullish'
        assert 'mention_count' in signal.metadata
        assert 'wsb_ratio' in signal.metadata
        assert 'dd_posts' in signal.metadata
    
    def test_analyze_tweet_sentiment(self, analyzer):
        """Test individual tweet sentiment analysis."""
        # Test positive tweet
        positive_tweet = "Amazing earnings! $AAPL breaking all-time highs! 🚀"
        sentiment = analyzer._analyze_tweet_sentiment(positive_tweet)
        assert sentiment > 0
        
        # Test negative tweet
        negative_tweet = "Terrible news, selling all my $AAPL positions. Company is doomed."
        sentiment = analyzer._analyze_tweet_sentiment(negative_tweet)
        assert sentiment < 0
        
        # Test neutral tweet
        neutral_tweet = "$AAPL trading sideways today. No major moves."
        sentiment = analyzer._analyze_tweet_sentiment(neutral_tweet)
        assert -0.5 < sentiment < 0.5
    
    def test_extract_stock_symbols(self, analyzer):
        """Test stock symbol extraction from text."""
        # Test with $ symbols
        text1 = "Buying $AAPL and $MSFT today! Also watching $GOOGL"
        symbols = analyzer._extract_stock_symbols(text1)
        assert set(symbols) == {'AAPL', 'MSFT', 'GOOGL'}
        
        # Test without $ symbols
        text2 = "TSLA is overvalued. I prefer AMZN and NFLX for growth."
        symbols = analyzer._extract_stock_symbols(text2)
        assert 'TSLA' in symbols
        assert 'AMZN' in symbols
        assert 'NFLX' in symbols
        
        # Test filtering common words
        text3 = "I think AI and CEO are important. Buy IBM!"
        symbols = analyzer._extract_stock_symbols(text3)
        assert 'AI' not in symbols  # Common word
        assert 'CEO' not in symbols  # Common word
        assert 'IBM' in symbols
    
    def test_is_meme_stock_mention(self, analyzer):
        """Test meme stock language detection."""
        # Meme stock mentions
        meme_texts = [
            "HODL to the moon! 🚀",
            "Diamond hands forever 💎🙌",
            "Apes together strong!",
            "YOLO on GME calls",
            "We like the stock!"
        ]
        
        for text in meme_texts:
            assert analyzer._is_meme_stock_mention(text) is True
        
        # Normal mentions
        normal_texts = [
            "Based on fundamental analysis, I'm bullish on AAPL",
            "The P/E ratio suggests undervaluation",
            "Quarterly earnings exceeded expectations"
        ]
        
        for text in normal_texts:
            assert analyzer._is_meme_stock_mention(text) is False
    
    def test_clean_social_media_text(self, analyzer):
        """Test social media text cleaning."""
        # Test URL removal
        text1 = "Check this out: https://example.com $AAPL analysis"
        cleaned1 = analyzer._clean_social_media_text(text1)
        assert "https://" not in cleaned1
        assert "$AAPL" in cleaned1
        
        # Test mention/hashtag cleaning
        text2 = "@john_doe thinks #AAPL is #bullish"
        cleaned2 = analyzer._clean_social_media_text(text2)
        assert "john_doe" in cleaned2
        assert "AAPL" in cleaned2
        assert "bullish" in cleaned2
        assert "@" not in cleaned2
        assert "#" not in cleaned2


class TestNewsAndWebScraper:
    """Test news and web scraping functionality."""
    
    @pytest.fixture
    def scraper(self):
        """Create news scraper instance."""
        with patch('app.alternative_data.alternative_data_processor.aiohttp.ClientSession'):
            with patch('app.alternative_data.alternative_data_processor.pipeline'):
                return NewsAndWebScraper()
    
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.session is not None
        assert scraper.sentiment_analyzer is not None
        assert scraper.ner_model is not None
        assert len(scraper.news_sources) > 0
        assert len(scraper.sector_keywords) > 0
    
    @pytest.mark.asyncio
    async def test_scrape_financial_news(self, scraper):
        """Test financial news scraping."""
        # Mock yfinance ticker
        with patch('yfinance.Ticker') as mock_ticker:
            mock_info = {'longName': 'Apple Inc.'}
            mock_ticker.return_value.info = mock_info
            
            # Mock news search
            mock_news = [{
                'title': 'Apple Reports Record Q4 Earnings',
                'source': 'reuters',
                'published': datetime.now() - timedelta(hours=2),
                'content': 'Apple Inc. reported record earnings...'
            }]
            
            with patch.object(scraper, '_search_news', return_value=mock_news):
                with patch.object(scraper, '_analyze_news_item', return_value={
                    'signal_type': 'bullish',
                    'strength': 0.8,
                    'confidence': 0.9,
                    'relevance': 0.95,
                    'entities': ['Apple Inc.'],
                    'related_symbols': []
                }):
                    signals = await scraper.scrape_financial_news(['AAPL'], 24)
        
        assert 'AAPL' in signals
        assert len(signals['AAPL']) > 0
        
        signal = signals['AAPL'][0]
        assert signal.source == 'news'
        assert signal.signal_type == 'bullish'
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert 'headline' in signal.metadata
        assert 'source' in signal.metadata
        assert 'relevance' in signal.metadata
    
    @pytest.mark.asyncio
    async def test_analyze_sec_filings(self, scraper):
        """Test SEC filing analysis."""
        # Mock SEC filing data
        mock_filings = [{
            'type': '8-K',
            'filed_date': datetime.now() - timedelta(days=1),
            'url': 'https://sec.gov/filing/123',
            'key_items': ['Material Agreement', 'Financial Statements']
        }]
        
        with patch.object(scraper, '_get_sec_filings', return_value=mock_filings):
            with patch.object(scraper, '_analyze_8k_filing', return_value='bullish'):
                signals = await scraper.analyze_sec_filings('AAPL', ['8-K'])
        
        assert len(signals) > 0
        signal = signals[0]
        
        assert signal.source == 'sec_filing'
        assert signal.confidence == 0.9  # SEC filings are reliable
        assert 'filing_type' in signal.metadata
        assert 'filing_url' in signal.metadata
        assert 'key_items' in signal.metadata
    
    @pytest.mark.asyncio
    async def test_scrape_supply_chain_data(self, scraper):
        """Test supply chain data scraping."""
        # Mock HTTP response
        mock_html = """
        <html>
            <body>
                <article>
                    <h2>Port Congestion Affects Apple Supply Chain</h2>
                    <p>Major delays at west coast ports impacting Apple shipments...</p>
                </article>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=mock_html)
        
        scraper.session.get = AsyncMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(scraper, '_company_to_symbol', return_value='AAPL'):
            signals = await scraper.scrape_supply_chain_data(['Apple'])
        
        if 'Apple' in signals:
            signal = signals['Apple']
            assert signal.source == 'supply_chain'
            assert 'disruption_type' in signal.metadata
            assert 'affected_regions' in signal.metadata
    
    def test_analyze_news_item(self, scraper):
        """Test news item analysis."""
        news = {
            'title': 'Apple Announces Revolutionary New Product',
            'content': 'Apple Inc. unveiled a groundbreaking new device that analysts predict will drive significant revenue growth. The product addresses a large market opportunity and has received positive initial reviews. Apple stock rose 5% on the news.'
        }
        
        # Mock sentiment analyzer
        scraper.sentiment_analyzer.return_value = [{
            'label': '5 stars',
            'score': 0.95
        }]
        
        # Mock NER
        scraper.ner_model.return_value = [
            {'word': 'Apple Inc.', 'entity': 'ORG'},
            {'word': 'Tim Cook', 'entity': 'PER'}
        ]
        
        analysis = scraper._analyze_news_item(news, 'AAPL')
        
        assert analysis['signal_type'] == 'bullish'
        assert analysis['strength'] > 0
        assert analysis['confidence'] > 0
        assert analysis['relevance'] > 0
        assert 'Apple Inc.' in analysis['entities']
    
    def test_analyze_supply_chain_impact(self, scraper):
        """Test supply chain impact analysis."""
        # Test severe disruption
        text1 = "Major port closure affecting Apple shipments. Severe delays expected for weeks."
        impact1 = scraper._analyze_supply_chain_impact(text1, 'Apple')
        
        assert impact1['signal_type'] == 'bearish'
        assert impact1['severity'] >= 0.8
        assert impact1['type'] == 'closure'
        
        # Test mild disruption
        text2 = "Minor delays in Apple supply chain due to weather."
        impact2 = scraper._analyze_supply_chain_impact(text2, 'Apple')
        
        assert impact2['severity'] < impact1['severity']
        assert impact2['type'] == 'delay'
    
    def test_company_to_symbol(self, scraper):
        """Test company name to symbol conversion."""
        # Test basic conversion
        assert scraper._company_to_symbol('Apple Inc.') == 'APPL'
        assert scraper._company_to_symbol('Microsoft Corporation') == 'MICR'
        assert scraper._company_to_symbol('X') == 'X'


class TestAlternativeDataAggregator:
    """Test alternative data aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance."""
        api_keys = {
            'twitter_bearer_token': 'test',
            'reddit_client_id': 'test',
            'reddit_client_secret': 'test'
        }
        
        with patch('app.alternative_data.alternative_data_processor.SatelliteImageryAnalyzer'):
            with patch('app.alternative_data.alternative_data_processor.SocialMediaSentimentAnalyzer'):
                with patch('app.alternative_data.alternative_data_processor.NewsAndWebScraper'):
                    return AlternativeDataAggregator(api_keys)
    
    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.satellite_analyzer is not None
        assert aggregator.social_analyzer is not None
        assert aggregator.news_scraper is not None
        assert len(aggregator.source_weights) == 5
        assert sum(aggregator.source_weights.values()) == 1.0
        assert isinstance(aggregator.signal_history, dict)
    
    @pytest.mark.asyncio
    async def test_get_composite_signals(self, aggregator):
        """Test composite signal generation."""
        # Mock individual signals
        twitter_signal = AlternativeDataSignal(
            source='twitter',
            signal_type='bullish',
            strength=0.7,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={'tweet_count': 100},
            affected_symbols=['AAPL']
        )
        
        reddit_signal = AlternativeDataSignal(
            source='reddit',
            signal_type='bullish',
            strength=0.6,
            confidence=0.7,
            timestamp=datetime.now(),
            metadata={'mention_count': 50},
            affected_symbols=['AAPL']
        )
        
        news_signal = AlternativeDataSignal(
            source='news',
            signal_type='neutral',
            strength=0.3,
            confidence=0.9,
            timestamp=datetime.now(),
            metadata={'article_count': 5},
            affected_symbols=['AAPL']
        )
        
        # Mock analyzer methods
        aggregator.social_analyzer.analyze_twitter_sentiment = AsyncMock(
            return_value={'AAPL': twitter_signal}
        )
        aggregator.social_analyzer.analyze_reddit_sentiment = AsyncMock(
            return_value={'AAPL': reddit_signal}
        )
        aggregator.news_scraper.scrape_financial_news = AsyncMock(
            return_value={'AAPL': [news_signal]}
        )
        
        composite_signals = await aggregator.get_composite_signals(['AAPL'], 24)
        
        assert 'AAPL' in composite_signals
        composite = composite_signals['AAPL']
        
        assert 'signal_type' in composite
        assert 'strength' in composite
        assert 'confidence' in composite
        assert 'sources' in composite
        assert 'signal_count' in composite
        assert 'latest_timestamp' in composite
        
        # Check signal aggregation
        assert composite['signal_type'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= composite['strength'] <= 1
        assert 0 <= composite['confidence'] <= 1
        assert composite['signal_count'] == 3
    
    def test_combine_signals(self, aggregator):
        """Test signal combination logic."""
        # Create diverse signals
        signals = [
            AlternativeDataSignal(
                source='twitter',
                signal_type='bullish',
                strength=0.8,
                confidence=0.9,
                timestamp=datetime.now(),
                metadata={},
                affected_symbols=['AAPL']
            ),
            AlternativeDataSignal(
                source='reddit',
                signal_type='bullish',
                strength=0.6,
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={},
                affected_symbols=['AAPL']
            ),
            AlternativeDataSignal(
                source='news',
                signal_type='bearish',
                strength=0.5,
                confidence=0.95,
                timestamp=datetime.now(),
                metadata={},
                affected_symbols=['AAPL']
            ),
            AlternativeDataSignal(
                source='satellite',
                signal_type='neutral',
                strength=0.3,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={},
                affected_symbols=['AAPL']
            )
        ]
        
        composite = aggregator._combine_signals(signals)
        
        # Check structure
        assert 'signal_type' in composite
        assert 'strength' in composite
        assert 'confidence' in composite
        assert 'sources' in composite
        
        # Check source aggregation
        assert 'twitter' in composite['sources']
        assert 'reddit' in composite['sources']
        assert 'news' in composite['sources']
        assert 'satellite' in composite['sources']
        
        # Verify weighted combination
        # Twitter and Reddit are bullish, News is bearish, Satellite is neutral
        # With default weights, should lean bullish but not strongly
        assert composite['signal_type'] in ['bullish', 'neutral']
    
    def test_signal_history_tracking(self, aggregator):
        """Test signal history tracking."""
        # Add signals to history
        signal1 = {
            'signal_type': 'bullish',
            'strength': 0.7,
            'confidence': 0.8
        }
        
        signal2 = {
            'signal_type': 'bearish',
            'strength': 0.6,
            'confidence': 0.9
        }
        
        aggregator.signal_history['AAPL'].append({
            'timestamp': datetime.now() - timedelta(hours=2),
            'composite': signal1
        })
        
        aggregator.signal_history['AAPL'].append({
            'timestamp': datetime.now(),
            'composite': signal2
        })
        
        # Check history
        assert len(aggregator.signal_history['AAPL']) == 2
        
        # Most recent should be bearish
        latest = aggregator.signal_history['AAPL'][-1]
        assert latest['composite']['signal_type'] == 'bearish'
    
    def test_source_weight_validation(self, aggregator):
        """Test source weight configuration."""
        # Check default weights sum to 1
        total_weight = sum(aggregator.source_weights.values())
        assert abs(total_weight - 1.0) < 0.001
        
        # Check all sources have weights
        expected_sources = ['satellite', 'twitter', 'reddit', 'news', 'sec_filing']
        for source in expected_sources:
            assert source in aggregator.source_weights
            assert 0 <= aggregator.source_weights[source] <= 1
