"""
Alternative Data Processing Module

Integrates and processes alternative data sources including:
- Satellite imagery for economic activity
- Social media sentiment
- Web scraping for supply chain data
- Weather patterns
- Mobile device location data
- Credit card transaction trends
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tweepy
import praw  # Reddit API
from textblob import TextBlob
import cv2
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from bs4 import BeautifulSoup
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataSignal:
    """Signal derived from alternative data."""
    
    source: str  # 'satellite', 'social', 'news', 'weather', etc.
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1 signal strength
    confidence: float  # 0-1 confidence level
    timestamp: datetime
    metadata: Dict[str, Any]
    affected_symbols: List[str]


class SatelliteImageryAnalyzer:
    """
    Analyzes satellite imagery for economic indicators like:
    - Parking lot occupancy (retail activity)
    - Ship traffic in ports
    - Agricultural yields
    - Construction activity
    - Oil storage levels
    """
    
    def __init__(self):
        # Load pre-trained ResNet for image classification
        self.model = models.resnet50(pretrained=True)
        
        # Modify for satellite imagery analysis
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 economic activity classes
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Economic activity mappings
        self.activity_classes = {
            0: 'low_retail_activity',
            1: 'high_retail_activity',
            2: 'low_shipping_activity',
            3: 'high_shipping_activity',
            4: 'construction_increase',
            5: 'construction_decrease',
            6: 'agricultural_growth',
            7: 'agricultural_decline',
            8: 'oil_storage_increase',
            9: 'oil_storage_decrease'
        }
        
        # Symbol mappings for different activities
        self.activity_symbol_map = {
            'retail_activity': ['XRT', 'WMT', 'TGT', 'COST', 'AMZN'],
            'shipping_activity': ['FDX', 'UPS', 'EXPD', 'CHRW'],
            'construction': ['CAT', 'DE', 'VMC', 'MLM'],
            'agricultural': ['ADM', 'BG', 'CORN', 'WEAT'],
            'oil_storage': ['XLE', 'USO', 'XOM', 'CVX']
        }
    
    async def analyze_satellite_image(
        self,
        image_path: str,
        location: Dict[str, float],
        image_type: str = 'parking_lot'
    ) -> AlternativeDataSignal:
        """Analyze a satellite image for economic signals."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = float(probabilities[0][prediction])
        
        # Interpret prediction
        activity_class = self.activity_classes[prediction]
        signal_type, strength = self._interpret_activity(activity_class, image_type)
        
        # Get affected symbols
        activity_category = activity_class.split('_')[1]
        affected_symbols = self.activity_symbol_map.get(
            f"{activity_category}_activity",
            []
        )
        
        return AlternativeDataSignal(
            source='satellite',
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata={
                'location': location,
                'image_type': image_type,
                'activity_class': activity_class
            },
            affected_symbols=affected_symbols
        )
    
    def analyze_parking_lot_occupancy(
        self,
        image: np.ndarray,
        baseline_occupancy: float = 0.7
    ) -> Dict[str, float]:
        """Analyze parking lot occupancy from satellite image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to identify cars (simplified)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours (potential cars)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size (car-sized objects)
        car_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 500:  # Typical car size in pixels
                car_contours.append(contour)
        
        # Calculate occupancy
        total_parking_area = image.shape[0] * image.shape[1]
        occupied_area = sum(cv2.contourArea(c) for c in car_contours)
        occupancy_rate = occupied_area / total_parking_area
        
        # Compare to baseline
        occupancy_change = (occupancy_rate - baseline_occupancy) / baseline_occupancy
        
        return {
            'occupancy_rate': occupancy_rate,
            'occupancy_change': occupancy_change,
            'car_count': len(car_contours),
            'signal_strength': abs(occupancy_change)
        }
    
    def analyze_port_activity(
        self,
        image_path: str,
        port_boundaries: Polygon
    ) -> Dict[str, Any]:
        """Analyze shipping activity at ports."""
        # Open satellite image
        with rasterio.open(image_path) as src:
            image = src.read()
            transform = src.transform
        
        # Detect ships using computer vision
        ship_detector = cv2.CascadeClassifier('haarcascade_ship.xml')  # Placeholder
        
        # Count ships in port area
        # Simplified - would use proper ship detection model
        ship_count = np.random.randint(10, 50)  # Placeholder
        
        # Historical average
        historical_avg = 30
        
        activity_change = (ship_count - historical_avg) / historical_avg
        
        return {
            'ship_count': ship_count,
            'activity_change': activity_change,
            'port_utilization': min(ship_count / 60, 1.0),  # Max capacity 60 ships
            'signal': 'bullish' if activity_change > 0.1 else 'bearish'
        }
    
    def _interpret_activity(
        self,
        activity_class: str,
        image_type: str
    ) -> Tuple[str, float]:
        """Interpret activity class into trading signal."""
        # Map activity to signal
        if 'high' in activity_class or 'increase' in activity_class or 'growth' in activity_class:
            signal_type = 'bullish'
            strength = 0.7
        elif 'low' in activity_class or 'decrease' in activity_class or 'decline' in activity_class:
            signal_type = 'bearish'
            strength = 0.7
        else:
            signal_type = 'neutral'
            strength = 0.3
        
        return signal_type, strength


class SocialMediaSentimentAnalyzer:
    """
    Analyzes social media sentiment from multiple sources:
    - Twitter/X
    - Reddit (WallStreetBets, investing subreddits)
    - StockTwits
    - Discord trading servers
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        # Twitter API
        self.twitter_api = tweepy.Client(
            bearer_token=api_keys.get('twitter_bearer_token')
        )
        
        # Reddit API
        self.reddit = praw.Reddit(
            client_id=api_keys.get('reddit_client_id'),
            client_secret=api_keys.get('reddit_client_secret'),
            user_agent='TradingBot'
        )
        
        # Sentiment analysis models
        self.financial_bert = pipeline(
            'sentiment-analysis',
            model='ProsusAI/finbert'
        )
        
        self.vader = SentimentIntensityAnalyzer()
        
        # Meme stock detector
        self.meme_stock_patterns = [
            'moon', 'rocket', 'diamond hands', 'apes', 'squeeze',
            'yolo', 'tendies', 'hodl', 'to the moon', 'we like the stock'
        ]
        
        # Influence scoring
        self.influencer_scores = {}  # User -> influence score
        
    async def analyze_twitter_sentiment(
        self,
        symbols: List[str],
        lookback_hours: int = 24
    ) -> Dict[str, AlternativeDataSignal]:
        """Analyze Twitter sentiment for given symbols."""
        signals = {}
        
        for symbol in symbols:
            # Search tweets
            query = f"${symbol} -is:retweet lang:en"
            
            try:
                tweets = self.twitter_api.search_recent_tweets(
                    query=query,
                    max_results=100,
                    tweet_fields=['created_at', 'author_id', 'public_metrics']
                )
                
                if not tweets.data:
                    continue
                
                # Analyze sentiment
                sentiments = []
                total_engagement = 0
                
                for tweet in tweets.data:
                    # Get sentiment
                    sentiment = self._analyze_tweet_sentiment(tweet.text)
                    
                    # Weight by engagement
                    engagement = (
                        tweet.public_metrics['like_count'] +
                        tweet.public_metrics['retweet_count'] * 2 +
                        tweet.public_metrics['reply_count']
                    )
                    
                    sentiments.append({
                        'sentiment': sentiment,
                        'engagement': engagement,
                        'is_meme': self._is_meme_stock_mention(tweet.text)
                    })
                    
                    total_engagement += engagement
                
                # Calculate weighted sentiment
                if total_engagement > 0:
                    weighted_sentiment = sum(
                        s['sentiment'] * s['engagement']
                        for s in sentiments
                    ) / total_engagement
                else:
                    weighted_sentiment = np.mean([s['sentiment'] for s in sentiments])
                
                # Determine signal
                if weighted_sentiment > 0.2:
                    signal_type = 'bullish'
                elif weighted_sentiment < -0.2:
                    signal_type = 'bearish'
                else:
                    signal_type = 'neutral'
                
                # Check for meme stock activity
                meme_percentage = sum(
                    1 for s in sentiments if s['is_meme']
                ) / len(sentiments)
                
                signals[symbol] = AlternativeDataSignal(
                    source='twitter',
                    signal_type=signal_type,
                    strength=abs(weighted_sentiment),
                    confidence=min(len(sentiments) / 100, 1.0),
                    timestamp=datetime.now(),
                    metadata={
                        'tweet_count': len(sentiments),
                        'total_engagement': total_engagement,
                        'meme_percentage': meme_percentage,
                        'avg_sentiment': weighted_sentiment
                    },
                    affected_symbols=[symbol]
                )
                
            except Exception as e:
                logger.error(f"Error analyzing Twitter sentiment for {symbol}: {e}")
        
        return signals
    
    async def analyze_reddit_sentiment(
        self,
        subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing'],
        limit: int = 1000
    ) -> Dict[str, AlternativeDataSignal]:
        """Analyze Reddit sentiment from trading subreddits."""
        symbol_mentions = defaultdict(list)
        
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            for submission in subreddit.hot(limit=limit):
                # Extract mentioned symbols
                symbols = self._extract_stock_symbols(
                    submission.title + ' ' + submission.selftext
                )
                
                # Analyze sentiment
                sentiment = self._analyze_reddit_sentiment(submission)
                
                for symbol in symbols:
                    symbol_mentions[symbol].append({
                        'sentiment': sentiment,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'subreddit': subreddit_name,
                        'is_dd': 'DD' in submission.title or 'Due Diligence' in submission.title
                    })
        
        # Generate signals
        signals = {}
        
        for symbol, mentions in symbol_mentions.items():
            if len(mentions) < 3:  # Minimum mentions threshold
                continue
            
            # Calculate weighted sentiment
            total_weight = sum(m['score'] + m['num_comments'] for m in mentions)
            if total_weight > 0:
                weighted_sentiment = sum(
                    m['sentiment'] * (m['score'] + m['num_comments'])
                    for m in mentions
                ) / total_weight
            else:
                weighted_sentiment = np.mean([m['sentiment'] for m in mentions])
            
            # WSB specific adjustments
            wsb_mentions = [m for m in mentions if m['subreddit'] == 'wallstreetbets']
            wsb_ratio = len(wsb_mentions) / len(mentions)
            
            # Determine signal
            if weighted_sentiment > 0.3:
                signal_type = 'bullish'
            elif weighted_sentiment < -0.3:
                signal_type = 'bearish'
            else:
                signal_type = 'neutral'
            
            signals[symbol] = AlternativeDataSignal(
                source='reddit',
                signal_type=signal_type,
                strength=abs(weighted_sentiment) * (1 + wsb_ratio * 0.5),
                confidence=min(len(mentions) / 50, 1.0),
                timestamp=datetime.now(),
                metadata={
                    'mention_count': len(mentions),
                    'wsb_ratio': wsb_ratio,
                    'dd_posts': sum(1 for m in mentions if m['is_dd']),
                    'avg_sentiment': weighted_sentiment
                },
                affected_symbols=[symbol]
            )
        
        return signals
    
    def _analyze_tweet_sentiment(self, text: str) -> float:
        """Analyze sentiment of a tweet."""
        # Clean text
        text = self._clean_social_media_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_sentiment = vader_scores['compound']
        
        # FinBERT sentiment (if text is long enough)
        if len(text.split()) > 5:
            try:
                finbert_result = self.financial_bert(text)[0]
                finbert_sentiment = (
                    1.0 if finbert_result['label'] == 'positive'
                    else -1.0 if finbert_result['label'] == 'negative'
                    else 0.0
                )
                finbert_confidence = finbert_result['score']
                
                # Weighted average
                sentiment = (
                    0.3 * vader_sentiment +
                    0.7 * finbert_sentiment * finbert_confidence
                )
            except:
                sentiment = vader_sentiment
        else:
            sentiment = vader_sentiment
        
        return sentiment
    
    def _analyze_reddit_sentiment(self, submission) -> float:
        """Analyze sentiment of Reddit submission."""
        # Combine title and body
        text = submission.title + ' ' + submission.selftext
        
        # Basic sentiment
        sentiment = self._analyze_tweet_sentiment(text)
        
        # Adjust for Reddit-specific signals
        if submission.link_flair_text:
            if 'bullish' in submission.link_flair_text.lower():
                sentiment += 0.3
            elif 'bearish' in submission.link_flair_text.lower():
                sentiment -= 0.3
            elif 'dd' in submission.link_flair_text.lower():
                sentiment *= 1.5  # DD posts are more significant
        
        # Clamp to [-1, 1]
        return max(-1, min(1, sentiment))
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        import re
        
        # Pattern for stock symbols ($XXX or XXX in caps)
        pattern = r'\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{2,5})(?:\s|$)'
        
        matches = re.findall(pattern, text)
        symbols = []
        
        for match in matches:
            symbol = match[0] if match[0] else match[1]
            if symbol and len(symbol) <= 5 and symbol.isupper():
                # Validate against common words
                if symbol not in ['I', 'A', 'DD', 'CEO', 'CFO', 'IPO', 'ETF']:
                    symbols.append(symbol)
        
        return list(set(symbols))
    
    def _is_meme_stock_mention(self, text: str) -> bool:
        """Check if text contains meme stock language."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.meme_stock_patterns)
    
    def _clean_social_media_text(self, text: str) -> str:
        """Clean social media text for analysis."""
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive spaces
        text = ' '.join(text.split())
        
        return text


class NewsAndWebScraper:
    """
    Scrapes and analyzes news from multiple sources:
    - Financial news sites
    - Company press releases
    - SEC filings
    - Earnings call transcripts
    - Supply chain data
    """
    
    def __init__(self):
        self.session = aiohttp.ClientSession()
        
        # NLP models
        self.sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='nlptown/bert-base-multilingual-uncased-sentiment'
        )
        
        self.ner_model = pipeline(
            'ner',
            model='dslim/bert-base-NER'
        )
        
        # News sources
        self.news_sources = {
            'reuters': 'https://www.reuters.com/business/',
            'bloomberg': 'https://www.bloomberg.com/markets',
            'wsj': 'https://www.wsj.com/market-data',
            'ft': 'https://www.ft.com/markets'
        }
        
        # Keywords for different sectors
        self.sector_keywords = {
            'technology': ['AI', 'cloud', 'semiconductor', 'software', 'cybersecurity'],
            'finance': ['interest rate', 'Fed', 'banking', 'loan', 'mortgage'],
            'energy': ['oil', 'gas', 'renewable', 'OPEC', 'crude'],
            'retail': ['consumer', 'sales', 'e-commerce', 'inventory'],
            'healthcare': ['FDA', 'drug', 'vaccine', 'clinical trial', 'biotech']
        }
    
    async def scrape_financial_news(
        self,
        symbols: List[str],
        hours_back: int = 24
    ) -> Dict[str, List[AlternativeDataSignal]]:
        """Scrape and analyze financial news."""
        signals = defaultdict(list)
        
        for symbol in symbols:
            # Get company info
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
            
            # Search news for company
            news_items = await self._search_news(company_name, hours_back)
            
            for news in news_items:
                # Analyze news sentiment and relevance
                analysis = self._analyze_news_item(news, symbol)
                
                if analysis['relevance'] > 0.5:
                    signal = AlternativeDataSignal(
                        source='news',
                        signal_type=analysis['signal_type'],
                        strength=analysis['strength'],
                        confidence=analysis['confidence'],
                        timestamp=news['published'],
                        metadata={
                            'headline': news['title'],
                            'source': news['source'],
                            'relevance': analysis['relevance'],
                            'entities': analysis['entities']
                        },
                        affected_symbols=[symbol] + analysis['related_symbols']
                    )
                    
                    signals[symbol].append(signal)
        
        return dict(signals)
    
    async def analyze_sec_filings(
        self,
        symbol: str,
        filing_types: List[str] = ['10-K', '10-Q', '8-K']
    ) -> List[AlternativeDataSignal]:
        """Analyze recent SEC filings."""
        signals = []
        
        # Use SEC EDGAR API (simplified)
        filings = await self._get_sec_filings(symbol, filing_types)
        
        for filing in filings:
            # Extract key information
            if filing['type'] == '8-K':
                # Material events
                signal_strength = 0.8
                signal_type = self._analyze_8k_filing(filing)
            elif filing['type'] in ['10-K', '10-Q']:
                # Financial reports
                signal_strength = 0.6
                signal_type = self._analyze_financial_filing(filing)
            
            signal = AlternativeDataSignal(
                source='sec_filing',
                signal_type=signal_type,
                strength=signal_strength,
                confidence=0.9,  # SEC filings are reliable
                timestamp=filing['filed_date'],
                metadata={
                    'filing_type': filing['type'],
                    'filing_url': filing['url'],
                    'key_items': filing.get('key_items', [])
                },
                affected_symbols=[symbol]
            )
            
            signals.append(signal)
        
        return signals
    
    async def scrape_supply_chain_data(
        self,
        companies: List[str]
    ) -> Dict[str, AlternativeDataSignal]:
        """Scrape supply chain disruption data."""
        signals = {}
        
        # Supply chain news sources
        supply_chain_urls = [
            'https://www.supplychaindive.com/',
            'https://www.freightwaves.com/',
            'https://www.joc.com/'
        ]
        
        for url in supply_chain_urls:
            try:
                async with self.session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract supply chain news
                    articles = soup.find_all('article', limit=20)
                    
                    for article in articles:
                        # Check if any monitored companies are mentioned
                        article_text = article.get_text()
                        
                        for company in companies:
                            if company.lower() in article_text.lower():
                                # Analyze impact
                                impact = self._analyze_supply_chain_impact(
                                    article_text,
                                    company
                                )
                                
                                if impact['severity'] > 0.3:
                                    signals[company] = AlternativeDataSignal(
                                        source='supply_chain',
                                        signal_type=impact['signal_type'],
                                        strength=impact['severity'],
                                        confidence=0.7,
                                        timestamp=datetime.now(),
                                        metadata={
                                            'disruption_type': impact['type'],
                                            'affected_regions': impact['regions'],
                                            'estimated_duration': impact['duration']
                                        },
                                        affected_symbols=[
                                            self._company_to_symbol(company)
                                        ]
                                    )
            
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        return signals
    
    async def _search_news(
        self,
        query: str,
        hours_back: int
    ) -> List[Dict[str, Any]]:
        """Search for news articles."""
        # Placeholder - would use news API
        return [
            {
                'title': f"Sample news about {query}",
                'source': 'reuters',
                'published': datetime.now() - timedelta(hours=1),
                'content': f"Article content about {query}..."
            }
        ]
    
    def _analyze_news_item(
        self,
        news: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Analyze a news item for trading signals."""
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(news['content'][:512])[0]
        
        # Convert to trading signal
        if sentiment_result['label'] in ['5 stars', '4 stars']:
            signal_type = 'bullish'
        elif sentiment_result['label'] in ['1 star', '2 stars']:
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'
        
        # Extract entities
        entities = self.ner_model(news['content'][:512])
        
        # Calculate relevance
        title_mentions = news['title'].lower().count(symbol.lower())
        content_mentions = news['content'].lower().count(symbol.lower())
        relevance = min(1.0, (title_mentions * 2 + content_mentions) / 10)
        
        return {
            'signal_type': signal_type,
            'strength': sentiment_result['score'] * relevance,
            'confidence': sentiment_result['score'],
            'relevance': relevance,
            'entities': [e['word'] for e in entities if e['entity'] == 'ORG'],
            'related_symbols': []  # Would extract from entities
        }
    
    async def _get_sec_filings(
        self,
        symbol: str,
        filing_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Get recent SEC filings."""
        # Placeholder - would use SEC EDGAR API
        return []
    
    def _analyze_8k_filing(self, filing: Dict[str, Any]) -> str:
        """Analyze 8-K filing for material events."""
        # Placeholder logic
        return 'neutral'
    
    def _analyze_financial_filing(self, filing: Dict[str, Any]) -> str:
        """Analyze financial filing."""
        # Placeholder logic
        return 'neutral'
    
    def _analyze_supply_chain_impact(
        self,
        text: str,
        company: str
    ) -> Dict[str, Any]:
        """Analyze supply chain disruption impact."""
        # Keywords indicating disruption
        disruption_keywords = {
            'shortage': 0.8,
            'delay': 0.6,
            'disruption': 0.7,
            'closure': 0.9,
            'strike': 0.8,
            'congestion': 0.6
        }
        
        severity = 0.0
        disruption_type = 'unknown'
        
        text_lower = text.lower()
        for keyword, weight in disruption_keywords.items():
            if keyword in text_lower:
                severity = max(severity, weight)
                disruption_type = keyword
        
        # Determine signal type
        if severity > 0.6:
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'
        
        return {
            'signal_type': signal_type,
            'severity': severity,
            'type': disruption_type,
            'regions': [],  # Would extract from text
            'duration': 'unknown'
        }
    
    def _company_to_symbol(self, company_name: str) -> str:
        """Convert company name to stock symbol."""
        # Placeholder - would use symbol lookup service
        return company_name[:4].upper()


class AlternativeDataAggregator:
    """
    Aggregates signals from all alternative data sources
    and generates composite trading signals.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.satellite_analyzer = SatelliteImageryAnalyzer()
        self.social_analyzer = SocialMediaSentimentAnalyzer(api_keys)
        self.news_scraper = NewsAndWebScraper()
        
        # Signal weights by source
        self.source_weights = {
            'satellite': 0.25,
            'twitter': 0.15,
            'reddit': 0.15,
            'news': 0.25,
            'sec_filing': 0.20
        }
        
        # Signal history for trend analysis
        self.signal_history = defaultdict(list)
        
    async def get_composite_signals(
        self,
        symbols: List[str],
        lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """Get composite signals from all alternative data sources."""
        composite_signals = {}
        
        # Gather signals from all sources in parallel
        tasks = []
        
        # Social media signals
        tasks.append(self.social_analyzer.analyze_twitter_sentiment(
            symbols, lookback_hours
        ))
        tasks.append(self.social_analyzer.analyze_reddit_sentiment())
        
        # News signals
        tasks.append(self.news_scraper.scrape_financial_news(
            symbols, lookback_hours
        ))
        
        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine signals by symbol
        all_signals = defaultdict(list)
        
        # Process results
        for result in results:
            if isinstance(result, dict):
                for symbol, signal in result.items():
                    if isinstance(signal, AlternativeDataSignal):
                        all_signals[symbol].append(signal)
                    elif isinstance(signal, list):
                        all_signals[symbol].extend(signal)
        
        # Generate composite signals
        for symbol, signals in all_signals.items():
            if not signals:
                continue
            
            composite = self._combine_signals(signals)
            composite_signals[symbol] = composite
            
            # Store in history
            self.signal_history[symbol].append({
                'timestamp': datetime.now(),
                'composite': composite
            })
        
        return composite_signals
    
    def _combine_signals(
        self,
        signals: List[AlternativeDataSignal]
    ) -> Dict[str, Any]:
        """Combine multiple signals into a composite signal."""
        # Group by source
        by_source = defaultdict(list)
        for signal in signals:
            by_source[signal.source].append(signal)
        
        # Calculate weighted sentiment by source
        source_sentiments = {}
        
        for source, source_signals in by_source.items():
            # Average sentiment for this source
            bullish_strength = sum(
                s.strength for s in source_signals
                if s.signal_type == 'bullish'
            )
            bearish_strength = sum(
                s.strength for s in source_signals
                if s.signal_type == 'bearish'
            )
            
            net_sentiment = bullish_strength - bearish_strength
            avg_confidence = np.mean([s.confidence for s in source_signals])
            
            source_sentiments[source] = {
                'sentiment': net_sentiment,
                'confidence': avg_confidence,
                'signal_count': len(source_signals)
            }
        
        # Calculate weighted composite
        total_weight = 0
        weighted_sentiment = 0
        weighted_confidence = 0
        
        for source, sentiment_data in source_sentiments.items():
            weight = self.source_weights.get(source, 0.1)
            total_weight += weight
            
            weighted_sentiment += weight * sentiment_data['sentiment']
            weighted_confidence += weight * sentiment_data['confidence']
        
        if total_weight > 0:
            composite_sentiment = weighted_sentiment / total_weight
            composite_confidence = weighted_confidence / total_weight
        else:
            composite_sentiment = 0
            composite_confidence = 0
        
        # Determine composite signal
        if composite_sentiment > 0.2:
            signal_type = 'bullish'
        elif composite_sentiment < -0.2:
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'
        
        return {
            'signal_type': signal_type,
            'strength': abs(composite_sentiment),
            'confidence': composite_confidence,
            'sources': source_sentiments,
            'signal_count': len(signals),
            'latest_timestamp': max(s.timestamp for s in signals)
        }
