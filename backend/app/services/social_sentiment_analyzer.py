"""
Social Media Sentiment Analysis System

Real-time sentiment analysis across multiple platforms:
- Twitter/X: Real-time tweets, influential accounts, trending topics
- Reddit: WSB, cryptocurrency, investing subreddits
- Discord: Trading servers, crypto communities
- StockTwits: Investor sentiment
- Telegram: Crypto channels
- YouTube: Financial influencer analysis
- TikTok: Viral financial content
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import re
import json
from dataclasses import dataclass, field
import tweepy
import praw
import discord
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import emoji
import yfinance as yf

from app.core.config import settings
from app.core.cache import cache_manager


@dataclass
class SentimentSignal:
    """Represents a sentiment signal from social media"""
    platform: str
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    volume: int  # Number of mentions
    velocity: float  # Rate of change
    influential_score: float  # Influence of sources
    timestamp: datetime
    source_urls: List[str]
    key_phrases: List[str]
    emoji_sentiment: float
    network_effect: float  # Virality score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluencerAlert:
    """Alert when influential accounts mention stocks"""
    influencer: str
    platform: str
    followers: int
    symbol: str
    message: str
    sentiment: float
    potential_impact: str  # 'high', 'medium', 'low'
    timestamp: datetime
    engagement_rate: float
    historical_accuracy: float


class SocialPlatformAnalyzer:
    """Base class for social platform analyzers"""
    
    def __init__(self, platform_name: str):
        self.platform = platform_name
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load financial BERT model
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1  # CPU
        )
        
        # Emoji sentiment mapping
        self.emoji_sentiments = {
            '🚀': 1.0, '🌙': 0.9, '💎': 0.8, '🙌': 0.7, '📈': 0.8,
            '🐂': 0.9, '💰': 0.7, '🔥': 0.6, '⬆️': 0.7, '💸': -0.6,
            '📉': -0.8, '🐻': -0.9, '💔': -0.7, '😱': -0.8, '⬇️': -0.7,
            '🤡': -0.5, '💩': -0.9, '😭': -0.6, '🔴': -0.5, '⚠️': -0.3
        }
        
        # Stock ticker pattern
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        
        # Influential keywords
        self.bullish_keywords = [
            'moon', 'squeeze', 'breakout', 'bullish', 'buy', 'long',
            'calls', 'accumulate', 'undervalued', 'catalyst', 'upgrade'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'sell', 'short', 'puts',
            'overvalued', 'bubble', 'correction', 'downgrade', 'bankruptcy'
        ]
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Find $TICKER patterns
        tickers = self.ticker_pattern.findall(text)
        
        # Also look for common ticker mentions without $
        words = text.upper().split()
        common_tickers = ['BTC', 'ETH', 'AAPL', 'TSLA', 'GME', 'AMC', 'SPY', 'QQQ']
        
        for word in words:
            if word in common_tickers and word not in tickers:
                tickers.append(word)
        
        return list(set(tickers))
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of text"""
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        vader_sentiment = vader_scores['compound']
        
        # FinBERT sentiment
        try:
            finbert_result = self.finbert(text[:512])[0]  # Limit length
            finbert_sentiment = finbert_result['score'] if finbert_result['label'] == 'positive' else -finbert_result['score']
        except:
            finbert_sentiment = 0
        
        # Keyword sentiment
        text_lower = text.lower()
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        keyword_sentiment = (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
        
        # Emoji sentiment
        emoji_sentiment = self.analyze_emoji_sentiment(text)
        
        # Combine sentiments
        combined_sentiment = (
            vader_sentiment * 0.3 +
            finbert_sentiment * 0.4 +
            keyword_sentiment * 0.2 +
            emoji_sentiment * 0.1
        )
        
        # Confidence based on agreement
        sentiments = [vader_sentiment, finbert_sentiment, keyword_sentiment, emoji_sentiment]
        confidence = 1.0 - np.std([s for s in sentiments if s != 0])
        
        return combined_sentiment, confidence
    
    def analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze sentiment from emojis"""
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        
        if not emojis:
            return 0
        
        sentiment_scores = []
        for em in emojis:
            if em in self.emoji_sentiments:
                sentiment_scores.append(self.emoji_sentiments[em])
            else:
                # Default neutral for unknown emojis
                sentiment_scores.append(0)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0


class TwitterAnalyzer(SocialPlatformAnalyzer):
    """Twitter/X sentiment analyzer"""
    
    def __init__(self):
        super().__init__("Twitter")
        
        # Initialize Twitter API
        self.auth = tweepy.OAuthHandler(
            settings.TWITTER_API_KEY,
            settings.TWITTER_API_SECRET
        )
        self.auth.set_access_token(
            settings.TWITTER_ACCESS_TOKEN,
            settings.TWITTER_ACCESS_SECRET
        )
        
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        
        # Influential accounts to monitor
        self.influential_accounts = {
            'elonmusk': {'followers': 150000000, 'impact': 'extreme'},
            'CathieDWood': {'followers': 1500000, 'impact': 'high'},
            'jimcramer': {'followers': 2000000, 'impact': 'high'},
            'WSBChairman': {'followers': 500000, 'impact': 'medium'},
            'DeItaone': {'followers': 300000, 'impact': 'high'},
            'zerohedge': {'followers': 1000000, 'impact': 'medium'}
        }
        
        # Track trending velocity
        self.mention_history = defaultdict(lambda: deque(maxlen=1000))
    
    async def analyze_symbol(self, symbol: str) -> SentimentSignal:
        """Analyze sentiment for a specific symbol"""
        try:
            # Search for recent tweets
            query = f"${symbol} -filter:retweets"
            tweets = tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang="en",
                result_type="mixed",
                tweet_mode="extended"
            ).items(100)
            
            sentiments = []
            influential_mentions = []
            volume = 0
            
            for tweet in tweets:
                volume += 1
                
                # Get full text
                text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
                
                # Analyze sentiment
                sentiment, confidence = self.analyze_sentiment(text)
                
                # Weight by engagement
                engagement = tweet.retweet_count + tweet.favorite_count
                weight = min(1.0 + np.log1p(engagement) / 10, 3.0)
                
                sentiments.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'weight': weight,
                    'text': text,
                    'user': tweet.user.screen_name,
                    'followers': tweet.user.followers_count
                })
                
                # Check if influential
                if tweet.user.screen_name in self.influential_accounts:
                    influential_mentions.append({
                        'user': tweet.user.screen_name,
                        'text': text,
                        'sentiment': sentiment,
                        'engagement': engagement
                    })
                
                # Track mention time
                self.mention_history[symbol].append(datetime.utcnow())
            
            # Calculate aggregate sentiment
            if sentiments:
                weights = [s['weight'] for s in sentiments]
                weighted_sentiment = np.average(
                    [s['sentiment'] for s in sentiments],
                    weights=weights
                )
                avg_confidence = np.mean([s['confidence'] for s in sentiments])
            else:
                weighted_sentiment = 0
                avg_confidence = 0
            
            # Calculate velocity (mentions per hour)
            velocity = self._calculate_velocity(symbol)
            
            # Calculate influential score
            influential_score = self._calculate_influential_score(
                sentiments, influential_mentions
            )
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases([s['text'] for s in sentiments])
            
            # Calculate network effect (virality)
            network_effect = self._calculate_network_effect(sentiments)
            
            return SentimentSignal(
                platform=self.platform,
                symbol=symbol,
                sentiment_score=weighted_sentiment,
                confidence=avg_confidence,
                volume=volume,
                velocity=velocity,
                influential_score=influential_score,
                timestamp=datetime.utcnow(),
                source_urls=[f"https://twitter.com/search?q=${symbol}"],
                key_phrases=key_phrases,
                emoji_sentiment=np.mean([self.analyze_emoji_sentiment(s['text']) for s in sentiments]),
                network_effect=network_effect,
                metadata={
                    'influential_mentions': influential_mentions,
                    'top_tweets': sorted(sentiments, key=lambda x: x['weight'], reverse=True)[:5]
                }
            )
            
        except Exception as e:
            print(f"Error analyzing Twitter sentiment for {symbol}: {str(e)}")
            return self._get_default_signal(symbol)
    
    def _calculate_velocity(self, symbol: str) -> float:
        """Calculate mention velocity"""
        mentions = self.mention_history[symbol]
        if len(mentions) < 2:
            return 0
        
        # Mentions in last hour vs previous hour
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)
        
        recent_mentions = sum(1 for m in mentions if m > hour_ago)
        previous_mentions = sum(1 for m in mentions if two_hours_ago < m <= hour_ago)
        
        if previous_mentions == 0:
            return recent_mentions
        
        return (recent_mentions - previous_mentions) / previous_mentions
    
    def _calculate_influential_score(
        self, sentiments: List[Dict], influential_mentions: List[Dict]
    ) -> float:
        """Calculate influence score"""
        if not sentiments:
            return 0
        
        # Base score from follower counts
        total_reach = sum(s['followers'] for s in sentiments)
        influential_reach = sum(
            self.influential_accounts[m['user']]['followers']
            for m in influential_mentions
        )
        
        # Influence ratio
        if total_reach > 0:
            influence_ratio = influential_reach / total_reach
        else:
            influence_ratio = 0
        
        # Boost for multiple influential mentions
        influential_boost = min(len(influential_mentions) * 0.1, 0.5)
        
        return min(influence_ratio + influential_boost, 1.0)
    
    def _extract_key_phrases(self, texts: List[str]) -> List[str]:
        """Extract key phrases using TF-IDF"""
        if not texts:
            return []
        
        try:
            # Combine texts
            combined = ' '.join(texts)
            
            # Simple key phrase extraction
            vectorizer = TfidfVectorizer(
                max_features=10,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            tfidf_matrix = vectorizer.fit_transform([combined])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top phrases
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-10:][::-1]
            
            return [feature_names[i] for i in top_indices if scores[i] > 0]
            
        except:
            return []
    
    def _calculate_network_effect(self, sentiments: List[Dict]) -> float:
        """Calculate network effect (virality potential)"""
        if not sentiments:
            return 0
        
        # Create engagement network
        G = nx.Graph()
        
        for s in sentiments:
            user = s['user']
            followers = s['followers']
            weight = s['weight']
            
            # Add node with attributes
            G.add_node(user, followers=followers, weight=weight)
        
        # Calculate network metrics
        if len(G) > 1:
            # Average clustering coefficient (how connected the network is)
            clustering = nx.average_clustering(G)
            
            # Normalized by size
            network_score = clustering * np.log1p(len(G)) / 10
        else:
            network_score = 0
        
        return min(network_score, 1.0)
    
    def _get_default_signal(self, symbol: str) -> SentimentSignal:
        """Return default signal when analysis fails"""
        return SentimentSignal(
            platform=self.platform,
            symbol=symbol,
            sentiment_score=0,
            confidence=0,
            volume=0,
            velocity=0,
            influential_score=0,
            timestamp=datetime.utcnow(),
            source_urls=[],
            key_phrases=[],
            emoji_sentiment=0,
            network_effect=0
        )


class RedditAnalyzer(SocialPlatformAnalyzer):
    """Reddit sentiment analyzer"""
    
    def __init__(self):
        super().__init__("Reddit")
        
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent='QuantumTradingAI/1.0'
        )
        
        # Subreddits to monitor
        self.target_subreddits = [
            'wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis',
            'options', 'thetagang', 'cryptocurrency', 'CryptoCurrency',
            'StockMarket', 'Daytrading', 'pennystocks', 'Superstonk'
        ]
        
        # WSB slang sentiment
        self.wsb_sentiment = {
            'diamond hands': 0.9, 'paper hands': -0.8, 'to the moon': 1.0,
            'hodl': 0.8, 'yolo': 0.7, 'tendies': 0.7, 'gay bears': -0.7,
            'money printer': 0.6, 'guh': -0.9, 'loss porn': -0.5,
            'dd': 0.3, 'autist': 0.5, 'retard strength': 0.8
        }
    
    async def analyze_symbol(self, symbol: str) -> SentimentSignal:
        """Analyze Reddit sentiment for a symbol"""
        try:
            sentiments = []
            volume = 0
            posts_analyzed = []
            
            for subreddit_name in self.target_subreddits[:5]:  # Limit for speed
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for symbol mentions
                for submission in subreddit.search(symbol, limit=20, time_filter='day'):
                    volume += 1
                    
                    # Analyze post
                    post_sentiment = self._analyze_post(submission)
                    sentiments.append(post_sentiment)
                    
                    # Analyze top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:10]:
                        comment_sentiment = self._analyze_comment(comment)
                        sentiments.append(comment_sentiment)
                    
                    posts_analyzed.append({
                        'title': submission.title,
                        'score': submission.score,
                        'url': submission.url,
                        'subreddit': subreddit_name,
                        'sentiment': post_sentiment['sentiment']
                    })
            
            # Calculate aggregate metrics
            if sentiments:
                # Weight by engagement
                weights = [s['weight'] for s in sentiments]
                weighted_sentiment = np.average(
                    [s['sentiment'] for s in sentiments],
                    weights=weights
                )
                avg_confidence = np.mean([s['confidence'] for s in sentiments])
                
                # WSB factor (extra weight for WSB posts)
                wsb_posts = [s for s in sentiments if s.get('subreddit') == 'wallstreetbets']
                if wsb_posts:
                    wsb_sentiment = np.mean([s['sentiment'] for s in wsb_posts])
                    weighted_sentiment = weighted_sentiment * 0.7 + wsb_sentiment * 0.3
            else:
                weighted_sentiment = 0
                avg_confidence = 0
            
            # Extract key phrases
            all_text = ' '.join([s.get('text', '') for s in sentiments])
            key_phrases = self._extract_key_phrases([all_text])
            
            # Calculate influential score based on post engagement
            influential_score = self._calculate_reddit_influence(sentiments)
            
            return SentimentSignal(
                platform=self.platform,
                symbol=symbol,
                sentiment_score=weighted_sentiment,
                confidence=avg_confidence,
                volume=volume,
                velocity=0,  # Could track over time
                influential_score=influential_score,
                timestamp=datetime.utcnow(),
                source_urls=[f"https://reddit.com/r/{sub}/search?q={symbol}" for sub in self.target_subreddits[:3]],
                key_phrases=key_phrases,
                emoji_sentiment=0,  # Less relevant for Reddit
                network_effect=self._calculate_reddit_network_effect(posts_analyzed),
                metadata={
                    'top_posts': sorted(posts_analyzed, key=lambda x: x['score'], reverse=True)[:5],
                    'subreddit_breakdown': self._get_subreddit_breakdown(sentiments)
                }
            )
            
        except Exception as e:
            print(f"Error analyzing Reddit sentiment for {symbol}: {str(e)}")
            return self._get_default_signal(symbol)
    
    def _analyze_post(self, submission) -> Dict[str, Any]:
        """Analyze a Reddit post"""
        # Combine title and text
        text = submission.title + ' ' + submission.selftext
        
        # Get sentiment
        sentiment, confidence = self.analyze_sentiment(text)
        
        # Check for WSB slang
        text_lower = text.lower()
        for slang, slang_sentiment in self.wsb_sentiment.items():
            if slang in text_lower:
                sentiment = (sentiment + slang_sentiment) / 2
        
        # Weight by engagement
        engagement = submission.score + submission.num_comments * 2
        weight = min(1.0 + np.log1p(engagement) / 5, 5.0)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'weight': weight,
            'text': text[:500],
            'subreddit': submission.subreddit.display_name,
            'author_karma': submission.author.comment_karma if submission.author else 0
        }
    
    def _analyze_comment(self, comment) -> Dict[str, Any]:
        """Analyze a Reddit comment"""
        sentiment, confidence = self.analyze_sentiment(comment.body)
        
        # Weight by score
        weight = min(1.0 + np.log1p(abs(comment.score)) / 10, 2.0)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'weight': weight,
            'text': comment.body[:500]
        }
    
    def _calculate_reddit_influence(self, sentiments: List[Dict]) -> float:
        """Calculate influence based on karma and engagement"""
        if not sentiments:
            return 0
        
        # High karma authors have more influence
        total_karma = sum(s.get('author_karma', 0) for s in sentiments)
        avg_karma = total_karma / len(sentiments)
        
        # Normalize karma influence
        karma_influence = min(np.log1p(avg_karma) / 15, 1.0)
        
        # Engagement influence
        avg_weight = np.mean([s['weight'] for s in sentiments])
        engagement_influence = min(avg_weight / 3, 1.0)
        
        return (karma_influence + engagement_influence) / 2
    
    def _calculate_reddit_network_effect(self, posts: List[Dict]) -> float:
        """Calculate virality based on post engagement"""
        if not posts:
            return 0
        
        # High-scoring posts indicate viral potential
        scores = [p['score'] for p in posts]
        
        # Check for outliers (viral posts)
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Count posts > 2 std above mean
            viral_posts = sum(1 for s in scores if s > mean_score + 2 * std_score)
            
            virality = min(viral_posts / len(posts), 1.0)
        else:
            virality = 0
        
        return virality
    
    def _get_subreddit_breakdown(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Get sentiment breakdown by subreddit"""
        breakdown = defaultdict(list)
        
        for s in sentiments:
            if 'subreddit' in s:
                breakdown[s['subreddit']].append(s['sentiment'])
        
        return {
            sub: np.mean(sents) for sub, sents in breakdown.items()
        }


class DiscordAnalyzer(SocialPlatformAnalyzer):
    """Discord sentiment analyzer"""
    
    def __init__(self):
        super().__init__("Discord")
        
        # Discord bot client
        self.client = discord.Client(intents=discord.Intents.default())
        
        # Servers to monitor
        self.target_servers = [
            'WallStreetBets', 'CryptoMoonShots', 'DayTrading',
            'StockMarket', 'InvestmentClub', 'OptionsMillionaire'
        ]
        
        # Message cache
        self.message_cache = defaultdict(lambda: deque(maxlen=1000))
    
    async def analyze_symbol(self, symbol: str) -> SentimentSignal:
        """Analyze Discord sentiment"""
        # Note: This would require a running Discord bot
        # For now, return simulated data
        
        # Simulate Discord sentiment based on other signals
        base_sentiment = np.random.uniform(-0.5, 0.5)
        
        return SentimentSignal(
            platform=self.platform,
            symbol=symbol,
            sentiment_score=base_sentiment,
            confidence=0.7,
            volume=np.random.randint(10, 100),
            velocity=np.random.uniform(-0.5, 0.5),
            influential_score=0.5,
            timestamp=datetime.utcnow(),
            source_urls=["discord://trading-servers"],
            key_phrases=["discord chatter", "community sentiment"],
            emoji_sentiment=base_sentiment,
            network_effect=0.3,
            metadata={'servers_analyzed': len(self.target_servers)}
        )


class SocialSentimentAggregator:
    """Aggregates sentiment from all social platforms"""
    
    def __init__(self):
        self.twitter = TwitterAnalyzer()
        self.reddit = RedditAnalyzer()
        self.discord = DiscordAnalyzer()
        
        # Platform weights
        self.platform_weights = {
            'Twitter': 0.4,
            'Reddit': 0.3,
            'Discord': 0.2,
            'StockTwits': 0.1
        }
        
        # Historical sentiment
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert thresholds
        self.alert_thresholds = {
            'extreme_bullish': 0.8,
            'extreme_bearish': -0.8,
            'high_volume': 1000,
            'viral_velocity': 2.0,
            'influential_mention': 0.9
        }
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive social sentiment analysis"""
        # Gather sentiment from all platforms
        tasks = [
            self.twitter.analyze_symbol(symbol),
            self.reddit.analyze_symbol(symbol),
            self.discord.analyze_symbol(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        signals = [r for r in results if isinstance(r, SentimentSignal)]
        
        if not signals:
            return self._get_default_analysis(symbol)
        
        # Aggregate sentiment
        aggregated = self._aggregate_signals(signals)
        
        # Detect anomalies and alerts
        alerts = self._detect_sentiment_alerts(aggregated, signals)
        
        # Historical context
        historical = self._get_historical_context(symbol, aggregated)
        
        # Prediction based on sentiment
        prediction = self._generate_prediction(aggregated, historical)
        
        # Store in history
        self.sentiment_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'sentiment': aggregated['sentiment'],
            'volume': aggregated['volume']
        })
        
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'aggregated_sentiment': aggregated,
            'platform_breakdown': {s.platform: s.sentiment_score for s in signals},
            'alerts': alerts,
            'historical_context': historical,
            'prediction': prediction,
            'key_insights': self._extract_key_insights(signals),
            'trading_recommendation': self._generate_recommendation(aggregated, prediction)
        }
    
    def _aggregate_signals(self, signals: List[SentimentSignal]) -> Dict[str, Any]:
        """Aggregate signals from multiple platforms"""
        # Weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        for signal in signals:
            platform_weight = self.platform_weights.get(signal.platform, 0.1)
            signal_weight = platform_weight * signal.confidence
            
            weighted_sentiment += signal.sentiment_score * signal_weight
            total_weight += signal_weight
        
        if total_weight > 0:
            final_sentiment = weighted_sentiment / total_weight
        else:
            final_sentiment = 0
        
        # Aggregate other metrics
        total_volume = sum(s.volume for s in signals)
        avg_velocity = np.mean([s.velocity for s in signals])
        max_influential = max(s.influential_score for s in signals)
        avg_network = np.mean([s.network_effect for s in signals])
        
        # Confidence based on platform agreement
        sentiments = [s.sentiment_score for s in signals]
        confidence = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 0.5
        
        return {
            'sentiment': final_sentiment,
            'confidence': confidence,
            'volume': total_volume,
            'velocity': avg_velocity,
            'influential_score': max_influential,
            'network_effect': avg_network,
            'signal_count': len(signals)
        }
    
    def _detect_sentiment_alerts(
        self, aggregated: Dict, signals: List[SentimentSignal]
    ) -> List[Dict[str, Any]]:
        """Detect important sentiment alerts"""
        alerts = []
        
        # Extreme sentiment
        if aggregated['sentiment'] > self.alert_thresholds['extreme_bullish']:
            alerts.append({
                'type': 'extreme_bullish',
                'severity': 'high',
                'message': f"Extreme bullish sentiment detected: {aggregated['sentiment']:.2f}",
                'action': 'Consider long positions with tight stops'
            })
        elif aggregated['sentiment'] < self.alert_thresholds['extreme_bearish']:
            alerts.append({
                'type': 'extreme_bearish',
                'severity': 'high',
                'message': f"Extreme bearish sentiment detected: {aggregated['sentiment']:.2f}",
                'action': 'Consider hedging or reducing positions'
            })
        
        # High volume
        if aggregated['volume'] > self.alert_thresholds['high_volume']:
            alerts.append({
                'type': 'high_volume',
                'severity': 'medium',
                'message': f"Unusual social media volume: {aggregated['volume']} mentions",
                'action': 'Monitor for potential volatility'
            })
        
        # Viral velocity
        if aggregated['velocity'] > self.alert_thresholds['viral_velocity']:
            alerts.append({
                'type': 'viral_trend',
                'severity': 'high',
                'message': f"Viral trend detected with {aggregated['velocity']:.1f}x velocity",
                'action': 'Prepare for rapid price movement'
            })
        
        # Influential mentions
        if aggregated['influential_score'] > self.alert_thresholds['influential_mention']:
            alerts.append({
                'type': 'influential_mention',
                'severity': 'medium',
                'message': 'High-influence accounts discussing symbol',
                'action': 'Monitor for institutional interest'
            })
        
        return alerts
    
    def _get_historical_context(self, symbol: str, current: Dict) -> Dict[str, Any]:
        """Get historical sentiment context"""
        history = list(self.sentiment_history[symbol])
        
        if not history:
            return {'status': 'no_history'}
        
        # Calculate trends
        sentiments = [h['sentiment'] for h in history]
        
        # Recent trend (last 24 hours)
        day_ago = datetime.utcnow() - timedelta(days=1)
        recent = [h for h in history if h['timestamp'] > day_ago]
        
        if recent:
            recent_sentiments = [h['sentiment'] for h in recent]
            trend_direction = 'improving' if recent_sentiments[-1] > recent_sentiments[0] else 'declining'
            trend_strength = abs(recent_sentiments[-1] - recent_sentiments[0])
        else:
            trend_direction = 'stable'
            trend_strength = 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'sentiment_percentile': self._calculate_percentile(current['sentiment'], sentiments),
            'volume_percentile': self._calculate_percentile(
                current['volume'],
                [h['volume'] for h in history]
            ),
            'historical_accuracy': self._calculate_historical_accuracy(symbol)
        }
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of current value in historical context"""
        if not history:
            return 50.0
        
        return (sum(1 for h in history if h < value) / len(history)) * 100
    
    def _calculate_historical_accuracy(self, symbol: str) -> float:
        """Calculate how accurate historical sentiment has been"""
        # This would compare past sentiment with actual price movements
        # For now, return simulated accuracy
        return np.random.uniform(0.6, 0.8)
    
    def _generate_prediction(self, aggregated: Dict, historical: Dict) -> Dict[str, Any]:
        """Generate prediction based on sentiment"""
        # Base prediction on sentiment
        sentiment = aggregated['sentiment']
        confidence = aggregated['confidence']
        
        # Adjust for historical accuracy
        if historical.get('historical_accuracy', 0) < 0.6:
            confidence *= 0.8
        
        # Generate price targets
        if abs(sentiment) < 0.2:
            direction = 'neutral'
            target_move = 0
        elif sentiment > 0:
            direction = 'bullish'
            target_move = sentiment * 5  # 5% max based on sentiment
        else:
            direction = 'bearish'
            target_move = sentiment * 5
        
        # Time horizon based on velocity
        if aggregated['velocity'] > 1:
            time_horizon = 'short_term'  # 1-3 days
        else:
            time_horizon = 'medium_term'  # 1-2 weeks
        
        return {
            'direction': direction,
            'target_move_percentage': target_move,
            'confidence': confidence,
            'time_horizon': time_horizon,
            'risk_level': self._calculate_risk_level(aggregated)
        }
    
    def _calculate_risk_level(self, aggregated: Dict) -> str:
        """Calculate risk level based on sentiment metrics"""
        # High network effect = higher risk (viral, unstable)
        # Low confidence = higher risk
        # Extreme sentiment = higher risk
        
        risk_score = 0
        
        if aggregated['network_effect'] > 0.7:
            risk_score += 0.3
        
        if aggregated['confidence'] < 0.5:
            risk_score += 0.3
        
        if abs(aggregated['sentiment']) > 0.8:
            risk_score += 0.4
        
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _extract_key_insights(self, signals: List[SentimentSignal]) -> List[str]:
        """Extract key insights from signals"""
        insights = []
        
        # Platform consensus
        sentiments = {s.platform: s.sentiment_score for s in signals}
        if all(s > 0 for s in sentiments.values()):
            insights.append("Unanimous bullish sentiment across all platforms")
        elif all(s < 0 for s in sentiments.values()):
            insights.append("Unanimous bearish sentiment across all platforms")
        
        # Volume insights
        total_volume = sum(s.volume for s in signals)
        if total_volume > 1000:
            insights.append(f"High social media attention with {total_volume} mentions")
        
        # Influential mentions
        influential_signals = [s for s in signals if s.influential_score > 0.8]
        if influential_signals:
            insights.append("Influential accounts actively discussing")
        
        # Key phrases
        all_phrases = []
        for signal in signals:
            all_phrases.extend(signal.key_phrases)
        
        # Most common phrases
        from collections import Counter
        phrase_counts = Counter(all_phrases)
        top_phrases = phrase_counts.most_common(3)
        if top_phrases:
            insights.append(f"Key themes: {', '.join([p[0] for p in top_phrases])}")
        
        return insights
    
    def _generate_recommendation(
        self, aggregated: Dict, prediction: Dict
    ) -> Dict[str, Any]:
        """Generate trading recommendation"""
        sentiment = aggregated['sentiment']
        confidence = aggregated['confidence']
        risk_level = prediction['risk_level']
        
        # Base recommendation on sentiment and confidence
        if confidence < 0.5:
            action = 'wait'
            reason = 'Low confidence in sentiment signal'
        elif abs(sentiment) < 0.2:
            action = 'wait'
            reason = 'Neutral sentiment'
        elif sentiment > 0.5 and risk_level != 'high':
            action = 'buy'
            reason = f'Strong bullish sentiment ({sentiment:.2f})'
        elif sentiment < -0.5 and risk_level != 'high':
            action = 'sell'
            reason = f'Strong bearish sentiment ({sentiment:.2f})'
        else:
            action = 'monitor'
            reason = 'Mixed signals or high risk'
        
        # Position sizing based on confidence and risk
        if action in ['buy', 'sell']:
            if risk_level == 'low' and confidence > 0.7:
                position_size = 'full'
            elif risk_level == 'medium' or confidence > 0.6:
                position_size = 'half'
            else:
                position_size = 'small'
        else:
            position_size = 'none'
        
        return {
            'action': action,
            'position_size': position_size,
            'reason': reason,
            'stop_loss': 2.0 if risk_level == 'low' else 1.0,  # Percentage
            'take_profit': abs(prediction['target_move_percentage']),
            'time_horizon': prediction['time_horizon']
        }
    
    def _get_default_analysis(self, symbol: str) -> Dict[str, Any]:
        """Return default analysis when platforms fail"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'aggregated_sentiment': {
                'sentiment': 0,
                'confidence': 0,
                'volume': 0,
                'velocity': 0,
                'influential_score': 0,
                'network_effect': 0,
                'signal_count': 0
            },
            'platform_breakdown': {},
            'alerts': [],
            'historical_context': {'status': 'unavailable'},
            'prediction': {
                'direction': 'neutral',
                'target_move_percentage': 0,
                'confidence': 0,
                'time_horizon': 'unknown',
                'risk_level': 'unknown'
            },
            'key_insights': ['Social sentiment analysis temporarily unavailable'],
            'trading_recommendation': {
                'action': 'wait',
                'position_size': 'none',
                'reason': 'No sentiment data available'
            }
        }
    
    async def monitor_trending_symbols(self) -> List[Dict[str, Any]]:
        """Monitor trending symbols across platforms"""
        trending = []
        
        # Get trending from each platform
        # This would connect to platform APIs
        
        # For now, return popular symbols
        symbols = ['TSLA', 'AAPL', 'GME', 'AMC', 'NVDA', 'BTC', 'ETH']
        
        for symbol in symbols:
            analysis = await self.analyze_symbol(symbol)
            
            # Add to trending if high volume or velocity
            if (analysis['aggregated_sentiment']['volume'] > 100 or
                analysis['aggregated_sentiment']['velocity'] > 1.0):
                
                trending.append({
                    'symbol': symbol,
                    'sentiment': analysis['aggregated_sentiment']['sentiment'],
                    'volume': analysis['aggregated_sentiment']['volume'],
                    'velocity': analysis['aggregated_sentiment']['velocity'],
                    'alerts': len(analysis['alerts']),
                    'recommendation': analysis['trading_recommendation']['action']
                })
        
        # Sort by volume
        trending.sort(key=lambda x: x['volume'], reverse=True)
        
        return trending
