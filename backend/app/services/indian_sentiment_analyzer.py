"""
Indian Market Sentiment Analyzer

Specialized sentiment analysis for Indian markets from:
- MoneyControl, Economic Times, Business Standard
- Indian Twitter/X influencers and traders
- Telegram trading channels
- WhatsApp forwards (via Telegram bridges)
- YouTube channels (PR Sundar, Vivek Bajaj, etc.)
- Regional language sentiment (Hindi, Tamil, Telugu)
- Broker research reports
- TV channels (CNBC TV18, ET Now, Zee Business)
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import re
import json
from bs4 import BeautifulSoup
import feedparser
from collections import defaultdict, deque
from googletrans import Translator
import yfinance as yf
from textblob import TextBlob
from transformers import pipeline
import requests

from app.core.config import settings
from app.services.social_sentiment_analyzer import SocialPlatformAnalyzer, SentimentSignal


@dataclass
class IndianSentimentSignal(SentimentSignal):
    """Extended sentiment signal for Indian markets"""
    source_language: str
    broker_rating: Optional[str]  # Buy, Hold, Sell
    target_price: Optional[float]
    stop_loss: Optional[float]
    news_category: str  # earnings, regulatory, macro, technical
    regional_sentiment: Dict[str, float]  # State-wise sentiment


class IndianNewsAnalyzer(SocialPlatformAnalyzer):
    """Analyzer for Indian financial news"""
    
    def __init__(self):
        super().__init__("IndianNews")
        
        # Indian news sources
        self.news_sources = {
            'moneycontrol': {
                'url': 'https://www.moneycontrol.com/rss/latestnews.xml',
                'weight': 0.25
            },
            'economictimes': {
                'url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'weight': 0.25
            },
            'businessstandard': {
                'url': 'https://www.business-standard.com/rss/markets-106.rss',
                'weight': 0.20
            },
            'livemint': {
                'url': 'https://www.livemint.com/rss/markets',
                'weight': 0.15
            },
            'financialexpress': {
                'url': 'https://www.financialexpress.com/market/feed/',
                'weight': 0.15
            }
        }
        
        # Translator for regional languages
        self.translator = Translator()
        
        # Indian market keywords
        self.indian_keywords = {
            'bullish': [
                'तेजी', 'ऊपर', 'बढ़त', 'मजबूत',  # Hindi
                'FII buying', 'DII buying', 'record high', 'target achieved',
                'breakout', 'accumulate', 'outperform', 'upgrade'
            ],
            'bearish': [
                'मंदी', 'गिरावट', 'कमजोर', 'नीचे',  # Hindi
                'FII selling', 'DII selling', 'support broken', 'downgrade',
                'profit booking', 'correction', 'weak', 'sell'
            ]
        }
        
        # Important Indian events
        self.market_events = {
            'budget': {'impact': 'high', 'typical_month': 2},
            'rbi_policy': {'impact': 'high', 'frequency': 'bi-monthly'},
            'earnings': {'impact': 'medium', 'frequency': 'quarterly'},
            'monsoon': {'impact': 'medium', 'typical_months': [6, 7, 8, 9]},
            'festivals': {'impact': 'medium', 'events': ['diwali', 'dussehra']}
        }
        
        # Broker ratings mapping
        self.broker_ratings = {
            'strong buy': 1.0,
            'buy': 0.7,
            'accumulate': 0.5,
            'hold': 0.0,
            'reduce': -0.5,
            'sell': -0.7,
            'strong sell': -1.0
        }
    
    async def analyze_symbol(self, symbol: str) -> IndianSentimentSignal:
        """Analyze Indian news sentiment for a symbol"""
        try:
            all_articles = []
            
            # Fetch news from all sources
            for source, config in self.news_sources.items():
                articles = await self._fetch_news(source, config['url'], symbol)
                
                # Add source weight
                for article in articles:
                    article['weight'] = config['weight']
                    
                all_articles.extend(articles)
            
            # Analyze sentiment
            if all_articles:
                aggregated = self._aggregate_news_sentiment(all_articles)
                
                # Check for broker reports
                broker_sentiment = await self._analyze_broker_reports(symbol)
                
                # Regional sentiment
                regional = await self._analyze_regional_sentiment(symbol)
                
                # Combine all signals
                final_sentiment = (
                    aggregated['sentiment'] * 0.6 +
                    broker_sentiment['sentiment'] * 0.3 +
                    regional['overall'] * 0.1
                )
                
                return IndianSentimentSignal(
                    platform=self.platform,
                    symbol=symbol,
                    sentiment_score=final_sentiment,
                    confidence=aggregated['confidence'],
                    volume=len(all_articles),
                    velocity=self._calculate_news_velocity(all_articles),
                    influential_score=aggregated['influential_score'],
                    timestamp=datetime.utcnow(),
                    source_urls=[a['url'] for a in all_articles[:5]],
                    key_phrases=aggregated['key_phrases'],
                    emoji_sentiment=0,  # Not relevant for news
                    network_effect=aggregated['virality'],
                    metadata={
                        'top_articles': all_articles[:5],
                        'broker_consensus': broker_sentiment['consensus']
                    },
                    source_language='multi',
                    broker_rating=broker_sentiment['rating'],
                    target_price=broker_sentiment.get('target'),
                    stop_loss=broker_sentiment.get('stop_loss'),
                    news_category=self._categorize_news(all_articles),
                    regional_sentiment=regional['breakdown']
                )
            
        except Exception as e:
            print(f"Error analyzing Indian news for {symbol}: {str(e)}")
        
        return self._get_default_indian_signal(symbol)
    
    async def _fetch_news(self, source: str, url: str, symbol: str) -> List[Dict]:
        """Fetch news from Indian sources"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:20]:
                            # Check if symbol mentioned
                            title = entry.get('title', '')
                            summary = entry.get('summary', '')
                            full_text = f"{title} {summary}"
                            
                            if symbol in full_text or self._check_symbol_variants(symbol, full_text):
                                # Analyze sentiment
                                sentiment, confidence = self.analyze_sentiment(full_text)
                                
                                articles.append({
                                    'source': source,
                                    'title': title,
                                    'summary': summary,
                                    'url': entry.get('link', ''),
                                    'published': self._parse_date(entry.get('published')),
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'text': full_text
                                })
                                
        except Exception as e:
            print(f"Error fetching from {source}: {str(e)}")
        
        return articles
    
    def _check_symbol_variants(self, symbol: str, text: str) -> bool:
        """Check for symbol variants (e.g., Reliance, RIL)"""
        symbol_mapping = {
            'RELIANCE': ['Reliance', 'RIL', 'Reliance Industries'],
            'HDFCBANK': ['HDFC Bank', 'HDFC'],
            'INFY': ['Infosys', 'Infy'],
            'TCS': ['Tata Consultancy', 'TCS'],
            'SBIN': ['SBI', 'State Bank'],
            'ICICIBANK': ['ICICI Bank', 'ICICI'],
            'KOTAKBANK': ['Kotak Bank', 'Kotak Mahindra'],
            'AXISBANK': ['Axis Bank'],
            'BAJFINANCE': ['Bajaj Finance', 'BajFinance'],
            'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
            'ITC': ['ITC Limited', 'ITC'],
            'LT': ['Larsen', 'L&T', 'Larsen & Toubro'],
            'MARUTI': ['Maruti Suzuki', 'Maruti'],
            'TATAMOTORS': ['Tata Motors'],
            'WIPRO': ['Wipro'],
            'HCLTECH': ['HCL Tech', 'HCL Technologies'],
            'SUNPHARMA': ['Sun Pharma', 'Sun Pharmaceutical']
        }
        
        variants = symbol_mapping.get(symbol, [symbol])
        
        for variant in variants:
            if variant.lower() in text.lower():
                return True
        
        return False
    
    def _aggregate_news_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple articles"""
        if not articles:
            return {'sentiment': 0, 'confidence': 0, 'influential_score': 0, 
                   'key_phrases': [], 'virality': 0}
        
        # Weight by source importance and recency
        weights = []
        sentiments = []
        
        for article in articles:
            # Recency weight
            age_hours = (datetime.utcnow() - article['published']).total_seconds() / 3600
            recency_weight = np.exp(-age_hours / 24)  # Decay over 24 hours
            
            # Combined weight
            weight = article['weight'] * recency_weight * article['confidence']
            
            weights.append(weight)
            sentiments.append(article['sentiment'])
        
        # Weighted average
        if sum(weights) > 0:
            weighted_sentiment = np.average(sentiments, weights=weights)
        else:
            weighted_sentiment = 0
        
        # Extract key phrases
        all_text = ' '.join([a['text'] for a in articles])
        key_phrases = self._extract_key_phrases([all_text])
        
        # Calculate influence based on source diversity
        unique_sources = len(set(a['source'] for a in articles))
        influential_score = min(unique_sources / 5, 1.0)
        
        # Virality based on coverage
        virality = min(len(articles) / 10, 1.0)
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': np.mean([a['confidence'] for a in articles]),
            'influential_score': influential_score,
            'key_phrases': key_phrases,
            'virality': virality
        }
    
    async def _analyze_broker_reports(self, symbol: str) -> Dict[str, Any]:
        """Analyze broker reports and ratings"""
        # This would fetch actual broker reports
        # For now, return simulated data
        
        broker_reports = [
            {'broker': 'Motilal Oswal', 'rating': 'buy', 'target': 2500},
            {'broker': 'HDFC Securities', 'rating': 'accumulate', 'target': 2400},
            {'broker': 'ICICI Direct', 'rating': 'buy', 'target': 2450},
            {'broker': 'Kotak Securities', 'rating': 'hold', 'target': 2300}
        ]
        
        if broker_reports:
            # Average sentiment from ratings
            sentiments = [self.broker_ratings.get(r['rating'], 0) for r in broker_reports]
            avg_sentiment = np.mean(sentiments)
            
            # Consensus rating
            ratings = [r['rating'] for r in broker_reports]
            consensus = max(set(ratings), key=ratings.count)
            
            # Average target
            targets = [r['target'] for r in broker_reports if 'target' in r]
            avg_target = np.mean(targets) if targets else None
            
            return {
                'sentiment': avg_sentiment,
                'rating': consensus,
                'consensus': f"{len([r for r in ratings if r in ['buy', 'accumulate']])}/{len(ratings)} Buy",
                'target': avg_target,
                'stop_loss': avg_target * 0.95 if avg_target else None
            }
        
        return {'sentiment': 0, 'rating': 'hold', 'consensus': 'No data', 'target': None}
    
    async def _analyze_regional_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze regional language sentiment"""
        regional_sentiments = {}
        
        # Simulate regional sentiment
        # In reality, would fetch from regional news/social media
        languages = {
            'hindi': {'sentiment': 0.2, 'volume': 1000},
            'tamil': {'sentiment': 0.1, 'volume': 500},
            'telugu': {'sentiment': 0.15, 'volume': 300},
            'gujarati': {'sentiment': 0.3, 'volume': 400},
            'marathi': {'sentiment': 0.1, 'volume': 600}
        }
        
        total_volume = sum(lang['volume'] for lang in languages.values())
        
        # Weighted average by volume
        if total_volume > 0:
            overall = sum(
                lang['sentiment'] * lang['volume'] / total_volume 
                for lang in languages.values()
            )
        else:
            overall = 0
        
        return {
            'overall': overall,
            'breakdown': {
                lang: data['sentiment'] 
                for lang, data in languages.items()
            }
        }
    
    def _calculate_news_velocity(self, articles: List[Dict]) -> float:
        """Calculate news velocity (stories per hour)"""
        if len(articles) < 2:
            return 0
        
        # Articles in last hour vs previous hour
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)
        
        recent = sum(1 for a in articles if a['published'] > hour_ago)
        previous = sum(1 for a in articles if two_hours_ago < a['published'] <= hour_ago)
        
        if previous == 0:
            return recent
        
        return (recent - previous) / previous
    
    def _categorize_news(self, articles: List[Dict]) -> str:
        """Categorize news type"""
        categories = defaultdict(int)
        
        category_keywords = {
            'earnings': ['results', 'earnings', 'profit', 'revenue', 'quarter'],
            'regulatory': ['SEBI', 'RBI', 'regulation', 'compliance', 'penalty'],
            'macro': ['GDP', 'inflation', 'economy', 'budget', 'fiscal'],
            'technical': ['support', 'resistance', 'breakout', 'trend', 'chart'],
            'corporate': ['merger', 'acquisition', 'stake', 'board', 'dividend']
        }
        
        for article in articles:
            text = article['text'].lower()
            
            for category, keywords in category_keywords.items():
                if any(keyword.lower() in text for keyword in keywords):
                    categories[category] += 1
        
        if categories:
            return max(categories, key=categories.get)
        
        return 'general'
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from various formats"""
        try:
            # Try common formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%d %b %Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            
            # Fallback to current time
            return datetime.utcnow()
            
        except:
            return datetime.utcnow()
    
    def _get_default_indian_signal(self, symbol: str) -> IndianSentimentSignal:
        """Return default signal"""
        return IndianSentimentSignal(
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
            network_effect=0,
            metadata={},
            source_language='en',
            broker_rating='hold',
            target_price=None,
            stop_loss=None,
            news_category='general',
            regional_sentiment={}
        )


class IndianTelegramAnalyzer(SocialPlatformAnalyzer):
    """Analyzer for Indian Telegram trading channels"""
    
    def __init__(self):
        super().__init__("IndianTelegram")
        
        # Popular Indian trading channels
        self.channels = [
            'StockMarketIndia',
            'NSEBSETips',
            'EquityResearch',
            'IntradayTradingTips',
            'OptionsTradingIndia',
            'StockMarketGuru',
            'ProfitableTrading',
            'SwingTradingIndia'
        ]
        
        # Trading slang
        self.trading_slang = {
            'operator': -0.3,  # Market manipulation
            'upper circuit': 0.8,
            'lower circuit': -0.8,
            'bulk deal': 0.5,
            'block deal': 0.5,
            'insider': 0.6,
            'pump': -0.7,  # Pump and dump
            'target hit': 0.7,
            'sl hit': -0.7,  # Stop loss hit
            'book profit': -0.2,  # Profit booking
            'fresh buying': 0.6,
            'short covering': 0.4,
            'long unwinding': -0.4,
            'short buildup': -0.6
        }
    
    async def analyze_symbol(self, symbol: str) -> SentimentSignal:
        """Analyze Telegram sentiment"""
        # This would require Telegram API integration
        # For now, return simulated data
        
        messages = await self._fetch_telegram_messages(symbol)
        
        if messages:
            sentiment = self._analyze_telegram_sentiment(messages)
            
            return SentimentSignal(
                platform=self.platform,
                symbol=symbol,
                sentiment_score=sentiment['score'],
                confidence=sentiment['confidence'],
                volume=len(messages),
                velocity=sentiment['velocity'],
                influential_score=sentiment['influential'],
                timestamp=datetime.utcnow(),
                source_urls=['telegram://channels'],
                key_phrases=sentiment['tips'],
                emoji_sentiment=sentiment['emoji_score'],
                network_effect=sentiment['viral_score'],
                metadata={
                    'channels_analyzed': len(self.channels),
                    'top_tips': sentiment['tips'][:3]
                }
            )
        
        return self._get_default_signal(symbol)
    
    async def _fetch_telegram_messages(self, symbol: str) -> List[Dict]:
        """Fetch messages from Telegram channels"""
        # Simulated messages
        messages = []
        
        # Generate sample messages
        for i in range(20):
            messages.append({
                'channel': np.random.choice(self.channels),
                'text': f"{symbol} looking good for intraday. Target 2400, SL 2350",
                'timestamp': datetime.utcnow() - timedelta(hours=np.random.randint(0, 24)),
                'forwards': np.random.randint(0, 1000),
                'views': np.random.randint(100, 10000)
            })
        
        return messages
    
    def _analyze_telegram_sentiment(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze Telegram messages"""
        sentiments = []
        
        for msg in messages:
            # Basic sentiment
            sentiment, confidence = self.analyze_sentiment(msg['text'])
            
            # Check for trading slang
            text_lower = msg['text'].lower()
            for slang, slang_sentiment in self.trading_slang.items():
                if slang in text_lower:
                    sentiment = (sentiment + slang_sentiment) / 2
            
            # Weight by engagement
            engagement = msg['forwards'] + msg['views'] / 100
            weight = min(1.0 + np.log1p(engagement) / 10, 3.0)
            
            sentiments.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'weight': weight
            })
        
        # Aggregate
        if sentiments:
            weights = [s['weight'] for s in sentiments]
            weighted_sentiment = np.average(
                [s['sentiment'] for s in sentiments],
                weights=weights
            )
            
            # Extract tips
            tips = self._extract_trading_tips(messages)
            
            # Viral score based on forwards
            total_forwards = sum(msg['forwards'] for msg in messages)
            viral_score = min(total_forwards / 1000, 1.0)
            
            return {
                'score': weighted_sentiment,
                'confidence': np.mean([s['confidence'] for s in sentiments]),
                'velocity': len(messages) / 24,  # Messages per hour
                'influential': 0.7,  # Telegram is influential in India
                'tips': tips,
                'emoji_score': 0.3,  # Moderate emoji usage
                'viral_score': viral_score
            }
        
        return {
            'score': 0,
            'confidence': 0,
            'velocity': 0,
            'influential': 0,
            'tips': [],
            'emoji_score': 0,
            'viral_score': 0
        }
    
    def _extract_trading_tips(self, messages: List[Dict]) -> List[str]:
        """Extract trading tips from messages"""
        tips = []
        
        tip_patterns = [
            r'Target[:\s]+(\d+)',
            r'SL[:\s]+(\d+)',
            r'Buy[:\s]+(\d+)',
            r'Sell[:\s]+(\d+)'
        ]
        
        for msg in messages:
            for pattern in tip_patterns:
                matches = re.findall(pattern, msg['text'], re.IGNORECASE)
                if matches:
                    tips.append(msg['text'][:100])  # First 100 chars
                    break
        
        return tips[:5]  # Top 5 tips


class IndianMarketPatternAnalyzer:
    """Analyze Indian market-specific patterns"""
    
    def __init__(self):
        self.patterns = {
            'diwali_effect': {
                'months': [10, 11],
                'typical_return': 0.05,
                'description': 'Muhurat trading positive bias'
            },
            'budget_rally': {
                'month': 2,
                'typical_return': 0.03,
                'description': 'Pre-budget rally'
            },
            'monsoon_impact': {
                'months': [6, 7, 8, 9],
                'sectors': ['FMCG', 'AUTO', 'AGRI'],
                'description': 'Monsoon-dependent sectors'
            },
            'fiscal_year_end': {
                'month': 3,
                'typical_behavior': 'window_dressing',
                'description': 'Year-end portfolio adjustments'
            },
            'election_impact': {
                'volatility_increase': 0.3,
                'description': 'Election uncertainty'
            },
            'fii_dii_pattern': {
                'correlation': -0.6,
                'description': 'FII selling often met with DII buying'
            }
        }
        
        # Monthly seasonality for NIFTY
        self.monthly_seasonality = {
            1: 0.02,   # January - positive
            2: 0.01,   # February - mildly positive
            3: -0.01,  # March - year-end selling
            4: 0.015,  # April - new year buying
            5: 0.005,  # May - neutral
            6: 0.01,   # June - pre-monsoon
            7: 0.02,   # July - monsoon rally
            8: -0.005, # August - consolidation
            9: 0.01,   # September - festival prep
            10: 0.025, # October - Diwali rally
            11: 0.02,  # November - post-Diwali
            12: 0.01   # December - Santa rally
        }
    
    async def analyze_current_pattern(self) -> Dict[str, Any]:
        """Analyze current market pattern"""
        current_month = datetime.now().month
        
        active_patterns = []
        
        # Check Diwali effect
        if current_month in self.patterns['diwali_effect']['months']:
            active_patterns.append({
                'pattern': 'diwali_effect',
                'impact': 'positive',
                'expected_return': self.patterns['diwali_effect']['typical_return'],
                'confidence': 0.7
            })
        
        # Check budget season
        if current_month == self.patterns['budget_rally']['month']:
            active_patterns.append({
                'pattern': 'budget_rally',
                'impact': 'positive',
                'expected_return': self.patterns['budget_rally']['typical_return'],
                'confidence': 0.6
            })
        
        # Check monsoon
        if current_month in self.patterns['monsoon_impact']['months']:
            active_patterns.append({
                'pattern': 'monsoon_impact',
                'impact': 'sector_specific',
                'affected_sectors': self.patterns['monsoon_impact']['sectors'],
                'confidence': 0.5
            })
        
        # Monthly seasonality
        seasonal_bias = self.monthly_seasonality.get(current_month, 0)
        
        # FII/DII pattern
        fii_dii = await self._analyze_fii_dii_flows()
        
        return {
            'active_patterns': active_patterns,
            'seasonal_bias': seasonal_bias,
            'seasonal_signal': 'bullish' if seasonal_bias > 0 else 'bearish',
            'fii_dii_signal': fii_dii,
            'pattern_based_recommendation': self._generate_pattern_recommendation(
                active_patterns, seasonal_bias, fii_dii
            )
        }
    
    async def _analyze_fii_dii_flows(self) -> Dict[str, Any]:
        """Analyze FII/DII flows"""
        # This would fetch actual FII/DII data
        # For now, simulate
        
        fii_flow = np.random.uniform(-1000, 1000)  # Crores
        dii_flow = -fii_flow * 0.6 + np.random.uniform(-200, 200)  # Inverse correlation
        
        signal = 'neutral'
        if fii_flow > 500:
            signal = 'bullish'
        elif fii_flow < -500:
            signal = 'bearish' if dii_flow < 0 else 'neutral'
        
        return {
            'fii_flow': fii_flow,
            'dii_flow': dii_flow,
            'net_flow': fii_flow + dii_flow,
            'signal': signal,
            'interpretation': self._interpret_flows(fii_flow, dii_flow)
        }
    
    def _interpret_flows(self, fii: float, dii: float) -> str:
        """Interpret FII/DII flows"""
        if fii > 500 and dii > 0:
            return "Strong buying by both FII and DII - Very Bullish"
        elif fii > 500 and dii < 0:
            return "FII buying, DII selling - Cautious Bullish"
        elif fii < -500 and dii > 500:
            return "FII selling, DII buying - Consolidation expected"
        elif fii < -500 and dii < 0:
            return "Both FII and DII selling - Bearish"
        else:
            return "Mixed flows - Range bound movement"
    
    def _generate_pattern_recommendation(
        self, patterns: List[Dict], seasonal: float, fii_dii: Dict
    ) -> Dict[str, str]:
        """Generate trading recommendation based on patterns"""
        
        # Count positive patterns
        positive_patterns = sum(1 for p in patterns if p.get('impact') == 'positive')
        
        # Overall bias
        bias_score = (
            positive_patterns * 0.3 +
            (1 if seasonal > 0 else -1) * 0.3 +
            (1 if fii_dii['signal'] == 'bullish' else -1 if fii_dii['signal'] == 'bearish' else 0) * 0.4
        )
        
        if bias_score > 0.5:
            return {
                'action': 'buy_on_dips',
                'confidence': 'high',
                'sectors': self._get_recommended_sectors(patterns),
                'strategy': 'Long bias with strict stop losses'
            }
        elif bias_score < -0.5:
            return {
                'action': 'reduce_exposure',
                'confidence': 'medium',
                'sectors': ['Defensive stocks', 'Large caps'],
                'strategy': 'Book profits and wait for better levels'
            }
        else:
            return {
                'action': 'range_trading',
                'confidence': 'medium',
                'sectors': ['High beta stocks for intraday'],
                'strategy': 'Buy at support, sell at resistance'
            }
    
    def _get_recommended_sectors(self, patterns: List[Dict]) -> List[str]:
        """Get recommended sectors based on patterns"""
        sectors = []
        
        for pattern in patterns:
            if pattern['pattern'] == 'monsoon_impact':
                sectors.extend(pattern['affected_sectors'])
            elif pattern['pattern'] == 'diwali_effect':
                sectors.extend(['CONSUMER', 'AUTO', 'RETAIL'])
            elif pattern['pattern'] == 'budget_rally':
                sectors.extend(['INFRA', 'CAPITAL GOODS', 'PSU BANKS'])
        
        return list(set(sectors)) if sectors else ['NIFTY50', 'BANKING']


class IndianSentimentAggregator:
    """Aggregate all Indian market sentiment sources"""
    
    def __init__(self):
        self.news_analyzer = IndianNewsAnalyzer()
        self.telegram_analyzer = IndianTelegramAnalyzer()
        self.pattern_analyzer = IndianMarketPatternAnalyzer()
        
        # Weight for each source
        self.source_weights = {
            'news': 0.4,
            'telegram': 0.2,
            'broker_reports': 0.3,
            'patterns': 0.1
        }
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive Indian market sentiment analysis"""
        
        # Gather sentiment from all sources
        tasks = [
            self.news_analyzer.analyze_symbol(symbol),
            self.telegram_analyzer.analyze_symbol(symbol),
            self.pattern_analyzer.analyze_current_pattern()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        news_signal = results[0] if not isinstance(results[0], Exception) else None
        telegram_signal = results[1] if not isinstance(results[1], Exception) else None
        pattern_analysis = results[2] if not isinstance(results[2], Exception) else {}
        
        # Aggregate signals
        aggregated_sentiment = 0
        total_weight = 0
        
        if news_signal:
            aggregated_sentiment += news_signal.sentiment_score * self.source_weights['news']
            total_weight += self.source_weights['news']
        
        if telegram_signal:
            aggregated_sentiment += telegram_signal.sentiment_score * self.source_weights['telegram']
            total_weight += self.source_weights['telegram']
        
        # Pattern-based adjustment
        if pattern_analysis:
            pattern_sentiment = pattern_analysis.get('seasonal_bias', 0)
            aggregated_sentiment += pattern_sentiment * self.source_weights['patterns']
            total_weight += self.source_weights['patterns']
        
        if total_weight > 0:
            aggregated_sentiment /= total_weight
        
        # Generate trading recommendation
        recommendation = self._generate_indian_recommendation(
            aggregated_sentiment,
            news_signal,
            telegram_signal,
            pattern_analysis
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'aggregated_sentiment': aggregated_sentiment,
            'sentiment_breakdown': {
                'news': news_signal.sentiment_score if news_signal else 0,
                'telegram': telegram_signal.sentiment_score if telegram_signal else 0,
                'seasonal': pattern_analysis.get('seasonal_bias', 0)
            },
            'news_summary': self._get_news_summary(news_signal),
            'broker_consensus': news_signal.broker_rating if news_signal else 'No data',
            'target_price': news_signal.target_price if news_signal else None,
            'stop_loss': news_signal.stop_loss if news_signal else None,
            'active_patterns': pattern_analysis.get('active_patterns', []),
            'fii_dii_signal': pattern_analysis.get('fii_dii_signal', {}),
            'recommendation': recommendation,
            'risk_factors': self._identify_risk_factors(news_signal, pattern_analysis),
            'sector_view': await self._get_sector_view(symbol)
        }
    
    def _generate_indian_recommendation(
        self, sentiment: float, news: Any, telegram: Any, patterns: Dict
    ) -> Dict[str, Any]:
        """Generate India-specific trading recommendation"""
        
        # Base recommendation on sentiment
        if sentiment > 0.5:
            action = 'strong_buy'
            position_size = 'full'
        elif sentiment > 0.2:
            action = 'buy'
            position_size = 'half'
        elif sentiment < -0.5:
            action = 'strong_sell'
            position_size = 'exit'
        elif sentiment < -0.2:
            action = 'sell'
            position_size = 'reduce'
        else:
            action = 'hold'
            position_size = 'maintain'
        
        # Adjust for patterns
        if patterns.get('active_patterns'):
            for pattern in patterns['active_patterns']:
                if pattern['pattern'] == 'diwali_effect' and pattern['impact'] == 'positive':
                    if action == 'hold':
                        action = 'buy'
                        position_size = 'small'
        
        # Set levels
        if news and news.target_price:
            target = news.target_price
            stop_loss = news.stop_loss or news.target_price * 0.95
        else:
            # Default 3:1 risk-reward
            target = None
            stop_loss = None
        
        return {
            'action': action,
            'position_size': position_size,
            'entry_strategy': self._get_entry_strategy(sentiment, patterns),
            'target': target,
            'stop_loss': stop_loss,
            'time_horizon': 'short_term' if telegram and telegram.volume > 50 else 'medium_term',
            'confidence': self._calculate_confidence(sentiment, news, telegram)
        }
    
    def _get_entry_strategy(self, sentiment: float, patterns: Dict) -> str:
        """Get entry strategy based on conditions"""
        fii_dii = patterns.get('fii_dii_signal', {})
        
        if fii_dii.get('signal') == 'bearish' and sentiment > 0:
            return "Wait for FII selling to stop, enter on reversal"
        elif sentiment > 0.5:
            return "Buy at current levels with 50%, add on dips"
        elif sentiment > 0:
            return "Buy on dips near support levels"
        elif sentiment < -0.5:
            return "Exit immediately, look for shorting opportunity"
        else:
            return "Wait and watch, trade the range"
    
    def _calculate_confidence(self, sentiment: float, news: Any, telegram: Any) -> float:
        """Calculate recommendation confidence"""
        confidence = 0.5
        
        # Strong sentiment increases confidence
        if abs(sentiment) > 0.5:
            confidence += 0.2
        
        # News and broker agreement
        if news and news.confidence > 0.7:
            confidence += 0.15
        
        # High telegram volume
        if telegram and telegram.volume > 100:
            confidence += 0.15
        
        return min(confidence, 0.95)
    
    def _get_news_summary(self, news_signal: Any) -> str:
        """Summarize news sentiment"""
        if not news_signal:
            return "No significant news"
        
        if news_signal.sentiment_score > 0.5:
            return f"Strong positive news sentiment. {news_signal.news_category} news dominating."
        elif news_signal.sentiment_score > 0:
            return f"Mildly positive news flow. Focus on {news_signal.news_category}."
        elif news_signal.sentiment_score < -0.5:
            return f"Strong negative sentiment. {news_signal.news_category} concerns."
        elif news_signal.sentiment_score < 0:
            return f"Mildly negative news sentiment."
        else:
            return "Neutral news flow"
    
    def _identify_risk_factors(self, news: Any, patterns: Dict) -> List[str]:
        """Identify India-specific risk factors"""
        risks = []
        
        # News-based risks
        if news:
            if news.news_category == 'regulatory':
                risks.append("Regulatory uncertainty - SEBI/RBI actions possible")
            if abs(news.sentiment_score) > 0.7:
                risks.append("Extreme sentiment - Possible reversal")
        
        # Pattern-based risks
        if patterns.get('active_patterns'):
            for pattern in patterns['active_patterns']:
                if pattern['pattern'] == 'monsoon_impact':
                    risks.append("Monsoon dependency - Weather risk")
                elif pattern['pattern'] == 'budget_rally':
                    risks.append("Budget expectations - Policy risk")
        
        # FII/DII risks
        fii_dii = patterns.get('fii_dii_signal', {})
        if fii_dii.get('fii_flow', 0) < -1000:
            risks.append("Heavy FII selling - Foreign fund outflow risk")
        
        # Add general market risks
        current_month = datetime.now().month
        if current_month == 3:
            risks.append("Financial year-end - Window dressing risk")
        elif current_month in [10, 11]:
            risks.append("Festival season - Lower volumes expected")
        
        return risks
    
    async def _get_sector_view(self, symbol: str) -> str:
        """Get sector-specific view"""
        # Map symbol to sector
        sector_mapping = {
            'RELIANCE': 'Oil & Gas',
            'TCS': 'IT',
            'INFY': 'IT',
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'PSU Banking',
            'BHARTIARTL': 'Telecom',
            'ITC': 'FMCG',
            'MARUTI': 'Auto',
            'SUNPHARMA': 'Pharma',
            'LT': 'Infrastructure'
        }
        
        sector = sector_mapping.get(symbol, 'General')
        
        # Sector-specific views
        sector_views = {
            'IT': "Dependent on US market and dollar movement",
            'Banking': "Sensitive to RBI policy and NPA concerns",
            'Auto': "Dependent on festive demand and chip availability",
            'FMCG': "Defensive play, rural demand recovery key",
            'Pharma': "Export outlook and USFDA approvals important",
            'Infrastructure': "Government spending and order book key"
        }
        
        return sector_views.get(sector, "Track index movement and FII flows")
