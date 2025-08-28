"""
GPT-4 Market Analysis Service

This revolutionary service integrates GPT-4 to provide unprecedented market analysis capabilities:
- Real-time news interpretation and impact assessment
- Complex pattern recognition in market narratives
- Earnings call transcript analysis
- Federal Reserve statement interpretation
- Global macro event analysis
- Market sentiment extraction from multiple sources
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import aiohttp
import numpy as np
from pydantic import BaseModel
import openai
from transformers import pipeline
import yfinance as yf
from textblob import TextBlob
import feedparser
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re
from urllib.parse import quote

from app.core.config import settings
from app.core.cache import cache_manager
from app.services.market_data import MarketDataService
from app.ml.model_manager import ModelManager


class MarketInsight(BaseModel):
    """Structured market insight from GPT-4 analysis"""
    timestamp: datetime
    source: str
    title: str
    summary: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    impact_prediction: str  # 'bullish', 'bearish', 'neutral'
    affected_sectors: List[str]
    affected_tickers: List[str]
    key_metrics: Dict[str, Any]
    trading_implications: str
    risk_factors: List[str]
    time_horizon: str  # 'immediate', 'short-term', 'long-term'


class GPTMarketAnalyzer:
    """Advanced market analysis using GPT-4 and multiple data sources"""
    
    def __init__(self):
        self.market_data = MarketDataService()
        self.model_manager = ModelManager()
        
        # Initialize OpenAI
        openai.api_key = settings.OPENAI_API_KEY
        
        # News sources
        self.news_sources = {
            'reuters': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'wsj': 'https://feeds.wsj.com/rss/RSSMarketsMain.xml',
            'ft': 'https://www.ft.com/rss/markets',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/'
        }
        
        # Sentiment models
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1  # CPU
        )
        
        # Market impact keywords
        self.impact_keywords = {
            'bullish': [
                'surge', 'rally', 'breakout', 'upgrade', 'beat expectations',
                'strong earnings', 'record high', 'expansion', 'growth',
                'positive guidance', 'merger', 'acquisition', 'breakthrough'
            ],
            'bearish': [
                'crash', 'plunge', 'downgrade', 'miss expectations', 'layoffs',
                'recession', 'decline', 'warning', 'investigation', 'lawsuit',
                'bankruptcy', 'default', 'sanctions', 'crisis'
            ]
        }
        
        # Sector mappings
        self.sector_keywords = {
            'technology': ['tech', 'software', 'AI', 'semiconductor', 'cloud'],
            'finance': ['bank', 'financial', 'insurance', 'fintech', 'payment'],
            'healthcare': ['pharma', 'biotech', 'medical', 'drug', 'FDA'],
            'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'nuclear'],
            'consumer': ['retail', 'consumer', 'shopping', 'brand', 'luxury'],
            'industrial': ['manufacturing', 'industrial', 'aerospace', 'defense']
        }
    
    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """Comprehensive market analysis using GPT-4"""
        try:
            # Gather data from multiple sources
            news_data = await self._fetch_all_news()
            market_data = await self._fetch_market_indicators()
            sentiment_data = await self._analyze_aggregate_sentiment()
            
            # GPT-4 analysis
            gpt_analysis = await self._gpt4_market_analysis(
                news_data, market_data, sentiment_data
            )
            
            # Combine insights
            insights = await self._generate_actionable_insights(gpt_analysis)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'market_state': gpt_analysis['market_state'],
                'key_themes': gpt_analysis['themes'],
                'risk_level': gpt_analysis['risk_level'],
                'opportunities': insights['opportunities'],
                'warnings': insights['warnings'],
                'sector_analysis': gpt_analysis['sectors'],
                'recommended_actions': insights['actions'],
                'confidence': gpt_analysis['confidence']
            }
            
        except Exception as e:
            print(f"Market analysis error: {str(e)}")
            return self._get_fallback_analysis()
    
    async def _fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from multiple sources"""
        all_news = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_rss_feed(session, source, url)
                for source, url in self.news_sources.items()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
        
        # Sort by timestamp and limit
        all_news.sort(key=lambda x: x['published'], reverse=True)
        return all_news[:100]  # Top 100 most recent
    
    async def _fetch_rss_feed(
        self, session: aiohttp.ClientSession, source: str, url: str
    ) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    news_items = []
                    for entry in feed.entries[:20]:  # Limit per source
                        news_items.append({
                            'source': source,
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': datetime.fromtimestamp(
                                entry.get('published_parsed', datetime.now().timestamp())
                            )
                        })
                    
                    return news_items
        except Exception as e:
            print(f"Error fetching {source}: {str(e)}")
            return []
    
    async def _fetch_market_indicators(self) -> Dict[str, Any]:
        """Fetch key market indicators"""
        indicators = {}
        
        # Major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX', '^TNX']
        
        for symbol in indices:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    indicators[symbol] = {
                        'price': hist['Close'][-1],
                        'change': hist['Close'][-1] - hist['Open'][-1],
                        'volume': hist['Volume'][-1],
                        'high': hist['High'][-1],
                        'low': hist['Low'][-1]
                    }
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
        
        return indicators
    
    async def _analyze_aggregate_sentiment(self) -> Dict[str, float]:
        """Analyze sentiment across multiple sources"""
        # This would connect to Twitter, Reddit, StockTwits APIs
        # For now, return sample data
        return {
            'overall': 0.15,  # Slightly bullish
            'twitter': 0.20,
            'reddit': 0.10,
            'news': 0.15,
            'options_flow': 0.25  # Based on put/call ratio
        }
    
    async def _gpt4_market_analysis(
        self, news: List[Dict], indicators: Dict, sentiment: Dict
    ) -> Dict[str, Any]:
        """Core GPT-4 analysis"""
        # Prepare context for GPT-4
        news_summary = self._summarize_news(news[:20])  # Top 20 news
        
        prompt = f"""
        You are an elite quantitative analyst. Analyze the current market conditions:
        
        LATEST NEWS:
        {news_summary}
        
        MARKET INDICATORS:
        S&P 500: {indicators.get('^GSPC', {}).get('change', 'N/A')}
        VIX: {indicators.get('^VIX', {}).get('price', 'N/A')}
        10Y Treasury: {indicators.get('^TNX', {}).get('price', 'N/A')}
        
        SENTIMENT SCORES:
        Overall: {sentiment['overall']}
        Twitter: {sentiment['twitter']}
        Options Flow: {sentiment['options_flow']}
        
        Provide a JSON response with:
        1. market_state: 'bullish', 'bearish', or 'neutral'
        2. themes: list of 5 key market themes
        3. risk_level: 'low', 'medium', 'high', 'extreme'
        4. sectors: dict of sector outlooks
        5. catalysts: upcoming events that could move markets
        6. confidence: your confidence level (0-1)
        7. rationale: brief explanation
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a top-tier market analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"GPT-4 analysis error: {str(e)}")
            # Fallback analysis
            return self._get_basic_analysis(news, indicators, sentiment)
    
    def _summarize_news(self, news_items: List[Dict]) -> str:
        """Summarize news for GPT-4 context"""
        summary = []
        for item in news_items:
            summary.append(f"- [{item['source']}] {item['title']}")
        return "\n".join(summary)
    
    async def _generate_actionable_insights(
        self, gpt_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate specific trading recommendations"""
        insights = {
            'opportunities': [],
            'warnings': [],
            'actions': []
        }
        
        # Based on market state and themes
        market_state = gpt_analysis.get('market_state', 'neutral')
        risk_level = gpt_analysis.get('risk_level', 'medium')
        
        if market_state == 'bullish' and risk_level in ['low', 'medium']:
            insights['opportunities'].extend([
                "Consider increasing long exposure in growth sectors",
                "Look for breakout patterns in technology stocks",
                "Options: Sell put spreads on major indices"
            ])
            insights['actions'].extend([
                "Scale into positions gradually",
                "Set trailing stops at 2-3%",
                "Monitor VIX for volatility spikes"
            ])
            
        elif market_state == 'bearish' or risk_level in ['high', 'extreme']:
            insights['warnings'].extend([
                "High risk of market correction",
                "Defensive positioning recommended",
                "Watch for support levels on major indices"
            ])
            insights['actions'].extend([
                "Reduce position sizes",
                "Consider protective puts",
                "Increase cash allocation to 30-40%"
            ])
        
        # Sector-specific insights
        sectors = gpt_analysis.get('sectors', {})
        for sector, outlook in sectors.items():
            if outlook == 'bullish':
                insights['opportunities'].append(
                    f"{sector.title()} sector showing strength - screen for leaders"
                )
        
        return insights
    
    async def analyze_earnings_impact(
        self, ticker: str, transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze earnings call impact using GPT-4"""
        # Fetch recent earnings data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get earnings history
        earnings = stock.earnings_history
        
        if transcript:
            # Analyze transcript with GPT-4
            prompt = f"""
            Analyze this earnings call transcript for {ticker}:
            
            {transcript[:3000]}  # Limit context
            
            Extract:
            1. Key positive points
            2. Key concerns
            3. Future guidance sentiment
            4. Management confidence level
            5. Likely stock impact (% move and direction)
            6. Trading recommendation
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        
        return {
            'ticker': ticker,
            'analysis': 'Earnings analysis requires transcript'
        }
    
    async def detect_market_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual market patterns and anomalies"""
        anomalies = []
        
        # Check for unusual volume patterns
        volume_anomalies = await self._detect_volume_anomalies()
        anomalies.extend(volume_anomalies)
        
        # Check for correlation breakdowns
        correlation_anomalies = await self._detect_correlation_breaks()
        anomalies.extend(correlation_anomalies)
        
        # Check for unusual options activity
        options_anomalies = await self._detect_options_anomalies()
        anomalies.extend(options_anomalies)
        
        # GPT-4 interpretation
        if anomalies:
            interpretation = await self._interpret_anomalies(anomalies)
            for i, anomaly in enumerate(anomalies):
                anomaly['interpretation'] = interpretation.get(i, '')
        
        return anomalies
    
    async def _detect_volume_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual volume patterns"""
        anomalies = []
        
        # Check major stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='30d')
                
                if len(hist) > 0:
                    avg_volume = hist['Volume'].mean()
                    today_volume = hist['Volume'][-1]
                    
                    if today_volume > avg_volume * 3:  # 3x normal volume
                        anomalies.append({
                            'type': 'volume_spike',
                            'symbol': symbol,
                            'metric': today_volume / avg_volume,
                            'severity': 'high' if today_volume > avg_volume * 5 else 'medium',
                            'timestamp': datetime.utcnow()
                        })
                        
            except Exception as e:
                print(f"Error checking {symbol}: {str(e)}")
        
        return anomalies
    
    async def _detect_correlation_breaks(self) -> List[Dict[str, Any]]:
        """Detect correlation breakdowns between related assets"""
        anomalies = []
        
        # Check key correlations
        correlations = [
            ('SPY', 'QQQ'),  # S&P vs NASDAQ
            ('GLD', 'TLT'),  # Gold vs Bonds
            ('VIX', 'SPY'),  # Volatility vs Market (inverse)
        ]
        
        for asset1, asset2 in correlations:
            try:
                # Calculate rolling correlation
                data1 = yf.download(asset1, period='60d', progress=False)['Close']
                data2 = yf.download(asset2, period='60d', progress=False)['Close']
                
                if len(data1) > 30 and len(data2) > 30:
                    # 30-day rolling correlation
                    rolling_corr = data1.rolling(30).corr(data2)
                    current_corr = rolling_corr.iloc[-1]
                    avg_corr = rolling_corr.mean()
                    
                    if abs(current_corr - avg_corr) > 0.4:  # Significant deviation
                        anomalies.append({
                            'type': 'correlation_break',
                            'pair': f"{asset1}/{asset2}",
                            'current': current_corr,
                            'average': avg_corr,
                            'deviation': abs(current_corr - avg_corr),
                            'severity': 'high' if abs(current_corr - avg_corr) > 0.6 else 'medium'
                        })
                        
            except Exception as e:
                print(f"Error checking correlation {asset1}/{asset2}: {str(e)}")
        
        return anomalies
    
    async def _detect_options_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual options activity"""
        # This would connect to options flow data
        # For now, return sample anomalies
        return [
            {
                'type': 'unusual_options',
                'symbol': 'NVDA',
                'strike': 500,
                'expiry': '2024-02-16',
                'volume': 50000,
                'open_interest': 10000,
                'put_call': 'call',
                'severity': 'high'
            }
        ]
    
    async def _interpret_anomalies(
        self, anomalies: List[Dict]
    ) -> Dict[int, str]:
        """Use GPT-4 to interpret anomalies"""
        anomaly_desc = json.dumps(anomalies, default=str)
        
        prompt = f"""
        Interpret these market anomalies and provide trading implications:
        
        {anomaly_desc}
        
        For each anomaly, explain:
        1. What it likely means
        2. Potential causes
        3. Trading implications
        4. Risk level
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Parse interpretations
            interpretations = {}
            lines = response.choices[0].message.content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    interpretations[i] = line.strip()
            
            return interpretations
            
        except Exception as e:
            print(f"Interpretation error: {str(e)}")
            return {}
    
    def _get_basic_analysis(
        self, news: List[Dict], indicators: Dict, sentiment: Dict
    ) -> Dict[str, Any]:
        """Fallback analysis without GPT-4"""
        # Simple rule-based analysis
        sentiment_score = sentiment.get('overall', 0)
        vix = indicators.get('^VIX', {}).get('price', 20)
        
        if sentiment_score > 0.3 and vix < 20:
            market_state = 'bullish'
            risk_level = 'low'
        elif sentiment_score < -0.3 or vix > 30:
            market_state = 'bearish'
            risk_level = 'high'
        else:
            market_state = 'neutral'
            risk_level = 'medium'
        
        return {
            'market_state': market_state,
            'themes': ['Market uncertainty', 'Volatility concerns'],
            'risk_level': risk_level,
            'sectors': {'technology': 'neutral', 'finance': 'neutral'},
            'catalysts': ['Fed meeting', 'Earnings season'],
            'confidence': 0.6,
            'rationale': 'Basic analysis based on sentiment and VIX'
        }
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback when analysis fails"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'market_state': 'neutral',
            'key_themes': ['Analysis temporarily unavailable'],
            'risk_level': 'medium',
            'opportunities': [],
            'warnings': ['Unable to fetch complete market data'],
            'sector_analysis': {},
            'recommended_actions': ['Maintain current positions', 'Wait for clearer signals'],
            'confidence': 0.3
        }
    
    async def generate_trading_signals(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific trading signals based on analysis"""
        signals = []
        
        market_state = analysis.get('market_state', 'neutral')
        risk_level = analysis.get('risk_level', 'medium')
        confidence = analysis.get('confidence', 0.5)
        
        # Generate signals based on conditions
        if market_state == 'bullish' and confidence > 0.7:
            signals.extend([
                {
                    'type': 'long',
                    'symbol': 'SPY',
                    'entry': 'market',
                    'size': 0.2,  # 20% of portfolio
                    'stop_loss': -2.0,  # 2% stop
                    'take_profit': 5.0,  # 5% target
                    'confidence': confidence,
                    'rationale': 'Broad market bullish signal'
                },
                {
                    'type': 'option',
                    'symbol': 'QQQ',
                    'strategy': 'bull_call_spread',
                    'expiry': '30d',
                    'size': 0.1,
                    'confidence': confidence * 0.9,
                    'rationale': 'Tech sector momentum play'
                }
            ])
        
        elif market_state == 'bearish' and risk_level == 'high':
            signals.extend([
                {
                    'type': 'hedge',
                    'symbol': 'VXX',
                    'entry': 'market',
                    'size': 0.05,  # 5% hedge
                    'confidence': confidence,
                    'rationale': 'Volatility hedge for protection'
                },
                {
                    'type': 'reduce',
                    'target_cash': 0.4,  # 40% cash
                    'confidence': confidence,
                    'rationale': 'Risk reduction in uncertain market'
                }
            ])
        
        return signals
