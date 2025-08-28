"""Market data service for fetching real-time stock and options data."""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.config import settings
from app.models.stock import Stock, PriceHistory
from app.models.option import Option

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching and processing market data."""
    
    def __init__(self):
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        
        if settings.ALPHA_VANTAGE_API_KEY:
            self.alpha_vantage_ts = TimeSeries(
                key=settings.ALPHA_VANTAGE_API_KEY,
                output_format='pandas'
            )
            self.alpha_vantage_fd = FundamentalData(
                key=settings.ALPHA_VANTAGE_API_KEY,
                output_format='pandas'
            )
    
    async def fetch_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time stock data."""
        try:
            # Use yfinance for real-time data
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if not info or 'currentPrice' not in info:
                return None
            
            # Get current quote
            quote = ticker.history(period="1d", interval="1m").tail(1)
            
            stock_data = {
                'symbol': symbol.upper(),
                'name': info.get('longName', info.get('shortName', symbol)),
                'exchange': info.get('exchange', ''),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open_price': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'week_52_high': info.get('fiftyTwoWeekHigh', 0),
                'week_52_low': info.get('fiftyTwoWeekLow', 0),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'is_optionable': len(ticker.options) > 0 if hasattr(ticker, 'options') else False,
            }
            
            # Calculate change
            if stock_data['previous_close'] and stock_data['current_price']:
                stock_data['change_amount'] = stock_data['current_price'] - stock_data['previous_close']
                stock_data['change_percent'] = (stock_data['change_amount'] / stock_data['previous_close']) * 100
            else:
                stock_data['change_amount'] = 0
                stock_data['change_percent'] = 0
            
            # Add technical indicators
            await self._add_technical_indicators(ticker, stock_data)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    async def _add_technical_indicators(self, ticker: yf.Ticker, stock_data: Dict[str, Any]):
        """Add technical indicators to stock data."""
        try:
            # Get historical data for indicators
            hist = ticker.history(period="3mo")
            
            if not hist.empty:
                close_prices = hist['Close']
                
                # RSI
                stock_data['rsi'] = self._calculate_rsi(close_prices)
                
                # Moving averages
                stock_data['sma_20'] = close_prices.rolling(window=20).mean().iloc[-1]
                stock_data['sma_50'] = close_prices.rolling(window=50).mean().iloc[-1]
                if len(close_prices) >= 200:
                    stock_data['sma_200'] = close_prices.rolling(window=200).mean().iloc[-1]
                
                # MACD
                ema_12 = close_prices.ewm(span=12, adjust=False).mean()
                ema_26 = close_prices.ewm(span=26, adjust=False).mean()
                stock_data['macd'] = (ema_12 - ema_26).iloc[-1]
                
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else None
    
    async def fetch_price_history(
        self, 
        symbol: str, 
        interval: str = "1d", 
        period: str = "1mo"
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch historical price data."""
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Map our intervals to yfinance intervals
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "60m",
                "1d": "1d",
                "1w": "1wk",
                "1M": "1mo"
            }
            
            yf_interval = interval_map.get(interval, "1d")
            
            # Fetch data
            hist = ticker.history(period=period, interval=yf_interval)
            
            if hist.empty:
                return None
            
            # Convert to list of dicts
            history_data = []
            for index, row in hist.iterrows():
                history_data.append({
                    'timestamp': index.to_pydatetime(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': int(row['Volume']),
                    'adjusted_close': row.get('Adj Close', row['Close'])
                })
            
            return history_data
            
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return None
    
    async def fetch_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch option chain data."""
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return None
            
            all_calls = []
            all_puts = []
            
            # Fetch options for each expiration
            for exp in expirations[:5]:  # Limit to first 5 expirations for performance
                opt = ticker.option_chain(exp)
                
                # Process calls
                for _, row in opt.calls.iterrows():
                    call_data = self._process_option_data(row, symbol, exp, 'call')
                    if call_data:
                        all_calls.append(call_data)
                
                # Process puts
                for _, row in opt.puts.iterrows():
                    put_data = self._process_option_data(row, symbol, exp, 'put')
                    if put_data:
                        all_puts.append(put_data)
            
            # Get unique strikes
            all_strikes = sorted(list(set(
                [opt['strike'] for opt in all_calls + all_puts]
            )))
            
            return {
                'underlying_symbol': symbol.upper(),
                'calls': all_calls,
                'puts': all_puts,
                'expirations': expirations,
                'strikes': all_strikes
            }
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return None
    
    def _process_option_data(
        self, 
        row: pd.Series, 
        symbol: str, 
        expiration: str, 
        option_type: str
    ) -> Optional[Dict[str, Any]]:
        """Process option data from yfinance."""
        try:
            return {
                'symbol': row.get('contractSymbol', ''),
                'underlying_symbol': symbol.upper(),
                'strike': float(row['strike']),
                'expiration': expiration,
                'option_type': option_type,
                'bid': float(row.get('bid', 0)),
                'ask': float(row.get('ask', 0)),
                'last_price': float(row.get('lastPrice', 0)),
                'volume': int(row.get('volume', 0)),
                'open_interest': int(row.get('openInterest', 0)),
                'implied_volatility': float(row.get('impliedVolatility', 0)),
                'delta': None,  # Would need to calculate
                'gamma': None,
                'theta': None,
                'vega': None,
                'rho': None,
            }
        except Exception as e:
            logger.error(f"Error processing option data: {e}")
            return None
    
    async def fetch_market_indicators(self) -> List[Dict[str, Any]]:
        """Fetch major market indicators."""
        indicators = []
        
        # Major indices
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX',
            '^TNX': '10-Year Treasury',
            'GC=F': 'Gold',
            'CL=F': 'Crude Oil',
            'BTC-USD': 'Bitcoin',
        }
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Open'].iloc[0]
                    
                    indicators.append({
                        'symbol': symbol,
                        'name': name,
                        'value': current,
                        'change_amount': current - previous,
                        'change_percent': ((current - previous) / previous) * 100
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching indicator {symbol}: {e}")
        
        return indicators
    
    async def stream_quotes(
        self, 
        symbols: List[str], 
        callback: callable
    ):
        """Stream real-time quotes for multiple symbols."""
        while True:
            try:
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m").tail(1)
                    
                    if not hist.empty:
                        quote = {
                            'symbol': symbol,
                            'price': hist['Close'].iloc[-1],
                            'volume': hist['Volume'].iloc[-1],
                            'timestamp': hist.index[-1].to_pydatetime()
                        }
                        
                        await callback(quote)
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error streaming quotes: {e}")
                await asyncio.sleep(10)  # Wait longer on error
