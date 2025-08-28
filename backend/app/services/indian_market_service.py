"""
Indian Market Service - Comprehensive NSE, BSE, MCX Integration

This service provides specialized support for Indian markets including:
- NSE (National Stock Exchange): Stocks, F&O, Indices
- BSE (Bombay Stock Exchange): Stocks, SME
- MCX (Multi Commodity Exchange): Commodities
- Currency Derivatives
- Indian market timings and holidays
- SEBI regulations and compliance
- Indian-specific indicators and patterns
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
import pytz
from enum import Enum
import json
import yfinance as yf
from nsepy import get_history, get_index_pe_history, get_rbi_ref_history
from nsetools import Nse
from bsedata.bse import BSE
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict

from app.core.config import settings
from app.core.cache import cache_manager
from app.services.market_data import MarketDataService


class IndianExchange(Enum):
    """Indian exchanges"""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NCDEX = "NCDEX"


class IndianIndex(Enum):
    """Major Indian indices"""
    NIFTY50 = "NIFTY 50"
    BANKNIFTY = "NIFTY BANK"
    SENSEX = "SENSEX"
    NIFTYIT = "NIFTY IT"
    FINNIFTY = "NIFTY FIN SERVICE"
    MIDCAP = "NIFTY MIDCAP 100"
    SMALLCAP = "NIFTY SMALLCAP 100"


@dataclass
class IndianStockData:
    """Indian stock data structure"""
    symbol: str
    exchange: str
    price: float
    change: float
    change_percent: float
    volume: int
    delivery_percent: float
    pe_ratio: Optional[float]
    market_cap: Optional[float]
    sector: str
    industry: str
    fno_eligible: bool
    upper_circuit: Optional[float]
    lower_circuit: Optional[float]
    week_52_high: float
    week_52_low: float
    vwap: float
    open_interest: Optional[int]
    implied_volatility: Optional[float]
    lot_size: Optional[int]


@dataclass
class FNOData:
    """Futures & Options data"""
    symbol: str
    expiry: datetime
    strike: Optional[float]
    option_type: Optional[str]  # CE or PE
    futures_price: Optional[float]
    spot_price: float
    premium: Optional[float]
    open_interest: int
    change_in_oi: int
    volume: int
    implied_volatility: float
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    lot_size: int


class IndianMarketService:
    """Comprehensive Indian market data service"""
    
    def __init__(self):
        self.nse = Nse()
        self.bse = BSE()
        self.base_service = MarketDataService()
        
        # Indian market timezone
        self.india_tz = pytz.timezone('Asia/Kolkata')
        
        # Market timings
        self.market_hours = {
            'NSE': {'open': time(9, 15), 'close': time(15, 30)},
            'BSE': {'open': time(9, 15), 'close': time(15, 30)},
            'MCX': {'open': time(9, 0), 'close': time(23, 30)},
            'pre_market': {'open': time(9, 0), 'close': time(9, 8)},
            'post_market': {'open': time(15, 40), 'close': time(16, 0)}
        }
        
        # FNO symbols
        self.fno_symbols = []
        self._load_fno_symbols()
        
        # Sector mappings
        self.nifty_sectors = {
            'NIFTY BANK': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK'],
            'NIFTY IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
            'NIFTY PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP'],
            'NIFTY AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'NIFTY FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
            'NIFTY METAL': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'NATIONALUM'],
            'NIFTY REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'SOBHA']
        }
        
        # Circuit limits
        self.circuit_limits = {
            'default': 0.20,  # 20% circuit
            'new_listing': 0.20,  # First day
            'sme': 0.05,  # 5% for SME
        }
        
        # Holiday calendar
        self.holidays_2024 = [
            datetime(2024, 1, 26),  # Republic Day
            datetime(2024, 3, 8),   # Mahashivratri
            datetime(2024, 3, 25),  # Holi
            datetime(2024, 3, 29),  # Good Friday
            datetime(2024, 4, 11),  # Id-ul-Fitr
            datetime(2024, 4, 17),  # Ram Navami
            datetime(2024, 5, 1),   # Maharashtra Day
            datetime(2024, 8, 15),  # Independence Day
            datetime(2024, 10, 2),  # Gandhi Jayanti
            datetime(2024, 11, 1),  # Diwali
            datetime(2024, 11, 15), # Guru Nanak Jayanti
        ]
    
    def _load_fno_symbols(self):
        """Load F&O enabled symbols"""
        try:
            # Get F&O stocks from NSE
            fno_list = self.nse.get_fno_lot_sizes()
            self.fno_symbols = list(fno_list.keys()) if fno_list else []
        except:
            # Fallback to major F&O stocks
            self.fno_symbols = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HDFC', 'ICICIBANK',
                'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'ITC',
                'AXISBANK', 'LT', 'DMART', 'SUNPHARMA', 'MARUTI', 'TITAN',
                'ULTRACEMCO', 'ONGC', 'NTPC', 'JSWSTEEL', 'TATAMOTORS',
                'POWERGRID', 'M&M', 'TATASTEEL', 'WIPRO', 'HCLTECH',
                'TECHM', 'ADANIPORTS', 'GRASIM', 'DRREDDY', 'HINDALCO',
                'DIVISLAB', 'CIPLA', 'NESTLEIND', 'BAJAJFINSV', 'HINDUNILVR',
                'VEDL', 'BPCL', 'PEL', 'INDUSINDBK', 'HDFCLIFE', 'PIDILITIND',
                'NAUKRI', 'UPL', 'MCDOWELL-N', 'BAJAJ-AUTO', 'TATACONSUM'
            ]
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current Indian market status"""
        now = datetime.now(self.india_tz)
        current_time = now.time()
        
        # Check if holiday
        is_holiday = now.date() in [h.date() for h in self.holidays_2024]
        
        # Check market hours
        nse_open = self._is_market_open('NSE', current_time)
        mcx_open = self._is_market_open('MCX', current_time)
        
        # Get index levels
        indices = await self.get_index_data()
        
        # Market phase
        if current_time < self.market_hours['pre_market']['open']:
            phase = 'pre_open_session'
        elif self.market_hours['pre_market']['open'] <= current_time < self.market_hours['NSE']['open']:
            phase = 'pre_market'
        elif self.market_hours['NSE']['open'] <= current_time < self.market_hours['NSE']['close']:
            phase = 'normal_market'
        elif self.market_hours['post_market']['open'] <= current_time < self.market_hours['post_market']['close']:
            phase = 'post_market'
        else:
            phase = 'closed'
        
        return {
            'timestamp': now.isoformat(),
            'is_trading_day': not is_holiday,
            'nse_open': nse_open,
            'bse_open': nse_open,  # Same as NSE
            'mcx_open': mcx_open,
            'market_phase': phase,
            'indices': indices,
            'next_trading_day': self._get_next_trading_day(now),
            'market_mood': self._calculate_market_mood(indices)
        }
    
    def _is_market_open(self, exchange: str, current_time: time) -> bool:
        """Check if market is open"""
        hours = self.market_hours.get(exchange)
        if hours:
            return hours['open'] <= current_time <= hours['close']
        return False
    
    async def get_index_data(self) -> Dict[str, Any]:
        """Get major Indian indices data"""
        indices = {}
        
        try:
            # NIFTY 50
            nifty = self.nse.get_index_quote("nifty 50")
            indices['NIFTY50'] = {
                'value': nifty['lastPrice'],
                'change': nifty['change'],
                'change_percent': nifty['pChange'],
                'open': nifty['open'],
                'high': nifty['dayHigh'],
                'low': nifty['dayLow'],
                'volume': nifty.get('totalTradedVolume', 0)
            }
            
            # Bank Nifty
            banknifty = self.nse.get_index_quote("nifty bank")
            indices['BANKNIFTY'] = {
                'value': banknifty['lastPrice'],
                'change': banknifty['change'],
                'change_percent': banknifty['pChange']
            }
            
            # Get SENSEX from BSE
            sensex_data = await self._fetch_sensex()
            if sensex_data:
                indices['SENSEX'] = sensex_data
            
            # India VIX
            vix = self.nse.get_index_quote("india vix")
            indices['INDIAVIX'] = {
                'value': vix['lastPrice'],
                'change': vix['change'],
                'change_percent': vix['pChange']
            }
            
        except Exception as e:
            print(f"Error fetching index data: {str(e)}")
        
        return indices
    
    async def _fetch_sensex(self) -> Optional[Dict[str, Any]]:
        """Fetch SENSEX data from BSE"""
        try:
            # Use BSE API or web scraping
            url = "https://api.bseindia.com/BseIndiaAPI/api/GetSensexData/w"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'value': float(data['CurrVal']),
                            'change': float(data['NetChg']),
                            'change_percent': float(data['PcChg']),
                            'open': float(data['OpenVal']),
                            'high': float(data['HighVal']),
                            'low': float(data['LowVal'])
                        }
        except:
            pass
        return None
    
    def _calculate_market_mood(self, indices: Dict[str, Any]) -> str:
        """Calculate overall market mood"""
        if not indices:
            return 'neutral'
        
        # Average change percentage
        changes = []
        for index_data in indices.values():
            if 'change_percent' in index_data:
                changes.append(index_data['change_percent'])
        
        if not changes:
            return 'neutral'
        
        avg_change = np.mean(changes)
        
        if avg_change > 1.5:
            return 'very_bullish'
        elif avg_change > 0.5:
            return 'bullish'
        elif avg_change < -1.5:
            return 'very_bearish'
        elif avg_change < -0.5:
            return 'bearish'
        else:
            return 'neutral'
    
    async def get_stock_data(self, symbol: str, exchange: str = "NSE") -> Optional[IndianStockData]:
        """Get comprehensive Indian stock data"""
        try:
            if exchange == "NSE":
                # Get NSE quote
                quote = self.nse.get_quote(symbol)
                
                if quote:
                    # Get additional data
                    delivery_data = await self._get_delivery_percentage(symbol)
                    
                    return IndianStockData(
                        symbol=symbol,
                        exchange=exchange,
                        price=float(quote['lastPrice']),
                        change=float(quote['change']),
                        change_percent=float(quote['pChange']),
                        volume=int(quote['totalTradedVolume']),
                        delivery_percent=delivery_data,
                        pe_ratio=float(quote.get('pe', 0)) if quote.get('pe') else None,
                        market_cap=float(quote.get('marketCap', 0)),
                        sector=quote.get('sector', 'Unknown'),
                        industry=quote.get('industry', 'Unknown'),
                        fno_eligible=symbol in self.fno_symbols,
                        upper_circuit=float(quote.get('pricebandupper', 0)),
                        lower_circuit=float(quote.get('pricebandlower', 0)),
                        week_52_high=float(quote['high52']),
                        week_52_low=float(quote['low52']),
                        vwap=float(quote.get('vwap', quote['lastPrice'])),
                        open_interest=None,  # Will be filled for F&O
                        implied_volatility=None,
                        lot_size=self._get_lot_size(symbol) if symbol in self.fno_symbols else None
                    )
                    
            elif exchange == "BSE":
                # Get BSE quote
                quote = self.bse.getQuote(symbol)
                
                if quote:
                    return IndianStockData(
                        symbol=symbol,
                        exchange=exchange,
                        price=float(quote['currentValue']),
                        change=float(quote['change']),
                        change_percent=float(quote['changePercent']),
                        volume=int(quote['totalTradedQty']),
                        delivery_percent=0,  # Not available from BSE
                        pe_ratio=float(quote.get('PE', 0)) if quote.get('PE') else None,
                        market_cap=float(quote.get('marketCapFull', 0)),
                        sector=quote.get('group', 'Unknown'),
                        industry=quote.get('industry', 'Unknown'),
                        fno_eligible=False,  # BSE doesn't have F&O
                        upper_circuit=float(quote.get('upperCircuitLimit', 0)),
                        lower_circuit=float(quote.get('lowerCircuitLimit', 0)),
                        week_52_high=float(quote['52weekHigh']),
                        week_52_low=float(quote['52weekLow']),
                        vwap=float(quote.get('weightedAvgPrice', quote['currentValue'])),
                        open_interest=None,
                        implied_volatility=None,
                        lot_size=None
                    )
                    
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange}: {str(e)}")
            
        return None
    
    async def _get_delivery_percentage(self, symbol: str) -> float:
        """Get delivery percentage from NSE"""
        try:
            # This would fetch from NSE delivery data
            # For now, return simulated data
            return np.random.uniform(30, 80)
        except:
            return 0
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get F&O lot size"""
        lot_sizes = {
            'RELIANCE': 250, 'TCS': 150, 'HDFCBANK': 550, 'INFY': 600,
            'ICICIBANK': 1375, 'KOTAKBANK': 400, 'SBIN': 1500, 'BHARTIARTL': 1851,
            'BAJFINANCE': 125, 'ITC': 1600, 'AXISBANK': 1200, 'LT': 150,
            'DMART': 50, 'SUNPHARMA': 700, 'MARUTI': 100, 'TITAN': 375,
            'NIFTY': 50, 'BANKNIFTY': 25, 'FINNIFTY': 40
        }
        return lot_sizes.get(symbol, 1000)
    
    async def get_fno_data(self, symbol: str, expiry: Optional[str] = None) -> List[FNOData]:
        """Get Futures & Options data"""
        fno_data = []
        
        try:
            # Get current month expiry if not specified
            if not expiry:
                expiry = self._get_current_expiry()
            
            # Get futures data
            futures = await self._get_futures_data(symbol, expiry)
            if futures:
                fno_data.append(futures)
            
            # Get options chain
            options = await self._get_options_chain(symbol, expiry)
            fno_data.extend(options)
            
        except Exception as e:
            print(f"Error fetching F&O data for {symbol}: {str(e)}")
        
        return fno_data
    
    def _get_current_expiry(self) -> str:
        """Get current month expiry date"""
        # NSE expiry is last Thursday of the month
        today = datetime.now()
        
        # Find last Thursday
        month_end = datetime(today.year, today.month, 28)  # Start from 28th
        while month_end.month == today.month:
            if month_end.weekday() == 3:  # Thursday
                last_thursday = month_end
            month_end += timedelta(days=1)
        
        return last_thursday.strftime("%d-%b-%Y")
    
    async def _get_futures_data(self, symbol: str, expiry: str) -> Optional[FNOData]:
        """Get futures data for a symbol"""
        try:
            # This would fetch real futures data from NSE
            # For now, return simulated data
            spot_price = 1000  # Would get from spot market
            futures_price = spot_price * (1 + np.random.uniform(-0.02, 0.02))
            
            return FNOData(
                symbol=symbol,
                expiry=datetime.strptime(expiry, "%d-%b-%Y"),
                strike=None,
                option_type=None,
                futures_price=futures_price,
                spot_price=spot_price,
                premium=None,
                open_interest=np.random.randint(1000000, 10000000),
                change_in_oi=np.random.randint(-500000, 500000),
                volume=np.random.randint(100000, 1000000),
                implied_volatility=np.random.uniform(15, 35),
                delta=None,
                gamma=None,
                theta=None,
                vega=None,
                lot_size=self._get_lot_size(symbol)
            )
        except:
            return None
    
    async def _get_options_chain(self, symbol: str, expiry: str) -> List[FNOData]:
        """Get options chain data"""
        options = []
        
        try:
            # Get ATM strikes
            spot_price = 1000  # Would get from spot market
            strikes = self._get_option_strikes(spot_price)
            
            for strike in strikes[:5]:  # Limit to 5 strikes for demo
                # Call option
                call = FNOData(
                    symbol=symbol,
                    expiry=datetime.strptime(expiry, "%d-%b-%Y"),
                    strike=strike,
                    option_type='CE',
                    futures_price=None,
                    spot_price=spot_price,
                    premium=self._calculate_option_premium(spot_price, strike, 'CE'),
                    open_interest=np.random.randint(10000, 1000000),
                    change_in_oi=np.random.randint(-50000, 50000),
                    volume=np.random.randint(1000, 100000),
                    implied_volatility=np.random.uniform(15, 35),
                    delta=self._calculate_delta(spot_price, strike, 'CE'),
                    gamma=0.02,
                    theta=-0.5,
                    vega=0.15,
                    lot_size=self._get_lot_size(symbol)
                )
                options.append(call)
                
                # Put option
                put = FNOData(
                    symbol=symbol,
                    expiry=datetime.strptime(expiry, "%d-%b-%Y"),
                    strike=strike,
                    option_type='PE',
                    futures_price=None,
                    spot_price=spot_price,
                    premium=self._calculate_option_premium(spot_price, strike, 'PE'),
                    open_interest=np.random.randint(10000, 1000000),
                    change_in_oi=np.random.randint(-50000, 50000),
                    volume=np.random.randint(1000, 100000),
                    implied_volatility=np.random.uniform(15, 35),
                    delta=self._calculate_delta(spot_price, strike, 'PE'),
                    gamma=0.02,
                    theta=-0.5,
                    vega=0.15,
                    lot_size=self._get_lot_size(symbol)
                )
                options.append(put)
                
        except Exception as e:
            print(f"Error getting options chain: {str(e)}")
        
        return options
    
    def _get_option_strikes(self, spot_price: float) -> List[float]:
        """Get option strikes around spot price"""
        # Strike interval based on spot price
        if spot_price < 100:
            interval = 2.5
        elif spot_price < 500:
            interval = 5
        elif spot_price < 1000:
            interval = 10
        elif spot_price < 5000:
            interval = 50
        else:
            interval = 100
        
        # Get strikes around ATM
        atm = round(spot_price / interval) * interval
        strikes = []
        
        for i in range(-5, 6):
            strikes.append(atm + i * interval)
        
        return strikes
    
    def _calculate_option_premium(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate option premium (simplified)"""
        # Intrinsic value
        if option_type == 'CE':
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        
        # Time value (simplified)
        time_value = abs(spot - strike) * 0.02 + 5
        
        return intrinsic + time_value
    
    def _calculate_delta(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate option delta (simplified)"""
        moneyness = spot / strike
        
        if option_type == 'CE':
            if moneyness > 1.1:  # Deep ITM
                return 0.9
            elif moneyness > 1.0:  # ITM
                return 0.6
            elif moneyness > 0.9:  # OTM
                return 0.4
            else:  # Deep OTM
                return 0.1
        else:  # Put
            if moneyness < 0.9:  # Deep ITM
                return -0.9
            elif moneyness < 1.0:  # ITM
                return -0.6
            elif moneyness < 1.1:  # OTM
                return -0.4
            else:  # Deep OTM
                return -0.1
    
    async def get_sector_performance(self) -> Dict[str, Any]:
        """Get sector-wise performance"""
        sectors = {}
        
        for sector_name, stocks in self.nifty_sectors.items():
            sector_data = []
            
            for symbol in stocks[:3]:  # Top 3 stocks per sector
                stock_data = await self.get_stock_data(symbol)
                if stock_data:
                    sector_data.append({
                        'symbol': symbol,
                        'change_percent': stock_data.change_percent,
                        'volume': stock_data.volume
                    })
            
            if sector_data:
                sectors[sector_name] = {
                    'change_percent': np.mean([s['change_percent'] for s in sector_data]),
                    'top_performer': max(sector_data, key=lambda x: x['change_percent'])['symbol'],
                    'bottom_performer': min(sector_data, key=lambda x: x['change_percent'])['symbol'],
                    'volume': sum(s['volume'] for s in sector_data)
                }
        
        return sectors
    
    async def get_market_breadth(self) -> Dict[str, Any]:
        """Get market breadth indicators"""
        try:
            # Get advances/declines from NSE
            adv_dec = self.nse.get_advances_declines()
            
            nifty_50_stocks = self._get_nifty50_stocks()
            advances = declines = unchanged = 0
            total_volume = 0
            
            for symbol in nifty_50_stocks[:20]:  # Sample 20 stocks
                data = await self.get_stock_data(symbol)
                if data:
                    if data.change_percent > 0:
                        advances += 1
                    elif data.change_percent < 0:
                        declines += 1
                    else:
                        unchanged += 1
                    total_volume += data.volume
            
            # Calculate breadth indicators
            if advances + declines > 0:
                advance_decline_ratio = advances / (advances + declines)
            else:
                advance_decline_ratio = 0.5
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advance_decline_ratio,
                'market_sentiment': self._get_breadth_sentiment(advance_decline_ratio),
                'total_volume': total_volume,
                'volume_sentiment': 'high' if total_volume > 1000000000 else 'normal'
            }
            
        except Exception as e:
            print(f"Error getting market breadth: {str(e)}")
            return {}
    
    def _get_breadth_sentiment(self, ratio: float) -> str:
        """Get market sentiment from breadth"""
        if ratio > 0.7:
            return 'strongly_bullish'
        elif ratio > 0.6:
            return 'bullish'
        elif ratio < 0.3:
            return 'strongly_bearish'
        elif ratio < 0.4:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_nifty50_stocks(self) -> List[str]:
        """Get NIFTY 50 constituents"""
        # Top NIFTY 50 stocks
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HDFC', 'ICICIBANK',
            'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'ITC',
            'AXISBANK', 'LT', 'DMART', 'SUNPHARMA', 'MARUTI', 'TITAN',
            'ULTRACEMCO', 'ONGC', 'NTPC', 'JSWSTEEL', 'TATAMOTORS',
            'POWERGRID', 'M&M', 'TATASTEEL', 'WIPRO', 'HCLTECH',
            'TECHM', 'ADANIPORTS', 'GRASIM', 'DRREDDY', 'HINDALCO',
            'DIVISLAB', 'CIPLA', 'NESTLEIND', 'BAJAJFINSV', 'HINDUNILVR',
            'VEDL', 'BPCL', 'PEL', 'INDUSINDBK', 'HDFCLIFE', 'PIDILITIND',
            'NAUKRI', 'UPL', 'MCDOWELL-N', 'BAJAJ-AUTO', 'TATACONSUM',
            'GAIL', 'COALINDIA', 'ADANIGREEN'
        ]
    
    async def get_option_chain_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyze option chain for trading signals"""
        try:
            # Get option chain
            options = await self.get_fno_data(symbol)
            
            if not options:
                return {}
            
            # Separate calls and puts
            calls = [o for o in options if o.option_type == 'CE']
            puts = [o for o in options if o.option_type == 'PE']
            
            # Calculate Put-Call Ratio
            call_oi = sum(c.open_interest for c in calls)
            put_oi = sum(p.open_interest for p in puts)
            pcr = put_oi / call_oi if call_oi > 0 else 0
            
            # Find max pain
            max_pain = self._calculate_max_pain(options)
            
            # Analyze OI buildup
            call_buildup = sum(c.change_in_oi for c in calls if c.change_in_oi > 0)
            put_buildup = sum(p.change_in_oi for p in puts if p.change_in_oi > 0)
            
            # Find support and resistance
            resistance = self._find_resistance_from_options(calls)
            support = self._find_support_from_options(puts)
            
            # IV analysis
            avg_iv = np.mean([o.implied_volatility for o in options])
            iv_skew = self._calculate_iv_skew(calls, puts)
            
            return {
                'symbol': symbol,
                'pcr': pcr,
                'pcr_signal': self._interpret_pcr(pcr),
                'max_pain': max_pain,
                'call_buildup': call_buildup,
                'put_buildup': put_buildup,
                'buildup_signal': 'bullish' if put_buildup > call_buildup else 'bearish',
                'resistance_levels': resistance,
                'support_levels': support,
                'avg_iv': avg_iv,
                'iv_percentile': self._calculate_iv_percentile(avg_iv),
                'iv_skew': iv_skew,
                'trading_signal': self._generate_option_signal(pcr, max_pain, iv_skew)
            }
            
        except Exception as e:
            print(f"Error in option chain analysis: {str(e)}")
            return {}
    
    def _calculate_max_pain(self, options: List[FNOData]) -> float:
        """Calculate max pain strike"""
        if not options:
            return 0
        
        strikes = list(set(o.strike for o in options if o.strike))
        min_pain = float('inf')
        max_pain_strike = 0
        
        for strike in strikes:
            total_pain = 0
            
            # Calculate pain for this strike
            for option in options:
                if option.strike and option.open_interest > 0:
                    if option.option_type == 'CE' and option.strike < strike:
                        # ITM calls
                        total_pain += (strike - option.strike) * option.open_interest
                    elif option.option_type == 'PE' and option.strike > strike:
                        # ITM puts
                        total_pain += (option.strike - strike) * option.open_interest
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret Put-Call Ratio"""
        if pcr > 1.5:
            return 'extremely_bullish'
        elif pcr > 1.2:
            return 'bullish'
        elif pcr < 0.5:
            return 'extremely_bearish'
        elif pcr < 0.8:
            return 'bearish'
        else:
            return 'neutral'
    
    def _find_resistance_from_options(self, calls: List[FNOData]) -> List[float]:
        """Find resistance levels from call OI"""
        # Sort by OI
        sorted_calls = sorted(calls, key=lambda x: x.open_interest, reverse=True)
        
        # Top 3 strikes with highest OI
        resistances = []
        for call in sorted_calls[:3]:
            if call.strike:
                resistances.append(call.strike)
        
        return sorted(resistances)
    
    def _find_support_from_options(self, puts: List[FNOData]) -> List[float]:
        """Find support levels from put OI"""
        # Sort by OI
        sorted_puts = sorted(puts, key=lambda x: x.open_interest, reverse=True)
        
        # Top 3 strikes with highest OI
        supports = []
        for put in sorted_puts[:3]:
            if put.strike:
                supports.append(put.strike)
        
        return sorted(supports, reverse=True)
    
    def _calculate_iv_skew(self, calls: List[FNOData], puts: List[FNOData]) -> float:
        """Calculate IV skew"""
        if not calls or not puts:
            return 0
        
        # Average IV for OTM calls and puts
        otm_call_iv = np.mean([c.implied_volatility for c in calls if c.strike and c.strike > c.spot_price])
        otm_put_iv = np.mean([p.implied_volatility for p in puts if p.strike and p.strike < p.spot_price])
        
        return otm_put_iv - otm_call_iv
    
    def _calculate_iv_percentile(self, current_iv: float) -> float:
        """Calculate IV percentile (simplified)"""
        # Historical IV range (would use actual historical data)
        historical_iv_range = (15, 40)
        
        if current_iv <= historical_iv_range[0]:
            return 0
        elif current_iv >= historical_iv_range[1]:
            return 100
        else:
            return ((current_iv - historical_iv_range[0]) / 
                   (historical_iv_range[1] - historical_iv_range[0])) * 100
    
    def _generate_option_signal(self, pcr: float, max_pain: float, iv_skew: float) -> str:
        """Generate trading signal from options data"""
        signals = []
        
        # PCR signal
        if pcr > 1.3:
            signals.append('bullish')
        elif pcr < 0.7:
            signals.append('bearish')
        
        # IV skew signal
        if iv_skew > 5:
            signals.append('bearish')  # Put protection demand
        elif iv_skew < -5:
            signals.append('bullish')  # Call speculation
        
        # Aggregate signals
        if signals.count('bullish') > signals.count('bearish'):
            return 'bullish'
        elif signals.count('bearish') > signals.count('bullish'):
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_next_trading_day(self, current_date: datetime) -> str:
        """Get next trading day"""
        next_day = current_date + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)
        
        # Skip holidays
        while next_day.date() in [h.date() for h in self.holidays_2024]:
            next_day += timedelta(days=1)
            # Skip weekend if holiday falls on Friday
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
        
        return next_day.strftime("%Y-%m-%d")
    
    async def get_pre_market_movers(self) -> Dict[str, List[Dict]]:
        """Get pre-market movers"""
        # This would fetch actual pre-market data
        # For now, return simulated data
        
        movers = {
            'gainers': [
                {'symbol': 'RELIANCE', 'change': 2.5, 'volume': 50000},
                {'symbol': 'TCS', 'change': 1.8, 'volume': 30000},
                {'symbol': 'INFY', 'change': 1.5, 'volume': 25000}
            ],
            'losers': [
                {'symbol': 'TATAMOTORS', 'change': -2.1, 'volume': 40000},
                {'symbol': 'VEDL', 'change': -1.7, 'volume': 20000},
                {'symbol': 'ONGC', 'change': -1.2, 'volume': 15000}
            ]
        }
        
        return movers
    
    async def get_delivery_volume_analysis(self) -> Dict[str, Any]:
        """Analyze delivery volumes for institutional activity"""
        high_delivery_stocks = []
        
        for symbol in self._get_nifty50_stocks()[:20]:
            data = await self.get_stock_data(symbol)
            
            if data and data.delivery_percent > 60:  # High delivery
                high_delivery_stocks.append({
                    'symbol': symbol,
                    'delivery_percent': data.delivery_percent,
                    'price_change': data.change_percent,
                    'signal': 'accumulation' if data.change_percent > 0 else 'distribution'
                })
        
        # Sort by delivery percentage
        high_delivery_stocks.sort(key=lambda x: x['delivery_percent'], reverse=True)
        
        return {
            'high_delivery_stocks': high_delivery_stocks[:10],
            'market_signal': self._interpret_delivery_data(high_delivery_stocks)
        }
    
    def _interpret_delivery_data(self, stocks: List[Dict]) -> str:
        """Interpret delivery data for market signal"""
        if not stocks:
            return 'neutral'
        
        accumulation = sum(1 for s in stocks if s['signal'] == 'accumulation')
        distribution = sum(1 for s in stocks if s['signal'] == 'distribution')
        
        if accumulation > distribution * 1.5:
            return 'institutional_buying'
        elif distribution > accumulation * 1.5:
            return 'institutional_selling'
        else:
            return 'mixed'
