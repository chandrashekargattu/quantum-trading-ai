"""NSE data service for fetching all NSE listed stocks."""

import aiohttp
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from functools import lru_cache
import asyncio


class NSEDataService:
    """Service to fetch NSE stock data from official and unofficial sources."""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        self._cache = {}
        self._cache_expiry = {}
    
    async def get_nse_equity_list(self) -> List[Dict[str, str]]:
        """Get list of all NSE equity stocks."""
        cache_key = 'nse_equity_list'
        
        # Check cache
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        
        try:
            # Alternative approach: Use NSE's equity list CSV
            csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(csv_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse CSV content
                        lines = content.strip().split('\n')
                        if len(lines) > 1:
                            headers = lines[0].split(',')
                            stocks = []
                            
                            for line in lines[1:]:
                                values = line.split(',')
                                if len(values) >= 2:
                                    symbol = values[0].strip()
                                    name = values[1].strip() if len(values) > 1 else symbol
                                    
                                    stocks.append({
                                        'symbol': f"{symbol}.NS",
                                        'name': name,
                                        'exchange': 'NSE'
                                    })
                            
                            # Cache for 24 hours
                            self._cache[cache_key] = stocks
                            self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=24)
                            
                            return stocks
        except Exception as e:
            print(f"Error fetching NSE equity list: {e}")
        
        # Fallback: Return empty list if fetching fails
        return []
    
    async def get_index_constituents(self, index_name: str = 'NIFTY 50') -> List[str]:
        """Get constituents of a specific index."""
        index_mapping = {
            'NIFTY 50': 'NIFTY 50',
            'NIFTY BANK': 'NIFTY BANK',
            'NIFTY IT': 'NIFTY IT',
            'NIFTY PHARMA': 'NIFTY PHARMA',
            'NIFTY AUTO': 'NIFTY AUTO',
            'NIFTY FMCG': 'NIFTY FMCG',
            'NIFTY METAL': 'NIFTY METAL',
            'NIFTY REALTY': 'NIFTY REALTY'
        }
        
        cache_key = f'index_{index_name}'
        
        # Check cache
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        
        try:
            # This is a placeholder - in production, you'd use proper NSE API
            # For now, return popular stocks based on index
            index_stocks = {
                'NIFTY 50': [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                    'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'ITC.NS',
                    'AXISBANK.NS', 'LT.NS', 'DMART.NS', 'SUNPHARMA.NS', 'MARUTI.NS',
                    'TITAN.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'NTPC.NS', 'JSWSTEEL.NS'
                ],
                'NIFTY BANK': [
                    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
                    'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS'
                ],
                'NIFTY IT': [
                    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
                    'LTTS.NS', 'MINDTREE.NS', 'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS'
                ],
                'NIFTY PHARMA': [
                    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS',
                    'AUROPHARMA.NS', 'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'GLENMARK.NS'
                ],
                'NIFTY AUTO': [
                    'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
                    'EICHERMOT.NS', 'ASHOKLEY.NS', 'TVSMOTOR.NS', 'ESCORTS.NS', 'BHARATFORG.NS'
                ],
                'NIFTY FMCG': [
                    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
                    'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'TATACONSUM.NS', 'UBL.NS'
                ]
            }
            
            stocks = index_stocks.get(index_name, [])
            
            # Cache for 24 hours
            self._cache[cache_key] = stocks
            self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=24)
            
            return stocks
            
        except Exception as e:
            print(f"Error fetching index constituents: {e}")
            return []
    
    async def search_nse_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search NSE stocks by symbol or name."""
        query_upper = query.upper()
        all_stocks = await self.get_nse_equity_list()
        
        matches = []
        
        for stock in all_stocks:
            symbol = stock['symbol'].replace('.NS', '')
            name = stock['name'].upper()
            
            # Calculate match score
            score = 0
            
            # Exact symbol match
            if symbol == query_upper:
                score = 100
            # Symbol starts with query
            elif symbol.startswith(query_upper):
                score = 90
            # Query in symbol
            elif query_upper in symbol:
                score = 80
            # Exact word match in name
            elif query_upper in name.split():
                score = 70
            # Any word starts with query
            elif any(word.startswith(query_upper) for word in name.split()):
                score = 60
            # Query anywhere in name
            elif query_upper in name:
                score = 50
            
            if score > 0:
                matches.append({
                    **stock,
                    'score': score
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:20]  # Return top 20 matches
