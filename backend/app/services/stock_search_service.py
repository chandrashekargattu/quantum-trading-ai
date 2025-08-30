"""Dynamic stock search service with intelligent fuzzy matching."""

from typing import List, Dict, Optional
import asyncio
import difflib
import re
from functools import lru_cache
import yfinance as yf
import aiohttp
from app.services.market_data import MarketDataService
from app.services.nse_data_service import NSEDataService


class StockSearchService:
    """Intelligent stock search service with dynamic symbol discovery."""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.nse_service = NSEDataService()
        self._symbol_cache = {}
        self._last_cache_update = None
    
    @lru_cache(maxsize=1)
    def get_nse_symbols(self) -> List[Dict[str, str]]:
        """Get all NSE stock symbols dynamically from Yahoo Finance."""
        try:
            # Common NSE indices to search for constituents
            indices = ['^NSEI', '^NSEBANK', '^NSMIDCP', '^NSESML']
            all_symbols = set()
            
            # Get ticker symbols from indices
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    info = ticker.info
                    # This approach might need adjustment based on yfinance updates
                except:
                    pass
            
            # For now, we'll use a more practical approach
            # In production, this should connect to NSE API or a proper data provider
            return []
        except Exception as e:
            print(f"Error fetching NSE symbols: {e}")
            return []
    
    async def search_symbols_from_yahoo(self, query: str) -> List[Dict[str, str]]:
        """Search for symbols using Yahoo Finance search API."""
        symbols = []
        
        # Yahoo Finance search endpoint (unofficial)
        search_url = f"https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 20,
            'newsCount': 0,
            'listsCount': 0,
            'enableFuzzyQuery': True,
            'quotesQueryId': 'tss_match_phrase_query',
            'multiQuoteQueryId': 'multi_quote_single_token_query',
            'region': 'IN',  # India region
            'lang': 'en-IN'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        quotes = data.get('quotes', [])
                        
                        for quote in quotes:
                            symbol = quote.get('symbol', '')
                            exchange = quote.get('exchange', '')
                            
                            # Filter for Indian stocks (NSE/BSE)
                            if symbol.endswith('.NS') or symbol.endswith('.BO') or exchange in ['NSI', 'BSE', 'BOM']:
                                # If no suffix but Indian exchange, add appropriate suffix
                                if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                                    if exchange in ['NSI', 'NSE']:
                                        symbol = f"{symbol}.NS"
                                    elif exchange in ['BSE', 'BOM']:
                                        symbol = f"{symbol}.BO"
                                
                                symbols.append({
                                    'symbol': symbol,
                                    'name': quote.get('longname') or quote.get('shortname', ''),
                                    'exchange': exchange,
                                    'type': quote.get('quoteType', 'EQUITY'),
                                    'sector': quote.get('sector', ''),
                                    'industry': quote.get('industry', '')
                                })
        except Exception as e:
            print(f"Error searching Yahoo Finance: {e}")
        
        return symbols
    
    def normalize_search_term(self, term: str) -> str:
        """Normalize search term for better matching."""
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^a-zA-Z0-9\s&]', '', term)
        normalized = ' '.join(normalized.split())
        return normalized.upper()
    
    def calculate_relevance_score(self, search_term: str, symbol: str, name: str) -> float:
        """Calculate relevance score for search results."""
        search_term = search_term.upper()
        symbol_clean = symbol.replace('.NS', '').replace('.BO', '').upper()
        name_upper = name.upper()
        
        # Exact symbol match
        if search_term == symbol_clean:
            return 1.0
        
        # Symbol starts with search term
        if symbol_clean.startswith(search_term):
            return 0.9
        
        # Search term in symbol
        if search_term in symbol_clean:
            return 0.8
        
        # Exact word match in name
        name_words = name_upper.split()
        if search_term in name_words:
            return 0.7
        
        # Any word in name starts with search term
        for word in name_words:
            if word.startswith(search_term):
                return 0.6
        
        # Fuzzy match on symbol
        symbol_ratio = difflib.SequenceMatcher(None, search_term, symbol_clean).ratio()
        
        # Fuzzy match on name words
        name_ratios = [
            difflib.SequenceMatcher(None, search_term, word).ratio() 
            for word in name_words
        ]
        max_name_ratio = max(name_ratios) if name_ratios else 0
        
        # Return the best score
        return max(symbol_ratio * 0.5, max_name_ratio * 0.4)
    
    async def intelligent_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Perform intelligent search across all available stocks."""
        normalized_query = self.normalize_search_term(query)
        
        # Search from multiple sources in parallel
        yahoo_task = self.search_symbols_from_yahoo(query)
        nse_task = self.nse_service.search_nse_stocks(query)
        
        # Wait for both searches to complete
        yahoo_results, nse_results = await asyncio.gather(yahoo_task, nse_task)
        
        # Combine results and remove duplicates
        all_results = {}
        
        # Add NSE results first (more reliable for Indian stocks)
        for result in nse_results:
            symbol = result['symbol']
            all_results[symbol] = result
        
        # Add Yahoo results if not already present
        for result in yahoo_results:
            symbol = result['symbol']
            if symbol not in all_results:
                all_results[symbol] = result
        
        # Calculate relevance scores
        scored_results = []
        for symbol, result in all_results.items():
            score = self.calculate_relevance_score(
                normalized_query,
                result['symbol'],
                result['name']
            )
            if score > 0.3:  # Minimum relevance threshold
                result['relevance_score'] = score
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Get detailed data for top results
        final_results = []
        for result in scored_results[:limit]:
            try:
                # Fetch real-time data
                stock_data = await self.market_service.fetch_stock_data(result['symbol'])
                
                if stock_data:
                    final_results.append({
                        'symbol': result['symbol'],
                        'name': stock_data.get('name') or result['name'],
                        'exchange': stock_data.get('exchange', 'NSE'),
                        'current_price': stock_data.get('current_price', 0),
                        'change_percent': stock_data.get('change_percent', 0),
                        'volume': stock_data.get('volume', 0),
                        'market_cap': stock_data.get('market_cap', 0),
                        'pe_ratio': stock_data.get('pe_ratio'),
                        'previous_close': stock_data.get('previous_close', 0),
                        'day_high': stock_data.get('day_high', 0),
                        'day_low': stock_data.get('day_low', 0),
                        'week_52_high': stock_data.get('week_52_high', 0),
                        'week_52_low': stock_data.get('week_52_low', 0),
                        'change_amount': stock_data.get('change_amount', 0),
                        'sector': result.get('sector', ''),
                        'industry': result.get('industry', ''),
                        'relevance_score': result['relevance_score']
                    })
            except Exception as e:
                print(f"Error fetching data for {result['symbol']}: {e}")
                # Still include basic data if real-time fetch fails
                final_results.append({
                    'symbol': result['symbol'],
                    'name': result['name'],
                    'exchange': result.get('exchange', 'NSE'),
                    'current_price': 0,
                    'change_percent': 0,
                    'relevance_score': result['relevance_score']
                })
        
        return final_results
    
    async def get_popular_stocks(self, category: str = 'all') -> List[Dict]:
        """Get popular stocks by category."""
        # Map categories to NSE indices
        nse_index_mapping = {
            'nifty50': 'NIFTY 50',
            'banking': 'NIFTY BANK',
            'it': 'NIFTY IT',
            'pharma': 'NIFTY PHARMA',
            'auto': 'NIFTY AUTO',
            'fmcg': 'NIFTY FMCG'
        }
        
        # Check if it's an NSE index category
        if category.lower() in nse_index_mapping:
            index_name = nse_index_mapping[category.lower()]
            symbols = await self.nse_service.get_index_constituents(index_name)
            
            results = []
            for symbol in symbols:
                results.append({
                    'symbol': symbol,
                    'name': symbol.replace('.NS', ''),
                    'exchange': 'NSE'
                })
        else:
            # Use Yahoo Finance for gainers/losers/active
            category_queries = {
                'gainers': 'day_gainers',
                'losers': 'day_losers',
                'active': 'most_actives'
            }
            
            query = category_queries.get(category.lower(), 'NIFTY')
            results = await self.search_symbols_from_yahoo(query)
        
        # Get detailed data for results
        detailed_results = []
        for result in results[:20]:  # Limit to top 20
            try:
                stock_data = await self.market_service.fetch_stock_data(result['symbol'])
                if stock_data:
                    detailed_results.append({
                        'symbol': result['symbol'],
                        'name': stock_data.get('name') or result['name'],
                        'exchange': stock_data.get('exchange', 'NSE'),
                        'current_price': stock_data.get('current_price', 0),
                        'change_percent': stock_data.get('change_percent', 0),
                        'volume': stock_data.get('volume', 0),
                        'market_cap': stock_data.get('market_cap', 0),
                        'category': category
                    })
            except:
                pass
        
        return detailed_results