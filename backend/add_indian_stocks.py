"""Script to add popular Indian stocks to the database."""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.models.stock import Stock
from app.core.config import settings
from app.services.market_data import MarketDataService

# Popular Indian stocks to add
INDIAN_STOCKS = [
    # NIFTY 50 Major Stocks
    {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries Ltd', 'exchange': 'NSE'},
    {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services Ltd', 'exchange': 'NSE'},
    {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank Ltd', 'exchange': 'NSE'},
    {'symbol': 'INFY.NS', 'name': 'Infosys Ltd', 'exchange': 'NSE'},
    {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank Ltd', 'exchange': 'NSE'},
    {'symbol': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank Ltd', 'exchange': 'NSE'},
    {'symbol': 'SBIN.NS', 'name': 'State Bank of India', 'exchange': 'NSE'},
    {'symbol': 'BHARTIARTL.NS', 'name': 'Bharti Airtel Ltd', 'exchange': 'NSE'},
    {'symbol': 'BAJFINANCE.NS', 'name': 'Bajaj Finance Ltd', 'exchange': 'NSE'},
    {'symbol': 'ITC.NS', 'name': 'ITC Ltd', 'exchange': 'NSE'},
    {'symbol': 'AXISBANK.NS', 'name': 'Axis Bank Ltd', 'exchange': 'NSE'},
    {'symbol': 'LT.NS', 'name': 'Larsen & Toubro Ltd', 'exchange': 'NSE'},
    {'symbol': 'DMART.NS', 'name': 'Avenue Supermarts Ltd', 'exchange': 'NSE'},
    {'symbol': 'SUNPHARMA.NS', 'name': 'Sun Pharmaceutical Industries Ltd', 'exchange': 'NSE'},
    {'symbol': 'MARUTI.NS', 'name': 'Maruti Suzuki India Ltd', 'exchange': 'NSE'},
    {'symbol': 'TITAN.NS', 'name': 'Titan Company Ltd', 'exchange': 'NSE'},
    {'symbol': 'ULTRACEMCO.NS', 'name': 'UltraTech Cement Ltd', 'exchange': 'NSE'},
    {'symbol': 'ONGC.NS', 'name': 'Oil and Natural Gas Corporation Ltd', 'exchange': 'NSE'},
    {'symbol': 'NTPC.NS', 'name': 'NTPC Ltd', 'exchange': 'NSE'},
    {'symbol': 'JSWSTEEL.NS', 'name': 'JSW Steel Ltd', 'exchange': 'NSE'},
    {'symbol': 'TATAMOTORS.NS', 'name': 'Tata Motors Ltd', 'exchange': 'NSE'},
    {'symbol': 'POWERGRID.NS', 'name': 'Power Grid Corporation of India Ltd', 'exchange': 'NSE'},
    {'symbol': 'M&M.NS', 'name': 'Mahindra & Mahindra Ltd', 'exchange': 'NSE'},
    {'symbol': 'TATASTEEL.NS', 'name': 'Tata Steel Ltd', 'exchange': 'NSE'},
    {'symbol': 'WIPRO.NS', 'name': 'Wipro Ltd', 'exchange': 'NSE'},
    {'symbol': 'HCLTECH.NS', 'name': 'HCL Technologies Ltd', 'exchange': 'NSE'},
    {'symbol': 'TECHM.NS', 'name': 'Tech Mahindra Ltd', 'exchange': 'NSE'},
    {'symbol': 'ADANIPORTS.NS', 'name': 'Adani Ports and SEZ Ltd', 'exchange': 'NSE'},
    {'symbol': 'GRASIM.NS', 'name': 'Grasim Industries Ltd', 'exchange': 'NSE'},
    {'symbol': 'DRREDDY.NS', 'name': 'Dr. Reddy\'s Laboratories Ltd', 'exchange': 'NSE'},
    {'symbol': 'HINDALCO.NS', 'name': 'Hindalco Industries Ltd', 'exchange': 'NSE'},
    {'symbol': 'DIVISLAB.NS', 'name': 'Divi\'s Laboratories Ltd', 'exchange': 'NSE'},
    {'symbol': 'CIPLA.NS', 'name': 'Cipla Ltd', 'exchange': 'NSE'},
    {'symbol': 'NESTLEIND.NS', 'name': 'Nestle India Ltd', 'exchange': 'NSE'},
    {'symbol': 'BAJAJFINSV.NS', 'name': 'Bajaj Finserv Ltd', 'exchange': 'NSE'},
    {'symbol': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever Ltd', 'exchange': 'NSE'},
    {'symbol': 'VEDL.NS', 'name': 'Vedanta Ltd', 'exchange': 'NSE'},
    {'symbol': 'BPCL.NS', 'name': 'Bharat Petroleum Corporation Ltd', 'exchange': 'NSE'},
    {'symbol': 'PEL.NS', 'name': 'Piramal Enterprises Ltd', 'exchange': 'NSE'},
    {'symbol': 'INDUSINDBK.NS', 'name': 'IndusInd Bank Ltd', 'exchange': 'NSE'},
    {'symbol': 'HDFCLIFE.NS', 'name': 'HDFC Life Insurance Company Ltd', 'exchange': 'NSE'},
    {'symbol': 'PIDILITIND.NS', 'name': 'Pidilite Industries Ltd', 'exchange': 'NSE'},
    {'symbol': 'NAUKRI.NS', 'name': 'Info Edge (India) Ltd', 'exchange': 'NSE'},
    {'symbol': 'UPL.NS', 'name': 'UPL Ltd', 'exchange': 'NSE'},
    {'symbol': 'MCDOWELL-N.NS', 'name': 'United Spirits Ltd', 'exchange': 'NSE'},
    {'symbol': 'BAJAJ-AUTO.NS', 'name': 'Bajaj Auto Ltd', 'exchange': 'NSE'},
    {'symbol': 'TATACONSUM.NS', 'name': 'Tata Consumer Products Ltd', 'exchange': 'NSE'},
    {'symbol': 'GAIL.NS', 'name': 'GAIL (India) Ltd', 'exchange': 'NSE'},
    {'symbol': 'COALINDIA.NS', 'name': 'Coal India Ltd', 'exchange': 'NSE'}
]


async def add_stocks():
    """Add Indian stocks to the database."""
    # Create async engine
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    market_service = MarketDataService()
    added_stocks = []
    failed_stocks = []
    
    async with async_session() as session:
        for stock_info in INDIAN_STOCKS:
            try:
                # Check if stock already exists
                from sqlalchemy import select
                result = await session.execute(
                    select(Stock).where(Stock.symbol == stock_info['symbol'])
                )
                existing = result.scalar_one_or_none()
                if existing:
                    print(f"Stock {stock_info['symbol']} already exists")
                    continue
                
                print(f"Fetching data for {stock_info['symbol']}...")
                
                # Fetch real-time data
                stock_data = await market_service.fetch_stock_data(stock_info['symbol'])
                
                if stock_data:
                    # Create stock
                    stock = Stock(
                        symbol=stock_info['symbol'],
                        name=stock_info['name'],
                        exchange=stock_info['exchange'],
                        current_price=stock_data.get('current_price', 0),
                        previous_close=stock_data.get('previous_close', 0),
                        open_price=stock_data.get('open_price', 0),
                        day_high=stock_data.get('day_high', 0),
                        day_low=stock_data.get('day_low', 0),
                        volume=stock_data.get('volume', 0),
                        avg_volume=stock_data.get('avg_volume', 0),
                        market_cap=stock_data.get('market_cap', 0),
                        pe_ratio=stock_data.get('pe_ratio'),
                        week_52_high=stock_data.get('week_52_high', 0),
                        week_52_low=stock_data.get('week_52_low', 0),
                        change_amount=stock_data.get('change_amount', 0),
                        change_percent=stock_data.get('change_percent', 0),
                        is_active=True,
                        is_optionable=True  # Most NIFTY 50 stocks have F&O
                    )
                    
                    session.add(stock)
                    added_stocks.append(stock_info['symbol'])
                    print(f"✓ Added {stock_info['symbol']} - {stock_info['name']}")
                else:
                    failed_stocks.append(stock_info['symbol'])
                    print(f"✗ Failed to fetch data for {stock_info['symbol']}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error adding {stock_info['symbol']}: {str(e)}")
                failed_stocks.append(stock_info['symbol'])
        
        # Commit all changes
        await session.commit()
    
    print(f"\n\nSummary:")
    print(f"Successfully added: {len(added_stocks)} stocks")
    print(f"Failed: {len(failed_stocks)} stocks")
    
    if failed_stocks:
        print(f"\nFailed stocks: {', '.join(failed_stocks)}")


if __name__ == "__main__":
    print("Adding popular Indian stocks to the database...")
    print("This may take a few minutes...\n")
    asyncio.run(add_stocks())
    print("\nDone!")
