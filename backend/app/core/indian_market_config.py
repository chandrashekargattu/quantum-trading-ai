"""Indian Market Configuration"""

# Indian Stock Exchanges
EXCHANGES = {
    'NSE': {
        'suffix': '.NS',
        'name': 'National Stock Exchange',
        'trading_hours': '09:15 - 15:30 IST',
        'currency': 'INR'
    },
    'BSE': {
        'suffix': '.BO',
        'name': 'Bombay Stock Exchange',
        'trading_hours': '09:15 - 15:30 IST',
        'currency': 'INR'
    }
}

# Major Indian Indices
INDICES = {
    '^NSEI': 'NIFTY 50',
    '^BSESN': 'SENSEX',
    '^NSEBANK': 'Bank Nifty',
    'NIFTY_FIN_SERVICE.NS': 'Nifty Financial Services',
    '^NSMIDCP': 'Nifty Midcap 100',
    '^CNXSC': 'Nifty Smallcap 100',
    '^CNXIT': 'Nifty IT',
    '^CNXPHARMA': 'Nifty Pharma',
    '^CNXAUTO': 'Nifty Auto',
    '^CNXFMCG': 'Nifty FMCG',
    '^CNXMETAL': 'Nifty Metal',
    '^CNXREALTY': 'Nifty Realty',
}

# Popular NSE Stocks
POPULAR_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HDFC.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
    'ITC.NS', 'AXISBANK.NS', 'LT.NS', 'DMART.NS', 'SUNPHARMA.NS',
    'MARUTI.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'NTPC.NS',
    'JSWSTEEL.NS', 'TATAMOTORS.NS', 'POWERGRID.NS', 'M&M.NS', 'TATASTEEL.NS',
    'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'ADANIPORTS.NS', 'GRASIM.NS',
    'DRREDDY.NS', 'HINDALCO.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'NESTLEIND.NS',
    'BAJAJFINSV.NS', 'HINDUNILVR.NS', 'VEDL.NS', 'BPCL.NS', 'PEL.NS',
    'INDUSINDBK.NS', 'HDFCLIFE.NS', 'PIDILITIND.NS', 'NAUKRI.NS', 'UPL.NS',
    'MCDOWELL-N.NS', 'BAJAJ-AUTO.NS', 'TATACONSUM.NS', 'GAIL.NS', 'COALINDIA.NS'
]

# F&O Lot Sizes (as of 2024)
FNO_LOT_SIZES = {
    'RELIANCE': 250,
    'TCS': 150,
    'HDFCBANK': 550,
    'INFY': 600,
    'ICICIBANK': 1375,
    'KOTAKBANK': 400,
    'SBIN': 1500,
    'BHARTIARTL': 1851,
    'BAJFINANCE': 125,
    'ITC': 1600,
    'AXISBANK': 1200,
    'LT': 150,
    'DMART': 50,
    'SUNPHARMA': 700,
    'MARUTI': 100,
    'TITAN': 375,
    'NIFTY': 50,
    'BANKNIFTY': 25,
    'FINNIFTY': 40
}

# Sector Mappings
SECTORS = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'INDUSINDBK'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'MPHASIS', 'COFORGE'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'BIOCON'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT'],
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO'],
    'Metal': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'NATIONALUM', 'SAIL'],
    'Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'PETRONET'],
    'Finance': ['HDFC', 'BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI'],
    'Cement': ['ULTRACEMCO', 'GRASIM', 'SHREECEM', 'DALMIACEM', 'RAMCOCEM'],
    'Power': ['NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIGREEN', 'ADANIPOWER']
}

# Market Timings (IST)
MARKET_TIMINGS = {
    'pre_open': {
        'start': '09:00',
        'end': '09:08'
    },
    'normal': {
        'start': '09:15',
        'end': '15:30'
    },
    'post_close': {
        'start': '15:40',
        'end': '16:00'
    },
    'muhurat': {  # Special Diwali trading session
        'duration': '1 hour'
    }
}

# Circuit Limits
CIRCUIT_LIMITS = {
    'index': {
        '10%': 'Trading halt for 45 minutes',
        '15%': 'Trading halt for 1 hour 45 minutes',
        '20%': 'Trading halted for the day'
    },
    'stocks': {
        'default': '20%',
        'sme': '5%',
        'new_listing': 'No limit on listing day'
    }
}

# Settlement Cycle
SETTLEMENT = {
    'equity': 'T+1',  # Changed from T+2 to T+1 in 2023
    'derivatives': 'T+1'
}

# Indian Market Holidays 2024
HOLIDAYS_2024 = [
    '2024-01-26',  # Republic Day
    '2024-03-08',  # Mahashivratri
    '2024-03-25',  # Holi
    '2024-03-29',  # Good Friday
    '2024-04-11',  # Id-ul-Fitr
    '2024-04-17',  # Ram Navami
    '2024-05-01',  # Maharashtra Day
    '2024-08-15',  # Independence Day
    '2024-10-02',  # Gandhi Jayanti
    '2024-11-01',  # Diwali - Laxmi Pujan
    '2024-11-15',  # Guru Nanak Jayanti
]
