# Indian Market Configuration Guide

## 🇮🇳 Overview

The platform has been configured to focus on Indian markets, showing NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) data with proper INR currency formatting.

## 🏗️ Changes Made

### 1. Backend Market Data Service

**File:** `backend/app/services/market_data.py`

Updated the `fetch_market_indicators` method to show Indian indices:
- **NIFTY 50** (^NSEI) - Main Indian index
- **SENSEX** (^BSESN) - BSE benchmark index
- **Bank Nifty** (^NSEBANK) - Banking sector index
- **Nifty Financial Services** - Financial sector
- **Nifty Midcap** - Mid-cap stocks
- **Gold**, **Silver**, **Crude Oil** - Commodities

### 2. Indian Market Service

**File:** `backend/app/services/indian_market_service.py`

Comprehensive service providing:
- NSE/BSE/MCX integration
- Real-time Indian market data
- F&O (Futures & Options) support
- Market timings and holidays
- Sector-wise analysis
- Pre-market and post-market data
- Delivery percentage analysis

### 3. Frontend Configuration

**File:** `frontend/src/config/market-config.ts`

Created market configuration with:
- INR currency formatting
- Indian market hours (9:15 AM - 3:30 PM IST)
- Exchange suffixes (.NS for NSE, .BO for BSE)
- Popular Indian stocks
- Market status helper functions

### 4. UI Updates

**File:** `frontend/src/components/dashboard/MarketOverview.tsx`

Enhanced to show:
- 8 Indian market indices
- Market status (Open/Closed/Pre-open)
- Proper number formatting for Indian locale
- Commodity prices in USD

### 5. Currency Formatting

**File:** `frontend/src/lib/utils.ts`

Already configured for:
- INR currency symbol (₹)
- Indian number formatting (en-IN locale)
- Proper decimal places

## 📊 Available Indian Indices

```javascript
{
  '^NSEI': 'NIFTY 50',
  '^BSESN': 'SENSEX',
  '^NSEBANK': 'Bank Nifty',
  'NIFTY_FIN_SERVICE.NS': 'Nifty Financial',
  '^NSMIDCP': 'Nifty Midcap',
  '^CNXIT': 'Nifty IT',
  '^CNXPHARMA': 'Nifty Pharma',
  '^CNXAUTO': 'Nifty Auto'
}
```

## 💹 Popular Indian Stocks

The platform supports all major Indian stocks with proper exchange suffixes:

- **NSE Stocks:** Add `.NS` suffix (e.g., RELIANCE.NS, TCS.NS)
- **BSE Stocks:** Add `.BO` suffix (e.g., RELIANCE.BO, TCS.BO)

Top stocks configured:
- RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
- HDFC, KOTAKBANK, SBIN, BHARTIARTL, BAJFINANCE
- ITC, AXISBANK, LT, DMART, SUNPHARMA
- And many more...

## ⏰ Market Timings (IST)

- **Pre-open Session:** 9:00 AM - 9:08 AM
- **Normal Trading:** 9:15 AM - 3:30 PM
- **Post-close Session:** 3:40 PM - 4:00 PM
- **Settlement:** T+1 (Trade + 1 day)

## 🚀 Features Enabled

1. **Real-time Indian Market Data**
   - Live prices from NSE/BSE
   - Automatic updates during market hours
   - Market status indicator

2. **F&O Support**
   - Options chain analysis
   - Futures data
   - Greeks calculation
   - Put-Call Ratio (PCR)
   - Max Pain analysis

3. **Sector Analysis**
   - Sector-wise performance
   - Top gainers/losers
   - Market breadth indicators

4. **Delivery Analysis**
   - Delivery percentage tracking
   - Institutional activity signals

5. **Indian Market Holidays**
   - Automatic holiday calendar
   - Next trading day calculation

## 📱 Dashboard View

The dashboard now shows:
- Indian market indices prominently
- Market open/closed status
- Prices in ₹ (INR)
- Commodities in $ (USD)
- Color-coded gains/losses

## 🔧 API Endpoints

All existing endpoints work with Indian stocks:
- `/api/v1/market-data/quote/RELIANCE.NS`
- `/api/v1/market-data/indicators` (returns Indian indices)
- `/api/v1/options/chain/NIFTY`

## 📈 Testing

To test Indian market data:

```bash
# Get Indian market indicators
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/market-data/indicators

# Get quote for an Indian stock
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/market-data/quote/RELIANCE.NS
```

## 🎯 Next Steps

1. **Zerodha Integration** - Already available in `backend/app/services/zerodha_integration.py`
2. **NSE/BSE Direct APIs** - Can be enabled for real-time data
3. **Indian Options Strategies** - Available in `indian_options_strategies.py`
4. **SEBI Compliance** - Available in `sebi_compliance_service.py`

The platform is now fully configured for Indian markets! 🇮🇳
