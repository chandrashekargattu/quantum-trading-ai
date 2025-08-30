# How to Add Stocks in Quantum Trading AI 📈

## 🔧 Quick Fix for Stock Adding

I've created the necessary endpoints and a script to add Indian stocks. Here's how to use them:

## 📌 Method 1: Using the API (Recommended)

### Add a Single Stock via API

```bash
# Add a stock (e.g., RELIANCE)
curl -X POST http://localhost:8000/api/v1/stocks/add \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "name": "Reliance Industries Ltd",
    "exchange": "NSE"
  }'
```

**Note**: The system will automatically:
- Add `.NS` suffix for NSE stocks
- Fetch real-time data from Yahoo Finance
- Store the stock in the database

## 📌 Method 2: Auto-Add via Search

When you search for a stock that doesn't exist in the database:

1. **Search for the stock** in the trading interface
2. If not found, the system will **automatically fetch and add it**
3. The stock will then appear in search results

Example:
- Search: "RELIANCE" or "TCS"
- System adds: "RELIANCE.NS" or "TCS.NS"

## 📌 Method 3: Bulk Add Popular Stocks

Run the provided script to add 50 popular Indian stocks:

```bash
cd backend
python add_indian_stocks.py
```

This will add major NIFTY 50 stocks including:
- RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
- KOTAKBANK, SBIN, BHARTIARTL, BAJFINANCE, ITC
- And 40 more popular stocks

## 🎯 How to Use in the Platform

### 1. **From Trading Panel**
- Click on "Trade" or "New Order"
- In the symbol field, type the stock name (e.g., "RELIANCE")
- The system will search and auto-add if needed

### 2. **From Portfolio**
- When creating trades, just enter the symbol
- System handles the rest

### 3. **Stock Symbol Format**
- **NSE Stocks**: Automatically adds `.NS` (e.g., RELIANCE → RELIANCE.NS)
- **BSE Stocks**: Use `.BO` suffix (e.g., RELIANCE.BO)

## 🔍 Supported Features

Once a stock is added, you can:
- ✅ View real-time prices
- ✅ Place buy/sell orders
- ✅ Track in portfolio
- ✅ View price history
- ✅ See technical indicators
- ✅ Add to watchlist

## ⚡ Quick Examples

### Search and Auto-Add:
```javascript
// In the UI, just search for:
"RELIANCE"    // Auto-converts to RELIANCE.NS
"TCS"         // Auto-converts to TCS.NS
"INFY"        // Auto-converts to INFY.NS
```

### Direct API Call:
```javascript
// From frontend code
const response = await fetch('/api/v1/stocks/add', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    symbol: 'WIPRO',
    name: 'Wipro Ltd',
    exchange: 'NSE'
  })
})
```

## 🚨 Troubleshooting

### If you get 404 errors:
1. **Stock not in database**: Use the add endpoint or search to auto-add
2. **Wrong symbol format**: Ensure you use `.NS` for NSE stocks
3. **Authentication issue**: Make sure you're logged in

### Common Issues:
- **"Stock already exists"**: The stock is already in the database
- **"Unable to fetch data"**: Yahoo Finance might be down or symbol is incorrect
- **404 on trading**: The stock needs to be added first

## 📊 Pre-Added Stocks

After running `add_indian_stocks.py`, these stocks will be available:

**Banking**: HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN
**IT**: TCS, INFY, WIPRO, HCLTECH, TECHM
**Pharma**: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB
**Auto**: MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO
**FMCG**: HINDUNILVR, ITC, NESTLEIND, BRITANNIA
**And many more...**

## 🎉 That's It!

You can now add and trade Indian stocks on the platform. The system handles NSE/BSE formatting automatically!
