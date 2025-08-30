# BSE Stock Support Guide 📈

## ✅ BSE Stocks Now Supported!

The platform now fully supports **BSE (Bombay Stock Exchange)** stocks alongside NSE stocks.

## 🎯 How to Add BSE Stocks

### From the Stocks Page:

1. **Navigate to Stocks**: http://localhost:3000/stocks
2. **Fill in the form**:
   - **Symbol**: Enter the BSE stock symbol (e.g., CITL)
   - **Company Name**: Optional (will be fetched automatically)
   - **Exchange**: Select **BSE** from the dropdown
3. **Click "Add Stock"**

### Via API:
```bash
curl -X POST http://localhost:8000/api/v1/stocks/add \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "CITL",
    "name": "Cambridge IT Labs",
    "exchange": "BSE"
  }'
```

## 📊 Automatic Exchange Detection

The system automatically handles exchange suffixes:
- **NSE stocks**: Adds `.NS` suffix (e.g., RELIANCE → RELIANCE.NS)
- **BSE stocks**: Adds `.BO` suffix (e.g., CITL → CITL.BO)

## 🔍 Search Functionality

The intelligent search now covers both NSE and BSE stocks:
- Searches across both exchanges
- Auto-detects the correct exchange
- Returns results from both NSE and BSE

### Example Searches:
```
Search: "bank"     → Returns banks from both NSE & BSE
Search: "citl"     → Returns CITL.BO (BSE stock)
Search: "reliance" → Returns RELIANCE.NS (NSE) and RELIANCE.BO (BSE)
```

## 💡 Tips

1. **Stock Symbol Format**:
   - For NSE: Just the symbol (e.g., RELIANCE)
   - For BSE: Just the symbol (e.g., CITL)
   - System adds the correct suffix automatically

2. **Exchange Selection**:
   - Default is NSE
   - Select BSE when adding BSE-only stocks
   - Some stocks are listed on both exchanges

3. **Real-time Data**:
   - BSE stock prices are fetched from Yahoo Finance
   - Updates in real-time
   - All features work the same as NSE stocks

## 🚀 Examples of BSE Stocks

### Small Cap BSE Stocks:
- CITL - Cambridge IT Labs
- GUJGAS - Gujarat Gas
- KALYANI - Kalyani Steels

### Mid Cap BSE Stocks:
- PRAJ - Praj Industries
- GRINDWELL - Grindwell Norton
- KIRLOSENG - Kirloskar Oil Engines

### Large Cap (Listed on both NSE & BSE):
- RELIANCE - Available as RELIANCE.NS and RELIANCE.BO
- TCS - Available as TCS.NS and TCS.BO
- INFY - Available as INFY.NS and INFY.BO

## 🛠 Technical Details

### Backend Changes:
- Stock search supports `.BO` suffix for BSE
- Add stock endpoint accepts "BSE" exchange parameter
- Yahoo Finance integration works with BSE stocks

### Frontend Changes:
- Exchange dropdown in Add Stock form
- Automatic exchange detection
- Display exchange badge on stock cards

## ✨ Benefits

- **Complete Coverage**: Trade stocks from both major Indian exchanges
- **Unified Interface**: Same UI for NSE and BSE stocks
- **Real-time Data**: Live prices for all stocks
- **Smart Search**: Find stocks across both exchanges

Now you can trade the entire universe of Indian stocks - both NSE and BSE! 🎉
