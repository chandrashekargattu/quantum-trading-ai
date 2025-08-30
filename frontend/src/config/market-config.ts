/**
 * Indian Market Configuration
 */

export const MARKET_CONFIG = {
  // Default currency
  currency: 'INR',
  currencySymbol: '₹',
  locale: 'en-IN',
  
  // Market hours (IST)
  marketHours: {
    preOpen: { start: '09:00', end: '09:08' },
    normal: { start: '09:15', end: '15:30' },
    postClose: { start: '15:40', end: '16:00' }
  },
  
  // Exchange suffixes
  exchanges: {
    NSE: '.NS',
    BSE: '.BO'
  },
  
  // Major indices
  indices: [
    { symbol: '^NSEI', name: 'NIFTY 50', shortName: 'NIFTY' },
    { symbol: '^BSESN', name: 'SENSEX', shortName: 'SENSEX' },
    { symbol: '^NSEBANK', name: 'Bank Nifty', shortName: 'BANKNIFTY' },
    { symbol: 'NIFTY_FIN_SERVICE.NS', name: 'Nifty Financial', shortName: 'FINNIFTY' },
    { symbol: '^NSMIDCP', name: 'Nifty Midcap', shortName: 'MIDCAP' },
    { symbol: '^CNXIT', name: 'Nifty IT', shortName: 'IT' },
    { symbol: '^CNXPHARMA', name: 'Nifty Pharma', shortName: 'PHARMA' },
    { symbol: '^CNXAUTO', name: 'Nifty Auto', shortName: 'AUTO' }
  ],
  
  // Popular stocks for quick access
  popularStocks: [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HDFC.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS'
  ],
  
  // Sectors
  sectors: {
    Banking: ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN'],
    IT: ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
    Pharma: ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP'],
    Auto: ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
    FMCG: ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR']
  },
  
  // F&O expiry (last Thursday of the month)
  getMonthlyExpiry: (date: Date = new Date()) => {
    const year = date.getFullYear()
    const month = date.getMonth()
    const lastDay = new Date(year, month + 1, 0).getDate()
    
    // Find last Thursday
    for (let day = lastDay; day >= 1; day--) {
      const checkDate = new Date(year, month, day)
      if (checkDate.getDay() === 4) { // Thursday
        return checkDate
      }
    }
    return null
  },
  
  // Format helpers
  formatPrice: (value: number): string => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  },
  
  formatVolume: (value: number): string => {
    if (value >= 10000000) {
      return `${(value / 10000000).toFixed(2)} Cr`
    } else if (value >= 100000) {
      return `${(value / 100000).toFixed(2)} L`
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(2)} K`
    }
    return value.toString()
  },
  
  // Add exchange suffix if not present
  normalizeSymbol: (symbol: string, exchange: 'NSE' | 'BSE' = 'NSE'): string => {
    if (symbol.includes('.')) {
      return symbol
    }
    return `${symbol}${MARKET_CONFIG.exchanges[exchange]}`
  }
}

// Helper function to check if market is open
export function isMarketOpen(): boolean {
  const now = new Date()
  const hours = now.getHours()
  const minutes = now.getMinutes()
  const currentTime = hours * 60 + minutes
  
  const marketStart = 9 * 60 + 15 // 9:15 AM
  const marketEnd = 15 * 60 + 30 // 3:30 PM
  
  const day = now.getDay()
  const isWeekend = day === 0 || day === 6 // Sunday or Saturday
  
  return !isWeekend && currentTime >= marketStart && currentTime <= marketEnd
}

// Helper to get market status
export function getMarketStatus(): {
  status: 'pre-open' | 'open' | 'closed' | 'post-close'
  message: string
} {
  const now = new Date()
  const hours = now.getHours()
  const minutes = now.getMinutes()
  const currentTime = hours * 60 + minutes
  
  const preOpenStart = 9 * 60 // 9:00 AM
  const preOpenEnd = 9 * 60 + 8 // 9:08 AM
  const marketStart = 9 * 60 + 15 // 9:15 AM
  const marketEnd = 15 * 60 + 30 // 3:30 PM
  const postCloseStart = 15 * 60 + 40 // 3:40 PM
  const postCloseEnd = 16 * 60 // 4:00 PM
  
  const day = now.getDay()
  const isWeekend = day === 0 || day === 6
  
  if (isWeekend) {
    return { status: 'closed', message: 'Market closed (Weekend)' }
  }
  
  if (currentTime >= preOpenStart && currentTime < preOpenEnd) {
    return { status: 'pre-open', message: 'Pre-open session' }
  }
  
  if (currentTime >= marketStart && currentTime <= marketEnd) {
    return { status: 'open', message: 'Market open' }
  }
  
  if (currentTime >= postCloseStart && currentTime <= postCloseEnd) {
    return { status: 'post-close', message: 'Post-close session' }
  }
  
  return { status: 'closed', message: 'Market closed' }
}
