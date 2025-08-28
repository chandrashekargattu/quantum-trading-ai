export interface Stock {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap?: number
  high?: number
  low?: number
  open?: number
  previousClose?: number
  timestamp: string
}

export interface Option {
  symbol: string
  strike: number
  expiration: string
  type: 'CALL' | 'PUT'
  bid: number
  ask: number
  last?: number
  volume: number
  openInterest: number
  impliedVolatility: number
  delta?: number
  gamma?: number
  theta?: number
  vega?: number
  rho?: number
}

export interface OptionChain {
  calls: Option[]
  puts: Option[]
  expirations: string[]
  strikes: number[]
}

export interface MarketQuote {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  bid: number
  ask: number
  timestamp: string
}

class MarketService {
  async getStock(symbol: string): Promise<Stock> {
    const response = await fetch(`/api/v1/market/stocks/${symbol}`)
    if (!response.ok) throw new Error('Failed to fetch stock data')
    return response.json()
  }

  async getBatchQuotes(symbols: string[]): Promise<MarketQuote[]> {
    const response = await fetch('/api/v1/market/quotes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbols })
    })
    if (!response.ok) throw new Error('Failed to fetch quotes')
    return response.json()
  }

  async getOptionChain(symbol: string): Promise<OptionChain> {
    const response = await fetch(`/api/v1/options/chain/${symbol}`)
    if (!response.ok) throw new Error('Failed to fetch option chain')
    return response.json()
  }

  async searchSymbols(query: string): Promise<Stock[]> {
    const response = await fetch(`/api/v1/market/search?q=${encodeURIComponent(query)}`)
    if (!response.ok) throw new Error('Failed to search symbols')
    return response.json()
  }

  async getMarketOverview(): Promise<{
    indices: any[]
    sectors: any[]
    topGainers: Stock[]
    topLosers: Stock[]
    mostActive: Stock[]
  }> {
    const response = await fetch('/api/v1/market/overview')
    if (!response.ok) throw new Error('Failed to fetch market overview')
    return response.json()
  }

  async getHistoricalData(
    symbol: string,
    interval: string,
    start: Date,
    end: Date
  ): Promise<{
    timestamps: string[]
    open: number[]
    high: number[]
    low: number[]
    close: number[]
    volume: number[]
  }> {
    const params = new URLSearchParams({
      symbol,
      interval,
      start: start.toISOString(),
      end: end.toISOString()
    })
    const response = await fetch(`/api/v1/market/historical?${params}`)
    if (!response.ok) throw new Error('Failed to fetch historical data')
    return response.json()
  }
}

export const marketService = new MarketService()