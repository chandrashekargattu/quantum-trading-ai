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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class MarketService {
  private getHeaders() {
    const token = localStorage.getItem('access_token')
    const tokenType = localStorage.getItem('token_type') || 'Bearer'
    return {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `${tokenType} ${token}` } : {})
    }
  }

  async getStock(symbol: string): Promise<Stock> {
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/stocks/${symbol}`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch stock data')
    return response.json()
  }

  async getMarketIndicators(): Promise<any[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/indicators`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch market indicators')
    return response.json()
  }

  async getBatchQuotes(symbols: string[]): Promise<MarketQuote[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/quotes`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ symbols })
    })
    if (!response.ok) throw new Error('Failed to fetch quotes')
    return response.json()
  }

  async getOptionChain(symbol: string): Promise<OptionChain> {
    const response = await fetch(`${API_BASE_URL}/api/v1/options/chain/${symbol}`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch option chain')
    return response.json()
  }

  async searchSymbols(query: string): Promise<Stock[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/search?q=${encodeURIComponent(query)}`, {
      headers: this.getHeaders()
    })
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
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/overview`, {
      headers: this.getHeaders()
    })
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
    const response = await fetch(`${API_BASE_URL}/api/v1/market-data/historical?${params}`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch historical data')
    return response.json()
  }
}

// Note: This service has been replaced with the optimized version
// Use import { marketService } from './market-optimized' instead
export const legacyMarketService = new MarketService()