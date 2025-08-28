import { apiClient } from './client'

export interface Stock {
  id: string
  symbol: string
  name: string
  exchange: string
  current_price: number
  previous_close: number
  open_price: number
  day_high: number
  day_low: number
  volume: number
  change_amount: number
  change_percent: number
  market_cap: number
  pe_ratio: number
  week_52_high: number
  week_52_low: number
  is_optionable: boolean
  last_updated: string
}

export interface Option {
  id: string
  symbol: string
  underlying_symbol: string
  strike: number
  expiration: string
  option_type: 'call' | 'put'
  bid: number
  ask: number
  last_price: number
  volume: number
  open_interest: number
  implied_volatility: number
  delta: number
  gamma: number
  theta: number
  vega: number
  rho: number
}

export interface OptionChain {
  underlying_symbol: string
  calls: Option[]
  puts: Option[]
  expirations: string[]
  strikes: number[]
}

export interface PriceHistory {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface MarketIndicator {
  symbol: string
  name: string
  value: number
  change_amount: number
  change_percent: number
}

export const marketService = {
  // Stock endpoints
  async getStock(symbol: string): Promise<Stock> {
    return apiClient.get<Stock>(`/stocks/${symbol}`)
  },

  async searchStocks(query: string): Promise<Stock[]> {
    return apiClient.get<Stock[]>('/stocks/search', {
      params: { q: query },
    })
  },

  async getTopMovers(): Promise<{
    gainers: Stock[]
    losers: Stock[]
    most_active: Stock[]
  }> {
    return apiClient.get('/stocks/movers')
  },

  // Options endpoints
  async getOptionChain(symbol: string): Promise<OptionChain> {
    return apiClient.get<OptionChain>(`/options/chain/${symbol}`)
  },

  async getOption(optionId: string): Promise<Option> {
    return apiClient.get<Option>(`/options/${optionId}`)
  },

  async getOptionsByExpiration(
    symbol: string,
    expiration: string
  ): Promise<{
    calls: Option[]
    puts: Option[]
  }> {
    return apiClient.get(`/options/${symbol}/${expiration}`)
  },

  // Historical data
  async getPriceHistory(
    symbol: string,
    interval: '1m' | '5m' | '15m' | '30m' | '1h' | '1d',
    period: '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y'
  ): Promise<PriceHistory[]> {
    return apiClient.get<PriceHistory[]>(`/market-data/history/${symbol}`, {
      params: { interval, period },
    })
  },

  // Market indicators
  async getMarketIndicators(): Promise<MarketIndicator[]> {
    return apiClient.get<MarketIndicator[]>('/market-data/indicators')
  },

  // Real-time quotes
  async getQuote(symbol: string): Promise<{
    symbol: string
    price: number
    bid: number
    ask: number
    bid_size: number
    ask_size: number
    timestamp: string
  }> {
    return apiClient.get(`/market-data/quote/${symbol}`)
  },

  // Batch quotes
  async getBatchQuotes(symbols: string[]): Promise<Record<string, any>> {
    return apiClient.post('/market-data/quotes/batch', { symbols })
  },
}
