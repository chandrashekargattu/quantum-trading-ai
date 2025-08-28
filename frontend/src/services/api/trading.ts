import { apiClient } from './client'

export interface Trade {
  id: string
  trade_id: string
  symbol: string
  asset_type: 'stock' | 'option' | 'etf'
  side: 'buy' | 'sell'
  quantity: number
  price: number
  total_amount: number
  commission: number
  fees: number
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected'
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit'
  time_in_force: 'day' | 'gtc' | 'ioc' | 'fok'
  created_at: string
  executed_at?: string
}

export interface Position {
  id: string
  symbol: string
  asset_type: 'stock' | 'option' | 'etf'
  quantity: number
  avg_cost: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_percent: number
  realized_pnl: number
  is_open: boolean
  opened_at: string
}

export interface Portfolio {
  id: string
  name: string
  total_value: number
  cash_balance: number
  buying_power: number
  total_return: number
  total_return_percent: number
  daily_return: number
  daily_return_percent: number
  positions: Position[]
}

export interface OrderRequest {
  symbol: string
  asset_type: 'stock' | 'option'
  side: 'buy' | 'sell'
  quantity: number
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit'
  limit_price?: number
  stop_price?: number
  time_in_force?: 'day' | 'gtc' | 'ioc' | 'fok'
  option_id?: string
}

export interface OrderResponse {
  order_id: string
  status: string
  message?: string
}

export const tradingService = {
  // Portfolio endpoints
  async getPortfolios(): Promise<Portfolio[]> {
    return apiClient.get<Portfolio[]>('/portfolios')
  },

  async getPortfolio(portfolioId: string): Promise<Portfolio> {
    return apiClient.get<Portfolio>(`/portfolios/${portfolioId}`)
  },

  async createPortfolio(data: {
    name: string
    description?: string
    portfolio_type?: string
  }): Promise<Portfolio> {
    return apiClient.post<Portfolio>('/portfolios', data)
  },

  // Positions
  async getPositions(portfolioId?: string): Promise<Position[]> {
    const url = portfolioId 
      ? `/portfolios/${portfolioId}/positions`
      : '/positions'
    return apiClient.get<Position[]>(url)
  },

  async getPosition(positionId: string): Promise<Position> {
    return apiClient.get<Position>(`/positions/${positionId}`)
  },

  // Trading
  async placeOrder(order: OrderRequest): Promise<OrderResponse> {
    return apiClient.post<OrderResponse>('/trades/order', order)
  },

  async cancelOrder(orderId: string): Promise<{ message: string }> {
    return apiClient.delete(`/trades/order/${orderId}`)
  },

  async getTrades(params?: {
    portfolio_id?: string
    symbol?: string
    status?: string
    limit?: number
    offset?: number
  }): Promise<Trade[]> {
    return apiClient.get<Trade[]>('/trades', { params })
  },

  async getTrade(tradeId: string): Promise<Trade> {
    return apiClient.get<Trade>(`/trades/${tradeId}`)
  },

  // Orders
  async getOpenOrders(): Promise<any[]> {
    return apiClient.get('/trades/orders/open')
  },

  async getOrderHistory(params?: {
    limit?: number
    offset?: number
  }): Promise<any[]> {
    return apiClient.get('/trades/orders/history', { params })
  },

  // Performance
  async getPortfolioPerformance(
    portfolioId: string,
    period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | 'all'
  ): Promise<{
    dates: string[]
    values: number[]
    returns: number[]
  }> {
    return apiClient.get(`/portfolios/${portfolioId}/performance`, {
      params: { period },
    })
  },

  // Risk metrics
  async getPortfolioRisk(portfolioId: string): Promise<{
    var_95: number
    var_99: number
    sharpe_ratio: number
    max_drawdown: number
    beta: number
    alpha: number
  }> {
    return apiClient.get(`/portfolios/${portfolioId}/risk`)
  },
}
