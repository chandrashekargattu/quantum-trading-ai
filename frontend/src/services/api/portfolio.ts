export interface Portfolio {
  id: string
  name: string
  initialCapital: number
  currentValue: number
  totalReturn: number
  totalReturnPercent: number
  dayChange: number
  dayChangePercent: number
  cashBalance: number
  investedAmount: number
  createdAt: string
  updatedAt: string
}

export interface Position {
  id: string
  portfolioId: string
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  marketValue: number
  costBasis: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  realizedPnL: number
  dayChange: number
  dayChangePercent: number
  openedAt: string
  closedAt?: string
}

export interface Performance {
  totalReturn: number
  totalReturnPercent: number
  dailyReturn: number
  dailyReturnPercent: number
  monthlyReturn: number
  monthlyReturnPercent: number
  yearlyReturn: number
  yearlyReturnPercent: number
  sharpeRatio: number
  sortinoRatio: number
  maxDrawdown: number
  maxDrawdownPercent: number
  winRate: number
  profitFactor: number
  avgWin: number
  avgLoss: number
  bestDay: number
  worstDay: number
  volatility: number
  beta: number
  alpha: number
  chartData: {
    timestamps: string[]
    values: number[]
    returns: number[]
  }
}

export interface Transaction {
  id: string
  portfolioId: string
  type: 'BUY' | 'SELL' | 'DEPOSIT' | 'WITHDRAWAL' | 'DIVIDEND' | 'FEE'
  symbol?: string
  quantity?: number
  price?: number
  amount: number
  timestamp: string
  description?: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class PortfolioService {
  private getHeaders() {
    const token = localStorage.getItem('access_token')
    const tokenType = localStorage.getItem('token_type') || 'Bearer'
    return {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `${tokenType} ${token}` } : {})
    }
  }

  async getPortfolios(): Promise<Portfolio[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch portfolios')
    return response.json()
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${id}`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch portfolio')
    return response.json()
  }

  async createPortfolio(data: {
    name: string
    initialCapital: number
  }): Promise<Portfolio> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('Failed to create portfolio')
    return response.json()
  }

  async updatePortfolio(id: string, data: Partial<Portfolio>): Promise<Portfolio> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${id}`, {
      method: 'PATCH',
      headers: this.getHeaders(),
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('Failed to update portfolio')
    return response.json()
  }

  async deletePortfolio(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${id}`, {
      method: 'DELETE',
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to delete portfolio')
  }

  async getPositions(portfolioId: string): Promise<Position[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${portfolioId}/positions`, {
      headers: this.getHeaders()
    })
    if (!response.ok) throw new Error('Failed to fetch positions')
    return response.json()
  }

  async closePosition(portfolioId: string, positionId: string): Promise<void> {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/portfolios/${portfolioId}/positions/${positionId}/close`,
      { method: 'POST', headers: this.getHeaders() }
    )
    if (!response.ok) throw new Error('Failed to close position')
  }

  async getPerformance(portfolioId: string, period: string): Promise<Performance> {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/portfolios/${portfolioId}/performance?period=${period}`,
      { headers: this.getHeaders() }
    )
    if (!response.ok) throw new Error('Failed to fetch performance')
    return response.json()
  }

  async getTransactions(portfolioId: string, limit?: number): Promise<Transaction[]> {
    const url = limit 
      ? `${API_BASE_URL}/api/v1/portfolios/${portfolioId}/transactions?limit=${limit}`
      : `${API_BASE_URL}/api/v1/portfolios/${portfolioId}/transactions`
    const response = await fetch(url, { headers: this.getHeaders() })
    if (!response.ok) throw new Error('Failed to fetch transactions')
    return response.json()
  }

  async addFunds(portfolioId: string, amount: number): Promise<Transaction> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${portfolioId}/deposit`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ amount })
    })
    if (!response.ok) throw new Error('Failed to add funds')
    return response.json()
  }

  async withdrawFunds(portfolioId: string, amount: number): Promise<Transaction> {
    const response = await fetch(`${API_BASE_URL}/api/v1/portfolios/${portfolioId}/withdraw`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ amount })
    })
    if (!response.ok) throw new Error('Failed to withdraw funds')
    return response.json()
  }
}

// Note: This service has been replaced with the optimized version
// Use import { portfolioService } from './portfolio-optimized' instead
export const legacyPortfolioService = new PortfolioService()
