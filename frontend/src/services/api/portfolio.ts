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

class PortfolioService {
  async getPortfolios(): Promise<Portfolio[]> {
    const response = await fetch('/api/v1/portfolios')
    if (!response.ok) throw new Error('Failed to fetch portfolios')
    return response.json()
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    const response = await fetch(`/api/v1/portfolios/${id}`)
    if (!response.ok) throw new Error('Failed to fetch portfolio')
    return response.json()
  }

  async createPortfolio(data: {
    name: string
    initialCapital: number
  }): Promise<Portfolio> {
    const response = await fetch('/api/v1/portfolios', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('Failed to create portfolio')
    return response.json()
  }

  async updatePortfolio(id: string, data: Partial<Portfolio>): Promise<Portfolio> {
    const response = await fetch(`/api/v1/portfolios/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!response.ok) throw new Error('Failed to update portfolio')
    return response.json()
  }

  async deletePortfolio(id: string): Promise<void> {
    const response = await fetch(`/api/v1/portfolios/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete portfolio')
  }

  async getPositions(portfolioId: string): Promise<Position[]> {
    const response = await fetch(`/api/v1/portfolios/${portfolioId}/positions`)
    if (!response.ok) throw new Error('Failed to fetch positions')
    return response.json()
  }

  async closePosition(portfolioId: string, positionId: string): Promise<void> {
    const response = await fetch(
      `/api/v1/portfolios/${portfolioId}/positions/${positionId}/close`,
      { method: 'POST' }
    )
    if (!response.ok) throw new Error('Failed to close position')
  }

  async getPerformance(portfolioId: string, period: string): Promise<Performance> {
    const response = await fetch(
      `/api/v1/portfolios/${portfolioId}/performance?period=${period}`
    )
    if (!response.ok) throw new Error('Failed to fetch performance')
    return response.json()
  }

  async getTransactions(portfolioId: string, limit?: number): Promise<Transaction[]> {
    const url = limit 
      ? `/api/v1/portfolios/${portfolioId}/transactions?limit=${limit}`
      : `/api/v1/portfolios/${portfolioId}/transactions`
    const response = await fetch(url)
    if (!response.ok) throw new Error('Failed to fetch transactions')
    return response.json()
  }

  async addFunds(portfolioId: string, amount: number): Promise<Transaction> {
    const response = await fetch(`/api/v1/portfolios/${portfolioId}/deposit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount })
    })
    if (!response.ok) throw new Error('Failed to add funds')
    return response.json()
  }

  async withdrawFunds(portfolioId: string, amount: number): Promise<Transaction> {
    const response = await fetch(`/api/v1/portfolios/${portfolioId}/withdraw`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount })
    })
    if (!response.ok) throw new Error('Failed to withdraw funds')
    return response.json()
  }
}

export const portfolioService = new PortfolioService()
