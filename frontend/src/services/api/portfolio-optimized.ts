import { cachedFetch, prefetch, clearCache } from '@/lib/api-cache'

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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class OptimizedPortfolioService {
  private getHeaders() {
    // Only return content-type, auth is handled by authenticatedFetch
    return {
      'Content-Type': 'application/json'
    }
  }

  async getPortfolios(): Promise<Portfolio[]> {
    // Use cached fetch with 2 minute TTL for portfolios
    return cachedFetch<Portfolio[]>(
      `${API_BASE_URL}/api/v1/portfolios/`,
      { headers: this.getHeaders() },
      { ttl: 2 * 60 * 1000 } // 2 minutes
    )
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    return cachedFetch<Portfolio>(
      `${API_BASE_URL}/api/v1/portfolios/${id}`,
      { headers: this.getHeaders() },
      { ttl: 2 * 60 * 1000 }
    )
  }

  async createPortfolio(data: {
    name: string
    initialCapital: number
  }): Promise<Portfolio> {
    // Import authenticatedFetch at the top of the file
    const { authenticatedFetch } = await import('@/lib/auth-interceptor')
    
    const response = await authenticatedFetch(`${API_BASE_URL}/api/v1/portfolios/`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        name: data.name,
        initial_cash: data.initialCapital // Backend expects initial_cash, not initialCapital
      })
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('Portfolio creation error:', response.status, errorText)
      throw new Error(`Failed to create portfolio: ${response.status}`)
    }
    
    // Clear portfolio cache after creation
    const result = await response.json()
    
    // Clear the cache for portfolios to force refresh
    clearCache('portfolios')
    
    // Prefetch the updated portfolio list
    prefetch(`${API_BASE_URL}/api/v1/portfolios/`, { headers: this.getHeaders() })
    
    return result
  }

  // Batch operations for better performance
  async getPortfoliosSummary(): Promise<{
    portfolios: Portfolio[]
    totalValue: number
    totalGain: number
    bestPerformer?: Portfolio
  }> {
    const portfolios = await this.getPortfolios()
    
    const totalValue = portfolios.reduce((sum, p) => sum + p.currentValue, 0)
    const totalGain = portfolios.reduce((sum, p) => sum + p.totalReturn, 0)
    const bestPerformer = portfolios.reduce((best, p) => 
      !best || p.totalReturnPercent > best.totalReturnPercent ? p : best, 
      null as Portfolio | null
    ) || undefined

    return { portfolios, totalValue, totalGain, bestPerformer }
  }

  // Prefetch portfolios for faster initial load
  prefetchPortfolios() {
    prefetch(`${API_BASE_URL}/api/v1/portfolios/`, { headers: this.getHeaders() })
  }
}

export const portfolioService = new OptimizedPortfolioService()
// Also export with the old name for backward compatibility
export const optimizedPortfolioService = portfolioService
