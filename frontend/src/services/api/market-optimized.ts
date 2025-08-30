import { cachedFetch, prefetch } from '@/lib/api-cache'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Batch multiple indicator requests into one
interface BatchRequest {
  symbols: string[]
  resolve: (value: any) => void
  reject: (error: any) => void
}

class OptimizedMarketService {
  private batchQueue: BatchRequest[] = []
  private batchTimer: NodeJS.Timeout | null = null
  private readonly BATCH_DELAY = 50 // 50ms delay to collect requests

  private getHeaders() {
    // Only return content-type, auth is handled by authenticatedFetch
    return {
      'Content-Type': 'application/json'
    }
  }

  async getMarketIndicators(): Promise<any[]> {
    // Market indicators change frequently, cache for 30 seconds
    return cachedFetch<any[]>(
      `${API_BASE_URL}/api/v1/market-data/indicators`,
      { headers: this.getHeaders() },
      { ttl: 30 * 1000 } // 30 seconds
    )
  }

  async getStock(symbol: string): Promise<any> {
    // Individual stocks, cache for 1 minute
    return cachedFetch<any>(
      `${API_BASE_URL}/api/v1/market-data/stocks/${symbol}`,
      { headers: this.getHeaders() },
      { ttl: 60 * 1000 }
    )
  }

  // Batch multiple quote requests
  async getBatchQuotesOptimized(symbols: string[]): Promise<Map<string, any>> {
    return new Promise((resolve, reject) => {
      this.batchQueue.push({ symbols, resolve, reject })
      
      if (!this.batchTimer) {
        this.batchTimer = setTimeout(() => this.processBatchQueue(), this.BATCH_DELAY)
      }
    })
  }

  private async processBatchQueue() {
    const queue = [...this.batchQueue]
    this.batchQueue = []
    this.batchTimer = null

    if (queue.length === 0) return

    // Collect all unique symbols
    const allSymbols = new Set<string>()
    queue.forEach(req => req.symbols.forEach(s => allSymbols.add(s)))

    try {
      // Make single batch request
      const quotes = await cachedFetch<any[]>(
        `${API_BASE_URL}/api/v1/market-data/quotes/batch`,
        {
          method: 'POST',
          headers: this.getHeaders(),
          body: JSON.stringify({ symbols: Array.from(allSymbols) })
        },
        { ttl: 30 * 1000 } // 30 second cache
      )

      // Create a map for quick lookup
      const quoteMap = new Map(quotes.map(q => [q.symbol, q]))

      // Resolve each request with its requested symbols
      queue.forEach(req => {
        const result = new Map()
        req.symbols.forEach(symbol => {
          const quote = quoteMap.get(symbol)
          if (quote) result.set(symbol, quote)
        })
        req.resolve(result)
      })
    } catch (error) {
      // Reject all requests in the batch
      queue.forEach(req => req.reject(error))
    }
  }

  // Optimized option chain with pagination
  async getOptionChain(symbol: string, expiration?: string): Promise<any> {
    const url = expiration 
      ? `${API_BASE_URL}/api/v1/options/chain/${symbol}?expiration=${expiration}`
      : `${API_BASE_URL}/api/v1/options/chain/${symbol}`
    
    return cachedFetch<any>(
      url,
      { headers: this.getHeaders() },
      { ttl: 5 * 60 * 1000 } // 5 minute cache for options
    )
  }

  // Get market overview - using indicators for now
  async getMarketOverviewOptimized(): Promise<any> {
    // Just return indicators as overview for now
    return this.getMarketIndicators()
  }

  // Prefetch critical market data
  prefetchMarketData() {
    const headers = { headers: this.getHeaders() }
    
    // Prefetch indicators
    prefetch(`${API_BASE_URL}/api/v1/market-data/indicators`, headers)
  }

  // Get trending options with pagination
  async getTrendingOptions(page: number = 1, limit: number = 10): Promise<{
    options: any[]
    hasMore: boolean
    total: number
  }> {
    const offset = (page - 1) * limit
    const data = await cachedFetch<any>(
      `${API_BASE_URL}/api/v1/options/trending?offset=${offset}&limit=${limit}`,
      { headers: this.getHeaders() },
      { ttl: 2 * 60 * 1000 } // 2 minute cache
    )

    return {
      options: data.items || [],
      hasMore: data.has_more || false,
      total: data.total || 0
    }
  }

  // WebSocket-like updates using polling with exponential backoff
  startMarketUpdates(callback: (data: any) => void, symbols: string[] = []) {
    let interval = 5000 // Start with 5 second updates
    const maxInterval = 30000 // Max 30 seconds
    
    const update = async () => {
      try {
        if (symbols.length > 0) {
          const quotes = await this.getBatchQuotesOptimized(symbols)
          callback({ type: 'quotes', data: quotes })
        } else {
          const indicators = await this.getMarketIndicators()
          callback({ type: 'indicators', data: indicators })
        }
        
        // Reset interval on success
        interval = 5000
      } catch (error) {
        // Exponential backoff on error
        interval = Math.min(interval * 2, maxInterval)
      }
    }

    // Initial update
    update()

    // Set up polling
    const intervalId = setInterval(update, interval)

    // Return cleanup function
    return () => clearInterval(intervalId)
  }
}

export const marketService = new OptimizedMarketService()
// Also export with the old name for backward compatibility
export const optimizedMarketService = marketService
