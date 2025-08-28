export type OrderType = 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT'
export type OrderSide = 'BUY' | 'SELL'
export type OrderStatus = 'PENDING' | 'OPEN' | 'PARTIALLY_FILLED' | 'FILLED' | 'CANCELLED' | 'REJECTED'
export type TimeInForce = 'GTC' | 'IOC' | 'FOK' | 'DAY'

export interface Order {
  id: string
  portfolioId: string
  symbol: string
  type: OrderType
  side: OrderSide
  quantity: number
  filledQuantity: number
  remainingQuantity: number
  price?: number
  stopPrice?: number
  avgFillPrice?: number
  status: OrderStatus
  timeInForce: TimeInForce
  createdAt: string
  updatedAt: string
  filledAt?: string
  cancelledAt?: string
  rejectReason?: string
}

export interface Trade {
  id: string
  orderId: string
  portfolioId: string
  symbol: string
  side: OrderSide
  quantity: number
  price: number
  commission: number
  timestamp: string
}

export interface OrderBook {
  symbol: string
  bids: Array<{
    price: number
    quantity: number
    orders?: number
  }>
  asks: Array<{
    price: number
    quantity: number
    orders?: number
  }>
  spread: number
  spreadPercent: number
  timestamp: string
}

export interface MarketDepth {
  symbol: string
  totalBidVolume: number
  totalAskVolume: number
  imbalance: number
  levels: number
  timestamp: string
}

export interface ExecutionReport {
  orderId: string
  execId: string
  execType: 'NEW' | 'PARTIAL_FILL' | 'FILL' | 'CANCELLED' | 'REJECTED'
  orderStatus: OrderStatus
  symbol: string
  side: OrderSide
  lastQty?: number
  lastPrice?: number
  leavesQty: number
  cumQty: number
  avgPrice?: number
  timestamp: string
}

class TradingService {
  async placeOrder(order: Partial<Order>): Promise<Order> {
    const response = await fetch('/api/v1/orders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(order)
    })
    if (!response.ok) throw new Error('Failed to place order')
    return response.json()
  }

  async getOpenOrders(portfolioId?: string): Promise<Order[]> {
    const url = portfolioId 
      ? `/api/v1/orders/open?portfolioId=${portfolioId}`
      : '/api/v1/orders/open'
    const response = await fetch(url)
    if (!response.ok) throw new Error('Failed to fetch open orders')
    return response.json()
  }

  async getOrderHistory(limit = 100, portfolioId?: string): Promise<Order[]> {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (portfolioId) params.append('portfolioId', portfolioId)
    
    const response = await fetch(`/api/v1/orders/history?${params}`)
    if (!response.ok) throw new Error('Failed to fetch order history')
    return response.json()
  }

  async getOrder(orderId: string): Promise<Order> {
    const response = await fetch(`/api/v1/orders/${orderId}`)
    if (!response.ok) throw new Error('Failed to fetch order')
    return response.json()
  }

  async cancelOrder(orderId: string): Promise<void> {
    const response = await fetch(`/api/v1/orders/${orderId}/cancel`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to cancel order')
  }

  async cancelAllOrders(portfolioId?: string): Promise<void> {
    const url = portfolioId
      ? `/api/v1/orders/cancel-all?portfolioId=${portfolioId}`
      : '/api/v1/orders/cancel-all'
    const response = await fetch(url, { method: 'POST' })
    if (!response.ok) throw new Error('Failed to cancel all orders')
  }

  async modifyOrder(orderId: string, updates: {
    quantity?: number
    price?: number
    stopPrice?: number
  }): Promise<Order> {
    const response = await fetch(`/api/v1/orders/${orderId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    })
    if (!response.ok) throw new Error('Failed to modify order')
    return response.json()
  }

  async getRecentTrades(symbol?: string, limit = 50): Promise<Trade[]> {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (symbol) params.append('symbol', symbol)
    
    const response = await fetch(`/api/v1/trades?${params}`)
    if (!response.ok) throw new Error('Failed to fetch trades')
    return response.json()
  }

  async getOrderBook(symbol: string, depth = 10): Promise<OrderBook> {
    const response = await fetch(
      `/api/v1/market/orderbook/${symbol}?depth=${depth}`
    )
    if (!response.ok) throw new Error('Failed to fetch order book')
    return response.json()
  }

  async getMarketDepth(symbol: string): Promise<MarketDepth> {
    const response = await fetch(`/api/v1/market/depth/${symbol}`)
    if (!response.ok) throw new Error('Failed to fetch market depth')
    return response.json()
  }

  async getExecutionReports(
    orderId?: string,
    startTime?: Date,
    endTime?: Date
  ): Promise<ExecutionReport[]> {
    const params = new URLSearchParams()
    if (orderId) params.append('orderId', orderId)
    if (startTime) params.append('startTime', startTime.toISOString())
    if (endTime) params.append('endTime', endTime.toISOString())
    
    const response = await fetch(`/api/v1/executions?${params}`)
    if (!response.ok) throw new Error('Failed to fetch execution reports')
    return response.json()
  }

  // WebSocket connection for real-time order updates
  connectOrderUpdates(
    onUpdate: (update: ExecutionReport) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/orders`)
    
    ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data)
        onUpdate(update)
      } catch (error) {
        onError?.(new Error('Failed to parse order update'))
      }
    }
    
    ws.onerror = () => {
      onError?.(new Error('WebSocket connection error'))
    }
    
    return () => ws.close()
  }
}

export const tradingService = new TradingService()