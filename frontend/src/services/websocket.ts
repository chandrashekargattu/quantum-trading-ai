import { io, Socket } from 'socket.io-client'
import { useMarketStore } from '@/store/useMarketStore'

class WebSocketService {
  private socket: Socket | null = null
  private subscriptions: Set<string> = new Set()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5

  connect(token?: string) {
    if (this.socket?.connected) {
      return
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
    
    this.socket = io(wsUrl, {
      path: '/ws/socket.io',
      transports: ['websocket'],
      auth: token ? { token } : undefined,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 1000,
    })

    this.setupEventHandlers()
  }

  private setupEventHandlers() {
    if (!this.socket) return

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      
      // Re-subscribe to all symbols
      this.subscriptions.forEach(symbol => {
        this.subscribeToSymbol(symbol)
      })
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
    })

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error)
    })

    // Market data handlers
    this.socket.on('price_update', (data) => {
      this.handlePriceUpdate(data)
    })

    this.socket.on('market_update', (data) => {
      this.handleMarketUpdate(data)
    })

    this.socket.on('trade_update', (data) => {
      this.handleTradeUpdate(data)
    })

    this.socket.on('alert', (data) => {
      this.handleAlert(data)
    })
  }

  subscribeToSymbol(symbol: string) {
    if (!this.socket?.connected) {
      console.warn('Socket not connected')
      return
    }

    this.socket.emit('subscribe_symbol', { symbol })
    this.subscriptions.add(symbol)
  }

  unsubscribeFromSymbol(symbol: string) {
    if (!this.socket?.connected) {
      console.warn('Socket not connected')
      return
    }

    this.socket.emit('unsubscribe_symbol', { symbol })
    this.subscriptions.delete(symbol)
  }

  subscribeToPortfolio(portfolioId: string) {
    if (!this.socket?.connected) {
      console.warn('Socket not connected')
      return
    }

    this.socket.emit('subscribe_portfolio', { portfolio_id: portfolioId })
  }

  private handlePriceUpdate(data: any) {
    const { symbol, data: priceData } = data
    
    // Update store with new price data
    useMarketStore.getState().updateStockData(symbol, {
      current_price: priceData.price,
      change_amount: priceData.change,
      change_percent: priceData.change_percent,
      volume: priceData.volume,
      last_updated: new Date().toISOString(),
    })
  }

  private handleMarketUpdate(data: any) {
    // Handle general market updates
    console.log('Market update:', data)
  }

  private handleTradeUpdate(data: any) {
    // Handle trade updates
    console.log('Trade update:', data)
    // Could update a trading store here
  }

  private handleAlert(data: any) {
    // Handle alerts
    console.log('Alert:', data)
    // Show notification to user
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
      this.subscriptions.clear()
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false
  }
}

// Export singleton instance
export const wsService = new WebSocketService()
