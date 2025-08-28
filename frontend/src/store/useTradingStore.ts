import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { tradingService, Order, Trade, OrderBook, OrderType, OrderSide } from '@/services/api/trading'

interface TradingState {
  // Orders
  openOrders: Order[]
  orderHistory: Order[]
  
  // Trades
  recentTrades: Trade[]
  
  // Order book
  orderBook: OrderBook | null
  
  // Order placement
  orderForm: {
    symbol: string
    type: OrderType
    side: OrderSide
    quantity: number
    price?: number
    stopPrice?: number
    timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY'
  }
  
  // Loading states
  isLoadingOrders: boolean
  isLoadingTrades: boolean
  isLoadingOrderBook: boolean
  isPlacingOrder: boolean
  
  // Actions
  loadOpenOrders: () => Promise<void>
  loadOrderHistory: (limit?: number) => Promise<void>
  loadRecentTrades: (symbol?: string) => Promise<void>
  loadOrderBook: (symbol: string) => Promise<void>
  placeOrder: (order: Partial<Order>) => Promise<Order>
  cancelOrder: (orderId: string) => Promise<void>
  cancelAllOrders: () => Promise<void>
  updateOrderForm: (updates: Partial<TradingState['orderForm']>) => void
  resetOrderForm: () => void
  updateOrderStatus: (orderId: string, status: string, filledQuantity?: number) => void
}

const defaultOrderForm: TradingState['orderForm'] = {
  symbol: '',
  type: 'LIMIT',
  side: 'BUY',
  quantity: 100,
  price: undefined,
  stopPrice: undefined,
  timeInForce: 'DAY'
}

export const useTradingStore = create<TradingState>()(
  subscribeWithSelector((set, get) => ({
    openOrders: [],
    orderHistory: [],
    recentTrades: [],
    orderBook: null,
    orderForm: { ...defaultOrderForm },
    isLoadingOrders: false,
    isLoadingTrades: false,
    isLoadingOrderBook: false,
    isPlacingOrder: false,

    loadOpenOrders: async () => {
      set({ isLoadingOrders: true })
      try {
        const orders = await tradingService.getOpenOrders()
        set({ openOrders: orders, isLoadingOrders: false })
      } catch (error) {
        set({ isLoadingOrders: false })
        throw error
      }
    },

    loadOrderHistory: async (limit = 100) => {
      set({ isLoadingOrders: true })
      try {
        const orders = await tradingService.getOrderHistory(limit)
        set({ orderHistory: orders, isLoadingOrders: false })
      } catch (error) {
        set({ isLoadingOrders: false })
        throw error
      }
    },

    loadRecentTrades: async (symbol?: string) => {
      set({ isLoadingTrades: true })
      try {
        const trades = await tradingService.getRecentTrades(symbol)
        set({ recentTrades: trades, isLoadingTrades: false })
      } catch (error) {
        set({ isLoadingTrades: false })
        throw error
      }
    },

    loadOrderBook: async (symbol: string) => {
      set({ isLoadingOrderBook: true })
      try {
        const orderBook = await tradingService.getOrderBook(symbol)
        set({ orderBook, isLoadingOrderBook: false })
      } catch (error) {
        set({ isLoadingOrderBook: false })
        throw error
      }
    },

    placeOrder: async (orderData: Partial<Order>) => {
      set({ isPlacingOrder: true })
      try {
        const order = await tradingService.placeOrder(orderData)
        
        // Add to open orders
        set(state => ({
          openOrders: [order, ...state.openOrders],
          isPlacingOrder: false
        }))
        
        // Reset form after successful order
        get().resetOrderForm()
        
        return order
      } catch (error) {
        set({ isPlacingOrder: false })
        throw error
      }
    },

    cancelOrder: async (orderId: string) => {
      await tradingService.cancelOrder(orderId)
      
      // Remove from open orders and update status
      set(state => ({
        openOrders: state.openOrders.filter(o => o.id !== orderId),
        orderHistory: state.orderHistory.map(o =>
          o.id === orderId ? { ...o, status: 'CANCELLED' } : o
        )
      }))
    },

    cancelAllOrders: async () => {
      await tradingService.cancelAllOrders()
      
      // Move all open orders to history with cancelled status
      set(state => ({
        openOrders: [],
        orderHistory: [
          ...state.openOrders.map(o => ({ ...o, status: 'CANCELLED' as const })),
          ...state.orderHistory
        ]
      }))
    },

    updateOrderForm: (updates: Partial<TradingState['orderForm']>) => {
      set(state => ({
        orderForm: { ...state.orderForm, ...updates }
      }))
      
      // Auto-adjust based on order type
      const { type } = get().orderForm
      if (type === 'MARKET') {
        set(state => ({
          orderForm: { ...state.orderForm, price: undefined }
        }))
      } else if (type === 'STOP_LOSS' || type === 'STOP_LIMIT') {
        // Ensure stop price is set
        const { stopPrice } = get().orderForm
        if (!stopPrice) {
          set(state => ({
            orderForm: { ...state.orderForm, stopPrice: 0 }
          }))
        }
      }
    },

    resetOrderForm: () => {
      set({ orderForm: { ...defaultOrderForm } })
    },

    updateOrderStatus: (orderId: string, status: string, filledQuantity?: number) => {
      set(state => {
        const updatedOpenOrders = state.openOrders.map(order => {
          if (order.id === orderId) {
            const updated = { ...order, status }
            if (filledQuantity !== undefined) {
              updated.filledQuantity = filledQuantity
            }
            return updated
          }
          return order
        })
        
        // If order is filled or cancelled, remove from open orders
        if (status === 'FILLED' || status === 'CANCELLED') {
          const completedOrder = updatedOpenOrders.find(o => o.id === orderId)
          if (completedOrder) {
            return {
              openOrders: updatedOpenOrders.filter(o => o.id !== orderId),
              orderHistory: [completedOrder, ...state.orderHistory]
            }
          }
        }
        
        return { openOrders: updatedOpenOrders }
      })
    }
  }))
)
