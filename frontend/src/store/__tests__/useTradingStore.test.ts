import { renderHook, act, waitFor } from '@testing-library/react'
import { useTradingStore } from '../useTradingStore'
import { tradingService } from '@/services/api/trading'

// Mock the trading service
jest.mock('@/services/api/trading', () => ({
  tradingService: {
    getOpenOrders: jest.fn(),
    getOrderHistory: jest.fn(),
    getRecentTrades: jest.fn(),
    getOrderBook: jest.fn(),
    placeOrder: jest.fn(),
    cancelOrder: jest.fn(),
    cancelAllOrders: jest.fn(),
  },
  Order: {},
  Trade: {},
  OrderBook: {},
  OrderType: {},
  OrderSide: {},
}))

describe('useTradingStore', () => {
  const mockOpenOrders = [
    {
      id: 'order-1',
      portfolioId: 'portfolio-1',
      symbol: 'AAPL',
      type: 'LIMIT' as const,
      side: 'BUY' as const,
      quantity: 100,
      filledQuantity: 0,
      remainingQuantity: 100,
      price: 150,
      status: 'OPEN' as const,
      timeInForce: 'DAY' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    },
    {
      id: 'order-2',
      portfolioId: 'portfolio-1',
      symbol: 'GOOGL',
      type: 'LIMIT' as const,
      side: 'SELL' as const,
      quantity: 50,
      filledQuantity: 25,
      remainingQuantity: 25,
      price: 2600,
      status: 'PARTIALLY_FILLED' as const,
      timeInForce: 'DAY' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
  ]

  const mockTrades = [
    {
      id: 'trade-1',
      orderId: 'order-3',
      symbol: 'AAPL',
      side: 'BUY',
      quantity: 100,
      price: 149.50,
      timestamp: new Date().toISOString()
    },
    {
      id: 'trade-2',
      orderId: 'order-4',
      symbol: 'MSFT',
      side: 'SELL',
      quantity: 75,
      price: 305.25,
      timestamp: new Date().toISOString()
    }
  ]

  const mockOrderBook = {
    symbol: 'AAPL',
    bids: [
      { price: 149.90, quantity: 1000 },
      { price: 149.85, quantity: 2000 },
      { price: 149.80, quantity: 1500 }
    ],
    asks: [
      { price: 150.00, quantity: 800 },
      { price: 150.05, quantity: 1200 },
      { price: 150.10, quantity: 900 }
    ],
    timestamp: new Date().toISOString()
  }

  beforeEach(() => {
    jest.clearAllMocks()
    // Reset store state
    useTradingStore.setState({
      openOrders: [],
      orderHistory: [],
      recentTrades: [],
      orderBook: null,
      orderForm: {
        symbol: '',
        type: 'LIMIT',
        side: 'BUY',
        quantity: 100,
        price: undefined,
        stopPrice: undefined,
        timeInForce: 'DAY'
      },
      isLoadingOrders: false,
      isLoadingTrades: false,
      isLoadingOrderBook: false,
      isPlacingOrder: false,
    })
  })

  describe('Order Management', () => {
    it('should load open orders', async () => {
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce(mockOpenOrders)
      const { result } = renderHook(() => useTradingStore())

      await act(async () => {
        await result.current.loadOpenOrders()
      })

      expect(result.current.openOrders).toEqual(mockOpenOrders)
      expect(result.current.isLoadingOrders).toBe(false)
      expect(tradingService.getOpenOrders).toHaveBeenCalled()
    })

    it('should load order history', async () => {
      const mockHistory = [...mockOpenOrders].map(o => ({ ...o, status: 'FILLED' }))
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValueOnce(mockHistory)
      const { result } = renderHook(() => useTradingStore())

      await act(async () => {
        await result.current.loadOrderHistory(50)
      })

      expect(result.current.orderHistory).toEqual(mockHistory)
      expect(tradingService.getOrderHistory).toHaveBeenCalledWith(50)
    })

    it('should place new order', async () => {
      const newOrder = {
        id: 'order-3',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        type: 'LIMIT' as const,
        side: 'BUY' as const,
        quantity: 100,
        filledQuantity: 0,
        remainingQuantity: 100,
        price: 150,
        status: 'OPEN' as const,
        timeInForce: 'DAY' as const,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      ;(tradingService.placeOrder as jest.Mock).mockResolvedValueOnce(newOrder)
      const { result } = renderHook(() => useTradingStore())

      // Set order form
      act(() => {
        result.current.updateOrderForm({
          symbol: 'AAPL',
          type: 'LIMIT',
          side: 'BUY',
          quantity: 100,
          price: 150
        })
      })

      await act(async () => {
        const placed = await result.current.placeOrder({
          symbol: 'AAPL',
          type: 'LIMIT',
          side: 'BUY',
          quantity: 100,
          price: 150
        })
        expect(placed).toEqual(newOrder)
      })

      expect(result.current.openOrders).toContainEqual(newOrder)
      expect(result.current.orderForm.symbol).toBe('') // Should reset
      expect(result.current.isPlacingOrder).toBe(false)
    })

    it('should cancel single order', async () => {
      ;(tradingService.cancelOrder as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useTradingStore())
      
      // Set initial state
      act(() => {
        useTradingStore.setState({ openOrders: mockOpenOrders })
      })

      await act(async () => {
        await result.current.cancelOrder('order-1')
      })

      expect(result.current.openOrders).toHaveLength(1)
      expect(result.current.openOrders[0].id).toBe('order-2')
      expect(tradingService.cancelOrder).toHaveBeenCalledWith('order-1')
    })

    it('should cancel all orders', async () => {
      ;(tradingService.cancelAllOrders as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useTradingStore())
      
      // Set initial state
      act(() => {
        useTradingStore.setState({ 
          openOrders: mockOpenOrders,
          orderHistory: []
        })
      })

      await act(async () => {
        await result.current.cancelAllOrders()
      })

      expect(result.current.openOrders).toHaveLength(0)
      expect(result.current.orderHistory).toHaveLength(2)
      expect(result.current.orderHistory[0].status).toBe('CANCELLED')
      expect(tradingService.cancelAllOrders).toHaveBeenCalled()
    })
  })

  describe('Order Form Management', () => {
    it('should update order form', () => {
      const { result } = renderHook(() => useTradingStore())

      act(() => {
        result.current.updateOrderForm({
          symbol: 'AAPL',
          quantity: 200,
          price: 155
        })
      })

      expect(result.current.orderForm.symbol).toBe('AAPL')
      expect(result.current.orderForm.quantity).toBe(200)
      expect(result.current.orderForm.price).toBe(155)
      expect(result.current.orderForm.type).toBe('LIMIT') // Unchanged
    })

    it('should clear price for market orders', () => {
      const { result } = renderHook(() => useTradingStore())

      // Set initial price
      act(() => {
        result.current.updateOrderForm({ price: 150 })
      })

      // Change to market order
      act(() => {
        result.current.updateOrderForm({ type: 'MARKET' })
      })

      expect(result.current.orderForm.type).toBe('MARKET')
      expect(result.current.orderForm.price).toBeUndefined()
    })

    it('should set stop price for stop orders', () => {
      const { result } = renderHook(() => useTradingStore())

      act(() => {
        result.current.updateOrderForm({ type: 'STOP_LIMIT' })
      })

      expect(result.current.orderForm.stopPrice).toBe(0)
    })

    it('should reset order form', () => {
      const { result } = renderHook(() => useTradingStore())

      // Set some values
      act(() => {
        result.current.updateOrderForm({
          symbol: 'AAPL',
          quantity: 200,
          price: 155
        })
      })

      act(() => {
        result.current.resetOrderForm()
      })

      expect(result.current.orderForm).toEqual({
        symbol: '',
        type: 'LIMIT',
        side: 'BUY',
        quantity: 100,
        price: undefined,
        stopPrice: undefined,
        timeInForce: 'DAY'
      })
    })
  })

  describe('Trade History', () => {
    it('should load recent trades', async () => {
      ;(tradingService.getRecentTrades as jest.Mock).mockResolvedValueOnce(mockTrades)
      const { result } = renderHook(() => useTradingStore())

      await act(async () => {
        await result.current.loadRecentTrades()
      })

      expect(result.current.recentTrades).toEqual(mockTrades)
      expect(result.current.isLoadingTrades).toBe(false)
      expect(tradingService.getRecentTrades).toHaveBeenCalledWith(undefined)
    })

    it('should load trades for specific symbol', async () => {
      ;(tradingService.getRecentTrades as jest.Mock).mockResolvedValueOnce(
        mockTrades.filter(t => t.symbol === 'AAPL')
      )
      const { result } = renderHook(() => useTradingStore())

      await act(async () => {
        await result.current.loadRecentTrades('AAPL')
      })

      expect(result.current.recentTrades).toHaveLength(1)
      expect(result.current.recentTrades[0].symbol).toBe('AAPL')
      expect(tradingService.getRecentTrades).toHaveBeenCalledWith('AAPL')
    })
  })

  describe('Order Book', () => {
    it('should load order book', async () => {
      ;(tradingService.getOrderBook as jest.Mock).mockResolvedValueOnce(mockOrderBook)
      const { result } = renderHook(() => useTradingStore())

      await act(async () => {
        await result.current.loadOrderBook('AAPL')
      })

      expect(result.current.orderBook).toEqual(mockOrderBook)
      expect(result.current.isLoadingOrderBook).toBe(false)
      expect(tradingService.getOrderBook).toHaveBeenCalledWith('AAPL')
    })

    it('should handle order book loading errors', async () => {
      const error = new Error('Order book unavailable')
      ;(tradingService.getOrderBook as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useTradingStore())

      await expect(
        act(async () => {
          await result.current.loadOrderBook('AAPL')
        })
      ).rejects.toThrow('Order book unavailable')

      expect(result.current.isLoadingOrderBook).toBe(false)
    })
  })

  describe('Order Status Updates', () => {
    it('should update order status', () => {
      const { result } = renderHook(() => useTradingStore())
      
      // Set initial state
      act(() => {
        useTradingStore.setState({ openOrders: mockOpenOrders })
      })

      act(() => {
        result.current.updateOrderStatus('order-1', 'PARTIALLY_FILLED', 50)
      })

      const updatedOrder = result.current.openOrders.find(o => o.id === 'order-1')
      expect(updatedOrder?.status).toBe('PARTIALLY_FILLED')
      expect(updatedOrder?.filledQuantity).toBe(50)
    })

    it('should move filled orders to history', () => {
      const { result } = renderHook(() => useTradingStore())
      
      // Set initial state
      act(() => {
        useTradingStore.setState({ 
          openOrders: mockOpenOrders,
          orderHistory: []
        })
      })

      act(() => {
        result.current.updateOrderStatus('order-1', 'FILLED', 100)
      })

      expect(result.current.openOrders).toHaveLength(1)
      expect(result.current.orderHistory).toHaveLength(1)
      expect(result.current.orderHistory[0].id).toBe('order-1')
      expect(result.current.orderHistory[0].status).toBe('FILLED')
    })

    it('should move cancelled orders to history', () => {
      const { result } = renderHook(() => useTradingStore())
      
      // Set initial state
      act(() => {
        useTradingStore.setState({ 
          openOrders: mockOpenOrders,
          orderHistory: []
        })
      })

      act(() => {
        result.current.updateOrderStatus('order-2', 'CANCELLED')
      })

      expect(result.current.openOrders).toHaveLength(1)
      expect(result.current.orderHistory).toHaveLength(1)
      expect(result.current.orderHistory[0].id).toBe('order-2')
      expect(result.current.orderHistory[0].status).toBe('CANCELLED')
    })
  })

  describe('Loading States', () => {
    it('should manage placing order loading state', async () => {
      ;(tradingService.placeOrder as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockOpenOrders[0]), 100))
      )
      
      const { result } = renderHook(() => useTradingStore())

      act(() => {
        result.current.placeOrder({})
      })

      expect(result.current.isPlacingOrder).toBe(true)

      await waitFor(() => {
        expect(result.current.isPlacingOrder).toBe(false)
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle order placement errors', async () => {
      const error = new Error('Insufficient funds')
      ;(tradingService.placeOrder as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useTradingStore())

      await expect(
        act(async () => {
          await result.current.placeOrder({})
        })
      ).rejects.toThrow('Insufficient funds')

      expect(result.current.isPlacingOrder).toBe(false)
      expect(result.current.openOrders).toHaveLength(0)
    })

    it('should handle order cancellation errors', async () => {
      const error = new Error('Order already filled')
      ;(tradingService.cancelOrder as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useTradingStore())

      await expect(
        act(async () => {
          await result.current.cancelOrder('order-1')
        })
      ).rejects.toThrow('Order already filled')
    })
  })

  describe('Complex Workflows', () => {
    it('should handle complete trading workflow', async () => {
      const newOrder = {
        id: 'order-new',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        type: 'LIMIT' as const,
        side: 'BUY' as const,
        quantity: 100,
        filledQuantity: 0,
        remainingQuantity: 100,
        price: 150,
        status: 'OPEN' as const,
        timeInForce: 'DAY' as const,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce([])
      ;(tradingService.placeOrder as jest.Mock).mockResolvedValueOnce(newOrder)
      ;(tradingService.getOrderBook as jest.Mock).mockResolvedValueOnce(mockOrderBook)
      
      const { result } = renderHook(() => useTradingStore())

      // Load initial empty orders
      await act(async () => {
        await result.current.loadOpenOrders()
      })
      expect(result.current.openOrders).toHaveLength(0)

      // Load order book
      await act(async () => {
        await result.current.loadOrderBook('AAPL')
      })
      expect(result.current.orderBook).toBeTruthy()

      // Place order
      await act(async () => {
        await result.current.placeOrder(newOrder)
      })
      expect(result.current.openOrders).toHaveLength(1)

      // Simulate partial fill
      act(() => {
        result.current.updateOrderStatus('order-new', 'PARTIALLY_FILLED', 50)
      })
      expect(result.current.openOrders[0].filledQuantity).toBe(50)

      // Simulate complete fill
      act(() => {
        result.current.updateOrderStatus('order-new', 'FILLED', 100)
      })
      expect(result.current.openOrders).toHaveLength(0)
      expect(result.current.orderHistory).toHaveLength(1)
    })
  })
})
