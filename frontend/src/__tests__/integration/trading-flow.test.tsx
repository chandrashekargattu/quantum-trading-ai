import React from 'react'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useMarketStore } from '@/store/useMarketStore'
import { useTradingStore } from '@/store/useTradingStore'
import { usePortfolioStore } from '@/store/usePortfolioStore'
import { marketService } from '@/services/api/market-optimized'
import { tradingService } from '@/services/api/trading'
import { portfolioService } from '@/services/api/portfolio-optimized'
import toast from 'react-hot-toast'

// Mock services
jest.mock('@/services/api/market-optimized')
jest.mock('@/services/api/trading')
jest.mock('@/services/api/portfolio-optimized')
jest.mock('react-hot-toast')

// Mock WebSocket
class MockWebSocket {
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  
  constructor(public url: string) {
    setTimeout(() => {
      if (this.onopen) this.onopen(new Event('open'))
    }, 100)
  }
  
  send(data: string) {
    // Mock sending data
  }
  
  close() {
    if (this.onclose) this.onclose(new CloseEvent('close'))
  }
}

global.WebSocket = MockWebSocket as any

// Trading Dashboard Component (simplified for testing)
const TradingDashboard = () => {
  const { selectedSymbol, selectedStock, loadOptionChain, selectSymbol } = useMarketStore()
  const { orderForm, placeOrder, updateOrderForm, loadOpenOrders, openOrders } = useTradingStore()
  const { activePortfolio, positions } = usePortfolioStore()
  
  React.useEffect(() => {
    loadOpenOrders()
  }, [loadOpenOrders])
  
  return (
    <div>
      <div data-testid="portfolio-value">
        Portfolio Value: ${activePortfolio?.currentValue || 0}
      </div>
      
      <div data-testid="symbol-search">
        <input
          type="text"
          placeholder="Search symbol..."
          onChange={(e) => {
            if (e.target.value.length >= 2) {
              selectSymbol(e.target.value.toUpperCase())
            }
          }}
        />
      </div>
      
      {selectedStock && (
        <div data-testid="stock-info">
          <h2>{selectedStock.symbol}</h2>
          <p>Price: ${selectedStock.price}</p>
          <p>Change: {selectedStock.change} ({selectedStock.changePercent}%)</p>
        </div>
      )}
      
      <div data-testid="order-form">
        <select
          value={orderForm.type}
          onChange={(e) => updateOrderForm({ type: e.target.value as any })}
        >
          <option value="MARKET">Market</option>
          <option value="LIMIT">Limit</option>
          <option value="STOP_LIMIT">Stop Limit</option>
        </select>
        
        <select
          value={orderForm.side}
          onChange={(e) => updateOrderForm({ side: e.target.value as any })}
        >
          <option value="BUY">Buy</option>
          <option value="SELL">Sell</option>
        </select>
        
        <input
          type="number"
          placeholder="Quantity"
          value={orderForm.quantity}
          onChange={(e) => updateOrderForm({ quantity: parseInt(e.target.value) })}
        />
        
        {orderForm.type !== 'MARKET' && (
          <input
            type="number"
            placeholder="Price"
            value={orderForm.price || ''}
            onChange={(e) => updateOrderForm({ price: parseFloat(e.target.value) })}
          />
        )}
        
        <button
          onClick={() => placeOrder({
            symbol: selectedSymbol!,
            ...orderForm
          })}
          disabled={!selectedSymbol}
        >
          Place Order
        </button>
      </div>
      
      <div data-testid="open-orders">
        <h3>Open Orders</h3>
        {openOrders.map(order => (
          <div key={order.id} data-testid={`order-${order.id}`}>
            {order.symbol} - {order.side} {order.quantity} @ ${order.price || 'Market'}
            <span data-testid={`order-status-${order.id}`}>{order.status}</span>
          </div>
        ))}
      </div>
      
      <div data-testid="positions">
        <h3>Positions</h3>
        {positions.map(position => (
          <div key={position.id} data-testid={`position-${position.id}`}>
            {position.symbol} - {position.quantity} shares
            <span>P&L: ${position.unrealizedPnL}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

describe('Trading Flow Integration', () => {
  const mockPortfolio = {
    id: 'portfolio-1',
    name: 'Main Portfolio',
    currentValue: 100000,
    cashBalance: 50000,
  }
  
  const mockStock = {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 150,
    change: 2.5,
    changePercent: 1.7,
    volume: 1000000,
    timestamp: new Date().toISOString(),
  }
  
  beforeEach(() => {
    jest.clearAllMocks()
    
    // Reset stores
    useMarketStore.setState({
      watchlist: [],
      watchlistData: {},
      selectedSymbol: null,
      selectedStock: null,
      optionChain: null,
      isLoadingStock: false,
      isLoadingOptions: false,
    })
    
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
        timeInForce: 'DAY',
      },
      isLoadingOrders: false,
      isPlacingOrder: false,
    })
    
    usePortfolioStore.setState({
      portfolios: [mockPortfolio],
      activePortfolio: mockPortfolio as any,
      positions: [],
      performance: null,
      isLoadingPortfolios: false,
    })
  })
  
  describe('Symbol Search and Selection', () => {
    it('should search and select a symbol', async () => {
      const user = userEvent.setup()
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      
      render(<TradingDashboard />)
      
      const searchInput = screen.getByPlaceholderText(/search symbol/i)
      await user.type(searchInput, 'AAPL')
      
      await waitFor(() => {
        expect(marketService.getStock).toHaveBeenCalledWith('AAPL')
      })
      
      // Check stock info display
      expect(screen.getByTestId('stock-info')).toBeInTheDocument()
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('Price: $150')).toBeInTheDocument()
    })
    
    it('should handle symbol not found', async () => {
      const user = userEvent.setup()
      ;(marketService.getStock as jest.Mock).mockRejectedValueOnce(
        new Error('Symbol not found')
      )
      
      render(<TradingDashboard />)
      
      const searchInput = screen.getByPlaceholderText(/search symbol/i)
      await user.type(searchInput, 'INVALID')
      
      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Symbol not found')
      })
    })
  })
  
  describe('Order Placement', () => {
    it('should place a market buy order', async () => {
      const user = userEvent.setup()
      const mockOrder = {
        id: 'order-1',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        type: 'MARKET',
        side: 'BUY',
        quantity: 100,
        filledQuantity: 0,
        remainingQuantity: 100,
        status: 'PENDING',
        timeInForce: 'DAY',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      ;(tradingService.placeOrder as jest.Mock).mockResolvedValueOnce(mockOrder)
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce([])
      
      render(<TradingDashboard />)
      
      // Select symbol first
      const searchInput = screen.getByPlaceholderText(/search symbol/i)
      await user.type(searchInput, 'AAPL')
      
      await waitFor(() => {
        expect(screen.getByTestId('stock-info')).toBeInTheDocument()
      })
      
      // Fill order form
      const typeSelect = screen.getByRole('combobox', { name: '' }).parentElement!.querySelector('select')!
      const quantityInput = screen.getByPlaceholderText(/quantity/i)
      const placeOrderButton = screen.getByRole('button', { name: /place order/i })
      
      await user.selectOptions(typeSelect, 'MARKET')
      await user.clear(quantityInput)
      await user.type(quantityInput, '100')
      await user.click(placeOrderButton)
      
      expect(tradingService.placeOrder).toHaveBeenCalledWith({
        symbol: 'AAPL',
        type: 'MARKET',
        side: 'BUY',
        quantity: 100,
        timeInForce: 'DAY',
      })
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Order placed successfully')
      })
    })
    
    it('should place a limit sell order', async () => {
      const user = userEvent.setup()
      const mockOrder = {
        id: 'order-2',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        type: 'LIMIT',
        side: 'SELL',
        quantity: 50,
        price: 155,
        filledQuantity: 0,
        remainingQuantity: 50,
        status: 'OPEN',
        timeInForce: 'DAY',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      ;(tradingService.placeOrder as jest.Mock).mockResolvedValueOnce(mockOrder)
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce([])
      
      render(<TradingDashboard />)
      
      // Select symbol
      const searchInput = screen.getByPlaceholderText(/search symbol/i)
      await user.type(searchInput, 'AAPL')
      
      await waitFor(() => {
        expect(screen.getByTestId('stock-info')).toBeInTheDocument()
      })
      
      // Fill order form
      const selects = screen.getAllByRole('combobox')
      const sideSelect = selects[1]
      const quantityInput = screen.getByPlaceholderText(/quantity/i)
      const priceInput = screen.getByPlaceholderText(/price/i)
      const placeOrderButton = screen.getByRole('button', { name: /place order/i })
      
      await user.selectOptions(sideSelect, 'SELL')
      await user.clear(quantityInput)
      await user.type(quantityInput, '50')
      await user.type(priceInput, '155')
      await user.click(placeOrderButton)
      
      expect(tradingService.placeOrder).toHaveBeenCalledWith({
        symbol: 'AAPL',
        type: 'LIMIT',
        side: 'SELL',
        quantity: 50,
        price: 155,
        timeInForce: 'DAY',
      })
    })
    
    it('should handle order placement errors', async () => {
      const user = userEvent.setup()
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      ;(tradingService.placeOrder as jest.Mock).mockRejectedValueOnce(
        new Error('Insufficient funds')
      )
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce([])
      
      render(<TradingDashboard />)
      
      // Select symbol and place order
      const searchInput = screen.getByPlaceholderText(/search symbol/i)
      await user.type(searchInput, 'AAPL')
      
      await waitFor(() => {
        expect(screen.getByTestId('stock-info')).toBeInTheDocument()
      })
      
      const placeOrderButton = screen.getByRole('button', { name: /place order/i })
      await user.click(placeOrderButton)
      
      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Insufficient funds')
      })
    })
  })
  
  describe('Order Management', () => {
    it('should display open orders', async () => {
      const mockOrders = [
        {
          id: 'order-1',
          portfolioId: 'portfolio-1',
          symbol: 'AAPL',
          type: 'LIMIT',
          side: 'BUY',
          quantity: 100,
          price: 149,
          filledQuantity: 0,
          remainingQuantity: 100,
          status: 'OPEN',
          timeInForce: 'DAY',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
        {
          id: 'order-2',
          portfolioId: 'portfolio-1',
          symbol: 'GOOGL',
          type: 'MARKET',
          side: 'SELL',
          quantity: 50,
          filledQuantity: 25,
          remainingQuantity: 25,
          status: 'PARTIALLY_FILLED',
          timeInForce: 'IOC',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      ]
      
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce(mockOrders)
      
      render(<TradingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('order-order-1')).toBeInTheDocument()
        expect(screen.getByTestId('order-order-2')).toBeInTheDocument()
      })
      
      // Check order details
      expect(screen.getByText(/AAPL - BUY 100 @ \$149/)).toBeInTheDocument()
      expect(screen.getByText(/GOOGL - SELL 50 @ Market/)).toBeInTheDocument()
      expect(screen.getByTestId('order-status-order-1')).toHaveTextContent('OPEN')
      expect(screen.getByTestId('order-status-order-2')).toHaveTextContent('PARTIALLY_FILLED')
    })
    
    it('should update order status via WebSocket', async () => {
      const mockOrder = {
        id: 'order-1',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        type: 'LIMIT',
        side: 'BUY',
        quantity: 100,
        price: 149,
        filledQuantity: 0,
        remainingQuantity: 100,
        status: 'OPEN',
        timeInForce: 'DAY',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(tradingService.getOpenOrders as jest.Mock).mockResolvedValueOnce([mockOrder])
      
      render(<TradingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('order-order-1')).toBeInTheDocument()
      })
      
      // Simulate WebSocket update
      act(() => {
        useTradingStore.getState().updateOrderStatus('order-1', 'FILLED', 100)
      })
      
      // Order should be moved to history
      await waitFor(() => {
        expect(screen.queryByTestId('order-order-1')).not.toBeInTheDocument()
      })
    })
  })
  
  describe('Position Management', () => {
    it('should display current positions', async () => {
      const mockPositions = [
        {
          id: 'pos-1',
          portfolioId: 'portfolio-1',
          symbol: 'AAPL',
          quantity: 100,
          avgPrice: 145,
          currentPrice: 150,
          marketValue: 15000,
          costBasis: 14500,
          unrealizedPnL: 500,
          unrealizedPnLPercent: 3.45,
          realizedPnL: 0,
          openedAt: new Date().toISOString(),
        },
        {
          id: 'pos-2',
          portfolioId: 'portfolio-1',
          symbol: 'GOOGL',
          quantity: 50,
          avgPrice: 2600,
          currentPrice: 2550,
          marketValue: 127500,
          costBasis: 130000,
          unrealizedPnL: -2500,
          unrealizedPnLPercent: -1.92,
          realizedPnL: 100,
          openedAt: new Date().toISOString(),
        },
      ]
      
      usePortfolioStore.setState({ positions: mockPositions })
      
      render(<TradingDashboard />)
      
      expect(screen.getByTestId('position-pos-1')).toBeInTheDocument()
      expect(screen.getByTestId('position-pos-2')).toBeInTheDocument()
      
      // Check position details
      expect(screen.getByText(/AAPL - 100 shares/)).toBeInTheDocument()
      expect(screen.getByText(/P&L: \$500/)).toBeInTheDocument()
      expect(screen.getByText(/GOOGL - 50 shares/)).toBeInTheDocument()
      expect(screen.getByText(/P&L: \$-2500/)).toBeInTheDocument()
    })
    
    it('should update position prices in real-time', async () => {
      const mockPosition = {
        id: 'pos-1',
        portfolioId: 'portfolio-1',
        symbol: 'AAPL',
        quantity: 100,
        avgPrice: 145,
        currentPrice: 150,
        marketValue: 15000,
        costBasis: 14500,
        unrealizedPnL: 500,
        unrealizedPnLPercent: 3.45,
        realizedPnL: 0,
        openedAt: new Date().toISOString(),
      }
      
      usePortfolioStore.setState({ positions: [mockPosition] })
      
      render(<TradingDashboard />)
      
      expect(screen.getByText(/P&L: \$500/)).toBeInTheDocument()
      
      // Simulate price update
      act(() => {
        usePortfolioStore.getState().updatePosition('pos-1', {
          currentPrice: 155,
          marketValue: 15500,
          unrealizedPnL: 1000,
          unrealizedPnLPercent: 6.90,
        })
      })
      
      expect(screen.getByText(/P&L: \$1000/)).toBeInTheDocument()
    })
  })
  
  describe('Real-time Updates', () => {
    it('should connect to WebSocket and receive updates', async () => {
      let wsInstance: MockWebSocket | null = null
      const originalWebSocket = global.WebSocket
      
      global.WebSocket = jest.fn().mockImplementation((url) => {
        wsInstance = new MockWebSocket(url)
        return wsInstance
      }) as any
      
      render(<TradingDashboard />)
      
      // Simulate WebSocket connection
      await waitFor(() => {
        expect(global.WebSocket).toHaveBeenCalledWith(
          expect.stringContaining('/ws/')
        )
      })
      
      // Simulate receiving market data update
      if (wsInstance && wsInstance.onmessage) {
        act(() => {
          wsInstance!.onmessage!(new MessageEvent('message', {
            data: JSON.stringify({
              type: 'MARKET_UPDATE',
              symbol: 'AAPL',
              price: 152,
              change: 2,
              changePercent: 1.33,
            })
          }))
        })
      }
      
      // Restore original WebSocket
      global.WebSocket = originalWebSocket
    })
  })
  
  describe('Portfolio Integration', () => {
    it('should update portfolio value after trades', async () => {
      const updatedPortfolio = {
        ...mockPortfolio,
        currentValue: 101000,
        cashBalance: 49000,
      }
      
      ;(portfolioService.getPortfolio as jest.Mock).mockResolvedValueOnce(updatedPortfolio)
      
      render(<TradingDashboard />)
      
      expect(screen.getByTestId('portfolio-value')).toHaveTextContent('Portfolio Value: $100000')
      
      // Simulate portfolio update after trade
      act(() => {
        usePortfolioStore.getState().refreshPortfolioData()
      })
      
      await waitFor(() => {
        expect(screen.getByTestId('portfolio-value')).toHaveTextContent('Portfolio Value: $101000')
      })
    })
  })
})
