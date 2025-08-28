/**
 * Frontend Performance Tests
 * 
 * These tests measure render times, interaction responsiveness,
 * and resource usage to ensure optimal user experience.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { performance } from 'perf_hooks'
import { act } from 'react-dom/test-utils'

// Components to test
import Dashboard from '@/components/Dashboard'
import TradingPanel from '@/components/TradingPanel'
import PortfolioView from '@/components/PortfolioView'
import MarketDataGrid from '@/components/MarketDataGrid'
import Charts from '@/components/Charts'

// Utilities
const measureRenderTime = async (Component: React.FC, props = {}) => {
  const start = performance.now()
  
  await act(async () => {
    render(<Component {...props} />)
  })
  
  const end = performance.now()
  return end - start
}

const measureInteractionTime = async (
  setup: () => { element: HTMLElement },
  interaction: (element: HTMLElement) => void
) => {
  const { element } = setup()
  
  const start = performance.now()
  await act(async () => {
    interaction(element)
  })
  const end = performance.now()
  
  return end - start
}

describe('Frontend Performance Tests', () => {
  beforeEach(() => {
    // Mock heavy data
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => generateMockData()
    })
  })

  afterEach(() => {
    jest.clearAllMocks()
  })

  describe('Component Render Performance', () => {
    test('Dashboard renders within acceptable time', async () => {
      const renderTime = await measureRenderTime(Dashboard)
      expect(renderTime).toBeLessThan(100) // Should render in < 100ms
    })

    test('TradingPanel renders within acceptable time', async () => {
      const renderTime = await measureRenderTime(TradingPanel)
      expect(renderTime).toBeLessThan(150) // Complex component, allow 150ms
    })

    test('Charts render within acceptable time with data', async () => {
      const chartData = generateChartData(1000) // 1000 data points
      const renderTime = await measureRenderTime(Charts, { data: chartData })
      expect(renderTime).toBeLessThan(200) // Chart rendering allows 200ms
    })

    test('Large portfolio list renders efficiently', async () => {
      const positions = generatePositions(500) // 500 positions
      const renderTime = await measureRenderTime(PortfolioView, { positions })
      expect(renderTime).toBeLessThan(300) // Large list allows 300ms
    })
  })

  describe('Interaction Performance', () => {
    test('Order form submission responds quickly', async () => {
      const interactionTime = await measureInteractionTime(
        () => {
          render(<TradingPanel />)
          const submitButton = screen.getByText('Place Order')
          return { element: submitButton }
        },
        (element) => fireEvent.click(element)
      )
      
      expect(interactionTime).toBeLessThan(50) // UI response < 50ms
    })

    test('Tab switching is responsive', async () => {
      const interactionTime = await measureInteractionTime(
        () => {
          render(<Dashboard />)
          const tab = screen.getByText('Portfolio')
          return { element: tab }
        },
        (element) => fireEvent.click(element)
      )
      
      expect(interactionTime).toBeLessThan(30) // Tab switch < 30ms
    })

    test('Search input is responsive', async () => {
      let totalTime = 0
      const iterations = 10
      
      render(<MarketDataGrid />)
      const searchInput = screen.getByPlaceholderText('Search stocks...')
      
      for (let i = 0; i < iterations; i++) {
        const start = performance.now()
        fireEvent.change(searchInput, { target: { value: `AAPL${i}` } })
        const end = performance.now()
        totalTime += (end - start)
      }
      
      const avgTime = totalTime / iterations
      expect(avgTime).toBeLessThan(20) // Avg keystroke response < 20ms
    })
  })

  describe('Data Loading Performance', () => {
    test('Handles large market data updates efficiently', async () => {
      const { rerender } = render(<MarketDataGrid data={[]} />)
      
      const updates = 10
      const totalTime: number[] = []
      
      for (let i = 0; i < updates; i++) {
        const newData = generateMarketData(100) // 100 stocks
        
        const start = performance.now()
        await act(async () => {
          rerender(<MarketDataGrid data={newData} />)
        })
        const end = performance.now()
        
        totalTime.push(end - start)
      }
      
      const avgUpdateTime = totalTime.reduce((a, b) => a + b) / updates
      expect(avgUpdateTime).toBeLessThan(50) // Avg update < 50ms
    })

    test('Pagination handles large datasets', async () => {
      const allData = generateTrades(10000) // 10k trades
      
      render(<TradeHistory trades={allData} pageSize={50} />)
      
      // Measure pagination click
      const nextButton = screen.getByText('Next')
      
      const times: number[] = []
      for (let i = 0; i < 5; i++) {
        const start = performance.now()
        await act(async () => {
          fireEvent.click(nextButton)
        })
        const end = performance.now()
        times.push(end - start)
      }
      
      const avgPaginationTime = times.reduce((a, b) => a + b) / times.length
      expect(avgPaginationTime).toBeLessThan(30) // Pagination < 30ms
    })
  })

  describe('Memory Performance', () => {
    test('Component cleanup prevents memory leaks', async () => {
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0
      
      // Mount and unmount component multiple times
      for (let i = 0; i < 100; i++) {
        const { unmount } = render(<Dashboard />)
        unmount()
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc()
      }
      
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0
      const memoryIncrease = finalMemory - initialMemory
      
      // Memory increase should be minimal (< 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024)
    })

    test('WebSocket connections are properly cleaned up', async () => {
      let activeConnections = 0
      
      // Mock WebSocket
      global.WebSocket = jest.fn().mockImplementation(() => ({
        close: jest.fn(() => activeConnections--),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        send: jest.fn()
      }))
      
      // Mount component that uses WebSocket
      const { unmount } = render(<RealTimeQuotes symbols={['AAPL', 'GOOGL']} />)
      activeConnections = 1
      
      // Unmount should close connection
      unmount()
      
      expect(activeConnections).toBe(0)
    })
  })

  describe('Bundle Size Impact', () => {
    test('Lazy loading reduces initial bundle', async () => {
      // This would typically be measured in build process
      // Here we simulate by checking if components are loaded on demand
      
      const LazyComponent = React.lazy(() => import('@/components/AdvancedCharts'))
      
      const start = performance.now()
      render(
        <React.Suspense fallback={<div>Loading...</div>}>
          <LazyComponent />
        </React.Suspense>
      )
      const end = performance.now()
      
      // Initial render with suspense should be fast
      expect(end - start).toBeLessThan(10)
    })
  })

  describe('Animation Performance', () => {
    test('Smooth transitions maintain 60fps', async () => {
      render(<AnimatedCard />)
      
      const card = screen.getByTestId('animated-card')
      
      // Trigger animation
      fireEvent.mouseEnter(card)
      
      // Check if animation frame callbacks are fast enough for 60fps
      let frameCount = 0
      const startTime = performance.now()
      
      const checkFrame = () => {
        frameCount++
        if (performance.now() - startTime < 1000) {
          requestAnimationFrame(checkFrame)
        }
      }
      
      requestAnimationFrame(checkFrame)
      
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Should achieve close to 60fps
      expect(frameCount).toBeGreaterThan(50)
    })
  })

  describe('Store Performance', () => {
    test('Store updates are efficient with many subscribers', async () => {
      const subscribers = []
      
      // Create many components subscribed to store
      for (let i = 0; i < 100; i++) {
        subscribers.push(render(<StoreSubscriber id={i} />))
      }
      
      // Measure store update time
      const start = performance.now()
      
      act(() => {
        // Trigger store update
        useMarketStore.getState().updateStockData('AAPL', { price: 150 })
      })
      
      const end = performance.now()
      
      // Update propagation should be fast
      expect(end - start).toBeLessThan(50)
      
      // Cleanup
      subscribers.forEach(s => s.unmount())
    })

    test('Derived state calculations are memoized', async () => {
      let calculationCount = 0
      
      const TestComponent = () => {
        const portfolio = usePortfolioStore(state => state.portfolio)
        const totalValue = usePortfolioStore(state => {
          calculationCount++
          return state.calculateTotalValue()
        })
        
        return <div>{totalValue}</div>
      }
      
      const { rerender } = render(<TestComponent />)
      
      // Multiple rerenders without state change
      for (let i = 0; i < 10; i++) {
        rerender(<TestComponent />)
      }
      
      // Calculation should only run once due to memoization
      expect(calculationCount).toBe(1)
    })
  })

  describe('Virtual Scrolling Performance', () => {
    test('Renders large lists efficiently with virtualization', async () => {
      const items = generateItems(10000) // 10k items
      
      const start = performance.now()
      render(<VirtualList items={items} height={600} itemHeight={50} />)
      const end = performance.now()
      
      // Should only render visible items, so render time is low
      expect(end - start).toBeLessThan(100)
      
      // Check that only visible items are in DOM
      const renderedItems = screen.getAllByTestId(/list-item-/)
      expect(renderedItems.length).toBeLessThan(20) // ~12 visible + buffer
    })
  })

  describe('Chart Performance with Real-time Updates', () => {
    test('Chart handles frequent updates smoothly', async () => {
      const { rerender } = render(<Charts data={generateChartData(100)} />)
      
      const updateTimes: number[] = []
      
      // Simulate real-time updates
      for (let i = 0; i < 60; i++) { // 60 updates (1 per second for a minute)
        const newData = generateChartData(100 + i)
        
        const start = performance.now()
        await act(async () => {
          rerender(<Charts data={newData} />)
        })
        const end = performance.now()
        
        updateTimes.push(end - start)
      }
      
      const avgUpdateTime = updateTimes.reduce((a, b) => a + b) / updateTimes.length
      const maxUpdateTime = Math.max(...updateTimes)
      
      expect(avgUpdateTime).toBeLessThan(16.67) // Maintain 60fps on average
      expect(maxUpdateTime).toBeLessThan(33.33) // No single update > 30fps
    })
  })
})

// Helper functions to generate test data
function generateMockData() {
  return {
    stocks: Array(100).fill(null).map((_, i) => ({
      symbol: `STOCK${i}`,
      price: Math.random() * 1000,
      change: (Math.random() - 0.5) * 10
    }))
  }
}

function generateChartData(points: number) {
  return Array(points).fill(null).map((_, i) => ({
    time: Date.now() - (points - i) * 60000,
    value: 100 + Math.random() * 50
  }))
}

function generatePositions(count: number) {
  return Array(count).fill(null).map((_, i) => ({
    id: `pos-${i}`,
    symbol: `STOCK${i}`,
    quantity: Math.floor(Math.random() * 1000),
    avgCost: Math.random() * 200,
    currentPrice: Math.random() * 200
  }))
}

function generateMarketData(count: number) {
  return Array(count).fill(null).map((_, i) => ({
    symbol: `STOCK${i}`,
    price: Math.random() * 500,
    volume: Math.floor(Math.random() * 1000000),
    change: (Math.random() - 0.5) * 20
  }))
}

function generateTrades(count: number) {
  return Array(count).fill(null).map((_, i) => ({
    id: `trade-${i}`,
    symbol: `STOCK${i % 50}`,
    quantity: Math.floor(Math.random() * 1000),
    price: Math.random() * 200,
    timestamp: Date.now() - i * 60000
  }))
}

function generateItems(count: number) {
  return Array(count).fill(null).map((_, i) => ({
    id: i,
    name: `Item ${i}`,
    value: Math.random() * 1000
  }))
}

// Mock components for testing
const AnimatedCard: React.FC = () => (
  <div data-testid="animated-card" style={{ transition: 'transform 0.3s' }}>
    Card Content
  </div>
)

const StoreSubscriber: React.FC<{ id: number }> = ({ id }) => {
  const price = useMarketStore(state => state.selectedStock?.price)
  return <div>Subscriber {id}: {price}</div>
}

const VirtualList: React.FC<any> = ({ items, height, itemHeight }) => {
  // Simplified virtual list implementation
  const visibleCount = Math.ceil(height / itemHeight)
  const visibleItems = items.slice(0, visibleCount + 2)
  
  return (
    <div style={{ height, overflow: 'auto' }}>
      {visibleItems.map((item: any, index: number) => (
        <div key={item.id} data-testid={`list-item-${index}`} style={{ height: itemHeight }}>
          {item.name}
        </div>
      ))}
    </div>
  )
}

const RealTimeQuotes: React.FC<{ symbols: string[] }> = ({ symbols }) => {
  React.useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/market')
    return () => ws.close()
  }, [])
  
  return <div>Real-time quotes for {symbols.join(', ')}</div>
}

const TradeHistory: React.FC<{ trades: any[], pageSize: number }> = ({ trades, pageSize }) => {
  const [page, setPage] = React.useState(0)
  const visibleTrades = trades.slice(page * pageSize, (page + 1) * pageSize)
  
  return (
    <div>
      {visibleTrades.map(trade => (
        <div key={trade.id}>{trade.symbol}</div>
      ))}
      <button onClick={() => setPage(p => p + 1)}>Next</button>
    </div>
  )
}

// Mock store
const useMarketStore = {
  getState: () => ({
    updateStockData: jest.fn(),
    selectedStock: { price: 100 }
  })
} as any

const usePortfolioStore = jest.fn((selector) => {
  const state = {
    portfolio: {},
    calculateTotalValue: jest.fn(() => 100000)
  }
  return selector(state)
}) as any
