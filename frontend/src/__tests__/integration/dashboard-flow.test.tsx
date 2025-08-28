import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { usePortfolioStore } from '@/store/usePortfolioStore'
import { useMarketStore } from '@/store/useMarketStore'
import { useAlertStore } from '@/store/useAlertStore'
import { portfolioService } from '@/services/api/portfolio'
import { marketService } from '@/services/api/market'
import { alertService } from '@/services/api/alerts'

// Mock services
jest.mock('@/services/api/portfolio')
jest.mock('@/services/api/market')
jest.mock('@/services/api/alerts')

// Mock Chart components
jest.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
}))

// Dashboard Component
const Dashboard = () => {
  const { 
    portfolios, 
    activePortfolio, 
    positions, 
    performance,
    loadPortfolios,
    loadPerformance,
    loadPositions,
  } = usePortfolioStore()
  
  const { watchlist, watchlistData, addToWatchlist, removeFromWatchlist } = useMarketStore()
  const { activeAlerts, loadActiveAlerts } = useAlertStore()
  
  React.useEffect(() => {
    loadPortfolios()
    loadActiveAlerts()
  }, [loadPortfolios, loadActiveAlerts])
  
  React.useEffect(() => {
    if (activePortfolio) {
      loadPositions(activePortfolio.id)
      loadPerformance(activePortfolio.id, '1M')
    }
  }, [activePortfolio, loadPositions, loadPerformance])
  
  return (
    <div>
      <h1>Trading Dashboard</h1>
      
      {/* Portfolio Summary */}
      <div data-testid="portfolio-summary">
        <h2>Portfolio Summary</h2>
        {activePortfolio && (
          <>
            <div data-testid="total-value">
              Total Value: ${activePortfolio.currentValue.toLocaleString()}
            </div>
            <div data-testid="cash-balance">
              Cash Balance: ${activePortfolio.cashBalance.toLocaleString()}
            </div>
            <div data-testid="daily-change">
              Daily Change: ${activePortfolio.dayChange?.toLocaleString() || '0'} 
              ({activePortfolio.dayChangePercent || 0}%)
            </div>
          </>
        )}
      </div>
      
      {/* Performance Chart */}
      {performance && (
        <div data-testid="performance-chart">
          <h3>Portfolio Performance</h3>
          <div data-testid="performance-metrics">
            <span>Total Return: {performance.totalReturnPercent}%</span>
            <span>Sharpe Ratio: {performance.sharpeRatio}</span>
            <span>Max Drawdown: {performance.maxDrawdownPercent}%</span>
          </div>
          <div data-testid="chart-container">
            {/* Chart would be rendered here */}
          </div>
        </div>
      )}
      
      {/* Positions */}
      <div data-testid="positions-section">
        <h3>Current Positions</h3>
        {positions.length === 0 ? (
          <p>No open positions</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Quantity</th>
                <th>Avg Price</th>
                <th>Current Price</th>
                <th>P&L</th>
                <th>P&L %</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(position => (
                <tr key={position.id} data-testid={`position-row-${position.symbol}`}>
                  <td>{position.symbol}</td>
                  <td>{position.quantity}</td>
                  <td>${position.avgPrice}</td>
                  <td>${position.currentPrice}</td>
                  <td className={position.unrealizedPnL >= 0 ? 'profit' : 'loss'}>
                    ${position.unrealizedPnL}
                  </td>
                  <td className={position.unrealizedPnLPercent >= 0 ? 'profit' : 'loss'}>
                    {position.unrealizedPnLPercent}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      {/* Watchlist */}
      <div data-testid="watchlist-section">
        <h3>Watchlist</h3>
        <input
          type="text"
          placeholder="Add symbol..."
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              const input = e.target as HTMLInputElement
              addToWatchlist(input.value.toUpperCase())
              input.value = ''
            }
          }}
        />
        <div>
          {watchlist.map(symbol => (
            <div key={symbol} data-testid={`watchlist-item-${symbol}`}>
              <span>{symbol}</span>
              {watchlistData[symbol] && (
                <>
                  <span>${watchlistData[symbol].price}</span>
                  <span className={watchlistData[symbol].change >= 0 ? 'profit' : 'loss'}>
                    {watchlistData[symbol].changePercent}%
                  </span>
                </>
              )}
              <button onClick={() => removeFromWatchlist(symbol)}>Remove</button>
            </div>
          ))}
        </div>
      </div>
      
      {/* Active Alerts */}
      <div data-testid="alerts-section">
        <h3>Active Alerts</h3>
        {activeAlerts.length === 0 ? (
          <p>No active alerts</p>
        ) : (
          <ul>
            {activeAlerts.map(alert => (
              <li key={alert.id} data-testid={`alert-${alert.id}`}>
                {alert.symbol} - {alert.type} {alert.condition} {alert.value}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

describe('Dashboard Integration', () => {
  const mockPortfolios = [
    {
      id: 'portfolio-1',
      name: 'Main Portfolio',
      initialCapital: 100000,
      currentValue: 125000,
      totalReturn: 25000,
      totalReturnPercent: 25,
      dayChange: 1500,
      dayChangePercent: 1.2,
      cashBalance: 20000,
      investedAmount: 105000,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ]
  
  const mockPositions = [
    {
      id: 'pos-1',
      portfolioId: 'portfolio-1',
      symbol: 'AAPL',
      quantity: 100,
      avgPrice: 140,
      currentPrice: 150,
      marketValue: 15000,
      costBasis: 14000,
      unrealizedPnL: 1000,
      unrealizedPnLPercent: 7.14,
      realizedPnL: 0,
      dayChange: 200,
      dayChangePercent: 1.35,
      openedAt: new Date().toISOString(),
    },
    {
      id: 'pos-2',
      portfolioId: 'portfolio-1',
      symbol: 'MSFT',
      quantity: 50,
      avgPrice: 300,
      currentPrice: 310,
      marketValue: 15500,
      costBasis: 15000,
      unrealizedPnL: 500,
      unrealizedPnLPercent: 3.33,
      realizedPnL: 100,
      dayChange: -100,
      dayChangePercent: -0.64,
      openedAt: new Date().toISOString(),
    },
  ]
  
  const mockPerformance = {
    totalReturn: 25000,
    totalReturnPercent: 25,
    dailyReturn: 1500,
    dailyReturnPercent: 1.2,
    monthlyReturn: 5000,
    monthlyReturnPercent: 4.2,
    yearlyReturn: 25000,
    yearlyReturnPercent: 25,
    sharpeRatio: 1.8,
    sortinoRatio: 2.1,
    maxDrawdown: -8500,
    maxDrawdownPercent: -6.8,
    winRate: 65,
    profitFactor: 2.1,
    avgWin: 500,
    avgLoss: -238,
    bestDay: 3000,
    worstDay: -2000,
    volatility: 15.2,
    beta: 0.85,
    alpha: 0.12,
    chartData: {
      timestamps: ['2024-01-01', '2024-01-02', '2024-01-03'],
      values: [100000, 102000, 105000],
      returns: [0, 2, 2.94],
    },
  }
  
  const mockAlerts = [
    {
      id: 'alert-1',
      userId: 'user-1',
      symbol: 'AAPL',
      type: 'PRICE' as const,
      condition: 'ABOVE' as const,
      value: 160,
      enabled: true,
      triggered: false,
      sendEmail: true,
      sendPush: true,
      sendSMS: false,
      priority: 'HIGH' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ]
  
  beforeEach(() => {
    jest.clearAllMocks()
    
    // Reset stores
    usePortfolioStore.setState({
      portfolios: [],
      activePortfolio: null,
      positions: [],
      performance: null,
      isLoadingPortfolios: false,
    })
    
    useMarketStore.setState({
      watchlist: [],
      watchlistData: {},
    })
    
    useAlertStore.setState({
      alerts: [],
      activeAlerts: [],
      triggeredAlerts: [],
    })
  })
  
  describe('Initial Load', () => {
    it('should load all dashboard data on mount', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<Dashboard />)
      
      // Check initial loading
      expect(portfolioService.getPortfolios).toHaveBeenCalled()
      expect(alertService.getActiveAlerts).toHaveBeenCalled()
      
      // Wait for portfolio data
      await waitFor(() => {
        expect(screen.getByTestId('total-value')).toHaveTextContent('Total Value: $125,000')
      })
      
      // Check that positions and performance are loaded
      expect(portfolioService.getPositions).toHaveBeenCalledWith('portfolio-1')
      expect(portfolioService.getPerformance).toHaveBeenCalledWith('portfolio-1', '1M')
      
      // Check positions display
      await waitFor(() => {
        expect(screen.getByTestId('position-row-AAPL')).toBeInTheDocument()
        expect(screen.getByTestId('position-row-MSFT')).toBeInTheDocument()
      })
      
      // Check performance metrics
      expect(screen.getByTestId('performance-metrics')).toHaveTextContent('Total Return: 25%')
      expect(screen.getByTestId('performance-metrics')).toHaveTextContent('Sharpe Ratio: 1.8')
      
      // Check alerts
      expect(screen.getByTestId('alert-alert-1')).toBeInTheDocument()
    })
    
    it('should handle empty portfolio gracefully', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([])
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText('No open positions')).toBeInTheDocument()
        expect(screen.getByText('No active alerts')).toBeInTheDocument()
      })
    })
  })
  
  describe('Portfolio Summary', () => {
    it('should display portfolio metrics correctly', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('total-value')).toHaveTextContent('Total Value: $125,000')
        expect(screen.getByTestId('cash-balance')).toHaveTextContent('Cash Balance: $20,000')
        expect(screen.getByTestId('daily-change')).toHaveTextContent('Daily Change: $1,500 (1.2%)')
      })
    })
    
    it('should update portfolio values in real-time', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('total-value')).toHaveTextContent('Total Value: $125,000')
      })
      
      // Simulate portfolio update
      const updatedPortfolio = {
        ...mockPortfolios[0],
        currentValue: 126500,
        dayChange: 3000,
        dayChangePercent: 2.4,
      }
      
      act(() => {
        usePortfolioStore.setState({
          activePortfolio: updatedPortfolio,
        })
      })
      
      expect(screen.getByTestId('total-value')).toHaveTextContent('Total Value: $126,500')
      expect(screen.getByTestId('daily-change')).toHaveTextContent('Daily Change: $3,000 (2.4%)')
    })
  })
  
  describe('Positions Display', () => {
    it('should show position details with P&L', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        const aaplRow = screen.getByTestId('position-row-AAPL')
        expect(aaplRow).toHaveTextContent('AAPL')
        expect(aaplRow).toHaveTextContent('100')
        expect(aaplRow).toHaveTextContent('$140')
        expect(aaplRow).toHaveTextContent('$150')
        expect(aaplRow).toHaveTextContent('$1000')
        expect(aaplRow).toHaveTextContent('7.14%')
        
        // Check profit/loss styling
        const pnlCells = aaplRow.querySelectorAll('.profit')
        expect(pnlCells).toHaveLength(2)
      })
    })
    
    it('should show negative P&L correctly', async () => {
      const lossPosition = {
        ...mockPositions[0],
        currentPrice: 130,
        marketValue: 13000,
        unrealizedPnL: -1000,
        unrealizedPnLPercent: -7.14,
      }
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([lossPosition])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        const aaplRow = screen.getByTestId('position-row-AAPL')
        const lossCells = aaplRow.querySelectorAll('.loss')
        expect(lossCells).toHaveLength(2)
      })
    })
  })
  
  describe('Watchlist Management', () => {
    it('should add symbols to watchlist', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      const watchlistInput = screen.getByPlaceholderText(/add symbol/i)
      
      await user.type(watchlistInput, 'TSLA')
      await user.keyboard('{Enter}')
      
      expect(screen.getByTestId('watchlist-item-TSLA')).toBeInTheDocument()
    })
    
    it('should remove symbols from watchlist', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      // Pre-populate watchlist
      useMarketStore.setState({
        watchlist: ['AAPL', 'GOOGL'],
        watchlistData: {
          AAPL: { symbol: 'AAPL', price: 150, change: 2, changePercent: 1.35 } as any,
          GOOGL: { symbol: 'GOOGL', price: 2500, change: -50, changePercent: -1.96 } as any,
        },
      })
      
      render(<Dashboard />)
      
      const aaplItem = screen.getByTestId('watchlist-item-AAPL')
      const removeButton = aaplItem.querySelector('button')!
      
      await user.click(removeButton)
      
      expect(screen.queryByTestId('watchlist-item-AAPL')).not.toBeInTheDocument()
      expect(screen.getByTestId('watchlist-item-GOOGL')).toBeInTheDocument()
    })
    
    it('should display watchlist prices and changes', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      useMarketStore.setState({
        watchlist: ['AAPL'],
        watchlistData: {
          AAPL: { 
            symbol: 'AAPL', 
            price: 150, 
            change: 2, 
            changePercent: 1.35,
            name: 'Apple Inc.',
            volume: 1000000,
            timestamp: new Date().toISOString()
          },
        },
      })
      
      render(<Dashboard />)
      
      const aaplItem = screen.getByTestId('watchlist-item-AAPL')
      expect(aaplItem).toHaveTextContent('AAPL')
      expect(aaplItem).toHaveTextContent('$150')
      expect(aaplItem).toHaveTextContent('1.35%')
      expect(aaplItem.querySelector('.profit')).toBeInTheDocument()
    })
  })
  
  describe('Performance Chart', () => {
    it('should display performance metrics', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        const metrics = screen.getByTestId('performance-metrics')
        expect(metrics).toHaveTextContent('Total Return: 25%')
        expect(metrics).toHaveTextContent('Sharpe Ratio: 1.8')
        expect(metrics).toHaveTextContent('Max Drawdown: -6.8%')
      })
    })
    
    it('should handle missing performance data', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(null)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.queryByTestId('performance-chart')).not.toBeInTheDocument()
      })
    })
  })
  
  describe('Alerts Display', () => {
    it('should show active alerts', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('alert-alert-1')).toHaveTextContent(
          'AAPL - PRICE ABOVE 160'
        )
      })
    })
    
    it('should handle triggered alerts', async () => {
      const triggeredAlert = {
        ...mockAlerts[0],
        triggered: true,
        triggeredAt: new Date().toISOString(),
        currentValue: 161,
      }
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce([])
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText('No active alerts')).toBeInTheDocument()
      })
      
      // Simulate alert trigger via WebSocket or polling
      act(() => {
        useAlertStore.getState().addNotification({
          id: 'notif-1',
          alertId: 'alert-1',
          title: 'Price Alert Triggered',
          message: 'AAPL reached $161',
          timestamp: new Date(),
          read: false,
          type: 'price',
        })
      })
    })
  })
  
  describe('Data Refresh', () => {
    it('should refresh data periodically', async () => {
      jest.useFakeTimers()
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValue(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValue(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValue(mockPerformance)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValue(mockAlerts)
      
      render(<Dashboard />)
      
      await waitFor(() => {
        expect(portfolioService.getPortfolios).toHaveBeenCalledTimes(1)
      })
      
      // Fast-forward 30 seconds (typical refresh interval)
      act(() => {
        jest.advanceTimersByTime(30000)
      })
      
      // Check that data is refreshed
      await waitFor(() => {
        expect(portfolioService.getPositions).toHaveBeenCalledTimes(2)
      })
      
      jest.useRealTimers()
    })
  })
})
