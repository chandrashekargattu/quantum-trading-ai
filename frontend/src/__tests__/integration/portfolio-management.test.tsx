import React from 'react'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { usePortfolioStore } from '@/store/usePortfolioStore'
import { useTradingStore } from '@/store/useTradingStore'
import { portfolioService } from '@/services/api/portfolio'
import { tradingService } from '@/services/api/trading'
import toast from 'react-hot-toast'

// Mock services
jest.mock('@/services/api/portfolio')
jest.mock('@/services/api/trading')
jest.mock('react-hot-toast')

// Portfolio Management Component
const PortfolioManagement = () => {
  const {
    portfolios,
    activePortfolio,
    positions,
    performance,
    loadPortfolios,
    createPortfolio,
    selectPortfolio,
    deletePortfolio,
    closePosition,
    refreshPortfolioData,
  } = usePortfolioStore()
  
  const { loadOrderHistory, orderHistory } = useTradingStore()
  
  const [showCreateModal, setShowCreateModal] = React.useState(false)
  const [newPortfolioName, setNewPortfolioName] = React.useState('')
  const [newPortfolioCapital, setNewPortfolioCapital] = React.useState('100000')
  const [selectedTimeframe, setSelectedTimeframe] = React.useState('1M')
  
  React.useEffect(() => {
    loadPortfolios()
  }, [loadPortfolios])
  
  React.useEffect(() => {
    if (activePortfolio) {
      loadOrderHistory(100, activePortfolio.id)
    }
  }, [activePortfolio, loadOrderHistory])
  
  const handleCreatePortfolio = async () => {
    try {
      await createPortfolio(newPortfolioName, parseFloat(newPortfolioCapital))
      setShowCreateModal(false)
      setNewPortfolioName('')
      setNewPortfolioCapital('100000')
      toast.success('Portfolio created successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to create portfolio')
    }
  }
  
  const handleDeletePortfolio = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this portfolio?')) {
      try {
        await deletePortfolio(id)
        toast.success('Portfolio deleted successfully')
      } catch (error: any) {
        toast.error(error.message || 'Failed to delete portfolio')
      }
    }
  }
  
  const handleClosePosition = async (positionId: string) => {
    if (window.confirm('Are you sure you want to close this position?')) {
      try {
        await closePosition(positionId)
        toast.success('Position closed successfully')
      } catch (error: any) {
        toast.error(error.message || 'Failed to close position')
      }
    }
  }
  
  return (
    <div>
      <h1>Portfolio Management</h1>
      
      {/* Portfolio Selector */}
      <div data-testid="portfolio-selector">
        <select
          value={activePortfolio?.id || ''}
          onChange={(e) => selectPortfolio(e.target.value)}
        >
          <option value="">Select Portfolio</option>
          {portfolios.map(portfolio => (
            <option key={portfolio.id} value={portfolio.id}>
              {portfolio.name} - ${portfolio.currentValue.toLocaleString()}
            </option>
          ))}
        </select>
        <button onClick={() => setShowCreateModal(true)}>Create New Portfolio</button>
      </div>
      
      {/* Create Portfolio Modal */}
      {showCreateModal && (
        <div data-testid="create-portfolio-modal">
          <h2>Create New Portfolio</h2>
          <input
            type="text"
            placeholder="Portfolio Name"
            value={newPortfolioName}
            onChange={(e) => setNewPortfolioName(e.target.value)}
          />
          <input
            type="number"
            placeholder="Initial Capital"
            value={newPortfolioCapital}
            onChange={(e) => setNewPortfolioCapital(e.target.value)}
          />
          <button onClick={handleCreatePortfolio}>Create</button>
          <button onClick={() => setShowCreateModal(false)}>Cancel</button>
        </div>
      )}
      
      {activePortfolio && (
        <>
          {/* Portfolio Details */}
          <div data-testid="portfolio-details">
            <h2>{activePortfolio.name}</h2>
            <div>
              <span>Total Value: ${activePortfolio.currentValue.toLocaleString()}</span>
              <span>Cash Balance: ${activePortfolio.cashBalance.toLocaleString()}</span>
              <span>Invested: ${activePortfolio.investedAmount.toLocaleString()}</span>
              <span>Total Return: ${activePortfolio.totalReturn.toLocaleString()} ({activePortfolio.totalReturnPercent}%)</span>
            </div>
            <button
              onClick={() => handleDeletePortfolio(activePortfolio.id)}
              data-testid="delete-portfolio-btn"
            >
              Delete Portfolio
            </button>
          </div>
          
          {/* Performance Timeframe Selector */}
          <div data-testid="timeframe-selector">
            <button
              className={selectedTimeframe === '1D' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('1D')}
            >
              1D
            </button>
            <button
              className={selectedTimeframe === '1W' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('1W')}
            >
              1W
            </button>
            <button
              className={selectedTimeframe === '1M' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('1M')}
            >
              1M
            </button>
            <button
              className={selectedTimeframe === '3M' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('3M')}
            >
              3M
            </button>
            <button
              className={selectedTimeframe === '1Y' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('1Y')}
            >
              1Y
            </button>
            <button
              className={selectedTimeframe === 'ALL' ? 'active' : ''}
              onClick={() => setSelectedTimeframe('ALL')}
            >
              ALL
            </button>
          </div>
          
          {/* Performance Metrics */}
          {performance && (
            <div data-testid="performance-metrics">
              <h3>Performance Metrics</h3>
              <div className="metrics-grid">
                <div>
                  <label>Sharpe Ratio</label>
                  <span>{performance.sharpeRatio.toFixed(2)}</span>
                </div>
                <div>
                  <label>Sortino Ratio</label>
                  <span>{performance.sortinoRatio.toFixed(2)}</span>
                </div>
                <div>
                  <label>Max Drawdown</label>
                  <span>{performance.maxDrawdownPercent.toFixed(2)}%</span>
                </div>
                <div>
                  <label>Win Rate</label>
                  <span>{performance.winRate.toFixed(1)}%</span>
                </div>
                <div>
                  <label>Profit Factor</label>
                  <span>{performance.profitFactor.toFixed(2)}</span>
                </div>
                <div>
                  <label>Volatility</label>
                  <span>{performance.volatility.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}
          
          {/* Positions Table */}
          <div data-testid="positions-table">
            <h3>Open Positions</h3>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Shares</th>
                  <th>Avg Cost</th>
                  <th>Current Price</th>
                  <th>Market Value</th>
                  <th>Unrealized P&L</th>
                  <th>% Change</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(position => (
                  <tr key={position.id} data-testid={`position-${position.id}`}>
                    <td>{position.symbol}</td>
                    <td>{position.quantity}</td>
                    <td>${position.avgPrice.toFixed(2)}</td>
                    <td>${position.currentPrice.toFixed(2)}</td>
                    <td>${position.marketValue.toLocaleString()}</td>
                    <td className={position.unrealizedPnL >= 0 ? 'profit' : 'loss'}>
                      ${position.unrealizedPnL.toFixed(2)}
                    </td>
                    <td className={position.unrealizedPnLPercent >= 0 ? 'profit' : 'loss'}>
                      {position.unrealizedPnLPercent.toFixed(2)}%
                    </td>
                    <td>
                      <button
                        onClick={() => handleClosePosition(position.id)}
                        data-testid={`close-position-${position.id}`}
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Transaction History */}
          <div data-testid="transaction-history">
            <h3>Recent Transactions</h3>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Symbol</th>
                  <th>Quantity</th>
                  <th>Price</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {orderHistory.slice(0, 10).map(order => (
                  <tr key={order.id} data-testid={`transaction-${order.id}`}>
                    <td>{new Date(order.createdAt).toLocaleDateString()}</td>
                    <td>{order.side}</td>
                    <td>{order.symbol}</td>
                    <td>{order.filledQuantity || order.quantity}</td>
                    <td>${order.avgFillPrice || order.price || 'Market'}</td>
                    <td>${((order.filledQuantity || order.quantity) * (order.avgFillPrice || order.price || 0)).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Portfolio Actions */}
          <div data-testid="portfolio-actions">
            <button onClick={() => refreshPortfolioData()}>Refresh Data</button>
            <button onClick={() => portfolioService.addFunds(activePortfolio.id, 10000)}>
              Add Funds
            </button>
            <button onClick={() => portfolioService.withdrawFunds(activePortfolio.id, 5000)}>
              Withdraw Funds
            </button>
          </div>
        </>
      )}
    </div>
  )
}

describe('Portfolio Management Integration', () => {
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
    {
      id: 'portfolio-2',
      name: 'Growth Portfolio',
      initialCapital: 50000,
      currentValue: 55000,
      totalReturn: 5000,
      totalReturnPercent: 10,
      dayChange: 200,
      dayChangePercent: 0.37,
      cashBalance: 10000,
      investedAmount: 45000,
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
      timestamps: [],
      values: [],
      returns: [],
    },
  }
  
  beforeEach(() => {
    jest.clearAllMocks()
    window.confirm = jest.fn(() => true)
    
    // Reset stores
    usePortfolioStore.setState({
      portfolios: [],
      activePortfolio: null,
      positions: [],
      performance: null,
      isLoadingPortfolios: false,
    })
    
    useTradingStore.setState({
      orderHistory: [],
    })
  })
  
  describe('Portfolio Creation', () => {
    it('should create new portfolio successfully', async () => {
      const user = userEvent.setup()
      const newPortfolio = {
        id: 'portfolio-3',
        name: 'Test Portfolio',
        initialCapital: 50000,
        currentValue: 50000,
        totalReturn: 0,
        totalReturnPercent: 0,
        cashBalance: 50000,
        investedAmount: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.createPortfolio as jest.Mock).mockResolvedValueOnce(newPortfolio)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      
      render(<PortfolioManagement />)
      
      await waitFor(() => {
        expect(screen.getByText('Create New Portfolio')).toBeInTheDocument()
      })
      
      await user.click(screen.getByText('Create New Portfolio'))
      
      const modal = screen.getByTestId('create-portfolio-modal')
      const nameInput = within(modal).getByPlaceholderText('Portfolio Name')
      const capitalInput = within(modal).getByPlaceholderText('Initial Capital')
      const createButton = within(modal).getByText('Create')
      
      await user.clear(nameInput)
      await user.type(nameInput, 'Test Portfolio')
      await user.clear(capitalInput)
      await user.type(capitalInput, '50000')
      await user.click(createButton)
      
      expect(portfolioService.createPortfolio).toHaveBeenCalledWith({
        name: 'Test Portfolio',
        initialCapital: 50000,
      })
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Portfolio created successfully')
      })
    })
    
    it('should handle portfolio creation errors', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.createPortfolio as jest.Mock).mockRejectedValueOnce(
        new Error('Portfolio name already exists')
      )
      
      render(<PortfolioManagement />)
      
      await user.click(screen.getByText('Create New Portfolio'))
      
      const modal = screen.getByTestId('create-portfolio-modal')
      const nameInput = within(modal).getByPlaceholderText('Portfolio Name')
      const createButton = within(modal).getByText('Create')
      
      await user.type(nameInput, 'Main Portfolio')
      await user.click(createButton)
      
      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Portfolio name already exists')
      })
    })
  })
  
  describe('Portfolio Selection', () => {
    it('should switch between portfolios', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock)
        .mockResolvedValueOnce(mockPositions)
        .mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument()
      })
      
      const selector = screen.getByRole('combobox')
      
      // Select first portfolio
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByText('Main Portfolio')).toBeInTheDocument()
        expect(screen.getByText('Total Value: $125,000')).toBeInTheDocument()
      })
      
      // Switch to second portfolio
      await user.selectOptions(selector, 'portfolio-2')
      
      await waitFor(() => {
        expect(screen.getByText('Growth Portfolio')).toBeInTheDocument()
        expect(screen.getByText('Total Value: $55,000')).toBeInTheDocument()
      })
      
      expect(portfolioService.getPositions).toHaveBeenCalledTimes(2)
    })
  })
  
  describe('Portfolio Deletion', () => {
    it('should delete portfolio with confirmation', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(portfolioService.deletePortfolio as jest.Mock).mockResolvedValueOnce(undefined)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      // Select portfolio
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('delete-portfolio-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('delete-portfolio-btn'))
      
      expect(window.confirm).toHaveBeenCalledWith('Are you sure you want to delete this portfolio?')
      expect(portfolioService.deletePortfolio).toHaveBeenCalledWith('portfolio-1')
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Portfolio deleted successfully')
      })
    })
    
    it('should cancel deletion if not confirmed', async () => {
      const user = userEvent.setup()
      window.confirm = jest.fn(() => false)
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('delete-portfolio-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('delete-portfolio-btn'))
      
      expect(portfolioService.deletePortfolio).not.toHaveBeenCalled()
    })
  })
  
  describe('Position Management', () => {
    it('should display positions correctly', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await userEvent.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        const positionRow = screen.getByTestId('position-pos-1')
        expect(positionRow).toBeInTheDocument()
        expect(within(positionRow).getByText('AAPL')).toBeInTheDocument()
        expect(within(positionRow).getByText('100')).toBeInTheDocument()
        expect(within(positionRow).getByText('$140.00')).toBeInTheDocument()
        expect(within(positionRow).getByText('$150.00')).toBeInTheDocument()
        expect(within(positionRow).getByText('$1000.00')).toBeInTheDocument()
        expect(within(positionRow).getByText('7.14%')).toBeInTheDocument()
      })
    })
    
    it('should close position with confirmation', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock)
        .mockResolvedValueOnce(mockPositions)
        .mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(portfolioService.closePosition as jest.Mock).mockResolvedValueOnce(undefined)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('close-position-pos-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('close-position-pos-1'))
      
      expect(window.confirm).toHaveBeenCalledWith('Are you sure you want to close this position?')
      expect(portfolioService.closePosition).toHaveBeenCalledWith('portfolio-1', 'pos-1')
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Position closed successfully')
      })
    })
  })
  
  describe('Performance Metrics', () => {
    it('should display performance metrics', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await userEvent.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        const metrics = screen.getByTestId('performance-metrics')
        expect(within(metrics).getByText('1.80')).toBeInTheDocument() // Sharpe
        expect(within(metrics).getByText('2.10')).toBeInTheDocument() // Sortino
        expect(within(metrics).getByText('-6.80%')).toBeInTheDocument() // Max DD
        expect(within(metrics).getByText('65.0%')).toBeInTheDocument() // Win Rate
        expect(within(metrics).getByText('2.10')).toBeInTheDocument() // Profit Factor
        expect(within(metrics).getByText('15.20%')).toBeInTheDocument() // Volatility
      })
    })
    
    it('should change performance timeframe', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('timeframe-selector')).toBeInTheDocument()
      })
      
      // Click different timeframes
      const timeframeSelector = screen.getByTestId('timeframe-selector')
      
      await user.click(within(timeframeSelector).getByText('1Y'))
      expect(within(timeframeSelector).getByText('1Y')).toHaveClass('active')
      
      await user.click(within(timeframeSelector).getByText('1D'))
      expect(within(timeframeSelector).getByText('1D')).toHaveClass('active')
    })
  })
  
  describe('Transaction History', () => {
    it('should display recent transactions', async () => {
      const mockOrders = [
        {
          id: 'order-1',
          portfolioId: 'portfolio-1',
          symbol: 'AAPL',
          type: 'LIMIT',
          side: 'BUY',
          quantity: 100,
          filledQuantity: 100,
          price: 140,
          avgFillPrice: 140,
          status: 'FILLED',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      ]
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValueOnce(mockOrders)
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await userEvent.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        const transactionRow = screen.getByTestId('transaction-order-1')
        expect(transactionRow).toBeInTheDocument()
        expect(within(transactionRow).getByText('BUY')).toBeInTheDocument()
        expect(within(transactionRow).getByText('AAPL')).toBeInTheDocument()
        expect(within(transactionRow).getByText('100')).toBeInTheDocument()
        expect(within(transactionRow).getByText('$140')).toBeInTheDocument()
        expect(within(transactionRow).getByText('$14000.00')).toBeInTheDocument()
      })
    })
  })
  
  describe('Portfolio Actions', () => {
    it('should refresh portfolio data', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock)
        .mockResolvedValueOnce(mockPortfolios)
        .mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValue(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValue(mockPerformance)
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByText('Refresh Data')).toBeInTheDocument()
      })
      
      await user.click(screen.getByText('Refresh Data'))
      
      expect(portfolioService.getPortfolios).toHaveBeenCalledTimes(2)
    })
    
    it('should add funds to portfolio', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(portfolioService.addFunds as jest.Mock).mockResolvedValueOnce({
        id: 'trans-1',
        portfolioId: 'portfolio-1',
        type: 'DEPOSIT',
        amount: 10000,
        timestamp: new Date().toISOString(),
      })
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByText('Add Funds')).toBeInTheDocument()
      })
      
      await user.click(screen.getByText('Add Funds'))
      
      expect(portfolioService.addFunds).toHaveBeenCalledWith('portfolio-1', 10000)
    })
    
    it('should withdraw funds from portfolio', async () => {
      const user = userEvent.setup()
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce(mockPortfolios)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      ;(portfolioService.withdrawFunds as jest.Mock).mockResolvedValueOnce({
        id: 'trans-2',
        portfolioId: 'portfolio-1',
        type: 'WITHDRAWAL',
        amount: 5000,
        timestamp: new Date().toISOString(),
      })
      ;(tradingService.getOrderHistory as jest.Mock).mockResolvedValue([])
      
      render(<PortfolioManagement />)
      
      const selector = screen.getByRole('combobox')
      await user.selectOptions(selector, 'portfolio-1')
      
      await waitFor(() => {
        expect(screen.getByText('Withdraw Funds')).toBeInTheDocument()
      })
      
      await user.click(screen.getByText('Withdraw Funds'))
      
      expect(portfolioService.withdrawFunds).toHaveBeenCalledWith('portfolio-1', 5000)
    })
  })
})
