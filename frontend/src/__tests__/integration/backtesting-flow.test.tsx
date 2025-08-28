import React from 'react'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useBacktestStore } from '@/store/useBacktestStore'
import { backtestService } from '@/services/api/backtest'
import toast from 'react-hot-toast'

// Mock services
jest.mock('@/services/api/backtest')
jest.mock('react-hot-toast')

// Mock Chart components
jest.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  Line: () => null,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
}))

// Backtesting Component
const BacktestingDashboard = () => {
  const {
    savedConfigs,
    activeConfig,
    results,
    activeResult,
    runningBacktests,
    configForm,
    compareMode,
    compareResults,
    loadConfigs,
    loadResults,
    createConfig,
    selectConfig,
    deleteConfig,
    runBacktest,
    stopBacktest,
    selectResult,
    toggleCompareMode,
    toggleCompareResult,
    updateConfigForm,
    resetConfigForm,
  } = useBacktestStore()
  
  const [showConfigForm, setShowConfigForm] = React.useState(false)
  const [selectedStrategy, setSelectedStrategy] = React.useState('ma-crossover')
  
  React.useEffect(() => {
    loadConfigs()
    loadResults()
  }, [loadConfigs, loadResults])
  
  const handleCreateConfig = async () => {
    try {
      await createConfig({
        ...configForm,
        strategy: selectedStrategy,
      })
      setShowConfigForm(false)
      toast.success('Configuration created successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to create configuration')
    }
  }
  
  const handleRunBacktest = async (configId: string) => {
    try {
      const backtestId = await runBacktest(configId)
      toast.success('Backtest started successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to start backtest')
    }
  }
  
  const strategies = {
    'ma-crossover': {
      name: 'Moving Average Crossover',
      params: ['fastPeriod', 'slowPeriod'],
    },
    'rsi': {
      name: 'RSI Strategy',
      params: ['period', 'oversoldLevel', 'overboughtLevel'],
    },
    'bollinger': {
      name: 'Bollinger Bands',
      params: ['period', 'stdDev'],
    },
  }
  
  return (
    <div>
      <h1>Backtesting Dashboard</h1>
      
      {/* Configuration Section */}
      <div data-testid="config-section">
        <h2>Configurations</h2>
        <button 
          onClick={() => setShowConfigForm(true)}
          data-testid="create-config-btn"
        >
          Create New Configuration
        </button>
        
        <select
          value={activeConfig?.id || ''}
          onChange={(e) => selectConfig(e.target.value)}
          data-testid="config-selector"
        >
          <option value="">Select Configuration</option>
          {savedConfigs.map(config => (
            <option key={config.id} value={config.id}>
              {config.name} - {config.strategy}
            </option>
          ))}
        </select>
        
        {activeConfig && (
          <div data-testid="active-config">
            <h3>{activeConfig.name}</h3>
            <p>Strategy: {activeConfig.strategy}</p>
            <p>Symbols: {activeConfig.symbols.join(', ')}</p>
            <p>Period: {new Date(activeConfig.startDate).toLocaleDateString()} - {new Date(activeConfig.endDate).toLocaleDateString()}</p>
            <p>Initial Capital: ${activeConfig.initialCapital.toLocaleString()}</p>
            
            <button
              onClick={() => handleRunBacktest(activeConfig.id)}
              data-testid="run-backtest-btn"
              disabled={runningBacktests.has(activeConfig.id)}
            >
              {runningBacktests.has(activeConfig.id) ? 'Running...' : 'Run Backtest'}
            </button>
            
            <button
              onClick={() => deleteConfig(activeConfig.id)}
              data-testid="delete-config-btn"
            >
              Delete Configuration
            </button>
          </div>
        )}
      </div>
      
      {/* Configuration Form */}
      {showConfigForm && (
        <div data-testid="config-form">
          <h2>Create Configuration</h2>
          
          <input
            type="text"
            placeholder="Configuration Name"
            value={configForm.name}
            onChange={(e) => updateConfigForm({ name: e.target.value })}
          />
          
          <select
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
          >
            {Object.entries(strategies).map(([key, strategy]) => (
              <option key={key} value={key}>{strategy.name}</option>
            ))}
          </select>
          
          <input
            type="text"
            placeholder="Symbols (comma separated)"
            value={configForm.symbols.join(', ')}
            onChange={(e) => updateConfigForm({ 
              symbols: e.target.value.split(',').map(s => s.trim()).filter(Boolean) 
            })}
          />
          
          <input
            type="date"
            value={configForm.startDate.toISOString().split('T')[0]}
            onChange={(e) => updateConfigForm({ startDate: new Date(e.target.value) })}
          />
          
          <input
            type="date"
            value={configForm.endDate.toISOString().split('T')[0]}
            onChange={(e) => updateConfigForm({ endDate: new Date(e.target.value) })}
          />
          
          <input
            type="number"
            placeholder="Initial Capital"
            value={configForm.initialCapital}
            onChange={(e) => updateConfigForm({ initialCapital: parseFloat(e.target.value) })}
          />
          
          <input
            type="number"
            placeholder="Position Size"
            value={configForm.positionSize}
            onChange={(e) => updateConfigForm({ positionSize: parseFloat(e.target.value) })}
          />
          
          {/* Strategy Parameters */}
          <div data-testid="strategy-params">
            <h3>Strategy Parameters</h3>
            {strategies[selectedStrategy].params.map(param => (
              <input
                key={param}
                type="number"
                placeholder={param}
                value={configForm.parameters[param] || ''}
                onChange={(e) => updateConfigForm({ 
                  parameters: { 
                    ...configForm.parameters, 
                    [param]: parseFloat(e.target.value) 
                  } 
                })}
              />
            ))}
          </div>
          
          <button onClick={handleCreateConfig}>Create</button>
          <button onClick={() => {
            setShowConfigForm(false)
            resetConfigForm()
          }}>Cancel</button>
        </div>
      )}
      
      {/* Running Backtests */}
      {runningBacktests.size > 0 && (
        <div data-testid="running-backtests">
          <h2>Running Backtests</h2>
          {Array.from(runningBacktests.entries()).map(([id, status]) => (
            <div key={id} data-testid={`running-backtest-${id}`}>
              <p>{status.message}</p>
              <progress value={status.progress} max={100} />
              <span>{status.progress}%</span>
              <button
                onClick={() => stopBacktest(id)}
                data-testid={`stop-backtest-${id}`}
              >
                Stop
              </button>
            </div>
          ))}
        </div>
      )}
      
      {/* Results Section */}
      <div data-testid="results-section">
        <h2>Backtest Results</h2>
        
        <button
          onClick={() => toggleCompareMode()}
          data-testid="toggle-compare-btn"
        >
          {compareMode ? 'Exit Compare Mode' : 'Compare Results'}
        </button>
        
        <table data-testid="results-table">
          <thead>
            <tr>
              {compareMode && <th>Compare</th>}
              <th>Configuration</th>
              <th>Total Return</th>
              <th>Sharpe Ratio</th>
              <th>Max Drawdown</th>
              <th>Win Rate</th>
              <th>Total Trades</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {results.map(result => (
              <tr key={result.id} data-testid={`result-row-${result.id}`}>
                {compareMode && (
                  <td>
                    <input
                      type="checkbox"
                      checked={compareResults.includes(result.id)}
                      onChange={() => toggleCompareResult(result.id)}
                      data-testid={`compare-checkbox-${result.id}`}
                    />
                  </td>
                )}
                <td>{result.config?.name || result.configId}</td>
                <td className={result.totalReturnPercent >= 0 ? 'profit' : 'loss'}>
                  {result.totalReturnPercent.toFixed(2)}%
                </td>
                <td>{result.sharpeRatio.toFixed(2)}</td>
                <td className="loss">{result.maxDrawdownPercent.toFixed(2)}%</td>
                <td>{result.winRate.toFixed(1)}%</td>
                <td>{result.totalTrades}</td>
                <td>
                  <button
                    onClick={() => selectResult(result.id)}
                    data-testid={`view-result-${result.id}`}
                  >
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Result Details */}
      {activeResult && (
        <div data-testid="result-details">
          <h2>Result Details</h2>
          
          <div data-testid="performance-summary">
            <h3>Performance Summary</h3>
            <div className="metrics-grid">
              <div>
                <label>Total Return</label>
                <span>${activeResult.totalReturn.toLocaleString()} ({activeResult.totalReturnPercent.toFixed(2)}%)</span>
              </div>
              <div>
                <label>Annualized Return</label>
                <span>{activeResult.annualizedReturn.toFixed(2)}%</span>
              </div>
              <div>
                <label>Sharpe Ratio</label>
                <span>{activeResult.sharpeRatio.toFixed(2)}</span>
              </div>
              <div>
                <label>Sortino Ratio</label>
                <span>{activeResult.sortinoRatio.toFixed(2)}</span>
              </div>
              <div>
                <label>Max Drawdown</label>
                <span>{activeResult.maxDrawdownPercent.toFixed(2)}%</span>
              </div>
              <div>
                <label>Win Rate</label>
                <span>{activeResult.winRate.toFixed(1)}%</span>
              </div>
              <div>
                <label>Profit Factor</label>
                <span>{activeResult.profitFactor.toFixed(2)}</span>
              </div>
              <div>
                <label>Avg Trade Duration</label>
                <span>{activeResult.avgTradeDuration.toFixed(1)} days</span>
              </div>
            </div>
          </div>
          
          <div data-testid="trade-stats">
            <h3>Trade Statistics</h3>
            <p>Total Trades: {activeResult.totalTrades}</p>
            <p>Winning Trades: {activeResult.winningTrades}</p>
            <p>Losing Trades: {activeResult.losingTrades}</p>
            <p>Average Win: ${activeResult.avgWin.toFixed(2)}</p>
            <p>Average Loss: ${activeResult.avgLoss.toFixed(2)}</p>
            <p>Expectancy: ${activeResult.expectancy.toFixed(2)}</p>
          </div>
          
          <div data-testid="equity-chart">
            <h3>Equity Curve</h3>
            {/* Chart would be rendered here */}
          </div>
          
          <div data-testid="trades-list">
            <h3>Trade History</h3>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Entry Date</th>
                  <th>Exit Date</th>
                  <th>Side</th>
                  <th>Quantity</th>
                  <th>Entry Price</th>
                  <th>Exit Price</th>
                  <th>P&L</th>
                  <th>P&L %</th>
                </tr>
              </thead>
              <tbody>
                {activeResult.trades.slice(0, 10).map((trade, idx) => (
                  <tr key={idx} data-testid={`trade-row-${idx}`}>
                    <td>{trade.symbol}</td>
                    <td>{new Date(trade.entryDate).toLocaleDateString()}</td>
                    <td>{new Date(trade.exitDate).toLocaleDateString()}</td>
                    <td>{trade.side}</td>
                    <td>{trade.quantity}</td>
                    <td>${trade.entryPrice.toFixed(2)}</td>
                    <td>${trade.exitPrice.toFixed(2)}</td>
                    <td className={trade.pnl >= 0 ? 'profit' : 'loss'}>
                      ${trade.pnl.toFixed(2)}
                    </td>
                    <td className={trade.pnlPercent >= 0 ? 'profit' : 'loss'}>
                      {trade.pnlPercent.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Comparison View */}
      {compareMode && compareResults.length > 1 && (
        <div data-testid="comparison-view">
          <h2>Comparison</h2>
          <div data-testid="comparison-chart">
            {/* Comparison charts would be rendered here */}
          </div>
        </div>
      )}
    </div>
  )
}

describe('Backtesting Integration', () => {
  const mockConfigs = [
    {
      id: 'config-1',
      name: 'MA Crossover Strategy',
      strategy: 'ma-crossover',
      symbols: ['AAPL', 'GOOGL'],
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 100000,
      positionSize: 10000,
      maxPositions: 10,
      commission: 0.001,
      slippage: 0.0005,
      parameters: {
        fastPeriod: 10,
        slowPeriod: 30,
      },
      userId: 'user-1',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: 'config-2',
      name: 'RSI Strategy',
      strategy: 'rsi',
      symbols: ['MSFT', 'AMZN'],
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 50000,
      positionSize: 5000,
      maxPositions: 10,
      commission: 0.001,
      slippage: 0.0005,
      parameters: {
        period: 14,
        oversoldLevel: 30,
        overboughtLevel: 70,
      },
      userId: 'user-1',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ]
  
  const mockResults = [
    {
      id: 'result-1',
      configId: 'config-1',
      config: mockConfigs[0],
      status: 'COMPLETED' as const,
      startedAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      totalReturn: 15000,
      totalReturnPercent: 15,
      annualizedReturn: 15.5,
      sharpeRatio: 1.8,
      sortinoRatio: 2.1,
      calmarRatio: 2.5,
      maxDrawdown: -5000,
      maxDrawdownPercent: -5,
      maxDrawdownDuration: 30,
      totalTrades: 150,
      winningTrades: 98,
      losingTrades: 52,
      winRate: 65.33,
      avgWin: 350,
      avgLoss: -150,
      profitFactor: 2.2,
      expectancy: 100,
      avgTradeDuration: 5.5,
      volatility: 12.5,
      var95: -2000,
      cvar95: -2500,
      equity: [],
      drawdown: [],
      trades: [
        {
          id: 'trade-1',
          symbol: 'AAPL',
          entryDate: '2023-01-15',
          exitDate: '2023-01-20',
          entryPrice: 150,
          exitPrice: 155,
          quantity: 66,
          side: 'LONG' as const,
          pnl: 330,
          pnlPercent: 3.33,
          commission: 10,
          slippage: 5,
          duration: 5,
          exitReason: 'SIGNAL' as const,
        },
      ],
      dailyReturns: [],
      monthlyReturns: {},
      yearlyReturns: {},
    },
    {
      id: 'result-2',
      configId: 'config-2',
      config: mockConfigs[1],
      status: 'COMPLETED' as const,
      startedAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      totalReturn: -2000,
      totalReturnPercent: -4,
      annualizedReturn: -4.1,
      sharpeRatio: -0.5,
      sortinoRatio: -0.3,
      calmarRatio: -0.8,
      maxDrawdown: -3000,
      maxDrawdownPercent: -6,
      maxDrawdownDuration: 45,
      totalTrades: 80,
      winningTrades: 30,
      losingTrades: 50,
      winRate: 37.5,
      avgWin: 200,
      avgLoss: -160,
      profitFactor: 0.75,
      expectancy: -25,
      avgTradeDuration: 3.2,
      volatility: 18.5,
      var95: -1500,
      cvar95: -2000,
      equity: [],
      drawdown: [],
      trades: [],
      dailyReturns: [],
      monthlyReturns: {},
      yearlyReturns: {},
    },
  ]
  
  const mockBacktestStatus = {
    id: 'backtest-1',
    status: 'RUNNING' as const,
    progress: 45,
    message: 'Processing trades for AAPL...',
    currentDate: '2023-06-15',
  }
  
  beforeEach(() => {
    jest.clearAllMocks()
    window.confirm = jest.fn(() => true)
    
    // Reset store
    useBacktestStore.setState({
      savedConfigs: [],
      activeConfig: null,
      results: [],
      activeResult: null,
      runningBacktests: new Map(),
      configForm: {
        name: '',
        strategy: '',
        symbols: [],
        startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
        endDate: new Date(),
        initialCapital: 100000,
        positionSize: 10000,
        maxPositions: 10,
        commission: 0.001,
        slippage: 0.0005,
        parameters: {},
      },
      isLoadingConfigs: false,
      isLoadingResults: false,
      isRunningBacktest: false,
      compareMode: false,
      compareResults: [],
    })
  })
  
  describe('Configuration Management', () => {
    it('should create new configuration', async () => {
      const user = userEvent.setup()
      const newConfig = {
        id: 'config-3',
        name: 'Test Strategy',
        strategy: 'ma-crossover',
        symbols: ['TSLA', 'NVDA'],
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        initialCapital: 50000,
        positionSize: 5000,
        maxPositions: 10,
        commission: 0.001,
        slippage: 0.0005,
        parameters: {
          fastPeriod: 5,
          slowPeriod: 20,
        },
        userId: 'user-1',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      ;(backtestService.createConfig as jest.Mock).mockResolvedValueOnce(newConfig)
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('create-config-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('create-config-btn'))
      
      const form = screen.getByTestId('config-form')
      const nameInput = within(form).getByPlaceholderText('Configuration Name')
      const symbolsInput = within(form).getByPlaceholderText('Symbols (comma separated)')
      const capitalInput = within(form).getByPlaceholderText('Initial Capital')
      
      await user.type(nameInput, 'Test Strategy')
      await user.type(symbolsInput, 'TSLA, NVDA')
      await user.clear(capitalInput)
      await user.type(capitalInput, '50000')
      
      // Fill strategy parameters
      const paramsSection = within(form).getByTestId('strategy-params')
      const paramInputs = within(paramsSection).getAllByRole('spinbutton')
      
      await user.type(paramInputs[0], '5') // fastPeriod
      await user.type(paramInputs[1], '20') // slowPeriod
      
      await user.click(within(form).getByText('Create'))
      
      expect(backtestService.createConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Test Strategy',
          strategy: 'ma-crossover',
          symbols: ['TSLA', 'NVDA'],
          initialCapital: 50000,
          parameters: expect.objectContaining({
            fastPeriod: 5,
            slowPeriod: 20,
          }),
        })
      )
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Configuration created successfully')
      })
    })
    
    it('should select and display configuration', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('config-selector')).toBeInTheDocument()
      })
      
      const selector = screen.getByTestId('config-selector')
      await user.selectOptions(selector, 'config-1')
      
      const activeConfig = screen.getByTestId('active-config')
      expect(within(activeConfig).getByText('MA Crossover Strategy')).toBeInTheDocument()
      expect(within(activeConfig).getByText('Strategy: ma-crossover')).toBeInTheDocument()
      expect(within(activeConfig).getByText('Symbols: AAPL, GOOGL')).toBeInTheDocument()
      expect(within(activeConfig).getByText('Initial Capital: $100,000')).toBeInTheDocument()
    })
    
    it('should delete configuration', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      ;(backtestService.deleteConfig as jest.Mock).mockResolvedValueOnce(undefined)
      
      render(<BacktestingDashboard />)
      
      const selector = screen.getByTestId('config-selector')
      await user.selectOptions(selector, 'config-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('delete-config-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('delete-config-btn'))
      
      expect(backtestService.deleteConfig).toHaveBeenCalledWith('config-1')
    })
  })
  
  describe('Backtest Execution', () => {
    it('should run backtest', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      ;(backtestService.runBacktest as jest.Mock).mockResolvedValueOnce('backtest-1')
      ;(backtestService.getBacktestStatus as jest.Mock).mockResolvedValueOnce(mockBacktestStatus)
      
      render(<BacktestingDashboard />)
      
      const selector = screen.getByTestId('config-selector')
      await user.selectOptions(selector, 'config-1')
      
      await waitFor(() => {
        expect(screen.getByTestId('run-backtest-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('run-backtest-btn'))
      
      expect(backtestService.runBacktest).toHaveBeenCalledWith('config-1')
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Backtest started successfully')
      })
      
      // Check running backtests display
      expect(screen.getByTestId('running-backtests')).toBeInTheDocument()
      expect(screen.getByTestId('running-backtest-backtest-1')).toBeInTheDocument()
    })
    
    it('should show backtest progress', async () => {
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      // Simulate running backtest
      const runningBacktests = new Map()
      runningBacktests.set('backtest-1', mockBacktestStatus)
      
      useBacktestStore.setState({ runningBacktests })
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        const runningSection = screen.getByTestId('running-backtests')
        expect(within(runningSection).getByText('Processing trades for AAPL...')).toBeInTheDocument()
        expect(within(runningSection).getByText('45%')).toBeInTheDocument()
      })
    })
    
    it('should stop running backtest', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      ;(backtestService.stopBacktest as jest.Mock).mockResolvedValueOnce(undefined)
      
      const runningBacktests = new Map()
      runningBacktests.set('backtest-1', mockBacktestStatus)
      
      useBacktestStore.setState({ runningBacktests })
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('stop-backtest-backtest-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('stop-backtest-backtest-1'))
      
      expect(backtestService.stopBacktest).toHaveBeenCalledWith('backtest-1')
    })
  })
  
  describe('Results Display', () => {
    it('should display backtest results', async () => {
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('results-table')).toBeInTheDocument()
      })
      
      // Check first result
      const result1Row = screen.getByTestId('result-row-result-1')
      expect(within(result1Row).getByText('MA Crossover Strategy')).toBeInTheDocument()
      expect(within(result1Row).getByText('15.00%')).toBeInTheDocument()
      expect(within(result1Row).getByText('1.80')).toBeInTheDocument() // Sharpe
      expect(within(result1Row).getByText('-5.00%')).toBeInTheDocument() // Max DD
      expect(within(result1Row).getByText('65.3%')).toBeInTheDocument() // Win Rate
      expect(within(result1Row).getByText('150')).toBeInTheDocument() // Total Trades
      
      // Check styling for profit/loss
      expect(within(result1Row).getByText('15.00%')).toHaveClass('profit')
      
      // Check second result (loss)
      const result2Row = screen.getByTestId('result-row-result-2')
      expect(within(result2Row).getByText('-4.00%')).toHaveClass('loss')
    })
    
    it('should view result details', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      ;(backtestService.getResult as jest.Mock).mockResolvedValueOnce(mockResults[0])
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('view-result-result-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('view-result-result-1'))
      
      await waitFor(() => {
        expect(screen.getByTestId('result-details')).toBeInTheDocument()
      })
      
      // Check performance summary
      const perfSummary = screen.getByTestId('performance-summary')
      expect(within(perfSummary).getByText('$15,000 (15.00%)')).toBeInTheDocument()
      expect(within(perfSummary).getByText('15.50%')).toBeInTheDocument() // Annualized
      
      // Check trade statistics
      const tradeStats = screen.getByTestId('trade-stats')
      expect(within(tradeStats).getByText('Total Trades: 150')).toBeInTheDocument()
      expect(within(tradeStats).getByText('Winning Trades: 98')).toBeInTheDocument()
      expect(within(tradeStats).getByText('Average Win: $350.00')).toBeInTheDocument()
      
      // Check trades list
      const tradeRow = screen.getByTestId('trade-row-0')
      expect(within(tradeRow).getByText('AAPL')).toBeInTheDocument()
      expect(within(tradeRow).getByText('LONG')).toBeInTheDocument()
      expect(within(tradeRow).getByText('$330.00')).toBeInTheDocument()
      expect(within(tradeRow).getByText('3.33%')).toBeInTheDocument()
    })
  })
  
  describe('Results Comparison', () => {
    it('should enter compare mode', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      render(<BacktestingDashboard />)
      
      await waitFor(() => {
        expect(screen.getByTestId('toggle-compare-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('toggle-compare-btn'))
      
      // Check compare checkboxes appear
      expect(screen.getByTestId('compare-checkbox-result-1')).toBeInTheDocument()
      expect(screen.getByTestId('compare-checkbox-result-2')).toBeInTheDocument()
      
      // Button text should change
      expect(screen.getByTestId('toggle-compare-btn')).toHaveTextContent('Exit Compare Mode')
    })
    
    it('should select results for comparison', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      render(<BacktestingDashboard />)
      
      // Enter compare mode
      await user.click(screen.getByTestId('toggle-compare-btn'))
      
      // Select results
      await user.click(screen.getByTestId('compare-checkbox-result-1'))
      await user.click(screen.getByTestId('compare-checkbox-result-2'))
      
      // Comparison view should appear
      await waitFor(() => {
        expect(screen.getByTestId('comparison-view')).toBeInTheDocument()
      })
    })
    
    it('should limit comparison to 4 results', async () => {
      const user = userEvent.setup()
      const manyResults = [
        ...mockResults,
        { ...mockResults[0], id: 'result-3' },
        { ...mockResults[0], id: 'result-4' },
        { ...mockResults[0], id: 'result-5' },
      ]
      
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(manyResults)
      
      render(<BacktestingDashboard />)
      
      await user.click(screen.getByTestId('toggle-compare-btn'))
      
      // Select 5 results
      await user.click(screen.getByTestId('compare-checkbox-result-1'))
      await user.click(screen.getByTestId('compare-checkbox-result-2'))
      await user.click(screen.getByTestId('compare-checkbox-result-3'))
      await user.click(screen.getByTestId('compare-checkbox-result-4'))
      await user.click(screen.getByTestId('compare-checkbox-result-5'))
      
      // Only 4 should be selected
      expect(useBacktestStore.getState().compareResults).toHaveLength(4)
    })
  })
  
  describe('Strategy Parameter Validation', () => {
    it('should validate strategy parameters', async () => {
      const user = userEvent.setup()
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce(mockConfigs)
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce(mockResults)
      
      render(<BacktestingDashboard />)
      
      await user.click(screen.getByTestId('create-config-btn'))
      
      const form = screen.getByTestId('config-form')
      const strategySelect = within(form).getAllByRole('combobox')[0]
      
      // Switch to RSI strategy
      await user.selectOptions(strategySelect, 'rsi')
      
      // Check RSI parameters appear
      const paramsSection = within(form).getByTestId('strategy-params')
      const paramInputs = within(paramsSection).getAllByRole('spinbutton')
      
      expect(paramInputs).toHaveLength(3) // period, oversoldLevel, overboughtLevel
      expect(paramInputs[0]).toHaveAttribute('placeholder', 'period')
      expect(paramInputs[1]).toHaveAttribute('placeholder', 'oversoldLevel')
      expect(paramInputs[2]).toHaveAttribute('placeholder', 'overboughtLevel')
    })
  })
})
