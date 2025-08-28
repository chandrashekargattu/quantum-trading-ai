import { renderHook, act, waitFor } from '@testing-library/react'
import { useBacktestStore } from '../useBacktestStore'
import { backtestService } from '@/services/api/backtest'

// Mock the backtest service
jest.mock('@/services/api/backtest', () => ({
  backtestService: {
    getConfigs: jest.fn(),
    getResults: jest.fn(),
    createConfig: jest.fn(),
    updateConfig: jest.fn(),
    deleteConfig: jest.fn(),
    runBacktest: jest.fn(),
    stopBacktest: jest.fn(),
    getBacktestStatus: jest.fn(),
    getResult: jest.fn(),
    deleteResult: jest.fn(),
    exportResults: jest.fn(),
  },
  BacktestConfig: {},
  BacktestResult: {},
  BacktestStatus: {},
}))

describe('useBacktestStore', () => {
  const mockConfig = {
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
      slowPeriod: 30
    },
    createdAt: new Date().toISOString()
  }

  const mockResult = {
    id: 'result-1',
    configId: 'config-1',
    status: 'COMPLETED',
    totalReturn: 15000,
    totalReturnPercent: 15,
    sharpeRatio: 1.8,
    maxDrawdown: -5.2,
    winRate: 65,
    totalTrades: 150,
    profitableTrades: 98,
    profitFactor: 2.1,
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    completedAt: new Date().toISOString()
  }

  const mockStatus = {
    id: 'backtest-1',
    status: 'RUNNING',
    progress: 45,
    message: 'Processing trades...',
    currentDate: '2023-06-15'
  }

  beforeEach(() => {
    jest.clearAllMocks()
    // Reset store state
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
        parameters: {}
      },
      isLoadingConfigs: false,
      isLoadingResults: false,
      isRunningBacktest: false,
      selectedConfigId: null,
      selectedResultId: null,
      compareMode: false,
      compareResults: [],
    })
  })

  describe('Configuration Management', () => {
    it('should load configurations', async () => {
      ;(backtestService.getConfigs as jest.Mock).mockResolvedValueOnce([mockConfig])
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        await result.current.loadConfigs()
      })

      expect(result.current.savedConfigs).toEqual([mockConfig])
      expect(result.current.isLoadingConfigs).toBe(false)
      expect(backtestService.getConfigs).toHaveBeenCalled()
    })

    it('should create new configuration', async () => {
      const newConfig = { ...mockConfig, id: 'config-2', name: 'RSI Strategy' }
      ;(backtestService.createConfig as jest.Mock).mockResolvedValueOnce(newConfig)
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        const created = await result.current.createConfig(newConfig)
        expect(created).toEqual(newConfig)
      })

      expect(result.current.savedConfigs).toContainEqual(newConfig)
      expect(result.current.configForm.name).toBe('') // Should reset
    })

    it('should update configuration', async () => {
      const updatedConfig = { ...mockConfig, name: 'Updated Strategy' }
      ;(backtestService.updateConfig as jest.Mock).mockResolvedValueOnce(updatedConfig)
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({ savedConfigs: [mockConfig] })
      })

      await act(async () => {
        await result.current.updateConfig('config-1', { name: 'Updated Strategy' })
      })

      expect(result.current.savedConfigs[0].name).toBe('Updated Strategy')
      expect(backtestService.updateConfig).toHaveBeenCalledWith('config-1', { name: 'Updated Strategy' })
    })

    it('should delete configuration', async () => {
      ;(backtestService.deleteConfig as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({
          savedConfigs: [mockConfig],
          activeConfig: mockConfig,
          selectedConfigId: 'config-1'
        })
      })

      await act(async () => {
        await result.current.deleteConfig('config-1')
      })

      expect(result.current.savedConfigs).toHaveLength(0)
      expect(result.current.activeConfig).toBeNull()
      expect(result.current.selectedConfigId).toBeNull()
    })

    it('should select configuration and load into form', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({ savedConfigs: [mockConfig] })
      })

      act(() => {
        result.current.selectConfig('config-1')
      })

      expect(result.current.selectedConfigId).toBe('config-1')
      expect(result.current.activeConfig).toEqual(mockConfig)
      expect(result.current.configForm.name).toBe('MA Crossover Strategy')
      expect(result.current.configForm.strategy).toBe('ma-crossover')
      expect(result.current.configForm.symbols).toEqual(['AAPL', 'GOOGL'])
    })
  })

  describe('Backtest Execution', () => {
    it('should run backtest', async () => {
      ;(backtestService.runBacktest as jest.Mock).mockResolvedValueOnce('backtest-1')
      ;(backtestService.getBacktestStatus as jest.Mock)
        .mockResolvedValueOnce(mockStatus)
        .mockResolvedValueOnce({ ...mockStatus, status: 'COMPLETED' })
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce([mockResult])
      
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        const backtestId = await result.current.runBacktest('config-1')
        expect(backtestId).toBe('backtest-1')
      })

      expect(result.current.runningBacktests.has('backtest-1')).toBe(true)
      expect(result.current.isRunningBacktest).toBe(false)

      // Wait for status polling
      await waitFor(() => {
        expect(backtestService.getBacktestStatus).toHaveBeenCalled()
      }, { timeout: 2000 })
    })

    it('should stop running backtest', async () => {
      ;(backtestService.stopBacktest as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        const running = new Map()
        running.set('backtest-1', mockStatus)
        useBacktestStore.setState({ runningBacktests: running })
      })

      await act(async () => {
        await result.current.stopBacktest('backtest-1')
      })

      expect(result.current.runningBacktests.has('backtest-1')).toBe(false)
      expect(backtestService.stopBacktest).toHaveBeenCalledWith('backtest-1')
    })

    it('should get backtest status', async () => {
      const updatedStatus = { ...mockStatus, progress: 75 }
      ;(backtestService.getBacktestStatus as jest.Mock).mockResolvedValueOnce(updatedStatus)
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        const status = await result.current.getBacktestStatus('backtest-1')
        expect(status).toEqual(updatedStatus)
      })

      expect(result.current.runningBacktests.get('backtest-1')).toEqual(updatedStatus)
    })

    it('should handle backtest errors', async () => {
      const error = new Error('Strategy not found')
      ;(backtestService.runBacktest as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useBacktestStore())

      await expect(
        act(async () => {
          await result.current.runBacktest('config-1')
        })
      ).rejects.toThrow('Strategy not found')

      expect(result.current.isRunningBacktest).toBe(false)
    })
  })

  describe('Results Management', () => {
    it('should load results', async () => {
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce([mockResult])
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        await result.current.loadResults()
      })

      expect(result.current.results).toEqual([mockResult])
      expect(result.current.isLoadingResults).toBe(false)
    })

    it('should load specific result', async () => {
      ;(backtestService.getResult as jest.Mock).mockResolvedValueOnce(mockResult)
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        await result.current.loadResult('result-1')
      })

      expect(result.current.activeResult).toEqual(mockResult)
      expect(backtestService.getResult).toHaveBeenCalledWith('result-1')
    })

    it('should select result', async () => {
      ;(backtestService.getResult as jest.Mock).mockResolvedValueOnce(mockResult)
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        await result.current.selectResult('result-1')
      })

      expect(result.current.selectedResultId).toBe('result-1')
      expect(result.current.activeResult).toEqual(mockResult)
    })

    it('should deselect result', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({
          selectedResultId: 'result-1',
          activeResult: mockResult
        })
      })

      act(() => {
        result.current.selectResult(null)
      })

      expect(result.current.selectedResultId).toBeNull()
      expect(result.current.activeResult).toBeNull()
    })

    it('should delete result', async () => {
      ;(backtestService.deleteResult as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({
          results: [mockResult],
          activeResult: mockResult,
          selectedResultId: 'result-1',
          compareResults: ['result-1']
        })
      })

      await act(async () => {
        await result.current.deleteResult('result-1')
      })

      expect(result.current.results).toHaveLength(0)
      expect(result.current.activeResult).toBeNull()
      expect(result.current.selectedResultId).toBeNull()
      expect(result.current.compareResults).toHaveLength(0)
    })

    it('should export results', async () => {
      ;(backtestService.exportResults as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useBacktestStore())

      await act(async () => {
        await result.current.exportResults(['result-1', 'result-2'])
      })

      expect(backtestService.exportResults).toHaveBeenCalledWith(['result-1', 'result-2'])
    })
  })

  describe('Comparison Mode', () => {
    it('should toggle compare mode', () => {
      const { result } = renderHook(() => useBacktestStore())

      act(() => {
        result.current.toggleCompareMode()
      })
      expect(result.current.compareMode).toBe(true)

      act(() => {
        result.current.toggleCompareMode()
      })
      expect(result.current.compareMode).toBe(false)
      expect(result.current.compareResults).toHaveLength(0)
    })

    it('should add result to comparison', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Enable compare mode
      act(() => {
        useBacktestStore.setState({ compareMode: true })
      })

      act(() => {
        result.current.toggleCompareResult('result-1')
      })

      expect(result.current.compareResults).toContain('result-1')
    })

    it('should remove result from comparison', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Set initial state
      act(() => {
        useBacktestStore.setState({
          compareMode: true,
          compareResults: ['result-1', 'result-2']
        })
      })

      act(() => {
        result.current.toggleCompareResult('result-1')
      })

      expect(result.current.compareResults).toEqual(['result-2'])
    })

    it('should limit comparison to 4 results', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Enable compare mode
      act(() => {
        useBacktestStore.setState({ compareMode: true })
      })

      // Add 5 results
      act(() => {
        result.current.toggleCompareResult('result-1')
        result.current.toggleCompareResult('result-2')
        result.current.toggleCompareResult('result-3')
        result.current.toggleCompareResult('result-4')
        result.current.toggleCompareResult('result-5')
      })

      expect(result.current.compareResults).toHaveLength(4)
      expect(result.current.compareResults).not.toContain('result-5')
    })
  })

  describe('Configuration Form', () => {
    it('should update config form', () => {
      const { result } = renderHook(() => useBacktestStore())

      act(() => {
        result.current.updateConfigForm({
          name: 'New Strategy',
          symbols: ['AAPL', 'MSFT'],
          initialCapital: 50000
        })
      })

      expect(result.current.configForm.name).toBe('New Strategy')
      expect(result.current.configForm.symbols).toEqual(['AAPL', 'MSFT'])
      expect(result.current.configForm.initialCapital).toBe(50000)
    })

    it('should update parameters', () => {
      const { result } = renderHook(() => useBacktestStore())

      act(() => {
        result.current.updateConfigForm({
          parameters: {
            fastPeriod: 5,
            slowPeriod: 20,
            rsiThreshold: 70
          }
        })
      })

      expect(result.current.configForm.parameters).toEqual({
        fastPeriod: 5,
        slowPeriod: 20,
        rsiThreshold: 70
      })
    })

    it('should reset config form', () => {
      const { result } = renderHook(() => useBacktestStore())

      // Set some values
      act(() => {
        result.current.updateConfigForm({
          name: 'Test Strategy',
          symbols: ['AAPL'],
          initialCapital: 200000
        })
      })

      act(() => {
        result.current.resetConfigForm()
      })

      expect(result.current.configForm.name).toBe('')
      expect(result.current.configForm.symbols).toHaveLength(0)
      expect(result.current.configForm.initialCapital).toBe(100000)
    })
  })

  describe('Loading States', () => {
    it('should manage config loading state', async () => {
      ;(backtestService.getConfigs as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve([mockConfig]), 100))
      )
      
      const { result } = renderHook(() => useBacktestStore())

      act(() => {
        result.current.loadConfigs()
      })

      expect(result.current.isLoadingConfigs).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoadingConfigs).toBe(false)
      })
    })

    it('should manage backtest running state', async () => {
      ;(backtestService.runBacktest as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve('backtest-1'), 100))
      )
      
      const { result } = renderHook(() => useBacktestStore())

      act(() => {
        result.current.runBacktest('config-1')
      })

      expect(result.current.isRunningBacktest).toBe(true)

      await waitFor(() => {
        expect(result.current.isRunningBacktest).toBe(false)
      })
    })
  })

  describe('Complex Workflows', () => {
    it('should handle complete backtest workflow', async () => {
      ;(backtestService.createConfig as jest.Mock).mockResolvedValueOnce(mockConfig)
      ;(backtestService.runBacktest as jest.Mock).mockResolvedValueOnce('backtest-1')
      ;(backtestService.getBacktestStatus as jest.Mock)
        .mockResolvedValueOnce(mockStatus)
        .mockResolvedValueOnce({ ...mockStatus, status: 'COMPLETED' })
      ;(backtestService.getResults as jest.Mock).mockResolvedValueOnce([mockResult])
      
      const { result } = renderHook(() => useBacktestStore())

      // Create configuration
      await act(async () => {
        await result.current.createConfig(mockConfig)
      })
      expect(result.current.savedConfigs).toHaveLength(1)

      // Run backtest
      await act(async () => {
        await result.current.runBacktest('config-1')
      })
      expect(result.current.runningBacktests.has('backtest-1')).toBe(true)

      // Load results after completion
      await act(async () => {
        await result.current.loadResults()
      })
      expect(result.current.results).toHaveLength(1)

      // Select result
      await act(async () => {
        await result.current.selectResult('result-1')
      })
      expect(result.current.activeResult).toBeTruthy()
    })

    it('should maintain state consistency during updates', () => {
      const { result } = renderHook(() => useBacktestStore())
      
      // Set complex state
      act(() => {
        useBacktestStore.setState({
          savedConfigs: [mockConfig],
          activeConfig: mockConfig,
          results: [mockResult],
          compareMode: true,
          compareResults: ['result-1']
        })
      })

      // Update active config
      act(() => {
        result.current.updateConfigForm({ name: 'Modified' })
      })

      // State should remain consistent
      expect(result.current.savedConfigs[0].name).toBe('MA Crossover Strategy') // Original unchanged
      expect(result.current.configForm.name).toBe('Modified') // Form updated
      expect(result.current.compareResults).toContain('result-1') // Comparison maintained
    })
  })
})
