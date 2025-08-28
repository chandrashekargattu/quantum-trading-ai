import { renderHook, act, waitFor } from '@testing-library/react'
import { usePortfolioStore } from '../usePortfolioStore'
import { portfolioService } from '@/services/api/portfolio'

// Mock the portfolio service
jest.mock('@/services/api/portfolio', () => ({
  portfolioService: {
    getPortfolios: jest.fn(),
    createPortfolio: jest.fn(),
    deletePortfolio: jest.fn(),
    getPositions: jest.fn(),
    getPerformance: jest.fn(),
    closePosition: jest.fn(),
  },
  Portfolio: {},
  Position: {},
  Performance: {},
}))

describe('usePortfolioStore', () => {
  const mockPortfolio = {
    id: 'portfolio-1',
    name: 'Main Portfolio',
    initialCapital: 100000,
    currentValue: 105000,
    totalReturn: 5000,
    totalReturnPercent: 5,
    createdAt: new Date().toISOString()
  }

  const mockPositions = [
    {
      id: 'pos-1',
      symbol: 'AAPL',
      quantity: 100,
      avgPrice: 150,
      currentPrice: 155,
      unrealizedPnL: 500,
      realizedPnL: 0
    },
    {
      id: 'pos-2',
      symbol: 'GOOGL',
      quantity: 50,
      avgPrice: 2500,
      currentPrice: 2550,
      unrealizedPnL: 2500,
      realizedPnL: 100
    }
  ]

  const mockPerformance = {
    totalReturn: 5000,
    totalReturnPercent: 5,
    dailyReturn: 200,
    dailyReturnPercent: 0.19,
    sharpeRatio: 1.5,
    maxDrawdown: -3.2,
    winRate: 65,
    profitFactor: 1.8,
    chartData: []
  }

  beforeEach(() => {
    jest.clearAllMocks()
    // Reset store state
    usePortfolioStore.setState({
      portfolios: [],
      activePortfolio: null,
      positions: [],
      performance: null,
      isLoadingPortfolios: false,
      isLoadingPositions: false,
      isLoadingPerformance: false,
      selectedPositionId: null,
    })
  })

  describe('Portfolio Management', () => {
    it('should load portfolios', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([mockPortfolio])
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        await result.current.loadPortfolios()
      })

      expect(result.current.portfolios).toEqual([mockPortfolio])
      expect(result.current.isLoadingPortfolios).toBe(false)
      expect(portfolioService.getPortfolios).toHaveBeenCalled()
    })

    it('should auto-select first portfolio if none selected', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([mockPortfolio])
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        await result.current.loadPortfolios()
      })

      expect(result.current.activePortfolio).toEqual(mockPortfolio)
    })

    it('should create new portfolio', async () => {
      const newPortfolio = {
        id: 'portfolio-2',
        name: 'Test Portfolio',
        initialCapital: 50000,
        currentValue: 50000,
        totalReturn: 0,
        totalReturnPercent: 0,
        createdAt: new Date().toISOString()
      }
      
      ;(portfolioService.createPortfolio as jest.Mock).mockResolvedValueOnce(newPortfolio)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        const created = await result.current.createPortfolio('Test Portfolio', 50000)
        expect(created).toEqual(newPortfolio)
      })

      expect(result.current.portfolios).toContainEqual(newPortfolio)
      expect(result.current.activePortfolio).toEqual(newPortfolio)
      expect(portfolioService.createPortfolio).toHaveBeenCalledWith({
        name: 'Test Portfolio',
        initialCapital: 50000
      })
    })

    it('should delete portfolio', async () => {
      ;(portfolioService.deletePortfolio as jest.Mock).mockResolvedValueOnce(undefined)
      
      const { result } = renderHook(() => usePortfolioStore())
      
      // Set initial state
      act(() => {
        usePortfolioStore.setState({
          portfolios: [mockPortfolio],
          activePortfolio: mockPortfolio
        })
      })

      await act(async () => {
        await result.current.deletePortfolio('portfolio-1')
      })

      expect(result.current.portfolios).toEqual([])
      expect(result.current.activePortfolio).toBeNull()
      expect(portfolioService.deletePortfolio).toHaveBeenCalledWith('portfolio-1')
    })

    it('should handle portfolio selection errors', async () => {
      const { result } = renderHook(() => usePortfolioStore())

      await expect(
        act(async () => {
          await result.current.selectPortfolio('non-existent')
        })
      ).rejects.toThrow('Portfolio not found')
    })

    it('should select another portfolio after deletion', async () => {
      const portfolio2 = { ...mockPortfolio, id: 'portfolio-2', name: 'Secondary' }
      
      ;(portfolioService.deletePortfolio as jest.Mock).mockResolvedValueOnce(undefined)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      
      const { result } = renderHook(() => usePortfolioStore())
      
      // Set initial state with 2 portfolios
      act(() => {
        usePortfolioStore.setState({
          portfolios: [mockPortfolio, portfolio2],
          activePortfolio: mockPortfolio
        })
      })

      await act(async () => {
        await result.current.deletePortfolio('portfolio-1')
      })

      expect(result.current.activePortfolio).toEqual(portfolio2)
    })
  })

  describe('Position Management', () => {
    it('should load positions for portfolio', async () => {
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        await result.current.loadPositions('portfolio-1')
      })

      expect(result.current.positions).toEqual(mockPositions)
      expect(result.current.isLoadingPositions).toBe(false)
      expect(portfolioService.getPositions).toHaveBeenCalledWith('portfolio-1')
    })

    it('should update position data', () => {
      const { result } = renderHook(() => usePortfolioStore())
      
      // Set initial positions
      act(() => {
        usePortfolioStore.setState({ positions: mockPositions })
      })

      act(() => {
        result.current.updatePosition('pos-1', { currentPrice: 160, unrealizedPnL: 1000 })
      })

      const updatedPosition = result.current.positions.find(p => p.id === 'pos-1')
      expect(updatedPosition?.currentPrice).toBe(160)
      expect(updatedPosition?.unrealizedPnL).toBe(1000)
    })

    it('should close position', async () => {
      ;(portfolioService.closePosition as jest.Mock).mockResolvedValueOnce(undefined)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(
        mockPositions.filter(p => p.id !== 'pos-1')
      )
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([mockPortfolio])
      
      const { result } = renderHook(() => usePortfolioStore())
      
      // Set initial state
      act(() => {
        usePortfolioStore.setState({
          activePortfolio: mockPortfolio,
          positions: mockPositions
        })
      })

      await act(async () => {
        await result.current.closePosition('pos-1')
      })

      expect(portfolioService.closePosition).toHaveBeenCalledWith('portfolio-1', 'pos-1')
      expect(result.current.positions).toHaveLength(1)
      expect(result.current.positions[0].id).toBe('pos-2')
    })

    it('should handle close position without active portfolio', async () => {
      const { result } = renderHook(() => usePortfolioStore())

      await expect(
        act(async () => {
          await result.current.closePosition('pos-1')
        })
      ).rejects.toThrow('No active portfolio')
    })

    it('should select and deselect position', () => {
      const { result } = renderHook(() => usePortfolioStore())

      act(() => {
        result.current.selectPosition('pos-1')
      })
      expect(result.current.selectedPositionId).toBe('pos-1')

      act(() => {
        result.current.selectPosition(null)
      })
      expect(result.current.selectedPositionId).toBeNull()
    })
  })

  describe('Performance Loading', () => {
    it('should load performance data', async () => {
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        await result.current.loadPerformance('portfolio-1', '1M')
      })

      expect(result.current.performance).toEqual(mockPerformance)
      expect(result.current.isLoadingPerformance).toBe(false)
      expect(portfolioService.getPerformance).toHaveBeenCalledWith('portfolio-1', '1M')
    })

    it('should handle performance loading errors', async () => {
      const error = new Error('Performance API Error')
      ;(portfolioService.getPerformance as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => usePortfolioStore())

      await expect(
        act(async () => {
          await result.current.loadPerformance('portfolio-1', '1M')
        })
      ).rejects.toThrow('Performance API Error')

      expect(result.current.isLoadingPerformance).toBe(false)
    })
  })

  describe('Portfolio Refresh', () => {
    it('should refresh portfolio data', async () => {
      const updatedPortfolio = { ...mockPortfolio, currentValue: 110000 }
      
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([updatedPortfolio])
      
      const { result } = renderHook(() => usePortfolioStore())
      
      // Set initial state
      act(() => {
        usePortfolioStore.setState({
          portfolios: [mockPortfolio],
          activePortfolio: mockPortfolio
        })
      })

      await act(async () => {
        await result.current.refreshPortfolioData()
      })

      expect(result.current.activePortfolio?.currentValue).toBe(110000)
      expect(result.current.portfolios[0].currentValue).toBe(110000)
    })

    it('should handle refresh without active portfolio', async () => {
      const { result } = renderHook(() => usePortfolioStore())

      await act(async () => {
        await result.current.refreshPortfolioData()
      })

      expect(portfolioService.getPortfolios).not.toHaveBeenCalled()
    })
  })

  describe('Loading States', () => {
    it('should manage loading states correctly', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve([mockPortfolio]), 100))
      )
      
      const { result } = renderHook(() => usePortfolioStore())

      act(() => {
        result.current.loadPortfolios()
      })

      expect(result.current.isLoadingPortfolios).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoadingPortfolios).toBe(false)
      })
    })
  })

  describe('Complex Workflows', () => {
    it('should handle complete portfolio workflow', async () => {
      ;(portfolioService.getPortfolios as jest.Mock).mockResolvedValueOnce([])
      ;(portfolioService.createPortfolio as jest.Mock).mockResolvedValueOnce(mockPortfolio)
      ;(portfolioService.getPositions as jest.Mock).mockResolvedValueOnce(mockPositions)
      ;(portfolioService.getPerformance as jest.Mock).mockResolvedValueOnce(mockPerformance)
      
      const { result } = renderHook(() => usePortfolioStore())

      // Load empty portfolios
      await act(async () => {
        await result.current.loadPortfolios()
      })
      expect(result.current.portfolios).toHaveLength(0)

      // Create new portfolio
      await act(async () => {
        await result.current.createPortfolio('Main Portfolio', 100000)
      })
      expect(result.current.portfolios).toHaveLength(1)
      expect(result.current.activePortfolio).toBeTruthy()
      expect(result.current.positions).toEqual(mockPositions)
      expect(result.current.performance).toEqual(mockPerformance)
    })

    it('should maintain state consistency across operations', async () => {
      const { result } = renderHook(() => usePortfolioStore())
      
      // Multiple rapid operations
      act(() => {
        usePortfolioStore.setState({
          portfolios: [mockPortfolio],
          activePortfolio: mockPortfolio,
          positions: mockPositions
        })
      })

      // Update position
      act(() => {
        result.current.updatePosition('pos-1', { currentPrice: 160 })
      })

      // Select position
      act(() => {
        result.current.selectPosition('pos-1')
      })

      // Verify state consistency
      expect(result.current.positions[0].currentPrice).toBe(160)
      expect(result.current.selectedPositionId).toBe('pos-1')
      expect(result.current.activePortfolio).toEqual(mockPortfolio)
    })
  })
})
