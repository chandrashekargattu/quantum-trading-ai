import { renderHook, act, waitFor } from '@testing-library/react'
import { useMarketStore } from '../useMarketStore'
import { marketService } from '@/services/api/market'

// Mock the market service
jest.mock('@/services/api/market', () => ({
  marketService: {
    getStock: jest.fn(),
    getOptionChain: jest.fn(),
  },
  Stock: {},
  Option: {},
}))

describe('useMarketStore', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    // Reset the store state
    useMarketStore.setState({
      watchlist: [],
      watchlistData: {},
      selectedSymbol: null,
      selectedStock: null,
      selectedExpiration: null,
      optionChain: null,
      isLoadingStock: false,
      isLoadingOptions: false,
    })
  })

  describe('Watchlist Management', () => {
    it('should add symbol to watchlist', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
      })

      expect(result.current.watchlist).toContain('AAPL')
      expect(result.current.watchlist).toHaveLength(1)
    })

    it('should not add duplicate symbols to watchlist', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
        result.current.addToWatchlist('AAPL')
      })

      expect(result.current.watchlist).toHaveLength(1)
    })

    it('should add multiple unique symbols to watchlist', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
        result.current.addToWatchlist('GOOGL')
        result.current.addToWatchlist('MSFT')
      })

      expect(result.current.watchlist).toEqual(['AAPL', 'GOOGL', 'MSFT'])
    })

    it('should remove symbol from watchlist', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
        result.current.addToWatchlist('GOOGL')
        result.current.removeFromWatchlist('AAPL')
      })

      expect(result.current.watchlist).toEqual(['GOOGL'])
      expect(result.current.watchlist).not.toContain('AAPL')
    })

    it('should remove symbol data when removed from watchlist', () => {
      const { result } = renderHook(() => useMarketStore())

      // Add symbol with data
      act(() => {
        useMarketStore.setState({
          watchlist: ['AAPL', 'GOOGL'],
          watchlistData: {
            AAPL: { symbol: 'AAPL', price: 150 } as any,
            GOOGL: { symbol: 'GOOGL', price: 2500 } as any,
          },
        })
      })

      act(() => {
        result.current.removeFromWatchlist('AAPL')
      })

      expect(result.current.watchlistData).not.toHaveProperty('AAPL')
      expect(result.current.watchlistData).toHaveProperty('GOOGL')
    })

    it('should handle removing non-existent symbol', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
        result.current.removeFromWatchlist('GOOGL')
      })

      expect(result.current.watchlist).toEqual(['AAPL'])
    })
  })

  describe('Symbol Selection', () => {
    const mockStock = {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150,
      change: 2.5,
      changePercent: 1.7,
      volume: 1000000,
    }

    it('should select symbol and load stock data', async () => {
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      const { result } = renderHook(() => useMarketStore())

      await act(async () => {
        await result.current.selectSymbol('AAPL')
      })

      expect(result.current.selectedSymbol).toBe('AAPL')
      expect(result.current.selectedStock).toEqual(mockStock)
      expect(result.current.isLoadingStock).toBe(false)
      expect(marketService.getStock).toHaveBeenCalledWith('AAPL')
    })

    it('should set loading state while fetching stock', async () => {
      ;(marketService.getStock as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockStock), 100))
      )
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.selectSymbol('AAPL')
      })

      expect(result.current.isLoadingStock).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoadingStock).toBe(false)
      })
    })

    it('should clear previous selection when selecting new symbol', async () => {
      ;(marketService.getStock as jest.Mock).mockResolvedValue(mockStock)
      const { result } = renderHook(() => useMarketStore())

      // Set initial state
      act(() => {
        useMarketStore.setState({
          selectedStock: { symbol: 'GOOGL' } as any,
          optionChain: { calls: [], puts: [] } as any,
        })
      })

      await act(async () => {
        await result.current.selectSymbol('AAPL')
      })

      expect(result.current.selectedStock).toEqual(mockStock)
      expect(result.current.optionChain).toBeNull()
    })

    it('should handle API errors when selecting symbol', async () => {
      const error = new Error('API Error')
      ;(marketService.getStock as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useMarketStore())

      await expect(
        act(async () => {
          await result.current.selectSymbol('AAPL')
        })
      ).rejects.toThrow('API Error')

      expect(result.current.isLoadingStock).toBe(false)
      expect(result.current.selectedStock).toBeNull()
    })

    it('should update watchlist data if symbol is in watchlist', async () => {
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(mockStock)
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
      })

      await act(async () => {
        await result.current.selectSymbol('AAPL')
      })

      expect(result.current.watchlistData['AAPL']).toEqual(mockStock)
    })
  })

  describe('Option Chain', () => {
    const mockOptionChain = {
      calls: [
        { strike: 150, bid: 5, ask: 5.5, volume: 100 },
        { strike: 155, bid: 3, ask: 3.5, volume: 50 },
      ],
      puts: [
        { strike: 145, bid: 2, ask: 2.5, volume: 75 },
        { strike: 140, bid: 1, ask: 1.5, volume: 25 },
      ],
      expirations: ['2024-01-19', '2024-02-16'],
      strikes: [140, 145, 150, 155],
    }

    it('should load option chain for symbol', async () => {
      ;(marketService.getOptionChain as jest.Mock).mockResolvedValueOnce(mockOptionChain)
      const { result } = renderHook(() => useMarketStore())

      await act(async () => {
        await result.current.loadOptionChain('AAPL')
      })

      expect(result.current.optionChain).toEqual(mockOptionChain)
      expect(result.current.isLoadingOptions).toBe(false)
      expect(marketService.getOptionChain).toHaveBeenCalledWith('AAPL')
    })

    it('should set loading state while fetching options', async () => {
      ;(marketService.getOptionChain as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockOptionChain), 100))
      )
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.loadOptionChain('AAPL')
      })

      expect(result.current.isLoadingOptions).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoadingOptions).toBe(false)
      })
    })

    it('should handle API errors when loading options', async () => {
      const error = new Error('Options API Error')
      ;(marketService.getOptionChain as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useMarketStore())

      await expect(
        act(async () => {
          await result.current.loadOptionChain('AAPL')
        })
      ).rejects.toThrow('Options API Error')

      expect(result.current.isLoadingOptions).toBe(false)
    })
  })

  describe('Stock Data Updates', () => {
    it('should update selected stock data', () => {
      const { result } = renderHook(() => useMarketStore())
      const initialStock = { symbol: 'AAPL', price: 150, volume: 1000000 }
      const update = { price: 155, volume: 1100000 }

      act(() => {
        useMarketStore.setState({ selectedStock: initialStock as any })
      })

      act(() => {
        result.current.updateStockData('AAPL', update)
      })

      expect(result.current.selectedStock).toEqual({
        symbol: 'AAPL',
        price: 155,
        volume: 1100000,
      })
    })

    it('should update watchlist stock data', () => {
      const { result } = renderHook(() => useMarketStore())
      const initialData = {
        AAPL: { symbol: 'AAPL', price: 150 },
        GOOGL: { symbol: 'GOOGL', price: 2500 },
      }

      act(() => {
        useMarketStore.setState({ watchlistData: initialData as any })
      })

      act(() => {
        result.current.updateStockData('AAPL', { price: 155 })
      })

      expect(result.current.watchlistData['AAPL'].price).toBe(155)
      expect(result.current.watchlistData['GOOGL'].price).toBe(2500)
    })

    it('should not update if symbol not in watchlist or selected', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.updateStockData('TSLA', { price: 200 })
      })

      expect(result.current.selectedStock).toBeNull()
      expect(result.current.watchlistData).toEqual({})
    })

    it('should handle partial updates', () => {
      const { result } = renderHook(() => useMarketStore())
      const initialStock = {
        symbol: 'AAPL',
        price: 150,
        volume: 1000000,
        change: 2.5,
        changePercent: 1.7,
      }

      act(() => {
        useMarketStore.setState({ selectedStock: initialStock as any })
      })

      act(() => {
        result.current.updateStockData('AAPL', { price: 155 })
      })

      expect(result.current.selectedStock).toEqual({
        ...initialStock,
        price: 155,
      })
    })
  })

  describe('Clear Selection', () => {
    it('should clear all selection state', () => {
      const { result } = renderHook(() => useMarketStore())

      // Set some state
      act(() => {
        useMarketStore.setState({
          selectedSymbol: 'AAPL',
          selectedStock: { symbol: 'AAPL' } as any,
          selectedExpiration: '2024-01-19',
          optionChain: { calls: [], puts: [] } as any,
        })
      })

      act(() => {
        result.current.clearSelection()
      })

      expect(result.current.selectedSymbol).toBeNull()
      expect(result.current.selectedStock).toBeNull()
      expect(result.current.selectedExpiration).toBeNull()
      expect(result.current.optionChain).toBeNull()
    })

    it('should not affect watchlist when clearing selection', () => {
      const { result } = renderHook(() => useMarketStore())

      act(() => {
        result.current.addToWatchlist('AAPL')
        result.current.addToWatchlist('GOOGL')
        useMarketStore.setState({
          selectedSymbol: 'AAPL',
          selectedStock: { symbol: 'AAPL' } as any,
        })
      })

      act(() => {
        result.current.clearSelection()
      })

      expect(result.current.watchlist).toEqual(['AAPL', 'GOOGL'])
      expect(result.current.watchlistData).toEqual({})
    })
  })

  describe('Store Persistence', () => {
    it('should maintain state across multiple hook instances', () => {
      const { result: result1 } = renderHook(() => useMarketStore())
      const { result: result2 } = renderHook(() => useMarketStore())

      act(() => {
        result1.current.addToWatchlist('AAPL')
      })

      expect(result2.current.watchlist).toContain('AAPL')
    })
  })

  describe('Edge Cases', () => {
    it('should handle empty symbol string', async () => {
      ;(marketService.getStock as jest.Mock).mockResolvedValueOnce(null)
      const { result } = renderHook(() => useMarketStore())

      await act(async () => {
        await result.current.selectSymbol('')
      })

      expect(result.current.selectedSymbol).toBe('')
      expect(marketService.getStock).toHaveBeenCalledWith('')
    })

    it('should handle very long watchlist', () => {
      const { result } = renderHook(() => useMarketStore())
      const symbols = Array.from({ length: 100 }, (_, i) => `SYM${i}`)

      act(() => {
        symbols.forEach(symbol => result.current.addToWatchlist(symbol))
      })

      expect(result.current.watchlist).toHaveLength(100)
    })

    it('should handle rapid selection changes', async () => {
      const stocks = {
        AAPL: { symbol: 'AAPL', price: 150 },
        GOOGL: { symbol: 'GOOGL', price: 2500 },
        MSFT: { symbol: 'MSFT', price: 300 },
      }
      
      ;(marketService.getStock as jest.Mock).mockImplementation(
        (symbol) => Promise.resolve(stocks[symbol])
      )
      
      const { result } = renderHook(() => useMarketStore())

      // Rapid selections
      await act(async () => {
        const promises = [
          result.current.selectSymbol('AAPL'),
          result.current.selectSymbol('GOOGL'),
          result.current.selectSymbol('MSFT'),
        ]
        await Promise.all(promises)
      })

      // Should have the last selection
      expect(result.current.selectedSymbol).toBe('MSFT')
      expect(result.current.selectedStock).toEqual(stocks.MSFT)
    })
  })
})
