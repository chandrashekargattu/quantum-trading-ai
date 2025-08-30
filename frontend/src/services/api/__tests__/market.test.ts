import { marketService } from '../market'

// Mock fetch globally
global.fetch = jest.fn()

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
global.localStorage = localStorageMock as any

describe('MarketService', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    localStorageMock.getItem.mockReturnValue('test-token')
  })

  describe('getMarketIndicators', () => {
    it('should fetch market indicators with correct URL', async () => {
      const mockIndicators = [
        {
          symbol: '^NSEI',
          name: 'NIFTY 50',
          value: 21000,
          change_amount: 150,
          change_percent: 0.72
        },
        {
          symbol: '^BSESN',
          name: 'SENSEX',
          value: 70000,
          change_amount: 500,
          change_percent: 0.72
        }
      ]

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockIndicators
      })

      const result = await marketService.getMarketIndicators()

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/market-data/indicators',
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockIndicators)
    })

    it('should handle market indicators error', async () => {
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500
      })

      await expect(marketService.getMarketIndicators()).rejects.toThrow('Failed to fetch market indicators')
    })
  })

  describe('getStock', () => {
    it('should fetch stock data with correct URL', async () => {
      const symbol = 'RELIANCE'
      const mockStock = {
        symbol,
        name: 'Reliance Industries',
        price: 2500,
        change: 50,
        changePercent: 2.04,
        volume: 1000000,
        timestamp: '2024-01-04T10:00:00'
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockStock
      })

      const result = await marketService.getStock(symbol)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/market-data/stocks/${symbol}`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockStock)
    })
  })

  describe('getBatchQuotes', () => {
    it('should fetch batch quotes with POST request', async () => {
      const symbols = ['INFY', 'TCS', 'WIPRO']
      const mockQuotes = symbols.map(symbol => ({
        symbol,
        price: 1500,
        change: 20,
        changePercent: 1.35,
        volume: 500000,
        bid: 1499,
        ask: 1501,
        timestamp: '2024-01-04T10:00:00'
      }))

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockQuotes
      })

      const result = await marketService.getBatchQuotes(symbols)

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/market-data/quotes',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          },
          body: JSON.stringify({ symbols })
        }
      )
      expect(result).toEqual(mockQuotes)
    })
  })

  describe('getOptionChain', () => {
    it('should fetch option chain data', async () => {
      const symbol = 'NIFTY'
      const mockOptionChain = {
        calls: [
          {
            symbol: 'NIFTY24JAN21000CE',
            strike: 21000,
            expiration: '2024-01-25',
            type: 'CALL' as const,
            bid: 150,
            ask: 155,
            last: 152,
            volume: 10000,
            openInterest: 50000,
            impliedVolatility: 0.15,
            delta: 0.6,
            gamma: 0.02,
            theta: -5,
            vega: 20,
            rho: 0.1
          }
        ],
        puts: [
          {
            symbol: 'NIFTY24JAN21000PE',
            strike: 21000,
            expiration: '2024-01-25',
            type: 'PUT' as const,
            bid: 100,
            ask: 105,
            last: 102,
            volume: 8000,
            openInterest: 40000,
            impliedVolatility: 0.16,
            delta: -0.4,
            gamma: 0.02,
            theta: -5,
            vega: 20,
            rho: -0.1
          }
        ],
        expirations: ['2024-01-25', '2024-02-29'],
        strikes: [20000, 20500, 21000, 21500, 22000]
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOptionChain
      })

      const result = await marketService.getOptionChain(symbol)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/options/chain/${symbol}`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockOptionChain)
    })
  })

  describe('searchSymbols', () => {
    it('should search symbols with encoded query', async () => {
      const query = 'Tata Motors'
      const mockResults = [
        {
          symbol: 'TATAMOTORS',
          name: 'Tata Motors Limited',
          price: 600,
          change: 10,
          changePercent: 1.69,
          volume: 2000000,
          timestamp: '2024-01-04T10:00:00'
        }
      ]

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResults
      })

      const result = await marketService.searchSymbols(query)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/market-data/search?q=${encodeURIComponent(query)}`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockResults)
    })
  })

  describe('getMarketOverview', () => {
    it('should fetch complete market overview', async () => {
      const mockOverview = {
        indices: [
          { symbol: '^NSEI', name: 'NIFTY 50', value: 21000, change: 150, changePercent: 0.72 }
        ],
        sectors: [
          { name: 'Banking', change: 1.2 },
          { name: 'IT', change: -0.5 }
        ],
        topGainers: [
          { symbol: 'ADANI', name: 'Adani Enterprises', price: 2500, change: 100, changePercent: 4.17, volume: 500000, timestamp: '2024-01-04' }
        ],
        topLosers: [
          { symbol: 'WIPRO', name: 'Wipro Limited', price: 400, change: -20, changePercent: -4.76, volume: 1000000, timestamp: '2024-01-04' }
        ],
        mostActive: [
          { symbol: 'RELIANCE', name: 'Reliance Industries', price: 2500, change: 50, changePercent: 2.04, volume: 5000000, timestamp: '2024-01-04' }
        ]
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockOverview
      })

      const result = await marketService.getMarketOverview()

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/market-data/overview',
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockOverview)
    })
  })

  describe('getHistoricalData', () => {
    it('should fetch historical data with date range', async () => {
      const symbol = 'INFY'
      const interval = '1d'
      const start = new Date('2024-01-01')
      const end = new Date('2024-01-04')

      const mockHistoricalData = {
        timestamps: ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        open: [1400, 1410, 1420, 1430],
        high: [1415, 1425, 1435, 1445],
        low: [1395, 1405, 1415, 1425],
        close: [1410, 1420, 1430, 1440],
        volume: [1000000, 1100000, 1200000, 1300000]
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockHistoricalData
      })

      const result = await marketService.getHistoricalData(symbol, interval, start, end)

      const expectedParams = new URLSearchParams({
        symbol,
        interval,
        start: start.toISOString(),
        end: end.toISOString()
      })

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/market-data/historical?${expectedParams}`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockHistoricalData)
    })
  })

  describe('error handling', () => {
    it('should handle network errors', async () => {
      ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))

      await expect(marketService.getMarketIndicators()).rejects.toThrow('Network error')
    })

    it('should handle 404 errors', async () => {
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      })

      await expect(marketService.getStock('INVALID')).rejects.toThrow('Failed to fetch stock data')
    })
  })

  describe('auth token handling', () => {
    it('should work without auth token', async () => {
      localStorageMock.getItem.mockReturnValue(null)

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => []
      })

      await marketService.getMarketIndicators()

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      )
      expect(fetch.mock.calls[0][1].headers).not.toHaveProperty('Authorization')
    })
  })
})
