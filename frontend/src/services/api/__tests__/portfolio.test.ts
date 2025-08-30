import { portfolioService } from '../portfolio'

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

describe('PortfolioService', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    localStorageMock.getItem.mockReturnValue('test-token')
  })

  describe('getPortfolios', () => {
    it('should fetch portfolios with correct URL and headers', async () => {
      const mockPortfolios = [
        {
          id: '1',
          name: 'Test Portfolio',
          initialCapital: 100000,
          currentValue: 110000,
          totalReturn: 10000,
          totalReturnPercent: 10,
          dayChange: 500,
          dayChangePercent: 0.5,
          cashBalance: 50000,
          investedAmount: 60000,
          createdAt: '2024-01-01',
          updatedAt: '2024-01-02'
        }
      ]

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPortfolios
      })

      const result = await portfolioService.getPortfolios()

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/portfolios/',
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockPortfolios)
    })

    it('should handle fetch error', async () => {
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500
      })

      await expect(portfolioService.getPortfolios()).rejects.toThrow('Failed to fetch portfolios')
    })

    it('should work without auth token', async () => {
      localStorageMock.getItem.mockReturnValue(null)

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => []
      })

      await portfolioService.getPortfolios()

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/portfolios/',
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )
    })
  })

  describe('createPortfolio', () => {
    it('should create portfolio with correct data', async () => {
      const portfolioData = {
        name: 'New Portfolio',
        initialCapital: 200000
      }

      const mockResponse = {
        id: '2',
        ...portfolioData,
        currentValue: 200000,
        totalReturn: 0,
        totalReturnPercent: 0,
        dayChange: 0,
        dayChangePercent: 0,
        cashBalance: 200000,
        investedAmount: 0,
        createdAt: '2024-01-03',
        updatedAt: '2024-01-03'
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      })

      const result = await portfolioService.createPortfolio(portfolioData)

      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/portfolios/',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          },
          body: JSON.stringify(portfolioData)
        }
      )
      expect(result).toEqual(mockResponse)
    })

    it('should handle create portfolio error', async () => {
      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Invalid portfolio data' })
      })

      await expect(
        portfolioService.createPortfolio({ name: '', initialCapital: 0 })
      ).rejects.toThrow('Failed to create portfolio')
    })
  })

  describe('getPortfolio', () => {
    it('should fetch single portfolio by id', async () => {
      const portfolioId = '123'
      const mockPortfolio = {
        id: portfolioId,
        name: 'Test Portfolio',
        currentValue: 150000
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPortfolio
      })

      const result = await portfolioService.getPortfolio(portfolioId)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/portfolios/${portfolioId}`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockPortfolio)
    })
  })

  describe('getPositions', () => {
    it('should fetch portfolio positions', async () => {
      const portfolioId = '123'
      const mockPositions = [
        {
          id: 'pos1',
          portfolioId,
          symbol: 'AAPL',
          quantity: 100,
          avgPrice: 150,
          currentPrice: 160,
          marketValue: 16000,
          costBasis: 15000,
          unrealizedPnL: 1000,
          unrealizedPnLPercent: 6.67,
          realizedPnL: 0,
          dayChange: 200,
          dayChangePercent: 1.25,
          openedAt: '2024-01-01'
        }
      ]

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPositions
      })

      const result = await portfolioService.getPositions(portfolioId)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/portfolios/${portfolioId}/positions`,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          }
        }
      )
      expect(result).toEqual(mockPositions)
    })
  })

  describe('addFunds', () => {
    it('should add funds to portfolio', async () => {
      const portfolioId = '123'
      const amount = 50000
      const mockTransaction = {
        id: 'trans1',
        portfolioId,
        type: 'DEPOSIT',
        amount,
        timestamp: '2024-01-04',
        description: 'Added funds'
      }

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockTransaction
      })

      const result = await portfolioService.addFunds(portfolioId, amount)

      expect(fetch).toHaveBeenCalledWith(
        `http://localhost:8000/api/v1/portfolios/${portfolioId}/deposit`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token'
          },
          body: JSON.stringify({ amount })
        }
      )
      expect(result).toEqual(mockTransaction)
    })
  })

  describe('auth token handling', () => {
    it('should use custom token type if available', async () => {
      localStorageMock.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'custom-token'
        if (key === 'token_type') return 'Custom'
        return null
      })

      ;(fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => []
      })

      await portfolioService.getPortfolios()

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Custom custom-token'
          })
        })
      )
    })
  })
})
