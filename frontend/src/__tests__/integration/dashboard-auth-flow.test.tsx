import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import DashboardPage from '@/app/dashboard/page'
import { useAuthStore } from '@/store/useAuthStore'
import { portfolioService } from '@/services/api/portfolio-optimized'
import { marketService } from '@/services/api/market-optimized'
import { authenticatedFetch } from '@/lib/auth-interceptor'
import { clearCache } from '@/lib/api-cache'
import { toast } from 'react-hot-toast'

// Mock dependencies
jest.mock('@/store/useAuthStore')
jest.mock('@/services/api/portfolio-optimized')
jest.mock('@/services/api/market-optimized')
jest.mock('@/lib/auth-interceptor')
jest.mock('@/lib/api-cache')
jest.mock('react-hot-toast')
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
}))

const mockUseAuthStore = useAuthStore as jest.MockedFunction<typeof useAuthStore>
const mockAuthenticatedFetch = authenticatedFetch as jest.MockedFunction<typeof authenticatedFetch>

// Test user data
const testUser = {
  id: '123',
  username: 'testuser',
  email: 'test@example.com',
  full_name: 'Test User',
  account_type: 'paper',
  is_active: true,
  is_verified: true
}

// Test portfolio data
const testPortfolio = {
  id: 'portfolio-123',
  name: 'Test Portfolio',
  initialCapital: 100000,
  currentValue: 105000,
  totalReturn: 5000,
  totalReturnPercent: 5,
  dayChange: 500,
  dayChangePercent: 0.5,
  cashBalance: 50000,
  investedAmount: 50000,
  createdAt: '2024-01-01',
  updatedAt: '2024-01-01'
}

// Test market data
const testMarketData = [
  {
    symbol: '^GSPC',
    name: 'S&P 500',
    value: 5000,
    change_amount: 50,
    change_percent: 1
  },
  {
    symbol: '^DJI',
    name: 'Dow Jones',
    value: 40000,
    change_amount: -100,
    change_percent: -0.25
  }
]

describe('Dashboard Authentication Flow', () => {
  let queryClient: QueryClient

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    
    jest.clearAllMocks()
    clearCache()
    
    // Setup localStorage mock
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn()
      },
      writable: true
    })
  })

  const renderDashboard = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <DashboardPage />
      </QueryClientProvider>
    )
  }

  describe('Authentication State', () => {
    it('should redirect to login when not authenticated', async () => {
      const mockPush = jest.fn()
      jest.mocked(require('next/navigation').useRouter).mockReturnValue({
        push: mockPush,
        replace: jest.fn(),
        prefetch: jest.fn()
      })

      mockUseAuthStore.mockReturnValue({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      localStorage.getItem = jest.fn().mockReturnValue(null)

      renderDashboard()

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/auth/login')
      })
    })

    it('should show loading state while checking auth', () => {
      mockUseAuthStore.mockReturnValue({
        user: null,
        isAuthenticated: false,
        isLoading: true,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      localStorage.getItem = jest.fn().mockReturnValue('test-token')

      renderDashboard()

      expect(screen.getByText(/Loading dashboard/i)).toBeInTheDocument()
    })

    it('should fetch user if token exists but user not loaded', async () => {
      const mockFetchUser = jest.fn()
      
      mockUseAuthStore.mockReturnValue({
        user: null,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: mockFetchUser,
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      localStorage.getItem = jest.fn().mockReturnValue('test-token')

      renderDashboard()

      await waitFor(() => {
        expect(mockFetchUser).toHaveBeenCalled()
      })
    })
  })

  describe('Authenticated Dashboard', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue({
        user: testUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      localStorage.getItem = jest.fn().mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        if (key === 'token_type') return 'Bearer'
        return null
      })
    })

    it('should display user information', async () => {
      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [testPortfolio],
        totalValue: 105000,
        totalGain: 5000,
        bestPerformer: testPortfolio
      })

      ;(marketService.getMarketIndicators as jest.Mock).mockResolvedValue(testMarketData)

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText(/Welcome back, Test User!/)).toBeInTheDocument()
        expect(screen.getByText(/paper/i)).toBeInTheDocument()
      })
    })

    it('should prefetch data on mount', async () => {
      const mockPrefetchMarket = jest.fn()
      const mockPrefetchPortfolios = jest.fn()
      
      ;(marketService.prefetchMarketData as jest.Mock) = mockPrefetchMarket
      ;(portfolioService.prefetchPortfolios as jest.Mock) = mockPrefetchPortfolios

      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [],
        totalValue: 0,
        totalGain: 0
      })

      ;(marketService.getMarketIndicators as jest.Mock).mockResolvedValue([])

      renderDashboard()

      await waitFor(() => {
        expect(mockPrefetchMarket).toHaveBeenCalled()
        expect(mockPrefetchPortfolios).toHaveBeenCalled()
      })
    })
  })

  describe('Market Data Loading', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue({
        user: testUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })
    })

    it('should handle market data loading errors', async () => {
      ;(marketService.getMarketIndicators as jest.Mock).mockRejectedValue(
        new Error('Failed to load market data')
      )

      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [],
        totalValue: 0,
        totalGain: 0
      })

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText(/Failed to load market data/i)).toBeInTheDocument()
      })
    })

    it('should display market indicators', async () => {
      ;(marketService.getMarketIndicators as jest.Mock).mockResolvedValue(testMarketData)

      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [],
        totalValue: 0,
        totalGain: 0
      })

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('S&P 500')).toBeInTheDocument()
        expect(screen.getByText('Dow Jones')).toBeInTheDocument()
      })
    })
  })

  describe('Portfolio Management', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue({
        user: testUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      ;(marketService.getMarketIndicators as jest.Mock).mockResolvedValue(testMarketData)
    })

    it('should show create portfolio when no portfolios exist', async () => {
      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [],
        totalValue: 0,
        totalGain: 0
      })

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText(/No portfolio found/i)).toBeInTheDocument()
        expect(screen.getByText(/Create Portfolio/i)).toBeInTheDocument()
      })
    })

    it('should display portfolio summary when portfolio exists', async () => {
      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [testPortfolio],
        totalValue: 105000,
        totalGain: 5000,
        bestPerformer: testPortfolio
      })

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
        expect(screen.getByText(/105,000/)).toBeInTheDocument()
      })
    })

    it('should handle portfolio creation', async () => {
      ;(portfolioService.getPortfoliosSummary as jest.Mock)
        .mockResolvedValueOnce({
          portfolios: [],
          totalValue: 0,
          totalGain: 0
        })
        .mockResolvedValueOnce({
          portfolios: [testPortfolio],
          totalValue: 105000,
          totalGain: 5000,
          bestPerformer: testPortfolio
        })

      ;(portfolioService.createPortfolio as jest.Mock).mockResolvedValue(testPortfolio)

      renderDashboard()

      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText(/Create Portfolio/i)).toBeInTheDocument()
      })

      // Click create portfolio
      fireEvent.click(screen.getByText(/Create Portfolio/i))

      // Fill in form
      const nameInput = screen.getByPlaceholderText(/My Trading Portfolio/i)
      const capitalInput = screen.getByPlaceholderText(/100000/i)

      fireEvent.change(nameInput, { target: { value: 'Test Portfolio' } })
      fireEvent.change(capitalInput, { target: { value: '100000' } })

      // Submit
      fireEvent.click(screen.getByRole('button', { name: /Create Portfolio/i }))

      await waitFor(() => {
        expect(portfolioService.createPortfolio).toHaveBeenCalledWith({
          name: 'Test Portfolio',
          initialCapital: 100000
        })
        expect(toast.success).toHaveBeenCalledWith('Portfolio created successfully!')
      })
    })
  })

  describe('Authentication Errors', () => {
    it('should handle 401 errors and redirect to login', async () => {
      mockUseAuthStore.mockReturnValue({
        user: testUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })

      // Mock authenticated fetch to return 401
      mockAuthenticatedFetch.mockResolvedValue({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized'
      } as Response)

      ;(marketService.getMarketIndicators as jest.Mock).mockRejectedValue(
        new Error('Authentication required')
      )

      // Mock window.location
      delete (window as any).location
      window.location = {
        href: '',
        pathname: '/dashboard'
      } as any

      renderDashboard()

      await waitFor(() => {
        expect(clearCache).toHaveBeenCalled()
        expect(localStorage.removeItem).toHaveBeenCalledWith('access_token')
        expect(localStorage.removeItem).toHaveBeenCalledWith('token_type')
      })
    })
  })

  describe('Cache Management', () => {
    beforeEach(() => {
      mockUseAuthStore.mockReturnValue({
        user: testUser,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        fetchUser: jest.fn(),
        login: jest.fn(),
        logout: jest.fn(),
        register: jest.fn(),
        updateProfile: jest.fn()
      })
    })

    it('should clear cache on refresh button click', async () => {
      ;(portfolioService.getPortfoliosSummary as jest.Mock).mockResolvedValue({
        portfolios: [],
        totalValue: 0,
        totalGain: 0
      })

      ;(marketService.getMarketIndicators as jest.Mock).mockResolvedValue(testMarketData)

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText(/Clear Cache & Refresh/i)).toBeInTheDocument()
      })

      // Click clear cache button
      fireEvent.click(screen.getByText(/Clear Cache & Refresh/i))

      expect(clearCache).toHaveBeenCalled()
      expect(toast.success).toHaveBeenCalledWith('Cache cleared! Refreshing...')
    })
  })
})
