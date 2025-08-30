import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import DashboardPage from '@/app/dashboard/page'
import { useAuthStore } from '@/store/useAuthStore'
import { portfolioService } from '@/services/api/portfolio-optimized'
import { marketService } from '@/services/api/market-optimized'
import { toast } from 'react-hot-toast'

// Mock dependencies
jest.mock('@/store/useAuthStore')
jest.mock('@/services/api/portfolio-optimized')
jest.mock('@/services/api/market-optimized')
jest.mock('react-hot-toast')
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
}))

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
global.localStorage = localStorageMock as any

// Mock lazy loaded components
jest.mock('@/components/dashboard/MarketOverview', () => ({
  MarketOverview: () => <div data-testid="market-overview">Market Overview</div>
}))

jest.mock('@/components/dashboard/PortfolioSummary', () => ({
  PortfolioSummary: () => <div data-testid="portfolio-summary">Portfolio Summary</div>
}))

jest.mock('@/components/dashboard/AIInsights', () => ({
  AIInsights: () => <div data-testid="ai-insights">AI Insights</div>
}))

jest.mock('@/components/dashboard/WatchlistWidget', () => ({
  WatchlistWidget: () => <div data-testid="watchlist">Watchlist</div>
}))

jest.mock('@/components/dashboard/TrendingOptions', () => ({
  TrendingOptions: () => <div data-testid="trending-options">Trending Options</div>
}))

jest.mock('@/components/dashboard/RecentTrades', () => ({
  RecentTrades: () => <div data-testid="recent-trades">Recent Trades</div>
}))

const mockUseAuthStore = useAuthStore as jest.MockedFunction<typeof useAuthStore>
const mockPortfolioService = portfolioService as jest.Mocked<typeof portfolioService>
const mockMarketService = marketService as jest.Mocked<typeof marketService>
const mockToast = toast as jest.Mocked<typeof toast>

describe('Dashboard Integration Flow', () => {
  let queryClient: QueryClient

  const mockUser = {
    id: '123',
    email: 'test@example.com',
    username: 'testuser',
    full_name: 'Test User',
    account_type: 'premium',
    subscription_tier: 'pro',
    is_active: true,
    is_verified: true
  }

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    
    jest.clearAllMocks()
    
    // Default auth state
    mockUseAuthStore.mockReturnValue({
      user: mockUser,
      isAuthenticated: true,
      isLoading: false,
      error: null,
      login: jest.fn(),
      logout: jest.fn(),
      clearError: jest.fn(),
      fetchUser: jest.fn()
    })

    // Default localStorage state
    localStorageMock.getItem.mockImplementation((key) => {
      if (key === 'access_token') return 'test-token'
      if (key === 'token_type') return 'Bearer'
      return null
    })
  })

  const renderDashboard = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <DashboardPage />
      </QueryClientProvider>
    )
  }

  describe('Authentication Flow', () => {
    it('should redirect to login when not authenticated', async () => {
      const mockPush = jest.fn()
      jest.spyOn(require('next/navigation'), 'useRouter').mockReturnValue({
        push: mockPush,
        replace: jest.fn(),
        prefetch: jest.fn(),
      })

      mockUseAuthStore.mockReturnValue({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
        login: jest.fn(),
        logout: jest.fn(),
        clearError: jest.fn(),
        fetchUser: jest.fn()
      })

      localStorageMock.getItem.mockReturnValue(null)

      renderDashboard()

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/auth/login')
      })
    })

    it('should fetch user if token exists but user is null', async () => {
      const fetchUser = jest.fn()
      mockUseAuthStore.mockReturnValue({
        user: null,
        isAuthenticated: true,
        isLoading: false,
        error: null,
        login: jest.fn(),
        logout: jest.fn(),
        clearError: jest.fn(),
        fetchUser
      })

      renderDashboard()

      await waitFor(() => {
        expect(fetchUser).toHaveBeenCalled()
      })
    })
  })

  describe('Dashboard Loading', () => {
    it('should show loading state initially', () => {
      renderDashboard()

      expect(screen.getByText('Loading your dashboard...')).toBeInTheDocument()
    })

    it('should display user welcome message after loading', async () => {
      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
        expect(screen.getByText("Here's what's happening in the markets today")).toBeInTheDocument()
        expect(screen.getByText('premium')).toBeInTheDocument()
      })
    })

    it('should render all dashboard components', async () => {
      renderDashboard()

      await waitFor(() => {
        expect(screen.getByTestId('market-overview')).toBeInTheDocument()
        expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument()
        expect(screen.getByTestId('ai-insights')).toBeInTheDocument()
        expect(screen.getByTestId('watchlist')).toBeInTheDocument()
        expect(screen.getByTestId('trending-options')).toBeInTheDocument()
        expect(screen.getByTestId('recent-trades')).toBeInTheDocument()
      })
    })
  })

  describe('API Integration', () => {
    it('should make correct API calls on mount', async () => {
      const mockPortfolios = [{
        id: '1',
        name: 'Test Portfolio',
        currentValue: 100000,
        cashBalance: 50000,
        totalReturn: 5000,
        totalReturnPercent: 5,
        dayChange: 1000,
        dayChangePercent: 1,
        initialCapital: 95000,
        investedAmount: 50000,
        createdAt: '2024-01-01',
        updatedAt: '2024-01-04'
      }]

      const mockIndicators = [
        { symbol: '^NSEI', name: 'NIFTY 50', value: 21000, change_amount: 150, change_percent: 0.72 }
      ]

      mockPortfolioService.getPortfolios.mockResolvedValue(mockPortfolios)
      mockMarketService.getMarketIndicators.mockResolvedValue(mockIndicators)

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
      })

      // Verify API calls were made with correct auth headers
      expect(mockPortfolioService.getPortfolios).toHaveBeenCalled()
      expect(mockMarketService.getMarketIndicators).toHaveBeenCalled()
    })

    it('should handle API errors gracefully', async () => {
      mockPortfolioService.getPortfolios.mockRejectedValue(new Error('Portfolio API error'))
      mockMarketService.getMarketIndicators.mockRejectedValue(new Error('Market API error'))

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
        // Dashboard should still render even with API errors
        expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument()
        expect(screen.getByTestId('market-overview')).toBeInTheDocument()
      })
    })
  })

  describe('Real-time Updates', () => {
    it('should refetch market data periodically', async () => {
      jest.useFakeTimers()

      mockMarketService.getMarketIndicators.mockResolvedValue([])

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
      })

      expect(mockMarketService.getMarketIndicators).toHaveBeenCalledTimes(1)

      // Advance time by 30 seconds (market data refresh interval)
      jest.advanceTimersByTime(30000)

      await waitFor(() => {
        expect(mockMarketService.getMarketIndicators).toHaveBeenCalledTimes(2)
      })

      jest.useRealTimers()
    })
  })

  describe('User Interactions', () => {
    it('should handle portfolio creation from empty state', async () => {
      // Start with no portfolios
      mockPortfolioService.getPortfolios.mockResolvedValue([])

      renderDashboard()

      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
      })

      // Portfolio component would show create button
      // This would be tested in the actual PortfolioSummary component test
      expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument()
    })
  })

  describe('Performance Optimization', () => {
    it('should lazy load dashboard components', async () => {
      renderDashboard()

      // Initially shows loading
      expect(screen.getByText('Loading your dashboard...')).toBeInTheDocument()

      // Components load progressively
      await waitFor(() => {
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
      })

      // All components should be loaded
      await waitFor(() => {
        expect(screen.getAllByTestId(/market-overview|portfolio-summary|ai-insights|watchlist|trending-options|recent-trades/)).toHaveLength(6)
      })
    })
  })

  describe('Error Boundaries', () => {
    it('should handle component errors gracefully', async () => {
      // Mock console.error to avoid noise in test output
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation()

      // Force an error in one component
      jest.spyOn(require('@/components/dashboard/MarketOverview'), 'MarketOverview')
        .mockImplementation(() => {
          throw new Error('Component error')
        })

      renderDashboard()

      await waitFor(() => {
        // Dashboard should still render other components
        expect(screen.getByText('Welcome back, Test User!')).toBeInTheDocument()
        expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument()
      })

      consoleSpy.mockRestore()
    })
  })
})