import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { PortfolioSummary } from '../PortfolioSummary'
import { portfolioService } from '@/services/api/portfolio'
import { toast } from 'react-hot-toast'

// Mock the services
jest.mock('@/services/api/portfolio')
jest.mock('react-hot-toast')

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => {
    return <a href={href}>{children}</a>
  }
})

// Mock utils
jest.mock('@/lib/utils', () => ({
  formatCurrency: (value: number) => `₹${value.toLocaleString('en-IN')}`,
  formatPercentage: (value: number) => `${(value * 100).toFixed(2)}%`
}))

const mockPortfolioService = portfolioService as jest.Mocked<typeof portfolioService>
const mockToast = toast as jest.Mocked<typeof toast>

describe('PortfolioSummary', () => {
  let queryClient: QueryClient

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    jest.clearAllMocks()
  })

  const renderComponent = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <PortfolioSummary />
      </QueryClientProvider>
    )
  }

  describe('Loading State', () => {
    it('should show loading skeleton while fetching portfolios', async () => {
      mockPortfolioService.getPortfolios.mockImplementation(() => 
        new Promise(() => {}) // Never resolves
      )

      renderComponent()

      expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
      expect(screen.getByTestId('portfolio-loading-skeleton')).toBeInTheDocument()
    })
  })

  describe('Empty State', () => {
    it('should show create portfolio button when no portfolios exist', async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([])

      renderComponent()

      await waitFor(() => {
        expect(screen.getByText('No portfolio found. Create one to start trading.')).toBeInTheDocument()
        expect(screen.getByText('Create Portfolio')).toBeInTheDocument()
      })
    })

    it('should show create portfolio form when button is clicked', async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([])

      renderComponent()

      await waitFor(() => {
        fireEvent.click(screen.getByText('Create Portfolio'))
      })

      expect(screen.getByText('Create New Portfolio')).toBeInTheDocument()
      expect(screen.getByLabelText('Portfolio Name')).toBeInTheDocument()
      expect(screen.getByLabelText('Initial Capital (₹)')).toBeInTheDocument()
      expect(screen.getByText('Cancel')).toBeInTheDocument()
    })
  })

  describe('Portfolio Creation', () => {
    beforeEach(async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([])
      renderComponent()
      
      await waitFor(() => {
        fireEvent.click(screen.getByText('Create Portfolio'))
      })
    })

    it('should create portfolio with valid data', async () => {
      const newPortfolio = {
        id: '1',
        name: 'My Trading Portfolio',
        initialCapital: 200000,
        currentValue: 200000,
        totalReturn: 0,
        totalReturnPercent: 0,
        dayChange: 0,
        dayChangePercent: 0,
        cashBalance: 200000,
        investedAmount: 0,
        createdAt: '2024-01-04',
        updatedAt: '2024-01-04'
      }

      mockPortfolioService.createPortfolio.mockResolvedValue(newPortfolio)
      mockPortfolioService.getPortfolios.mockResolvedValue([newPortfolio])

      const nameInput = screen.getByLabelText('Portfolio Name')
      const capitalInput = screen.getByLabelText('Initial Capital (₹)')

      fireEvent.change(nameInput, { target: { value: 'My Trading Portfolio' } })
      fireEvent.change(capitalInput, { target: { value: '200000' } })

      fireEvent.click(screen.getByRole('button', { name: 'Create Portfolio' }))

      await waitFor(() => {
        expect(mockPortfolioService.createPortfolio).toHaveBeenCalledWith({
          name: 'My Trading Portfolio',
          initialCapital: 200000
        })
        expect(mockToast.success).toHaveBeenCalledWith('Portfolio created successfully!')
      })
    })

    it('should show error for empty portfolio name', async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Create Portfolio' }))

      await waitFor(() => {
        expect(mockToast.error).toHaveBeenCalledWith('Please enter a portfolio name')
        expect(mockPortfolioService.createPortfolio).not.toHaveBeenCalled()
      })
    })

    it('should show error for invalid capital amount', async () => {
      const nameInput = screen.getByLabelText('Portfolio Name')
      const capitalInput = screen.getByLabelText('Initial Capital (₹)')

      fireEvent.change(nameInput, { target: { value: 'Test Portfolio' } })
      fireEvent.change(capitalInput, { target: { value: '-1000' } })

      fireEvent.click(screen.getByRole('button', { name: 'Create Portfolio' }))

      await waitFor(() => {
        expect(mockToast.error).toHaveBeenCalledWith('Please enter a valid initial capital amount')
        expect(mockPortfolioService.createPortfolio).not.toHaveBeenCalled()
      })
    })

    it('should handle portfolio creation error', async () => {
      mockPortfolioService.createPortfolio.mockRejectedValue(new Error('Server error'))

      const nameInput = screen.getByLabelText('Portfolio Name')
      fireEvent.change(nameInput, { target: { value: 'Test Portfolio' } })

      fireEvent.click(screen.getByRole('button', { name: 'Create Portfolio' }))

      await waitFor(() => {
        expect(mockToast.error).toHaveBeenCalledWith('Server error')
      })
    })

    it('should reset form on cancel', async () => {
      const nameInput = screen.getByLabelText('Portfolio Name')
      const capitalInput = screen.getByLabelText('Initial Capital (₹)')

      fireEvent.change(nameInput, { target: { value: 'Test Portfolio' } })
      fireEvent.change(capitalInput, { target: { value: '500000' } })

      fireEvent.click(screen.getByText('Cancel'))

      await waitFor(() => {
        expect(screen.getByText('Create Portfolio')).toBeInTheDocument()
        expect(screen.queryByLabelText('Portfolio Name')).not.toBeInTheDocument()
      })
    })
  })

  describe('Portfolio Display', () => {
    const mockPortfolio = {
      id: '1',
      name: 'My Portfolio',
      initialCapital: 100000,
      currentValue: 110000,
      totalReturn: 10000,
      totalReturnPercent: 10,
      dayChange: 500,
      dayChangePercent: 0.45,
      cashBalance: 50000,
      investedAmount: 60000,
      createdAt: '2024-01-01',
      updatedAt: '2024-01-04'
    }

    it('should display portfolio data correctly', async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([mockPortfolio])

      renderComponent()

      await waitFor(() => {
        expect(screen.getByText('Total Portfolio Value')).toBeInTheDocument()
        expect(screen.getByText('₹1,10,000')).toBeInTheDocument() // currentValue
        expect(screen.getByText('₹500')).toBeInTheDocument() // dayChange
        expect(screen.getByText('(45.00%)')).toBeInTheDocument() // dayChangePercent
        expect(screen.getByText('Cash Balance')).toBeInTheDocument()
        expect(screen.getByText('₹50,000')).toBeInTheDocument() // cashBalance
        expect(screen.getByText('Total Return')).toBeInTheDocument()
        expect(screen.getByText('1000.00%')).toBeInTheDocument() // totalReturnPercent
        expect(screen.getByText('Buying Power')).toBeInTheDocument()
      })
    })

    it('should show positive indicator for gains', async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([mockPortfolio])

      renderComponent()

      await waitFor(() => {
        const changeElement = screen.getByText('₹500').parentElement
        expect(changeElement).toHaveClass('text-bullish')
      })
    })

    it('should show negative indicator for losses', async () => {
      const lossPortfolio = {
        ...mockPortfolio,
        dayChange: -500,
        dayChangePercent: -0.45,
        totalReturn: -5000,
        totalReturnPercent: -5
      }

      mockPortfolioService.getPortfolios.mockResolvedValue([lossPortfolio])

      renderComponent()

      await waitFor(() => {
        const changeElement = screen.getByText('₹500').parentElement
        expect(changeElement).toHaveClass('text-bearish')
      })
    })

    it('should have link to portfolio details', async () => {
      mockPortfolioService.getPortfolios.mockResolvedValue([mockPortfolio])

      renderComponent()

      await waitFor(() => {
        const detailsLink = screen.getByText('View Details').closest('a')
        expect(detailsLink).toHaveAttribute('href', '/portfolio')
      })
    })
  })

  describe('Multiple Portfolios', () => {
    it('should use first portfolio as default', async () => {
      const portfolios = [
        {
          id: '1',
          name: 'Portfolio 1',
          currentValue: 100000,
          dayChange: 1000,
          dayChangePercent: 1,
          cashBalance: 40000,
          totalReturn: 5000,
          totalReturnPercent: 5,
          initialCapital: 95000,
          investedAmount: 60000,
          createdAt: '2024-01-01',
          updatedAt: '2024-01-04'
        },
        {
          id: '2',
          name: 'Portfolio 2',
          currentValue: 200000,
          dayChange: 2000,
          dayChangePercent: 1,
          cashBalance: 80000,
          totalReturn: 10000,
          totalReturnPercent: 5,
          initialCapital: 190000,
          investedAmount: 120000,
          createdAt: '2024-01-02',
          updatedAt: '2024-01-04'
        }
      ]

      mockPortfolioService.getPortfolios.mockResolvedValue(portfolios)

      renderComponent()

      await waitFor(() => {
        // Should display first portfolio's value
        expect(screen.getByText('₹1,00,000')).toBeInTheDocument()
        expect(screen.queryByText('₹2,00,000')).not.toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle portfolio fetch error gracefully', async () => {
      mockPortfolioService.getPortfolios.mockRejectedValue(new Error('Network error'))

      renderComponent()

      await waitFor(() => {
        // Should still show the component structure
        expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
      })
    })
  })
})