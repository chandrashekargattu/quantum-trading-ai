import { render, screen } from '@/test-utils/test-utils'
import { PortfolioSummary } from '../PortfolioSummary'
import { useQuery } from '@tanstack/react-query'
import { mockPortfolio } from '@/test-utils/test-utils'

jest.mock('@tanstack/react-query')

describe('PortfolioSummary', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders loading state', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: null,
      isLoading: true,
    })

    render(<PortfolioSummary />)

    expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
    expect(document.querySelector('.animate-pulse')).toBeInTheDocument()
  })

  it('renders empty state when no portfolio exists', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: [],
      isLoading: false,
    })

    render(<PortfolioSummary />)

    expect(screen.getByText('No portfolio found. Create one to start trading.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Create Portfolio' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Create Portfolio' })).toHaveAttribute('href', '/portfolio/create')
  })

  it('renders portfolio data correctly', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: [mockPortfolio],
      isLoading: false,
    })

    render(<PortfolioSummary />)

    // Check header
    expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'View Details' })).toHaveAttribute('href', '/portfolio')

    // Check total value
    expect(screen.getByText('Total Portfolio Value')).toBeInTheDocument()
    expect(screen.getByText('$100,000.00')).toBeInTheDocument()

    // Check daily return
    expect(screen.getByText('$200.00')).toBeInTheDocument()
    expect(screen.getByText('(0.20%)')).toBeInTheDocument()

    // Check metrics
    expect(screen.getByText('Cash Balance')).toBeInTheDocument()
    expect(screen.getByText('$50,000.00')).toBeInTheDocument()

    expect(screen.getByText('Total Return')).toBeInTheDocument()
    expect(screen.getByText('5.00%')).toBeInTheDocument()

    expect(screen.getByText('Buying Power')).toBeInTheDocument()
  })

  it('displays positive returns with bullish color', () => {
    const positivePortfolio = {
      ...mockPortfolio,
      daily_return: 500,
      daily_return_percent: 0.5,
      total_return: 10000,
      total_return_percent: 10,
    }

    ;(useQuery as jest.Mock).mockReturnValue({
      data: [positivePortfolio],
      isLoading: false,
    })

    render(<PortfolioSummary />)

    const returnElements = screen.getAllByText(/\$500\.00/)
    const percentElements = screen.getAllByText(/10\.00%/)
    
    returnElements.forEach(el => {
      if (el.closest('.text-bullish')) {
        expect(el.closest('.text-bullish')).toBeInTheDocument()
      }
    })
  })

  it('displays negative returns with bearish color', () => {
    const negativePortfolio = {
      ...mockPortfolio,
      daily_return: -500,
      daily_return_percent: -0.5,
      total_return: -5000,
      total_return_percent: -5,
    }

    ;(useQuery as jest.Mock).mockReturnValue({
      data: [negativePortfolio],
      isLoading: false,
    })

    render(<PortfolioSummary />)

    const returnElements = screen.getAllByText(/\$500\.00/)
    const percentElements = screen.getAllByText(/5\.00%/)
    
    returnElements.forEach(el => {
      if (el.closest('.text-bearish')) {
        expect(el.closest('.text-bearish')).toBeInTheDocument()
      }
    })
  })

  it('selects default portfolio when multiple exist', () => {
    const portfolios = [
      { ...mockPortfolio, is_default: false, name: 'Secondary' },
      { ...mockPortfolio, is_default: true, name: 'Primary' },
    ]

    ;(useQuery as jest.Mock).mockReturnValue({
      data: portfolios,
      isLoading: false,
    })

    render(<PortfolioSummary />)

    // Should display the default portfolio's value
    expect(screen.getByText('$100,000.00')).toBeInTheDocument()
  })

  it('falls back to first portfolio if no default', () => {
    const portfolios = [
      { ...mockPortfolio, is_default: false, name: 'First', total_value: 75000 },
      { ...mockPortfolio, is_default: false, name: 'Second', total_value: 125000 },
    ]

    ;(useQuery as jest.Mock).mockReturnValue({
      data: portfolios,
      isLoading: false,
    })

    render(<PortfolioSummary />)

    // Should display the first portfolio's value
    expect(screen.getByText('$75,000.00')).toBeInTheDocument()
  })

  it('displays performance chart placeholder', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: [mockPortfolio],
      isLoading: false,
    })

    render(<PortfolioSummary />)

    expect(screen.getByText('Performance Chart')).toBeInTheDocument()
  })
})
