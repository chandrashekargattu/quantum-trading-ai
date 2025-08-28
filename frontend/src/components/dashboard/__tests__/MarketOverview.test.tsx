import { render, screen } from '@/test-utils/test-utils'
import { MarketOverview } from '../MarketOverview'
import { useQuery } from '@tanstack/react-query'

jest.mock('@tanstack/react-query')

describe('MarketOverview', () => {
  const mockIndicators = [
    {
      symbol: '^GSPC',
      name: 'S&P 500',
      value: 4500.0,
      change_amount: 25.0,
      change_percent: 0.56,
    },
    {
      symbol: '^DJI',
      name: 'Dow Jones',
      value: 35000.0,
      change_amount: -150.0,
      change_percent: -0.43,
    },
    {
      symbol: '^IXIC',
      name: 'NASDAQ',
      value: 14000.0,
      change_amount: 100.0,
      change_percent: 0.72,
    },
    {
      symbol: '^VIX',
      name: 'VIX',
      value: 15.5,
      change_amount: 0.5,
      change_percent: 3.33,
    },
  ]

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders loading state', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: null,
      isLoading: true,
    })

    render(<MarketOverview />)

    // Should show 4 loading cards
    const loadingCards = screen.getAllByText(/^$/).filter(
      element => element.closest('.animate-pulse')
    )
    expect(loadingCards.length).toBeGreaterThan(0)
  })

  it('renders market indicators correctly', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: mockIndicators,
      isLoading: false,
    })

    render(<MarketOverview />)

    // Check that all indicators are displayed
    expect(screen.getByText('S&P 500')).toBeInTheDocument()
    expect(screen.getByText('Dow Jones')).toBeInTheDocument()
    expect(screen.getByText('NASDAQ')).toBeInTheDocument()
    expect(screen.getByText('VIX')).toBeInTheDocument()

    // Check values are formatted correctly
    expect(screen.getByText('$4,500.00')).toBeInTheDocument()
    expect(screen.getByText('$35,000.00')).toBeInTheDocument()
    expect(screen.getByText('$14,000.00')).toBeInTheDocument()
    expect(screen.getByText('$15.50')).toBeInTheDocument()

    // Check percentages
    expect(screen.getByText('0.56%')).toBeInTheDocument()
    expect(screen.getByText('0.43%')).toBeInTheDocument()
    expect(screen.getByText('0.72%')).toBeInTheDocument()
    expect(screen.getByText('3.33%')).toBeInTheDocument()
  })

  it('displays positive changes in green', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: mockIndicators,
      isLoading: false,
    })

    render(<MarketOverview />)

    const positiveChanges = screen.getAllByText(/\+?\d+\.\d+%/)
      .filter(el => el.classList.contains('text-bullish'))
    
    expect(positiveChanges.length).toBe(3) // S&P, NASDAQ, VIX
  })

  it('displays negative changes in red', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: mockIndicators,
      isLoading: false,
    })

    render(<MarketOverview />)

    const negativeChanges = screen.getAllByText(/\d+\.\d+%/)
      .filter(el => el.classList.contains('text-bearish'))
    
    expect(negativeChanges.length).toBe(1) // Dow Jones
  })

  it('shows change amounts with proper signs', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: mockIndicators,
      isLoading: false,
    })

    render(<MarketOverview />)

    expect(screen.getByText('+$25.00')).toBeInTheDocument()
    expect(screen.getByText('-$150.00')).toBeInTheDocument()
    expect(screen.getByText('+$100.00')).toBeInTheDocument()
    expect(screen.getByText('+$0.50')).toBeInTheDocument()
  })

  it('refetches data at intervals', () => {
    const mockUseQuery = useQuery as jest.Mock
    mockUseQuery.mockReturnValue({
      data: mockIndicators,
      isLoading: false,
    })

    render(<MarketOverview />)

    // Check that refetchInterval is set to 30 seconds
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        refetchInterval: 30000,
      })
    )
  })

  it('handles empty data gracefully', () => {
    ;(useQuery as jest.Mock).mockReturnValue({
      data: [],
      isLoading: false,
    })

    render(<MarketOverview />)

    // Should render but with no content
    const container = screen.getByTestId
    expect(() => screen.getByText('S&P 500')).toThrow()
  })
})
