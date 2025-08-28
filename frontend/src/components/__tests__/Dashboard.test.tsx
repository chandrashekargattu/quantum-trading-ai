import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard } from '../Dashboard';
import { useAuthStore } from '@/store/useAuthStore';
import { useMarketDataStore } from '@/store/useMarketDataStore';
import { usePortfolioStore } from '@/store/usePortfolioStore';
import { mockMarketData, mockPortfolio, mockUser } from '@/test/mocks';

// Mock stores
jest.mock('@/store/useAuthStore');
jest.mock('@/store/useMarketDataStore');
jest.mock('@/store/usePortfolioStore');

// Mock chart components
jest.mock('../charts/PriceChart', () => ({
  PriceChart: () => <div data-testid="price-chart">Price Chart</div>
}));

jest.mock('../charts/PortfolioChart', () => ({
  PortfolioChart: () => <div data-testid="portfolio-chart">Portfolio Chart</div>
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('Dashboard Component', () => {
  const mockAuthStore = useAuthStore as jest.MockedFunction<typeof useAuthStore>;
  const mockMarketStore = useMarketDataStore as jest.MockedFunction<typeof useMarketDataStore>;
  const mockPortfolioStore = usePortfolioStore as jest.MockedFunction<typeof usePortfolioStore>;

  beforeEach(() => {
    mockAuthStore.mockReturnValue({
      user: mockUser,
      isAuthenticated: true,
    });

    mockMarketStore.mockReturnValue({
      marketData: mockMarketData,
      isLoading: false,
      error: null,
      subscribeToSymbol: jest.fn(),
      unsubscribeFromSymbol: jest.fn(),
    });

    mockPortfolioStore.mockReturnValue({
      portfolio: mockPortfolio,
      isLoading: false,
      error: null,
      totalValue: 150000,
      dailyPnL: 2500,
      dailyPnLPercent: 1.69,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders dashboard with user greeting', () => {
    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByText(`Welcome back, ${mockUser.full_name}!`)).toBeInTheDocument();
  });

  it('displays portfolio summary correctly', () => {
    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
    expect(screen.getByText('$150,000.00')).toBeInTheDocument();
    expect(screen.getByText('Daily P&L')).toBeInTheDocument();
    expect(screen.getByText('+$2,500.00')).toBeInTheDocument();
    expect(screen.getByText('(+1.69%)')).toBeInTheDocument();
  });

  it('renders market watchlist', () => {
    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Market Watchlist')).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('$150.00')).toBeInTheDocument();
    expect(screen.getByText('+2.50%')).toBeInTheDocument();
  });

  it('allows adding symbols to watchlist', async () => {
    const user = userEvent.setup();
    const mockSubscribe = jest.fn();
    
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      subscribeToSymbol: mockSubscribe,
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    const addButton = screen.getByLabelText('Add symbol to watchlist');
    await user.click(addButton);
    
    const input = screen.getByPlaceholderText('Enter symbol');
    await user.type(input, 'GOOGL');
    
    const confirmButton = screen.getByText('Add');
    await user.click(confirmButton);
    
    expect(mockSubscribe).toHaveBeenCalledWith('GOOGL');
  });

  it('allows removing symbols from watchlist', async () => {
    const user = userEvent.setup();
    const mockUnsubscribe = jest.fn();
    
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      unsubscribeFromSymbol: mockUnsubscribe,
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    const removeButton = screen.getByLabelText('Remove AAPL from watchlist');
    await user.click(removeButton);
    
    expect(mockUnsubscribe).toHaveBeenCalledWith('AAPL');
  });

  it('displays loading state', () => {
    mockPortfolioStore.mockReturnValue({
      ...mockPortfolioStore(),
      isLoading: true,
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByTestId('dashboard-loading')).toBeInTheDocument();
  });

  it('displays error state', () => {
    mockPortfolioStore.mockReturnValue({
      ...mockPortfolioStore(),
      error: 'Failed to load portfolio',
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Failed to load portfolio')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('refreshes data on pull-to-refresh', async () => {
    const mockRefresh = jest.fn();
    mockPortfolioStore.mockReturnValue({
      ...mockPortfolioStore(),
      refreshPortfolio: mockRefresh,
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    const refreshButton = screen.getByLabelText('Refresh dashboard');
    await userEvent.click(refreshButton);
    
    expect(mockRefresh).toHaveBeenCalled();
  });

  it('navigates to portfolio details on click', async () => {
    const user = userEvent.setup();
    const mockNavigate = jest.fn();
    
    // Mock useNavigate
    jest.mock('react-router-dom', () => ({
      ...jest.requireActual('react-router-dom'),
      useNavigate: () => mockNavigate,
    }));

    render(<Dashboard />, { wrapper: createWrapper() });
    
    const portfolioCard = screen.getByTestId('portfolio-summary-card');
    await user.click(portfolioCard);
    
    expect(mockNavigate).toHaveBeenCalledWith('/portfolio');
  });

  it('updates market data in real-time', async () => {
    const { rerender } = render(<Dashboard />, { wrapper: createWrapper() });
    
    // Initial price
    expect(screen.getByText('$150.00')).toBeInTheDocument();
    
    // Update market data
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      marketData: {
        AAPL: {
          ...mockMarketData.AAPL,
          price: 155.00,
          change_percent: 5.33,
        },
      },
    });
    
    rerender(<Dashboard />);
    
    // Updated price
    expect(screen.getByText('$155.00')).toBeInTheDocument();
    expect(screen.getByText('+5.33%')).toBeInTheDocument();
  });

  it('shows AI predictions toggle', async () => {
    const user = userEvent.setup();
    
    render(<Dashboard />, { wrapper: createWrapper() });
    
    const aiToggle = screen.getByLabelText('Show AI predictions');
    expect(aiToggle).toBeInTheDocument();
    
    await user.click(aiToggle);
    
    expect(screen.getByText('AI Predictions')).toBeInTheDocument();
    expect(screen.getByTestId('ai-predictions-panel')).toBeInTheDocument();
  });

  it('displays quick actions menu', async () => {
    const user = userEvent.setup();
    
    render(<Dashboard />, { wrapper: createWrapper() });
    
    const quickActionsButton = screen.getByLabelText('Quick actions');
    await user.click(quickActionsButton);
    
    expect(screen.getByText('Place Order')).toBeInTheDocument();
    expect(screen.getByText('View Positions')).toBeInTheDocument();
    expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
    expect(screen.getByText('Backtest Strategy')).toBeInTheDocument();
  });

  it('handles WebSocket connection status', () => {
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      connectionStatus: 'connected',
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByLabelText('WebSocket connected')).toBeInTheDocument();
    expect(screen.getByLabelText('WebSocket connected')).toHaveClass('text-green-500');
  });

  it('filters watchlist by search term', async () => {
    const user = userEvent.setup();
    
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      marketData: {
        AAPL: mockMarketData.AAPL,
        GOOGL: { symbol: 'GOOGL', price: 2800, change_percent: 1.2 },
        MSFT: { symbol: 'MSFT', price: 380, change_percent: -0.5 },
      },
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    const searchInput = screen.getByPlaceholderText('Search symbols...');
    await user.type(searchInput, 'AA');
    
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.queryByText('GOOGL')).not.toBeInTheDocument();
    expect(screen.queryByText('MSFT')).not.toBeInTheDocument();
  });

  it('sorts watchlist by different criteria', async () => {
    const user = userEvent.setup();
    
    render(<Dashboard />, { wrapper: createWrapper() });
    
    const sortButton = screen.getByLabelText('Sort watchlist');
    await user.click(sortButton);
    
    const sortByChange = screen.getByText('Sort by % Change');
    await user.click(sortByChange);
    
    const symbols = screen.getAllByTestId('watchlist-symbol');
    expect(symbols[0]).toHaveTextContent('AAPL'); // Highest positive change
  });

  it('displays news feed', async () => {
    render(<Dashboard />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Market News')).toBeInTheDocument();
      expect(screen.getByText('Apple announces new AI features')).toBeInTheDocument();
    });
  });

  it('shows risk alerts', () => {
    mockPortfolioStore.mockReturnValue({
      ...mockPortfolioStore(),
      riskAlerts: [
        { id: '1', type: 'high_volatility', message: 'High volatility detected in TSLA' },
        { id: '2', type: 'concentration', message: 'Portfolio concentration exceeds 30% in tech sector' },
      ],
    });

    render(<Dashboard />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Risk Alerts')).toBeInTheDocument();
    expect(screen.getByText('High volatility detected in TSLA')).toBeInTheDocument();
    expect(screen.getByText('Portfolio concentration exceeds 30% in tech sector')).toBeInTheDocument();
  });
});
