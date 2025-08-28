import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TradingPanel } from '../TradingPanel';
import { useTradingStore } from '@/store/useTradingStore';
import { useMarketDataStore } from '@/store/useMarketDataStore';
import { usePortfolioStore } from '@/store/usePortfolioStore';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';

// Mock dependencies
jest.mock('@/store/useTradingStore');
jest.mock('@/store/useMarketDataStore');
jest.mock('@/store/usePortfolioStore');
jest.mock('@/lib/api');
jest.mock('react-hot-toast');

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

describe('TradingPanel Component', () => {
  const mockTradingStore = useTradingStore as jest.MockedFunction<typeof useTradingStore>;
  const mockMarketStore = useMarketDataStore as jest.MockedFunction<typeof useMarketDataStore>;
  const mockPortfolioStore = usePortfolioStore as jest.MockedFunction<typeof usePortfolioStore>;
  const mockApi = api as jest.Mocked<typeof api>;
  const mockToast = toast as jest.Mocked<typeof toast>;

  const defaultStoreValues = {
    selectedSymbol: 'AAPL',
    orderType: 'market',
    orderSide: 'buy',
    quantity: 100,
    limitPrice: null,
    stopPrice: null,
    timeInForce: 'day',
    setSelectedSymbol: jest.fn(),
    setOrderType: jest.fn(),
    setOrderSide: jest.fn(),
    setQuantity: jest.fn(),
    setLimitPrice: jest.fn(),
    setStopPrice: jest.fn(),
    setTimeInForce: jest.fn(),
    submitOrder: jest.fn(),
  };

  beforeEach(() => {
    mockTradingStore.mockReturnValue(defaultStoreValues);

    mockMarketStore.mockReturnValue({
      marketData: {
        AAPL: {
          symbol: 'AAPL',
          price: 150.00,
          bid: 149.95,
          ask: 150.05,
          change_percent: 2.5,
          volume: 1000000,
        },
      },
      getQuote: jest.fn().mockResolvedValue({
        price: 150.00,
        bid: 149.95,
        ask: 150.05,
      }),
    });

    mockPortfolioStore.mockReturnValue({
      portfolio: {
        cash_balance: 50000,
        buying_power: 50000,
        positions: {
          AAPL: { quantity: 100, avg_cost: 140 },
        },
      },
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders trading panel with symbol info', () => {
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('$150.00')).toBeInTheDocument();
    expect(screen.getByText('+2.50%')).toBeInTheDocument();
  });

  it('displays buy and sell tabs', () => {
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByRole('tab', { name: 'Buy' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Sell' })).toBeInTheDocument();
  });

  it('switches between buy and sell modes', async () => {
    const user = userEvent.setup();
    const mockSetOrderSide = jest.fn();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      setOrderSide: mockSetOrderSide,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const sellTab = screen.getByRole('tab', { name: 'Sell' });
    await user.click(sellTab);
    
    expect(mockSetOrderSide).toHaveBeenCalledWith('sell');
  });

  it('allows changing order type', async () => {
    const user = userEvent.setup();
    const mockSetOrderType = jest.fn();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      setOrderType: mockSetOrderType,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const orderTypeSelect = screen.getByLabelText('Order Type');
    await user.selectOptions(orderTypeSelect, 'limit');
    
    expect(mockSetOrderType).toHaveBeenCalledWith('limit');
  });

  it('shows limit price input for limit orders', () => {
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderType: 'limit',
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByLabelText('Limit Price')).toBeInTheDocument();
  });

  it('shows stop price input for stop orders', () => {
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderType: 'stop',
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByLabelText('Stop Price')).toBeInTheDocument();
  });

  it('shows both prices for stop-limit orders', () => {
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderType: 'stop_limit',
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByLabelText('Stop Price')).toBeInTheDocument();
    expect(screen.getByLabelText('Limit Price')).toBeInTheDocument();
  });

  it('updates quantity input', async () => {
    const user = userEvent.setup();
    const mockSetQuantity = jest.fn();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      setQuantity: mockSetQuantity,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const quantityInput = screen.getByLabelText('Quantity');
    await user.clear(quantityInput);
    await user.type(quantityInput, '200');
    
    expect(mockSetQuantity).toHaveBeenCalledWith(200);
  });

  it('calculates and displays order value', () => {
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Order Value')).toBeInTheDocument();
    expect(screen.getByText('$15,000.00')).toBeInTheDocument(); // 100 * 150
  });

  it('calculates and displays commission', () => {
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Commission')).toBeInTheDocument();
    expect(screen.getByText('$0.00')).toBeInTheDocument(); // Free trading
  });

  it('shows buying power for buy orders', () => {
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Buying Power')).toBeInTheDocument();
    expect(screen.getByText('$50,000.00')).toBeInTheDocument();
  });

  it('shows available shares for sell orders', () => {
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderSide: 'sell',
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Available Shares')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('validates insufficient buying power', async () => {
    const user = userEvent.setup();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      quantity: 1000, // $150,000 order value > $50,000 buying power
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const submitButton = screen.getByText('Place Buy Order');
    await user.click(submitButton);
    
    expect(screen.getByText('Insufficient buying power')).toBeInTheDocument();
  });

  it('validates insufficient shares for sell', async () => {
    const user = userEvent.setup();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderSide: 'sell',
      quantity: 200, // More than 100 available
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const submitButton = screen.getByText('Place Sell Order');
    await user.click(submitButton);
    
    expect(screen.getByText('Insufficient shares')).toBeInTheDocument();
  });

  it('submits market order successfully', async () => {
    const user = userEvent.setup();
    const mockSubmitOrder = jest.fn().mockResolvedValue({ id: '123' });
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      submitOrder: mockSubmitOrder,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const submitButton = screen.getByText('Place Buy Order');
    await user.click(submitButton);
    
    expect(mockSubmitOrder).toHaveBeenCalledWith({
      symbol: 'AAPL',
      side: 'buy',
      type: 'market',
      quantity: 100,
      time_in_force: 'day',
    });
    
    await waitFor(() => {
      expect(mockToast.success).toHaveBeenCalledWith('Order submitted successfully');
    });
  });

  it('submits limit order with price', async () => {
    const user = userEvent.setup();
    const mockSubmitOrder = jest.fn().mockResolvedValue({ id: '123' });
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      orderType: 'limit',
      limitPrice: 149.50,
      submitOrder: mockSubmitOrder,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const submitButton = screen.getByText('Place Buy Order');
    await user.click(submitButton);
    
    expect(mockSubmitOrder).toHaveBeenCalledWith({
      symbol: 'AAPL',
      side: 'buy',
      type: 'limit',
      quantity: 100,
      limit_price: 149.50,
      time_in_force: 'day',
    });
  });

  it('displays order preview modal', async () => {
    const user = userEvent.setup();
    
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const previewButton = screen.getByText('Preview Order');
    await user.click(previewButton);
    
    expect(screen.getByText('Order Preview')).toBeInTheDocument();
    expect(screen.getByText('Buy 100 shares of AAPL')).toBeInTheDocument();
    expect(screen.getByText('Market Order')).toBeInTheDocument();
    expect(screen.getByText('Total Cost: $15,000.00')).toBeInTheDocument();
  });

  it('allows symbol search and selection', async () => {
    const user = userEvent.setup();
    const mockSetSelectedSymbol = jest.fn();
    
    mockTradingStore.mockReturnValue({
      ...defaultStoreValues,
      setSelectedSymbol: mockSetSelectedSymbol,
    });

    mockApi.searchSymbols = jest.fn().mockResolvedValue([
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'GOOG', name: 'Alphabet Inc. Class C' },
    ]);

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const searchButton = screen.getByLabelText('Search symbols');
    await user.click(searchButton);
    
    const searchInput = screen.getByPlaceholderText('Search symbol or company name');
    await user.type(searchInput, 'GOOG');
    
    await waitFor(() => {
      expect(screen.getByText('GOOGL')).toBeInTheDocument();
      expect(screen.getByText('Alphabet Inc.')).toBeInTheDocument();
    });
    
    const googleOption = screen.getByText('GOOGL');
    await user.click(googleOption);
    
    expect(mockSetSelectedSymbol).toHaveBeenCalledWith('GOOGL');
  });

  it('shows advanced order options', async () => {
    const user = userEvent.setup();
    
    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const advancedButton = screen.getByText('Advanced Options');
    await user.click(advancedButton);
    
    expect(screen.getByLabelText('Time in Force')).toBeInTheDocument();
    expect(screen.getByLabelText('Extended Hours')).toBeInTheDocument();
    expect(screen.getByLabelText('All or None')).toBeInTheDocument();
  });

  it('displays real-time quote updates', async () => {
    const { rerender } = render(<TradingPanel />, { wrapper: createWrapper() });
    
    expect(screen.getByText('$150.00')).toBeInTheDocument();
    
    // Update market data
    mockMarketStore.mockReturnValue({
      ...mockMarketStore(),
      marketData: {
        AAPL: {
          symbol: 'AAPL',
          price: 151.50,
          bid: 151.45,
          ask: 151.55,
          change_percent: 3.5,
          volume: 1100000,
        },
      },
    });
    
    rerender(<TradingPanel />);
    
    expect(screen.getByText('$151.50')).toBeInTheDocument();
    expect(screen.getByText('+3.50%')).toBeInTheDocument();
  });

  it('shows AI trading suggestions', async () => {
    const user = userEvent.setup();
    
    mockApi.getAISuggestions = jest.fn().mockResolvedValue({
      action: 'buy',
      confidence: 0.85,
      reasoning: 'Strong momentum and positive sentiment',
      suggested_price: 149.75,
      suggested_quantity: 150,
    });

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const aiButton = screen.getByLabelText('Get AI suggestions');
    await user.click(aiButton);
    
    await waitFor(() => {
      expect(screen.getByText('AI Suggestion: Buy')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 85%')).toBeInTheDocument();
      expect(screen.getByText('Strong momentum and positive sentiment')).toBeInTheDocument();
    });
  });

  it('displays order history', async () => {
    const user = userEvent.setup();
    
    mockApi.getOrderHistory = jest.fn().mockResolvedValue([
      {
        id: '1',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 50,
        price: 148.50,
        status: 'filled',
        timestamp: '2024-01-10T10:30:00Z',
      },
      {
        id: '2',
        symbol: 'AAPL',
        side: 'sell',
        quantity: 25,
        price: 152.00,
        status: 'filled',
        timestamp: '2024-01-11T14:15:00Z',
      },
    ]);

    render(<TradingPanel />, { wrapper: createWrapper() });
    
    const historyTab = screen.getByRole('tab', { name: 'Order History' });
    await user.click(historyTab);
    
    await waitFor(() => {
      expect(screen.getByText('Buy 50 @ $148.50')).toBeInTheDocument();
      expect(screen.getByText('Sell 25 @ $152.00')).toBeInTheDocument();
    });
  });
});
