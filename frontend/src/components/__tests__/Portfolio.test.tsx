import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PortfolioOverview } from '../portfolio/PortfolioOverview';
import { PositionsTable } from '../portfolio/PositionsTable';
import { PerformanceMetrics } from '../portfolio/PerformanceMetrics';
import { AllocationChart } from '../portfolio/AllocationChart';
import { RiskAnalysis } from '../portfolio/RiskAnalysis';
import { usePortfolioStore } from '@/store/usePortfolioStore';
import { api } from '@/lib/api';
import { mockPortfolio, mockPositions, mockPerformance } from '@/test/mocks';

jest.mock('@/store/usePortfolioStore');
jest.mock('@/lib/api');

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

describe('PortfolioOverview Component', () => {
  const mockPortfolioStore = usePortfolioStore as jest.MockedFunction<typeof usePortfolioStore>;

  beforeEach(() => {
    mockPortfolioStore.mockReturnValue({
      portfolio: mockPortfolio,
      positions: mockPositions,
      performance: mockPerformance,
      isLoading: false,
      error: null,
      fetchPortfolio: jest.fn(),
      fetchPositions: jest.fn(),
      fetchPerformance: jest.fn(),
    });
  });

  it('renders portfolio summary', () => {
    render(<PortfolioOverview />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
    expect(screen.getByText('Total Value')).toBeInTheDocument();
    expect(screen.getByText('$150,000.00')).toBeInTheDocument();
    expect(screen.getByText('Cash Balance')).toBeInTheDocument();
    expect(screen.getByText('$50,000.00')).toBeInTheDocument();
  });

  it('displays performance metrics', () => {
    render(<PortfolioOverview />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Today\'s Return')).toBeInTheDocument();
    expect(screen.getByText('+$2,500.00')).toBeInTheDocument();
    expect(screen.getByText('(+1.69%)')).toBeInTheDocument();
    expect(screen.getByText('Total Return')).toBeInTheDocument();
    expect(screen.getByText('+$50,000.00')).toBeInTheDocument();
    expect(screen.getByText('(+50.00%)')).toBeInTheDocument();
  });

  it('shows portfolio allocation breakdown', () => {
    render(<PortfolioOverview />, { wrapper: createWrapper() });
    
    expect(screen.getByText('Asset Allocation')).toBeInTheDocument();
    expect(screen.getByText('Stocks: 66.7%')).toBeInTheDocument();
    expect(screen.getByText('Cash: 33.3%')).toBeInTheDocument();
  });

  it('allows time period selection', async () => {
    const user = userEvent.setup();
    const mockFetchPerformance = jest.fn();
    
    mockPortfolioStore.mockReturnValue({
      ...mockPortfolioStore(),
      fetchPerformance: mockFetchPerformance,
    });

    render(<PortfolioOverview />, { wrapper: createWrapper() });
    
    const periodSelector = screen.getByLabelText('Time period');
    await user.selectOptions(periodSelector, '1M');
    
    expect(mockFetchPerformance).toHaveBeenCalledWith('1M');
  });
});

describe('PositionsTable Component', () => {
  const defaultProps = {
    positions: mockPositions,
    onPositionClick: jest.fn(),
  };

  it('renders positions table with all columns', () => {
    render(<PositionsTable {...defaultProps} />);
    
    expect(screen.getByText('Symbol')).toBeInTheDocument();
    expect(screen.getByText('Quantity')).toBeInTheDocument();
    expect(screen.getByText('Avg Cost')).toBeInTheDocument();
    expect(screen.getByText('Current Price')).toBeInTheDocument();
    expect(screen.getByText('Market Value')).toBeInTheDocument();
    expect(screen.getByText('Unrealized P&L')).toBeInTheDocument();
    expect(screen.getByText('% of Portfolio')).toBeInTheDocument();
  });

  it('displays position data correctly', () => {
    render(<PositionsTable {...defaultProps} />);
    
    const appleRow = screen.getByTestId('position-AAPL');
    
    expect(within(appleRow).getByText('AAPL')).toBeInTheDocument();
    expect(within(appleRow).getByText('100')).toBeInTheDocument();
    expect(within(appleRow).getByText('$140.00')).toBeInTheDocument();
    expect(within(appleRow).getByText('$150.00')).toBeInTheDocument();
    expect(within(appleRow).getByText('$15,000.00')).toBeInTheDocument();
    expect(within(appleRow).getByText('+$1,000.00')).toBeInTheDocument();
    expect(within(appleRow).getByText('(+7.14%)')).toBeInTheDocument();
    expect(within(appleRow).getByText('10.0%')).toBeInTheDocument();
  });

  it('sorts positions by different columns', async () => {
    const user = userEvent.setup();
    
    render(<PositionsTable {...defaultProps} />);
    
    const symbolHeader = screen.getByText('Symbol');
    await user.click(symbolHeader);
    
    const rows = screen.getAllByTestId(/^position-/);
    expect(rows[0]).toHaveAttribute('data-testid', 'position-AAPL');
    
    await user.click(symbolHeader); // Reverse sort
    expect(rows[0]).toHaveAttribute('data-testid', 'position-TSLA');
  });

  it('filters positions by search term', async () => {
    const user = userEvent.setup();
    
    render(<PositionsTable {...defaultProps} />);
    
    const searchInput = screen.getByPlaceholderText('Search positions...');
    await user.type(searchInput, 'AA');
    
    expect(screen.getByTestId('position-AAPL')).toBeInTheDocument();
    expect(screen.queryByTestId('position-GOOGL')).not.toBeInTheDocument();
  });

  it('handles position click', async () => {
    const user = userEvent.setup();
    
    render(<PositionsTable {...defaultProps} />);
    
    const appleRow = screen.getByTestId('position-AAPL');
    await user.click(appleRow);
    
    expect(defaultProps.onPositionClick).toHaveBeenCalledWith('AAPL');
  });

  it('shows quick actions menu', async () => {
    const user = userEvent.setup();
    
    render(<PositionsTable {...defaultProps} />);
    
    const actionsButton = within(screen.getByTestId('position-AAPL'))
      .getByLabelText('Position actions');
    await user.click(actionsButton);
    
    expect(screen.getByText('Trade')).toBeInTheDocument();
    expect(screen.getByText('Add to Position')).toBeInTheDocument();
    expect(screen.getByText('Close Position')).toBeInTheDocument();
    expect(screen.getByText('Set Alert')).toBeInTheDocument();
  });

  it('exports positions to CSV', async () => {
    const user = userEvent.setup();
    const mockDownload = jest.fn();
    
    global.URL.createObjectURL = jest.fn(() => 'blob:url');
    global.document.createElement = jest.fn((tag) => {
      if (tag === 'a') {
        return { click: mockDownload, setAttribute: jest.fn() };
      }
      return document.createElement(tag);
    });
    
    render(<PositionsTable {...defaultProps} />);
    
    const exportButton = screen.getByLabelText('Export positions');
    await user.click(exportButton);
    
    expect(mockDownload).toHaveBeenCalled();
  });
});

describe('PerformanceMetrics Component', () => {
  const defaultProps = {
    performance: mockPerformance,
    period: '1M' as const,
  };

  it('displays key performance metrics', () => {
    render(<PerformanceMetrics {...defaultProps} />);
    
    expect(screen.getByText('Return')).toBeInTheDocument();
    expect(screen.getByText('+15.5%')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getByText('1.85')).toBeInTheDocument();
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
    expect(screen.getByText('-5.2%')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('68%')).toBeInTheDocument();
  });

  it('shows performance chart', () => {
    render(<PerformanceMetrics {...defaultProps} />);
    
    expect(screen.getByTestId('performance-chart')).toBeInTheDocument();
  });

  it('compares to benchmark', () => {
    render(<PerformanceMetrics {...defaultProps} showBenchmark />);
    
    expect(screen.getByText('vs S&P 500')).toBeInTheDocument();
    expect(screen.getByText('+5.2%')).toBeInTheDocument(); // Outperformance
  });

  it('displays detailed statistics on expansion', async () => {
    const user = userEvent.setup();
    
    render(<PerformanceMetrics {...defaultProps} />);
    
    const expandButton = screen.getByLabelText('Show detailed statistics');
    await user.click(expandButton);
    
    expect(screen.getByText('Sortino Ratio')).toBeInTheDocument();
    expect(screen.getByText('Calmar Ratio')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Information Ratio')).toBeInTheDocument();
  });

  it('shows monthly returns heatmap', async () => {
    const user = userEvent.setup();
    
    render(<PerformanceMetrics {...defaultProps} />);
    
    const viewToggle = screen.getByLabelText('Toggle view');
    await user.click(viewToggle);
    
    const heatmapOption = screen.getByText('Monthly Returns');
    await user.click(heatmapOption);
    
    expect(screen.getByTestId('monthly-returns-heatmap')).toBeInTheDocument();
  });
});

describe('AllocationChart Component', () => {
  const defaultProps = {
    positions: mockPositions,
    groupBy: 'asset' as const,
  };

  it('renders pie chart with allocations', () => {
    render(<AllocationChart {...defaultProps} />);
    
    expect(screen.getByTestId('allocation-pie-chart')).toBeInTheDocument();
  });

  it('allows grouping by different criteria', async () => {
    const user = userEvent.setup();
    
    render(<AllocationChart {...defaultProps} />);
    
    const groupSelector = screen.getByLabelText('Group by');
    await user.selectOptions(groupSelector, 'sector');
    
    expect(screen.getByText('Technology: 60%')).toBeInTheDocument();
    expect(screen.getByText('Automotive: 20%')).toBeInTheDocument();
    expect(screen.getByText('Finance: 20%')).toBeInTheDocument();
  });

  it('shows allocation details on hover', async () => {
    const user = userEvent.setup();
    
    render(<AllocationChart {...defaultProps} />);
    
    const techSlice = screen.getByTestId('allocation-slice-technology');
    await user.hover(techSlice);
    
    await waitFor(() => {
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
      expect(screen.getByText('Technology')).toBeInTheDocument();
      expect(screen.getByText('Value: $90,000')).toBeInTheDocument();
      expect(screen.getByText('Allocation: 60%')).toBeInTheDocument();
    });
  });

  it('displays allocation table view', async () => {
    const user = userEvent.setup();
    
    render(<AllocationChart {...defaultProps} />);
    
    const tableViewButton = screen.getByLabelText('Table view');
    await user.click(tableViewButton);
    
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByText('Category')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();
    expect(screen.getByText('Percentage')).toBeInTheDocument();
  });

  it('shows rebalancing suggestions', async () => {
    const user = userEvent.setup();
    
    render(<AllocationChart {...defaultProps} showRebalancing />);
    
    const rebalanceButton = screen.getByText('Suggest Rebalancing');
    await user.click(rebalanceButton);
    
    await waitFor(() => {
      expect(screen.getByText('Rebalancing Suggestions')).toBeInTheDocument();
      expect(screen.getByText('Reduce AAPL by 10 shares')).toBeInTheDocument();
      expect(screen.getByText('Increase JPM by 15 shares')).toBeInTheDocument();
    });
  });
});

describe('RiskAnalysis Component', () => {
  const defaultProps = {
    portfolioId: 'portfolio-123',
  };

  beforeEach(() => {
    (api.getRiskMetrics as jest.Mock).mockResolvedValue({
      var_95: 5000,
      cvar_95: 6500,
      sharpe_ratio: 1.85,
      sortino_ratio: 2.1,
      max_drawdown: 0.052,
      beta: 1.15,
      correlation_matrix: {},
      risk_score: 6.5,
    });
  });

  it('displays risk metrics', async () => {
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Value at Risk (95%)')).toBeInTheDocument();
      expect(screen.getByText('$5,000')).toBeInTheDocument();
      expect(screen.getByText('Conditional VaR')).toBeInTheDocument();
      expect(screen.getByText('$6,500')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Beta')).toBeInTheDocument();
      expect(screen.getByText('1.15')).toBeInTheDocument();
    });
  });

  it('shows risk score gauge', async () => {
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByTestId('risk-score-gauge')).toBeInTheDocument();
      expect(screen.getByText('Risk Score: 6.5/10')).toBeInTheDocument();
      expect(screen.getByText('Moderate Risk')).toBeInTheDocument();
    });
  });

  it('runs stress tests', async () => {
    const user = userEvent.setup();
    
    (api.runStressTest as jest.Mock).mockResolvedValue({
      scenarios: [
        { name: 'Market Crash', impact: -0.25, probability: 0.1 },
        { name: 'Recession', impact: -0.15, probability: 0.2 },
        { name: 'Normal Correction', impact: -0.08, probability: 0.3 },
      ],
    });
    
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      const stressTestButton = screen.getByText('Run Stress Tests');
      return user.click(stressTestButton);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Stress Test Results')).toBeInTheDocument();
      expect(screen.getByText('Market Crash: -25%')).toBeInTheDocument();
      expect(screen.getByText('Recession: -15%')).toBeInTheDocument();
    });
  });

  it('displays correlation matrix', async () => {
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Asset Correlations')).toBeInTheDocument();
      expect(screen.getByTestId('correlation-heatmap')).toBeInTheDocument();
    });
  });

  it('shows risk decomposition', async () => {
    const user = userEvent.setup();
    
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      const decompButton = screen.getByText('Risk Decomposition');
      return user.click(decompButton);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Risk Contributors')).toBeInTheDocument();
      expect(screen.getByText('TSLA: 35%')).toBeInTheDocument();
      expect(screen.getByText('AAPL: 25%')).toBeInTheDocument();
    });
  });

  it('provides risk recommendations', async () => {
    render(<RiskAnalysis {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Risk Recommendations')).toBeInTheDocument();
      expect(screen.getByText('Consider diversifying into defensive sectors')).toBeInTheDocument();
      expect(screen.getByText('High concentration in technology sector')).toBeInTheDocument();
    });
  });
});
