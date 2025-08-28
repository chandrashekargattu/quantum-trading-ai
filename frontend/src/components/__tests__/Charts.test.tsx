import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PriceChart } from '../charts/PriceChart';
import { VolumeChart } from '../charts/VolumeChart';
import { CandlestickChart } from '../charts/CandlestickChart';
import { TechnicalIndicatorsChart } from '../charts/TechnicalIndicatorsChart';
import { HeatmapChart } from '../charts/HeatmapChart';
import { mockPriceData, mockVolumeData, mockCandlestickData } from '@/test/mocks';

// Mock react-chartjs-2
jest.mock('react-chartjs-2', () => ({
  Line: ({ data, options, ...props }: any) => (
    <div data-testid="line-chart" {...props}>
      {JSON.stringify({ data, options })}
    </div>
  ),
  Bar: ({ data, options, ...props }: any) => (
    <div data-testid="bar-chart" {...props}>
      {JSON.stringify({ data, options })}
    </div>
  ),
  Scatter: ({ data, options, ...props }: any) => (
    <div data-testid="scatter-chart" {...props}>
      {JSON.stringify({ data, options })}
    </div>
  ),
}));

// Mock lightweight-charts
jest.mock('lightweight-charts', () => ({
  createChart: jest.fn(() => ({
    addCandlestickSeries: jest.fn(() => ({
      setData: jest.fn(),
      update: jest.fn(),
    })),
    addLineSeries: jest.fn(() => ({
      setData: jest.fn(),
      createPriceLine: jest.fn(),
    })),
    addHistogramSeries: jest.fn(() => ({
      setData: jest.fn(),
    })),
    timeScale: jest.fn(() => ({
      fitContent: jest.fn(),
      setVisibleRange: jest.fn(),
    })),
    priceScale: jest.fn(() => ({
      applyOptions: jest.fn(),
    })),
    applyOptions: jest.fn(),
    remove: jest.fn(),
  })),
}));

describe('PriceChart Component', () => {
  const defaultProps = {
    data: mockPriceData,
    symbol: 'AAPL',
    interval: '1d' as const,
    height: 400,
  };

  it('renders price chart with data', () => {
    render(<PriceChart {...defaultProps} />);
    
    const chart = screen.getByTestId('line-chart');
    expect(chart).toBeInTheDocument();
    
    const chartData = JSON.parse(chart.textContent || '{}');
    expect(chartData.data.datasets[0].label).toBe('AAPL');
    expect(chartData.data.datasets[0].data).toHaveLength(mockPriceData.length);
  });

  it('shows loading state', () => {
    render(<PriceChart {...defaultProps} isLoading />);
    
    expect(screen.getByTestId('chart-loading')).toBeInTheDocument();
    expect(screen.getByText('Loading chart data...')).toBeInTheDocument();
  });

  it('shows error state', () => {
    render(<PriceChart {...defaultProps} error="Failed to load data" />);
    
    expect(screen.getByText('Failed to load data')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('allows interval selection', async () => {
    const user = userEvent.setup();
    const onIntervalChange = jest.fn();
    
    render(<PriceChart {...defaultProps} onIntervalChange={onIntervalChange} />);
    
    const intervalSelector = screen.getByLabelText('Chart interval');
    await user.selectOptions(intervalSelector, '1h');
    
    expect(onIntervalChange).toHaveBeenCalledWith('1h');
  });

  it('toggles between line and area chart', async () => {
    const user = userEvent.setup();
    
    render(<PriceChart {...defaultProps} />);
    
    const toggleButton = screen.getByLabelText('Toggle chart type');
    await user.click(toggleButton);
    
    const chart = screen.getByTestId('line-chart');
    const chartData = JSON.parse(chart.textContent || '{}');
    expect(chartData.data.datasets[0].fill).toBe(true);
  });

  it('displays moving averages', async () => {
    const user = userEvent.setup();
    
    render(<PriceChart {...defaultProps} />);
    
    const maButton = screen.getByLabelText('Toggle moving averages');
    await user.click(maButton);
    
    const chart = screen.getByTestId('line-chart');
    const chartData = JSON.parse(chart.textContent || '{}');
    
    expect(chartData.data.datasets).toHaveLength(3); // Price + MA20 + MA50
    expect(chartData.data.datasets[1].label).toBe('MA20');
    expect(chartData.data.datasets[2].label).toBe('MA50');
  });

  it('shows price annotations', async () => {
    const user = userEvent.setup();
    
    render(<PriceChart {...defaultProps} showAnnotations />);
    
    expect(screen.getByText('High: $155.00')).toBeInTheDocument();
    expect(screen.getByText('Low: $145.00')).toBeInTheDocument();
    expect(screen.getByText('Current: $150.00')).toBeInTheDocument();
  });

  it('handles zoom and pan', async () => {
    const user = userEvent.setup();
    
    render(<PriceChart {...defaultProps} enableZoom />);
    
    const chart = screen.getByTestId('line-chart');
    const chartOptions = JSON.parse(chart.textContent || '{}').options;
    
    expect(chartOptions.plugins.zoom).toBeDefined();
    expect(chartOptions.plugins.zoom.zoom.wheel.enabled).toBe(true);
    expect(chartOptions.plugins.zoom.pan.enabled).toBe(true);
  });
});

describe('CandlestickChart Component', () => {
  const defaultProps = {
    data: mockCandlestickData,
    symbol: 'AAPL',
    height: 400,
  };

  it('renders candlestick chart', () => {
    render(<CandlestickChart {...defaultProps} />);
    
    expect(screen.getByTestId('candlestick-chart')).toBeInTheDocument();
  });

  it('displays volume bars', () => {
    render(<CandlestickChart {...defaultProps} showVolume />);
    
    expect(screen.getByTestId('volume-bars')).toBeInTheDocument();
  });

  it('shows technical indicators overlay', async () => {
    const user = userEvent.setup();
    
    render(<CandlestickChart {...defaultProps} />);
    
    const indicatorButton = screen.getByLabelText('Add indicator');
    await user.click(indicatorButton);
    
    const rsiOption = screen.getByText('RSI');
    await user.click(rsiOption);
    
    expect(screen.getByTestId('rsi-indicator')).toBeInTheDocument();
  });

  it('allows drawing tools', async () => {
    const user = userEvent.setup();
    
    render(<CandlestickChart {...defaultProps} enableDrawing />);
    
    const drawButton = screen.getByLabelText('Drawing tools');
    await user.click(drawButton);
    
    expect(screen.getByText('Trend Line')).toBeInTheDocument();
    expect(screen.getByText('Horizontal Line')).toBeInTheDocument();
    expect(screen.getByText('Fibonacci Retracement')).toBeInTheDocument();
  });

  it('exports chart as image', async () => {
    const user = userEvent.setup();
    const mockDownload = jest.fn();
    
    // Mock canvas toBlob
    HTMLCanvasElement.prototype.toBlob = jest.fn((callback) => {
      callback(new Blob(['image'], { type: 'image/png' }));
    });
    
    // Mock download
    global.URL.createObjectURL = jest.fn(() => 'blob:url');
    document.createElement = jest.fn((tag) => {
      if (tag === 'a') {
        return { click: mockDownload, setAttribute: jest.fn() };
      }
      return document.createElement(tag);
    });
    
    render(<CandlestickChart {...defaultProps} />);
    
    const exportButton = screen.getByLabelText('Export chart');
    await user.click(exportButton);
    
    expect(mockDownload).toHaveBeenCalled();
  });
});

describe('TechnicalIndicatorsChart Component', () => {
  const defaultProps = {
    symbol: 'AAPL',
    data: mockPriceData,
    indicators: ['RSI', 'MACD', 'BB'],
  };

  it('renders multiple indicators', () => {
    render(<TechnicalIndicatorsChart {...defaultProps} />);
    
    expect(screen.getByText('RSI (14)')).toBeInTheDocument();
    expect(screen.getByText('MACD (12, 26, 9)')).toBeInTheDocument();
    expect(screen.getByText('Bollinger Bands (20, 2)')).toBeInTheDocument();
  });

  it('allows customizing indicator parameters', async () => {
    const user = userEvent.setup();
    
    render(<TechnicalIndicatorsChart {...defaultProps} />);
    
    const rsiSettings = screen.getByLabelText('RSI settings');
    await user.click(rsiSettings);
    
    const periodInput = screen.getByLabelText('Period');
    await user.clear(periodInput);
    await user.type(periodInput, '21');
    
    const applyButton = screen.getByText('Apply');
    await user.click(applyButton);
    
    expect(screen.getByText('RSI (21)')).toBeInTheDocument();
  });

  it('displays indicator values on hover', async () => {
    const user = userEvent.setup();
    
    render(<TechnicalIndicatorsChart {...defaultProps} />);
    
    const chartArea = screen.getByTestId('indicator-chart-area');
    fireEvent.mouseMove(chartArea, { clientX: 100, clientY: 100 });
    
    await waitFor(() => {
      expect(screen.getByText('RSI: 65.4')).toBeInTheDocument();
      expect(screen.getByText('MACD: 0.45')).toBeInTheDocument();
    });
  });

  it('shows overbought/oversold zones for RSI', () => {
    render(<TechnicalIndicatorsChart {...defaultProps} indicators={['RSI']} />);
    
    expect(screen.getByTestId('rsi-overbought-zone')).toBeInTheDocument();
    expect(screen.getByTestId('rsi-oversold-zone')).toBeInTheDocument();
  });

  it('allows adding/removing indicators', async () => {
    const user = userEvent.setup();
    
    render(<TechnicalIndicatorsChart {...defaultProps} />);
    
    // Add indicator
    const addButton = screen.getByLabelText('Add indicator');
    await user.click(addButton);
    
    const stochButton = screen.getByText('Stochastic');
    await user.click(stochButton);
    
    expect(screen.getByText('Stochastic (14, 3, 3)')).toBeInTheDocument();
    
    // Remove indicator
    const removeButton = screen.getByLabelText('Remove RSI');
    await user.click(removeButton);
    
    expect(screen.queryByText('RSI (14)')).not.toBeInTheDocument();
  });
});

describe('HeatmapChart Component', () => {
  const defaultProps = {
    data: [
      { symbol: 'AAPL', sector: 'Technology', change: 2.5, volume: 1000000 },
      { symbol: 'GOOGL', sector: 'Technology', change: -1.2, volume: 800000 },
      { symbol: 'JPM', sector: 'Finance', change: 0.8, volume: 1200000 },
      { symbol: 'BAC', sector: 'Finance', change: -0.5, volume: 900000 },
    ],
  };

  it('renders heatmap with sectors', () => {
    render(<HeatmapChart {...defaultProps} />);
    
    expect(screen.getByText('Technology')).toBeInTheDocument();
    expect(screen.getByText('Finance')).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('JPM')).toBeInTheDocument();
  });

  it('colors cells based on performance', () => {
    render(<HeatmapChart {...defaultProps} />);
    
    const appleCell = screen.getByTestId('heatmap-cell-AAPL');
    const googleCell = screen.getByTestId('heatmap-cell-GOOGL');
    
    expect(appleCell).toHaveClass('bg-green-500');
    expect(googleCell).toHaveClass('bg-red-500');
  });

  it('shows tooltips on hover', async () => {
    const user = userEvent.setup();
    
    render(<HeatmapChart {...defaultProps} />);
    
    const appleCell = screen.getByTestId('heatmap-cell-AAPL');
    await user.hover(appleCell);
    
    await waitFor(() => {
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
      expect(screen.getByText('AAPL: +2.5%')).toBeInTheDocument();
      expect(screen.getByText('Volume: 1M')).toBeInTheDocument();
    });
  });

  it('allows clicking on cells for details', async () => {
    const user = userEvent.setup();
    const onCellClick = jest.fn();
    
    render(<HeatmapChart {...defaultProps} onCellClick={onCellClick} />);
    
    const appleCell = screen.getByTestId('heatmap-cell-AAPL');
    await user.click(appleCell);
    
    expect(onCellClick).toHaveBeenCalledWith({
      symbol: 'AAPL',
      sector: 'Technology',
      change: 2.5,
      volume: 1000000,
    });
  });

  it('allows switching between metrics', async () => {
    const user = userEvent.setup();
    
    render(<HeatmapChart {...defaultProps} />);
    
    const metricSelector = screen.getByLabelText('Display metric');
    await user.selectOptions(metricSelector, 'volume');
    
    // Cells should now be sized/colored by volume
    const appleCell = screen.getByTestId('heatmap-cell-AAPL');
    expect(appleCell).toHaveStyle({ width: '120px' }); // Larger due to higher volume
  });

  it('filters by performance range', async () => {
    const user = userEvent.setup();
    
    render(<HeatmapChart {...defaultProps} />);
    
    const filterButton = screen.getByLabelText('Filter');
    await user.click(filterButton);
    
    const minInput = screen.getByLabelText('Min change %');
    await user.type(minInput, '0');
    
    const applyButton = screen.getByText('Apply Filter');
    await user.click(applyButton);
    
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('JPM')).toBeInTheDocument();
    expect(screen.queryByText('GOOGL')).not.toBeInTheDocument();
    expect(screen.queryByText('BAC')).not.toBeInTheDocument();
  });
});

describe('VolumeChart Component', () => {
  const defaultProps = {
    data: mockVolumeData,
    symbol: 'AAPL',
  };

  it('renders volume bars', () => {
    render(<VolumeChart {...defaultProps} />);
    
    const chart = screen.getByTestId('bar-chart');
    expect(chart).toBeInTheDocument();
  });

  it('colors bars based on price change', () => {
    render(<VolumeChart {...defaultProps} />);
    
    const chart = screen.getByTestId('bar-chart');
    const chartData = JSON.parse(chart.textContent || '{}');
    
    const greenBars = chartData.data.datasets[0].backgroundColor.filter((c: string) => c === 'green');
    const redBars = chartData.data.datasets[0].backgroundColor.filter((c: string) => c === 'red');
    
    expect(greenBars.length).toBeGreaterThan(0);
    expect(redBars.length).toBeGreaterThan(0);
  });

  it('shows volume profile', async () => {
    const user = userEvent.setup();
    
    render(<VolumeChart {...defaultProps} showProfile />);
    
    expect(screen.getByTestId('volume-profile')).toBeInTheDocument();
    expect(screen.getByText('Volume by Price Level')).toBeInTheDocument();
  });

  it('displays average volume line', () => {
    render(<VolumeChart {...defaultProps} showAverage />);
    
    const chart = screen.getByTestId('bar-chart');
    const chartData = JSON.parse(chart.textContent || '{}');
    
    expect(chartData.data.datasets).toHaveLength(2);
    expect(chartData.data.datasets[1].label).toBe('Average Volume');
  });
});
