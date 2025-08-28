export const mockMarketData = {
  AAPL: {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 150,
    change: 2.5,
    changePercent: 1.69,
    volume: 75000000,
    marketCap: 2500000000000,
    high: 151.5,
    low: 148.2,
    open: 148.5,
    previousClose: 147.5,
    timestamp: new Date().toISOString()
  },
  GOOGL: {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    price: 2550,
    change: -25,
    changePercent: -0.97,
    volume: 25000000,
    marketCap: 1600000000000,
    high: 2580,
    low: 2540,
    open: 2575,
    previousClose: 2575,
    timestamp: new Date().toISOString()
  },
  MSFT: {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    price: 310,
    change: 5.5,
    changePercent: 1.81,
    volume: 30000000,
    marketCap: 2300000000000,
    high: 312,
    low: 305,
    open: 305.5,
    previousClose: 304.5,
    timestamp: new Date().toISOString()
  },
  TSLA: {
    symbol: 'TSLA',
    name: 'Tesla, Inc.',
    price: 750,
    change: 15,
    changePercent: 2.04,
    volume: 100000000,
    marketCap: 750000000000,
    high: 765,
    low: 735,
    open: 740,
    previousClose: 735,
    timestamp: new Date().toISOString()
  }
}

export const mockPortfolio = {
  id: 'portfolio-1',
  name: 'Main Portfolio',
  initialCapital: 100000,
  currentValue: 125000,
  totalReturn: 25000,
  totalReturnPercent: 25,
  dayChange: 1500,
  dayChangePercent: 1.21,
  cashBalance: 50000,
  investedAmount: 75000,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString()
}

export const mockPositions = [
  {
    id: 'pos-1',
    portfolioId: 'portfolio-1',
    symbol: 'AAPL',
    quantity: 100,
    avgPrice: 145,
    currentPrice: 150,
    marketValue: 15000,
    costBasis: 14500,
    unrealizedPnL: 500,
    unrealizedPnLPercent: 3.45,
    realizedPnL: 0,
    dayChange: 250,
    dayChangePercent: 1.69,
    openedAt: new Date().toISOString()
  },
  {
    id: 'pos-2',
    portfolioId: 'portfolio-1',
    symbol: 'GOOGL',
    quantity: 10,
    avgPrice: 2500,
    currentPrice: 2550,
    marketValue: 25500,
    costBasis: 25000,
    unrealizedPnL: 500,
    unrealizedPnLPercent: 2,
    realizedPnL: 100,
    dayChange: -250,
    dayChangePercent: -0.97,
    openedAt: new Date().toISOString()
  }
]

export const mockAlerts = [
  {
    id: 'alert-1',
    symbol: 'AAPL',
    type: 'PRICE',
    condition: 'ABOVE',
    value: 160,
    enabled: true,
    triggered: false,
    message: 'AAPL breakout alert',
    sendEmail: true,
    sendPush: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'alert-2',
    symbol: 'TSLA',
    type: 'VOLUME',
    condition: 'ABOVE',
    value: 50000000,
    enabled: true,
    triggered: false,
    message: 'High volume on TSLA',
    sendEmail: true,
    sendPush: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }
]

export const mockBacktestConfig = {
  id: 'config-1',
  name: 'MA Crossover Strategy',
  strategy: 'ma-crossover',
  symbols: ['AAPL', 'GOOGL', 'MSFT'],
  startDate: '2023-01-01',
  endDate: '2023-12-31',
  initialCapital: 100000,
  positionSize: 10000,
  maxPositions: 10,
  commission: 0.001,
  slippage: 0.0005,
  parameters: {
    fastPeriod: 10,
    slowPeriod: 30
  },
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString()
}

export const mockBacktestResult = {
  id: 'result-1',
  configId: 'config-1',
  status: 'COMPLETED',
  totalReturn: 15000,
  totalReturnPercent: 15,
  annualizedReturn: 15.5,
  sharpeRatio: 1.8,
  sortinoRatio: 2.1,
  maxDrawdown: -5000,
  maxDrawdownPercent: -5,
  totalTrades: 150,
  winningTrades: 98,
  losingTrades: 52,
  winRate: 65.33,
  avgWin: 350,
  avgLoss: -150,
  profitFactor: 2.2,
  expectancy: 100,
  avgTradeDuration: 5.5,
  startedAt: new Date().toISOString(),
  completedAt: new Date().toISOString()
}

export const mockOptionChain = {
  symbol: 'AAPL',
  expirations: ['2024-01-19', '2024-02-16', '2024-03-15'],
  strikes: [140, 145, 150, 155, 160],
  calls: [
    {
      symbol: 'AAPL240119C150',
      strike: 150,
      expiration: '2024-01-19',
      type: 'CALL',
      bid: 5.2,
      ask: 5.5,
      last: 5.35,
      volume: 10000,
      openInterest: 25000,
      impliedVolatility: 0.25,
      delta: 0.55,
      gamma: 0.02,
      theta: -0.05,
      vega: 0.15
    }
  ],
  puts: [
    {
      symbol: 'AAPL240119P150',
      strike: 150,
      expiration: '2024-01-19',
      type: 'PUT',
      bid: 4.8,
      ask: 5.1,
      last: 4.95,
      volume: 8000,
      openInterest: 20000,
      impliedVolatility: 0.24,
      delta: -0.45,
      gamma: 0.02,
      theta: -0.04,
      vega: 0.14
    }
  ]
}
