export type BacktestStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED'
export type StrategyType = 'MA_CROSSOVER' | 'RSI' | 'BOLLINGER_BANDS' | 'MACD' | 'CUSTOM'

export interface BacktestConfig {
  id: string
  name: string
  description?: string
  strategy: string
  strategyType?: StrategyType
  symbols: string[]
  startDate: string
  endDate: string
  initialCapital: number
  positionSize: number
  positionSizing?: 'FIXED' | 'PERCENT' | 'KELLY'
  maxPositions: number
  commission: number
  slippage: number
  stopLoss?: number
  takeProfit?: number
  parameters: Record<string, any>
  userId: string
  createdAt: string
  updatedAt: string
}

export interface BacktestResult {
  id: string
  configId: string
  config?: BacktestConfig
  status: BacktestStatus
  startedAt: string
  completedAt?: string
  duration?: number
  error?: string
  // Performance metrics
  totalReturn: number
  totalReturnPercent: number
  annualizedReturn: number
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  maxDrawdown: number
  maxDrawdownPercent: number
  maxDrawdownDuration: number
  // Trade statistics
  totalTrades: number
  winningTrades: number
  losingTrades: number
  winRate: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  expectancy: number
  avgTradeDuration: number
  // Risk metrics
  volatility: number
  var95: number
  cvar95: number
  beta?: number
  alpha?: number
  // Additional data
  equity: number[]
  drawdown: number[]
  trades: BacktestTrade[]
  dailyReturns: number[]
  monthlyReturns: Record<string, number>
  yearlyReturns: Record<string, number>
}

export interface BacktestTrade {
  id: string
  symbol: string
  entryDate: string
  exitDate: string
  entryPrice: number
  exitPrice: number
  quantity: number
  side: 'LONG' | 'SHORT'
  pnl: number
  pnlPercent: number
  commission: number
  slippage: number
  duration: number
  exitReason: 'SIGNAL' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'END_OF_DATA'
}

export interface BacktestProgress {
  id: string
  status: BacktestStatus
  progress: number
  currentDate?: string
  message?: string
  tradesProcessed?: number
  estimatedTimeRemaining?: number
}

export interface BacktestComparison {
  configs: BacktestConfig[]
  results: BacktestResult[]
  metrics: {
    metric: string
    values: number[]
    winner: number
  }[]
}

export interface StrategyTemplate {
  id: string
  name: string
  description: string
  type: StrategyType
  defaultParameters: Record<string, any>
  requiredData: string[]
}

class BacktestService {
  async getConfigs(): Promise<BacktestConfig[]> {
    const response = await fetch('/api/v1/backtest/configs')
    if (!response.ok) throw new Error('Failed to fetch backtest configs')
    return response.json()
  }

  async getConfig(id: string): Promise<BacktestConfig> {
    const response = await fetch(`/api/v1/backtest/configs/${id}`)
    if (!response.ok) throw new Error('Failed to fetch backtest config')
    return response.json()
  }

  async createConfig(config: Partial<BacktestConfig>): Promise<BacktestConfig> {
    const response = await fetch('/api/v1/backtest/configs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    })
    if (!response.ok) throw new Error('Failed to create backtest config')
    return response.json()
  }

  async updateConfig(id: string, updates: Partial<BacktestConfig>): Promise<BacktestConfig> {
    const response = await fetch(`/api/v1/backtest/configs/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    })
    if (!response.ok) throw new Error('Failed to update backtest config')
    return response.json()
  }

  async deleteConfig(id: string): Promise<void> {
    const response = await fetch(`/api/v1/backtest/configs/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete backtest config')
  }

  async runBacktest(configId: string): Promise<string> {
    const response = await fetch(`/api/v1/backtest/run/${configId}`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to start backtest')
    const { backtestId } = await response.json()
    return backtestId
  }

  async stopBacktest(backtestId: string): Promise<void> {
    const response = await fetch(`/api/v1/backtest/stop/${backtestId}`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to stop backtest')
  }

  async getBacktestStatus(backtestId: string): Promise<BacktestProgress> {
    const response = await fetch(`/api/v1/backtest/status/${backtestId}`)
    if (!response.ok) throw new Error('Failed to fetch backtest status')
    return response.json()
  }

  async getResults(): Promise<BacktestResult[]> {
    const response = await fetch('/api/v1/backtest/results')
    if (!response.ok) throw new Error('Failed to fetch backtest results')
    return response.json()
  }

  async getResult(id: string): Promise<BacktestResult> {
    const response = await fetch(`/api/v1/backtest/results/${id}`)
    if (!response.ok) throw new Error('Failed to fetch backtest result')
    return response.json()
  }

  async deleteResult(id: string): Promise<void> {
    const response = await fetch(`/api/v1/backtest/results/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete backtest result')
  }

  async compareResults(resultIds: string[]): Promise<BacktestComparison> {
    const response = await fetch('/api/v1/backtest/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resultIds })
    })
    if (!response.ok) throw new Error('Failed to compare backtest results')
    return response.json()
  }

  async exportResults(resultIds: string[]): Promise<void> {
    const response = await fetch('/api/v1/backtest/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resultIds })
    })
    if (!response.ok) throw new Error('Failed to export backtest results')
    
    // Handle file download
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `backtest-results-${new Date().toISOString()}.csv`
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  async getStrategyTemplates(): Promise<StrategyTemplate[]> {
    const response = await fetch('/api/v1/backtest/strategies')
    if (!response.ok) throw new Error('Failed to fetch strategy templates')
    return response.json()
  }

  async validateStrategy(code: string): Promise<{
    valid: boolean
    errors?: string[]
    warnings?: string[]
  }> {
    const response = await fetch('/api/v1/backtest/validate-strategy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code })
    })
    if (!response.ok) throw new Error('Failed to validate strategy')
    return response.json()
  }

  async optimizeParameters(
    configId: string,
    parameters: Record<string, { min: number; max: number; step: number }>
  ): Promise<{
    optimizationId: string
  }> {
    const response = await fetch(`/api/v1/backtest/optimize/${configId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters })
    })
    if (!response.ok) throw new Error('Failed to start parameter optimization')
    return response.json()
  }

  async getOptimizationResults(optimizationId: string): Promise<{
    status: BacktestStatus
    progress: number
    results?: Array<{
      parameters: Record<string, number>
      sharpeRatio: number
      totalReturn: number
      maxDrawdown: number
    }>
  }> {
    const response = await fetch(`/api/v1/backtest/optimize/${optimizationId}/results`)
    if (!response.ok) throw new Error('Failed to fetch optimization results')
    return response.json()
  }

  // WebSocket connection for real-time backtest progress
  connectBacktestProgress(
    backtestId: string,
    onProgress: (progress: BacktestProgress) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(
      `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/backtest/${backtestId}`
    )
    
    ws.onmessage = (event) => {
      try {
        const progress = JSON.parse(event.data)
        onProgress(progress)
      } catch (error) {
        onError?.(new Error('Failed to parse backtest progress'))
      }
    }
    
    ws.onerror = () => {
      onError?.(new Error('WebSocket connection error'))
    }
    
    return () => ws.close()
  }
}

export const backtestService = new BacktestService()
