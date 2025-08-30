'use client'

import { useState, useEffect, memo, useCallback } from 'react'
import { portfolioService } from '@/services/api/portfolio-optimized'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ArrowUpIcon, ArrowDownIcon, TrendingUp, DollarSign, PieChart } from 'lucide-react'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'react-hot-toast'

// Memoize metric display to prevent re-renders
const MetricDisplay = memo(({ 
  icon: Icon, 
  label, 
  value, 
  isPercentage = false,
  isPositive = true 
}: {
  icon: any
  label: string
  value: number
  isPercentage?: boolean
  isPositive?: boolean
}) => (
  <div className="space-y-1">
    <div className="flex items-center space-x-2 text-muted-foreground">
      <Icon className="h-4 w-4" />
      <p className="text-sm">{label}</p>
    </div>
    <p className={`text-xl font-semibold ${isPercentage && !isPositive ? 'text-bearish' : isPercentage && isPositive ? 'text-bullish' : ''}`}>
      {isPercentage ? formatPercentage(value / 100) : formatCurrency(value)}
    </p>
  </div>
))

MetricDisplay.displayName = 'MetricDisplay'

export function PortfolioSummary() {
  const [portfolioData, setPortfolioData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [portfolioName, setPortfolioName] = useState('')
  const [initialCapital, setInitialCapital] = useState('100000')
  const [isCreating, setIsCreating] = useState(false)

  // Memoize the load function to prevent recreating on each render
  const loadPortfolios = useCallback(async () => {
    try {
      setIsLoading(true)
      const summary = await portfolioService.getPortfoliosSummary()
      
      console.log('Loaded portfolios:', summary) // Debug log
      
      if (summary.portfolios.length > 0) {
        const portfolio = summary.portfolios[0]
        setPortfolioData({
          portfolio,
          totalValue: portfolio.currentValue,
          dayChange: portfolio.dayChange,
          dayChangePercent: portfolio.dayChangePercent,
          cashBalance: portfolio.cashBalance,
          totalReturn: portfolio.totalReturn,
          totalReturnPercent: portfolio.totalReturnPercent,
          buyingPower: portfolio.cashBalance
        })
      } else {
        setPortfolioData(null)
      }
    } catch (error) {
      console.error('Failed to load portfolios:', error) // Debug log
      toast.error('Failed to load portfolio data')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    // Prefetch on mount for faster initial load
    portfolioService.prefetchPortfolios()
    loadPortfolios()
  }, [loadPortfolios])

  const handleCreatePortfolio = useCallback(async () => {
    if (!portfolioName.trim()) {
      toast.error('Please enter a portfolio name')
      return
    }
    const capital = parseFloat(initialCapital)
    if (isNaN(capital) || capital <= 0) {
      toast.error('Please enter a valid initial capital amount')
      return
    }

    setIsCreating(true)
    try {
      await portfolioService.createPortfolio({
        name: portfolioName.trim(),
        initialCapital: capital
      })
      toast.success('Portfolio created successfully!')
      setIsCreateOpen(false)
      setPortfolioName('')
      setInitialCapital('100000')
      // Reload portfolios
      await loadPortfolios()
    } catch (error) {
      toast.error('Failed to create portfolio')
    } finally {
      setIsCreating(false)
    }
  }, [portfolioName, initialCapital, loadPortfolios])

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-20 bg-muted rounded" />
            <div className="grid grid-cols-3 gap-4">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-16 bg-muted rounded" />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!portfolioData) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Portfolio Summary</CardTitle>
          <Button 
            onClick={() => {
              setPortfolioData(null) // Clear current data to show loading
              loadPortfolios()
            }} 
            variant="ghost" 
            size="sm"
            className="h-8 w-8 p-0"
            title="Refresh portfolios"
          >
            🔄
          </Button>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">
              No portfolio found. Create one to start trading.
            </p>
            {!isCreateOpen ? (
              <Button onClick={() => setIsCreateOpen(true)}>Create Portfolio</Button>
            ) : (
              <div className="space-y-4 mt-4 max-w-md mx-auto">
                <div>
                  <Label htmlFor="portfolio-name">Portfolio Name</Label>
                  <Input
                    id="portfolio-name"
                    value={portfolioName}
                    onChange={(e) => setPortfolioName(e.target.value)}
                    placeholder="My Trading Portfolio"
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="initial-capital">Initial Capital (₹)</Label>
                  <Input
                    id="initial-capital"
                    type="number"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(e.target.value)}
                    placeholder="100000"
                    className="mt-1"
                  />
                </div>
                <div className="flex gap-2">
                  <Button 
                    onClick={handleCreatePortfolio}
                    disabled={isCreating}
                    className="flex-1"
                  >
                    {isCreating ? 'Creating...' : 'Create Portfolio'}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setIsCreateOpen(false)
                      setPortfolioName('')
                      setInitialCapital('100000')
                    }}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  const isPositive = portfolioData.dayChangePercent >= 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Portfolio Summary</CardTitle>
        <Button 
          onClick={loadPortfolios} 
          variant="ghost" 
          size="sm"
          className="h-8 w-8 p-0"
        >
          🔄
        </Button>
      </CardHeader>
      <CardContent>
        {/* Total Value - Main metric */}
        <div className="space-y-2 mb-6">
          <p className="text-sm text-muted-foreground">Total Portfolio Value</p>
          <div className="flex items-baseline space-x-3">
            <h2 className="text-3xl font-bold">
              {formatCurrency(portfolioData.totalValue)}
            </h2>
            <div className={`flex items-center space-x-1 ${isPositive ? 'text-bullish' : 'text-bearish'}`}>
              {isPositive ? (
                <ArrowUpIcon className="h-4 w-4" />
              ) : (
                <ArrowDownIcon className="h-4 w-4" />
              )}
              <span className="text-sm font-medium">
                {formatCurrency(Math.abs(portfolioData.dayChange))}
                {' '}
                ({formatPercentage(portfolioData.dayChangePercent / 100)})
              </span>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-4">
          <MetricDisplay
            icon={DollarSign}
            label="Cash Balance"
            value={portfolioData.cashBalance}
          />
          
          <MetricDisplay
            icon={TrendingUp}
            label="Total Return"
            value={portfolioData.totalReturnPercent}
            isPercentage
            isPositive={portfolioData.totalReturn >= 0}
          />
          
          <MetricDisplay
            icon={PieChart}
            label="Buying Power"
            value={portfolioData.buyingPower}
          />
        </div>
      </CardContent>
    </Card>
  )
}