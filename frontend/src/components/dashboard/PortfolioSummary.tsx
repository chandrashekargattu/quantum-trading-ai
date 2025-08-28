'use client'

import { useQuery } from '@tanstack/react-query'
import { tradingService } from '@/services/api/trading'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ArrowUpIcon, ArrowDownIcon, TrendingUp, DollarSign, PieChart } from 'lucide-react'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import Link from 'next/link'

export function PortfolioSummary() {
  const { data: portfolios, isLoading } = useQuery({
    queryKey: ['portfolios'],
    queryFn: () => tradingService.getPortfolios(),
  })

  const defaultPortfolio = portfolios?.find(p => p.is_default) || portfolios?.[0]

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

  if (!defaultPortfolio) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">
              No portfolio found. Create one to start trading.
            </p>
            <Link href="/portfolio/create">
              <Button>Create Portfolio</Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    )
  }

  const isPositive = defaultPortfolio.daily_return_percent >= 0

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Portfolio Summary</CardTitle>
        <Link href="/portfolio">
          <Button variant="outline" size="sm">View Details</Button>
        </Link>
      </CardHeader>
      <CardContent>
        {/* Total Value */}
        <div className="space-y-2 mb-6">
          <p className="text-sm text-muted-foreground">Total Portfolio Value</p>
          <div className="flex items-baseline space-x-3">
            <h2 className="text-3xl font-bold">
              {formatCurrency(defaultPortfolio.total_value)}
            </h2>
            <div className={`flex items-center space-x-1 ${isPositive ? 'text-bullish' : 'text-bearish'}`}>
              {isPositive ? (
                <ArrowUpIcon className="h-4 w-4" />
              ) : (
                <ArrowDownIcon className="h-4 w-4" />
              )}
              <span className="text-sm font-medium">
                {formatCurrency(Math.abs(defaultPortfolio.daily_return))}
                {' '}
                ({formatPercentage(defaultPortfolio.daily_return_percent / 100)})
              </span>
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-1">
            <div className="flex items-center space-x-2 text-muted-foreground">
              <DollarSign className="h-4 w-4" />
              <p className="text-sm">Cash Balance</p>
            </div>
            <p className="text-xl font-semibold">
              {formatCurrency(defaultPortfolio.cash_balance)}
            </p>
          </div>
          
          <div className="space-y-1">
            <div className="flex items-center space-x-2 text-muted-foreground">
              <TrendingUp className="h-4 w-4" />
              <p className="text-sm">Total Return</p>
            </div>
            <p className={`text-xl font-semibold ${defaultPortfolio.total_return >= 0 ? 'text-bullish' : 'text-bearish'}`}>
              {formatPercentage(defaultPortfolio.total_return_percent / 100)}
            </p>
          </div>
          
          <div className="space-y-1">
            <div className="flex items-center space-x-2 text-muted-foreground">
              <PieChart className="h-4 w-4" />
              <p className="text-sm">Buying Power</p>
            </div>
            <p className="text-xl font-semibold">
              {formatCurrency(defaultPortfolio.buying_power)}
            </p>
          </div>
        </div>

        {/* Portfolio Performance Chart would go here */}
        <div className="mt-6 h-48 bg-muted/20 rounded-lg flex items-center justify-center">
          <p className="text-muted-foreground text-sm">Performance Chart</p>
        </div>
      </CardContent>
    </Card>
  )
}
