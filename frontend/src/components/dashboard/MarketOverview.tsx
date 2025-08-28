'use client'

import { useQuery } from '@tanstack/react-query'
import { marketService } from '@/services/api/market'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowUpIcon, ArrowDownIcon } from 'lucide-react'
import { formatCurrency, formatPercentage } from '@/lib/utils'

interface MarketIndex {
  symbol: string
  name: string
  value: number
  change_amount: number
  change_percent: number
}

export function MarketOverview() {
  const { data: indicators, isLoading } = useQuery({
    queryKey: ['market-indicators'],
    queryFn: () => marketService.getMarketIndicators(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="animate-pulse">
                <div className="h-4 bg-muted rounded w-1/2 mb-2" />
                <div className="h-8 bg-muted rounded w-3/4" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  const majorIndices = indicators?.slice(0, 4) || []

  return (
    <div className="grid gap-4 md:grid-cols-4">
      {majorIndices.map((index: MarketIndex) => (
        <Card key={index.symbol}>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium text-muted-foreground">
                  {index.name}
                </p>
                <p className="text-2xl font-bold">
                  {formatCurrency(index.value, 'USD')}
                </p>
              </div>
              <div
                className={`flex items-center space-x-1 ${
                  index.change_percent >= 0 ? 'text-bullish' : 'text-bearish'
                }`}
              >
                {index.change_percent >= 0 ? (
                  <ArrowUpIcon className="h-4 w-4" />
                ) : (
                  <ArrowDownIcon className="h-4 w-4" />
                )}
                <span className="text-sm font-medium">
                  {formatPercentage(index.change_percent / 100)}
                </span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {index.change_amount >= 0 ? '+' : ''}
              {formatCurrency(index.change_amount, 'USD')}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
