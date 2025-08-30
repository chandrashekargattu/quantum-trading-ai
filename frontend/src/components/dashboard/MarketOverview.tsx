'use client'

import { useEffect, useState, memo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowUpIcon, ArrowDownIcon, Clock } from 'lucide-react'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import { marketService } from '@/services/api/market-optimized'
import { getMarketStatus } from '@/config/market-config'

interface MarketIndex {
  symbol: string
  name: string
  value: number
  change_amount: number
  change_percent: number
}

// Memoize individual indicator card to prevent re-renders
const IndicatorCard = memo(({ index }: { index: MarketIndex }) => {
  const isPositive = index.change_percent >= 0
  
  // Commodity indices show in USD
  const isCommodity = index.name.includes('Gold') || index.name.includes('Silver') || index.name.includes('Crude')
  const displayValue = isCommodity ? 
    `$${index.value.toFixed(2)}` : 
    index.value.toLocaleString('en-IN', { maximumFractionDigits: 2 })

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">
              {index.name}
            </p>
            <p className="text-2xl font-bold">
              {displayValue}
            </p>
          </div>
          <div className={`flex items-center space-x-1 ${isPositive ? 'text-bullish' : 'text-bearish'}`}>
            {isPositive ? (
              <ArrowUpIcon className="h-4 w-4" />
            ) : (
              <ArrowDownIcon className="h-4 w-4" />
            )}
            <div className="text-right">
              <p className="text-sm font-medium">
                {isPositive ? '+' : ''}{index.change_amount.toFixed(2)}
              </p>
              <p className="text-xs">
                {formatPercentage(index.change_percent / 100)}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
})

IndicatorCard.displayName = 'IndicatorCard'

export function MarketOverview() {
  const [indicators, setIndicators] = useState<MarketIndex[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [marketStatus, setMarketStatus] = useState(getMarketStatus())

  useEffect(() => {
    let cleanup: (() => void) | undefined

    const loadData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // Initial load
        const data = await marketService.getMarketIndicators()
        setIndicators(data.slice(0, 8)) // Show 8 indices for Indian markets
        setIsLoading(false)

        // Start real-time updates
        cleanup = marketService.startMarketUpdates((update) => {
          if (update.type === 'indicators') {
            setIndicators(update.data.slice(0, 8))
          }
        })
      } catch (err: any) {
        console.error('Market data error:', err)
        setError(err.message || 'Failed to load market data')
        setIsLoading(false)
      }
    }

    loadData()

    // Update market status every minute
    const statusInterval = setInterval(() => {
      setMarketStatus(getMarketStatus())
    }, 60000)

    return () => {
      cleanup?.()
      clearInterval(statusInterval)
    }
  }, [])

  if (error) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          {error}
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid gap-4 md:grid-cols-4">
          {[...Array(8)].map((_, i) => (
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
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Market Status Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Indian Market Indices</h2>
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className={`text-sm font-medium ${
            marketStatus.status === 'open' ? 'text-green-600' : 
            marketStatus.status === 'pre-open' || marketStatus.status === 'post-close' ? 'text-yellow-600' : 
            'text-red-600'
          }`}>
            {marketStatus.message}
          </span>
        </div>
      </div>

      {/* Indices Grid */}
      <div className="grid gap-4 md:grid-cols-4">
        {indicators.map((index) => (
          <IndicatorCard key={index.symbol} index={index} />
        ))}
      </div>
    </div>
  )
}