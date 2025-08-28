'use client'

import { useQuery } from '@tanstack/react-query'
import { marketService } from '@/services/api/market'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, Flame, Activity } from 'lucide-react'
import { formatCurrency, formatPercentage, formatNumber } from '@/lib/utils'
import Link from 'next/link'

interface TrendingOption {
  id: string
  symbol: string
  underlying_symbol: string
  strike: number
  expiration: string
  option_type: 'call' | 'put'
  volume: number
  open_interest: number
  implied_volatility: number
  last_price: number
  change_percent: number
}

// Mock data - in real app, this would come from API
const mockTrendingOptions: TrendingOption[] = [
  {
    id: '1',
    symbol: 'AAPL230120C175',
    underlying_symbol: 'AAPL',
    strike: 175,
    expiration: '2024-01-20',
    option_type: 'call',
    volume: 15420,
    open_interest: 8750,
    implied_volatility: 0.285,
    last_price: 3.45,
    change_percent: 12.5,
  },
  {
    id: '2',
    symbol: 'SPY231215P440',
    underlying_symbol: 'SPY',
    strike: 440,
    expiration: '2024-01-15',
    option_type: 'put',
    volume: 28900,
    open_interest: 45200,
    implied_volatility: 0.165,
    last_price: 2.18,
    change_percent: -5.2,
  },
  {
    id: '3',
    symbol: 'TSLA240119C250',
    underlying_symbol: 'TSLA',
    strike: 250,
    expiration: '2024-01-19',
    option_type: 'call',
    volume: 9800,
    open_interest: 12400,
    implied_volatility: 0.485,
    last_price: 8.75,
    change_percent: 18.9,
  },
]

export function TrendingOptions() {
  // In a real app, this would fetch trending options from the API
  const { data: options, isLoading } = useQuery({
    queryKey: ['trending-options'],
    queryFn: async () => {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      return mockTrendingOptions
    },
    staleTime: 60000, // Cache for 1 minute
  })

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Flame className="h-5 w-5 text-orange-500" />
            <span>Trending Options</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-16 bg-muted rounded" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const getVolumeIndicator = (volume: number, openInterest: number) => {
    const ratio = volume / openInterest
    if (ratio > 2) return { icon: <Flame className="h-3 w-3" />, color: 'text-orange-500' }
    if (ratio > 1) return { icon: <TrendingUp className="h-3 w-3" />, color: 'text-yellow-500' }
    return { icon: <Activity className="h-3 w-3" />, color: 'text-blue-500' }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center space-x-2">
          <Flame className="h-5 w-5 text-orange-500" />
          <span>Trending Options</span>
        </CardTitle>
        <Link href="/options/screener">
          <Badge variant="outline" className="cursor-pointer">Screener</Badge>
        </Link>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {options?.map((option) => {
            const volumeIndicator = getVolumeIndicator(option.volume, option.open_interest)
            const isPositive = option.change_percent >= 0
            
            return (
              <Link
                key={option.id}
                href={`/options/${option.underlying_symbol}`}
                className="block hover:bg-muted/50 -mx-2 px-2 py-2 rounded-md transition-colors"
              >
                <div className="space-y-2">
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium">{option.underlying_symbol}</span>
                      <Badge
                        variant={option.option_type === 'call' ? 'default' : 'destructive'}
                        className="text-xs"
                      >
                        {option.strike} {option.option_type}
                      </Badge>
                      <div className={`flex items-center ${volumeIndicator.color}`}>
                        {volumeIndicator.icon}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{formatCurrency(option.last_price)}</p>
                      <p className={`text-xs ${isPositive ? 'text-bullish' : 'text-bearish'}`}>
                        {isPositive ? '+' : ''}{formatPercentage(option.change_percent / 100)}
                      </p>
                    </div>
                  </div>
                  
                  {/* Metrics */}
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <div className="flex items-center space-x-3">
                      <span>Vol: {formatNumber(option.volume, 0)}</span>
                      <span>OI: {formatNumber(option.open_interest, 0)}</span>
                      <span>IV: {formatPercentage(option.implied_volatility)}</span>
                    </div>
                    <span className="text-xs">{option.expiration}</span>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
        
        <div className="mt-4 text-center">
          <Link
            href="/options/trending"
            className="text-xs text-primary hover:underline"
          >
            View all trending options →
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
