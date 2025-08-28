'use client'

import { useQuery } from '@tanstack/react-query'
import { tradingService } from '@/services/api/trading'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatCurrency, formatDate } from '@/lib/utils'
import Link from 'next/link'

export function RecentTrades() {
  const { data: trades, isLoading } = useQuery({
    queryKey: ['recent-trades'],
    queryFn: () => tradingService.getTrades({ limit: 5 }),
  })

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-12 bg-muted rounded" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!trades || trades.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-muted-foreground">No trades yet</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      filled: 'default',
      pending: 'secondary',
      cancelled: 'destructive',
      rejected: 'destructive',
    }
    
    return (
      <Badge variant={variants[status] || 'outline'} className="capitalize">
        {status}
      </Badge>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Recent Trades</CardTitle>
        <Link href="/trades">
          <Badge variant="outline" className="cursor-pointer">View All</Badge>
        </Link>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {trades.map((trade) => (
            <div key={trade.id} className="flex items-center justify-between py-2">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <p className="font-medium">{trade.symbol}</p>
                  <Badge
                    variant={trade.side === 'buy' ? 'default' : 'destructive'}
                    className="text-xs"
                  >
                    {trade.side}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {trade.quantity} @ {formatCurrency(trade.price)}
                </p>
              </div>
              <div className="text-right space-y-1">
                <p className="text-sm font-medium">
                  {formatCurrency(trade.total_amount)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatDate(trade.created_at, 'short')}
                </p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
