'use client'

import { useMarketStore } from '@/store/useMarketStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ArrowUpIcon, ArrowDownIcon, Plus } from 'lucide-react'
import { formatCurrency, formatPercentage } from '@/lib/utils'
import Link from 'next/link'

export function WatchlistWidget() {
  const { watchlist, watchlistData } = useMarketStore()

  if (watchlist.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Watchlist</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">
              Your watchlist is empty
            </p>
            <Link href="/stocks">
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Stocks
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Watchlist</CardTitle>
        <Link href="/watchlist">
          <Button variant="ghost" size="sm">View All</Button>
        </Link>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {watchlist.slice(0, 5).map((symbol) => {
            const stock = watchlistData[symbol]
            
            if (!stock) {
              return (
                <div key={symbol} className="flex items-center justify-between py-2">
                  <div>
                    <p className="font-medium">{symbol}</p>
                    <p className="text-sm text-muted-foreground">Loading...</p>
                  </div>
                </div>
              )
            }
            
            const isPositive = stock.change_percent >= 0
            
            return (
              <Link
                key={symbol}
                href={`/stocks/${symbol}`}
                className="block hover:bg-muted/50 -mx-2 px-2 py-2 rounded-md transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{stock.symbol}</p>
                    <p className="text-xs text-muted-foreground">{stock.name}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{formatCurrency(stock.current_price)}</p>
                    <div className={`flex items-center justify-end space-x-1 text-sm ${
                      isPositive ? 'text-bullish' : 'text-bearish'
                    }`}>
                      {isPositive ? (
                        <ArrowUpIcon className="h-3 w-3" />
                      ) : (
                        <ArrowDownIcon className="h-3 w-3" />
                      )}
                      <span>{formatPercentage(stock.change_percent / 100)}</span>
                    </div>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
