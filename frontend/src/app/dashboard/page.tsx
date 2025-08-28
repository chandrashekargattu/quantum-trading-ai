'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/useAuthStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { MarketOverview } from '@/components/dashboard/MarketOverview'
import { PortfolioSummary } from '@/components/dashboard/PortfolioSummary'
import { RecentTrades } from '@/components/dashboard/RecentTrades'
import { WatchlistWidget } from '@/components/dashboard/WatchlistWidget'
import { TrendingOptions } from '@/components/dashboard/TrendingOptions'
import { AIInsights } from '@/components/dashboard/AIInsights'

export default function DashboardPage() {
  const router = useRouter()
  const { user, isAuthenticated, fetchUser } = useAuthStore()

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login')
    } else if (!user) {
      fetchUser()
    }
  }, [isAuthenticated, user, router, fetchUser])

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-pulse">Loading...</div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Welcome back, {user.full_name || user.username}!</h1>
          <p className="text-muted-foreground">
            Here's what's happening in the markets today
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-muted-foreground">Account Type</p>
          <p className="font-semibold capitalize">{user.account_type}</p>
        </div>
      </div>

      {/* Market Overview */}
      <MarketOverview />

      {/* Main Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Portfolio Summary */}
        <div className="lg:col-span-2">
          <PortfolioSummary />
        </div>
        
        {/* AI Insights */}
        <div>
          <AIInsights />
        </div>
      </div>

      {/* Secondary Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Watchlist */}
        <div>
          <WatchlistWidget />
        </div>
        
        {/* Trending Options */}
        <div>
          <TrendingOptions />
        </div>
        
        {/* Recent Trades */}
        <div>
          <RecentTrades />
        </div>
      </div>
    </div>
  )
}
