'use client'

import { useEffect, useState, lazy, Suspense } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/useAuthStore'
import { marketService } from '@/services/api/market-optimized'
import { portfolioService } from '@/services/api/portfolio-optimized'

// Import components directly for critical above-the-fold content
import { MarketOverview } from '@/components/dashboard/MarketOverview'
import { PortfolioSummary } from '@/components/dashboard/PortfolioSummary'
import { Button } from '@/components/ui/button'
import { BarChart3 } from 'lucide-react'


// Lazy load non-critical components
const AIInsights = lazy(() => import('@/components/dashboard/AIInsights').then(m => ({ default: m.AIInsights })))
const WatchlistWidget = lazy(() => import('@/components/dashboard/WatchlistWidget').then(m => ({ default: m.WatchlistWidget })))
const TrendingOptions = lazy(() => import('@/components/dashboard/TrendingOptions').then(m => ({ default: m.TrendingOptions })))
const RecentTrades = lazy(() => import('@/components/dashboard/RecentTrades').then(m => ({ default: m.RecentTrades })))

// Skeleton loader for lazy components
const ComponentSkeleton = ({ height = 'h-64' }: { height?: string }) => (
  <div className={`animate-pulse bg-muted rounded-lg ${height}`} />
)

export default function DashboardPage() {
  const router = useRouter()
  const { user, isAuthenticated, fetchUser } = useAuthStore()
  const [isHydrated, setIsHydrated] = useState(false)

  useEffect(() => {
    setIsHydrated(true)
  }, [])

  useEffect(() => {
    if (!isHydrated) return
    
    const token = localStorage.getItem('access_token')
    
    if (!token && !isAuthenticated) {
      router.push('/auth/login')
    } else if (!user && (token || isAuthenticated)) {
      fetchUser()
    } else if (user) {
      // Prefetch data in background after user is loaded
      marketService.prefetchMarketData()
      portfolioService.prefetchPortfolios()
    }
  }, [isHydrated, isAuthenticated, user, router, fetchUser])

  // Show minimal loader
  if (!isHydrated || (!user && localStorage.getItem('access_token'))) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-sm text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    )
  }
  
  if (!user) {
    return null
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header - Render immediately */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Welcome back, {user.full_name || user.username}!</h1>
          <p className="text-muted-foreground">
            Here's what's happening in the markets today
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Button 
            onClick={() => router.push('/stocks')}
            variant="outline"
            className="flex items-center gap-2"
          >
            <BarChart3 className="h-4 w-4" />
            Browse Stocks
          </Button>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Account Type</p>
            <p className="font-semibold capitalize">{user.account_type}</p>
          </div>
        </div>
      </div>

      {/* Critical Content - Load immediately */}
      <MarketOverview />

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <PortfolioSummary />
        </div>
        
        {/* AI Insights - Lazy load */}
        <div>
          <Suspense fallback={<ComponentSkeleton />}>
            <AIInsights />
          </Suspense>
        </div>
      </div>

      {/* Secondary Content - Lazy load with progressive enhancement */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <div>
          <Suspense fallback={<ComponentSkeleton height="h-80" />}>
            <WatchlistWidget />
          </Suspense>
        </div>
        
        <div>
          <Suspense fallback={<ComponentSkeleton height="h-80" />}>
            <TrendingOptions />
          </Suspense>
        </div>
        
        <div>
          <Suspense fallback={<ComponentSkeleton height="h-80" />}>
            <RecentTrades />
          </Suspense>
        </div>
      </div>
    </div>
  )
}