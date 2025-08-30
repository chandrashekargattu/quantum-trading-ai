'use client'

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/useAuthStore'
import { marketService } from '@/services/api/market-optimized'
import { portfolioService } from '@/services/api/portfolio-optimized'
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Plus,
  RefreshCw,
  BarChart3,
  DollarSign,
  Trash2,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { formatCurrency, formatNumber } from '@/lib/utils'

interface Stock {
  symbol: string
  name: string
  exchange: string
  current_price: number
  change_percent: number
  volume?: number
  market_cap?: number
  pe_ratio?: number
  is_optionable?: boolean
}

const STOCKS_PER_PAGE = 12

export default function StocksPage() {
  const [stocks, setStocks] = useState<Stock[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [newStock, setNewStock] = useState({ symbol: '', name: '', exchange: 'NSE' })
  const [isAddingStock, setIsAddingStock] = useState(false)
  const [deletingStock, setDeletingStock] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalStocks, setTotalStocks] = useState(0)
  
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()

  // Calculate pagination
  const totalPages = Math.ceil(totalStocks / STOCKS_PER_PAGE)
  const paginatedStocks = stocks.slice(
    (currentPage - 1) * STOCKS_PER_PAGE,
    currentPage * STOCKS_PER_PAGE
  )

  // Check authentication
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login')
    }
  }, [isAuthenticated, router])

  // Load initial stocks
  useEffect(() => {
    loadStocks()
  }, [])

  const loadStocks = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/stocks/watchlist`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`
        }
      })
      
      if (!response.ok) throw new Error('Failed to load stocks')
      
      const data = await response.json()
      setStocks(data)
      setTotalStocks(data.length)
      setCurrentPage(1) // Reset to first page
    } catch (err) {
      setError('Failed to load stocks')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const searchStocks = async () => {
    if (!searchQuery.trim()) return
    
    setIsSearching(true)
    setError(null)
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/stocks/search?q=${encodeURIComponent(searchQuery)}`,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('access_token')}`
          }
        }
      )
      
      if (!response.ok) throw new Error('Search failed')
      
      const data = await response.json()
      setStocks(data)
      setTotalStocks(data.length)
      setCurrentPage(1) // Reset to first page
    } catch (err) {
      setError('Search failed')
      console.error(err)
    } finally {
      setIsSearching(false)
    }
  }

  const addStock = async () => {
    if (!newStock.symbol.trim()) return
    
    setIsAddingStock(true)
    setError(null)
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/stocks/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          symbol: newStock.symbol.toUpperCase(),
          name: newStock.name || newStock.symbol.toUpperCase(),
          exchange: newStock.exchange
        })
      })
      
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to add stock')
      }
      
      // Clear form and reload stocks
      setNewStock({ symbol: '', name: '', exchange: 'NSE' })
      await loadStocks()
    } catch (err: any) {
      setError(err.message || 'Failed to add stock')
      console.error(err)
    } finally {
      setIsAddingStock(false)
    }
  }

  const deleteStock = async (symbol: string) => {
    setDeletingStock(symbol)
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/stocks/${encodeURIComponent(symbol)}`,
        {
          method: 'DELETE',
          headers: {
            Authorization: `Bearer ${localStorage.getItem('access_token')}`
          }
        }
      )
      
      if (!response.ok) throw new Error('Failed to remove stock')
      
      // Remove from local state
      setStocks(stocks.filter(s => s.symbol !== symbol))
      setTotalStocks(totalStocks - 1)
      
      // If current page is empty after deletion, go to previous page
      if (paginatedStocks.length === 1 && currentPage > 1) {
        setCurrentPage(currentPage - 1)
      }
    } catch (err) {
      console.error('Failed to remove stock:', err)
    } finally {
      setDeletingStock(null)
    }
  }

  const navigateToTrade = (symbol: string) => {
    // Navigate to trading page with pre-filled symbol
    router.push(`/dashboard?trade=${symbol}`)
  }

  const goToPage = (page: number) => {
    setCurrentPage(page)
  }

  if (!isAuthenticated) return null

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Indian Stocks</h1>
        <p className="text-muted-foreground">Browse and add NSE/BSE stocks to your watchlist</p>
      </div>

      {/* Add Stock Section */}
      <Card className="p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Add New Stock</h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div>
            <Label htmlFor="symbol">Stock Symbol</Label>
            <Input
              id="symbol"
              placeholder="e.g., RELIANCE"
              value={newStock.symbol}
              onChange={(e) => setNewStock({ ...newStock, symbol: e.target.value })}
              onKeyDown={(e) => e.key === 'Enter' && addStock()}
            />
          </div>
          <div className="md:col-span-2">
            <Label htmlFor="name">Company Name (Optional)</Label>
            <Input
              id="name"
              placeholder="e.g., Reliance Industries Ltd"
              value={newStock.name}
              onChange={(e) => setNewStock({ ...newStock, name: e.target.value })}
              onKeyDown={(e) => e.key === 'Enter' && addStock()}
            />
          </div>
          <div>
            <Label htmlFor="exchange">Exchange</Label>
            <select
              id="exchange"
              value={newStock.exchange}
              onChange={(e) => setNewStock({ ...newStock, exchange: e.target.value })}
              className="w-full px-3 py-2 border border-input bg-background rounded-md focus:outline-none focus:ring-2 focus:ring-ring text-sm"
            >
              <option value="NSE">NSE</option>
              <option value="BSE">BSE</option>
            </select>
          </div>
          <div className="flex items-end">
            <Button 
              onClick={addStock} 
              disabled={isAddingStock || !newStock.symbol.trim()}
              className="w-full"
            >
              {isAddingStock ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Adding...
                </>
              ) : (
                <>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Stock
                </>
              )}
            </Button>
          </div>
        </div>
        {error && (
          <p className="text-sm text-destructive mt-2">{error}</p>
        )}
      </Card>

      {/* Search Section */}
      <Card className="p-6 mb-8">
        <div className="flex gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search stocks by symbol or name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && searchStocks()}
              className="pl-10"
            />
          </div>
          <Button onClick={searchStocks} disabled={isSearching}>
            {isSearching ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Searching...
              </>
            ) : (
              'Search'
            )}
          </Button>
          <Button onClick={loadStocks} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </Card>

      {/* Stocks Grid */}
      {isLoading ? (
        <div className="text-center py-8">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading stocks...</p>
        </div>
      ) : stocks.length === 0 ? (
        <Card className="p-8 text-center">
          <p className="text-muted-foreground mb-4">No stocks found</p>
          <p className="text-sm">Add stocks using the form above or search for existing ones</p>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
            {paginatedStocks.map((stock) => (
              <Card key={stock.symbol} className="p-4 hover:shadow-lg transition-shadow relative">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteStock(stock.symbol)}
                  disabled={deletingStock === stock.symbol}
                  className="absolute top-2 right-2 text-muted-foreground hover:text-destructive"
                >
                  {deletingStock === stock.symbol ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>

                <div className="flex justify-between items-start mb-3 pr-8">
                  <div>
                    <h3 className="font-semibold text-lg">{stock.symbol}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-1">{stock.name}</p>
                  </div>
                  <Badge variant="secondary" className="text-xs">
                    {stock.exchange}
                  </Badge>
                </div>
                
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <p className="text-2xl font-bold">{formatCurrency(stock.current_price)}</p>
                    <div className={`flex items-center text-sm ${
                      stock.change_percent >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stock.change_percent >= 0 ? (
                        <TrendingUp className="h-3 w-3 mr-1" />
                      ) : (
                        <TrendingDown className="h-3 w-3 mr-1" />
                      )}
                      {Math.abs(stock.change_percent).toFixed(2)}%
                    </div>
                  </div>
                  {stock.is_optionable && (
                    <Badge variant="outline" className="text-xs">
                      F&O
                    </Badge>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2 text-sm mb-4">
                  {stock.volume && (
                    <div>
                      <p className="text-muted-foreground">Volume</p>
                      <p className="font-medium">{formatNumber(stock.volume)}</p>
                    </div>
                  )}
                  {stock.market_cap && (
                    <div>
                      <p className="text-muted-foreground">Market Cap</p>
                      <p className="font-medium">{formatCurrency(stock.market_cap / 10000000)}Cr</p>
                    </div>
                  )}
                  {stock.pe_ratio && (
                    <div>
                      <p className="text-muted-foreground">P/E Ratio</p>
                      <p className="font-medium">{stock.pe_ratio.toFixed(2)}</p>
                    </div>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="flex-1"
                    onClick={() => router.push(`/stocks/${stock.symbol}`)}
                  >
                    <BarChart3 className="h-3 w-3 mr-1" />
                    Details
                  </Button>
                  <Button 
                    size="sm" 
                    className="flex-1"
                    onClick={() => navigateToTrade(stock.symbol)}
                  >
                    <DollarSign className="h-3 w-3 mr-1" />
                    Trade
                  </Button>
                </div>
              </Card>
            ))}
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </Button>

              <div className="flex gap-1">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => {
                  // Show first page, last page, current page, and pages around current
                  if (
                    page === 1 ||
                    page === totalPages ||
                    (page >= currentPage - 1 && page <= currentPage + 1)
                  ) {
                    return (
                      <Button
                        key={page}
                        variant={page === currentPage ? "default" : "outline"}
                        size="sm"
                        onClick={() => goToPage(page)}
                        className="w-10"
                      >
                        {page}
                      </Button>
                    )
                  } else if (
                    page === currentPage - 2 ||
                    page === currentPage + 2
                  ) {
                    return <span key={page} className="px-2">...</span>
                  }
                  return null
                })}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}

          <div className="text-center text-sm text-muted-foreground mt-4">
            Showing {((currentPage - 1) * STOCKS_PER_PAGE) + 1} - {Math.min(currentPage * STOCKS_PER_PAGE, totalStocks)} of {totalStocks} stocks
          </div>
        </>
      )}
    </div>
  )
}