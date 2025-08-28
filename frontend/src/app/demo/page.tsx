'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowRight, TrendingUp, TrendingDown, Activity } from 'lucide-react'

export default function DemoPage() {
  const [selectedStock, setSelectedStock] = useState('AAPL')
  
  // Mock data for demonstration
  const mockStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 185.52, change: 2.34, changePercent: 1.28, prediction: 'bullish' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.91, change: -1.23, changePercent: -0.32, prediction: 'neutral' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 141.80, change: 3.21, changePercent: 2.32, prediction: 'bullish' },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 251.60, change: -5.40, changePercent: -2.10, prediction: 'bearish' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 521.80, change: 12.30, changePercent: 2.41, prediction: 'bullish' }
  ]
  
  const mockOptions = [
    { type: 'CALL', strike: 190, expiry: '2024-01-19', bid: 2.45, ask: 2.50, volume: 15420, openInterest: 45320, iv: 0.285 },
    { type: 'CALL', strike: 195, expiry: '2024-01-19', bid: 0.98, ask: 1.02, volume: 8930, openInterest: 32100, iv: 0.312 },
    { type: 'PUT', strike: 180, expiry: '2024-01-19', bid: 1.15, ask: 1.20, volume: 12340, openInterest: 28900, iv: 0.298 },
    { type: 'PUT', strike: 175, expiry: '2024-01-19', bid: 0.45, ask: 0.48, volume: 5670, openInterest: 15600, iv: 0.325 }
  ]

  return (
    <div className="min-h-screen p-4">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="mb-4 text-4xl font-bold">Live Trading Demo</h1>
          <p className="text-lg text-muted-foreground">
            Experience the power of AI-driven options trading
          </p>
        </div>

        {/* Stock Watchlist */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>AI Stock Predictions</CardTitle>
            <CardDescription>Real-time analysis powered by machine learning</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {mockStocks.map(stock => (
                <div
                  key={stock.symbol}
                  className={`flex items-center justify-between rounded-lg border p-4 transition-colors hover:bg-muted/50 cursor-pointer ${
                    selectedStock === stock.symbol ? 'border-primary bg-muted/50' : ''
                  }`}
                  onClick={() => setSelectedStock(stock.symbol)}
                >
                  <div>
                    <div className="font-semibold">{stock.symbol}</div>
                    <div className="text-sm text-muted-foreground">{stock.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">${stock.price.toFixed(2)}</div>
                    <div className={`flex items-center text-sm ${stock.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {stock.change > 0 ? <TrendingUp className="mr-1 h-3 w-3" /> : <TrendingDown className="mr-1 h-3 w-3" />}
                      {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent}%)
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className={`rounded-full px-3 py-1 text-xs font-medium ${
                      stock.prediction === 'bullish' ? 'bg-green-100 text-green-800' :
                      stock.prediction === 'bearish' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {stock.prediction.toUpperCase()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Options Chain */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Options Chain - {selectedStock}</CardTitle>
            <CardDescription>AI-recommended strikes highlighted</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="p-2 text-left">Type</th>
                    <th className="p-2 text-left">Strike</th>
                    <th className="p-2 text-left">Expiry</th>
                    <th className="p-2 text-left">Bid/Ask</th>
                    <th className="p-2 text-left">Volume</th>
                    <th className="p-2 text-left">OI</th>
                    <th className="p-2 text-left">IV</th>
                  </tr>
                </thead>
                <tbody>
                  {mockOptions.map((option, idx) => (
                    <tr key={idx} className="border-b hover:bg-muted/50">
                      <td className="p-2">
                        <span className={`rounded px-2 py-1 text-xs font-medium ${
                          option.type === 'CALL' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {option.type}
                        </span>
                      </td>
                      <td className="p-2">${option.strike}</td>
                      <td className="p-2">{option.expiry}</td>
                      <td className="p-2">${option.bid} / ${option.ask}</td>
                      <td className="p-2">{option.volume.toLocaleString()}</td>
                      <td className="p-2">{option.openInterest.toLocaleString()}</td>
                      <td className="p-2">{(option.iv * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* AI Strategy Recommendation */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>AI Strategy Recommendation</CardTitle>
            <CardDescription>Based on current market conditions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg bg-primary/10 p-6">
              <div className="mb-4 flex items-center">
                <Activity className="mr-2 h-6 w-6 text-primary" />
                <h3 className="text-lg font-semibold">Bull Call Spread</h3>
              </div>
              <p className="mb-4 text-muted-foreground">
                The AI model suggests a bull call spread on {selectedStock} based on:
              </p>
              <ul className="mb-4 space-y-2 text-sm">
                <li>• 78% probability of upward movement in the next 7 days</li>
                <li>• Low implied volatility compared to historical average</li>
                <li>• Strong technical support at $182 level</li>
                <li>• Positive sentiment analysis from recent earnings</li>
              </ul>
              <div className="rounded-lg border bg-background p-4">
                <p className="mb-2 font-semibold">Recommended Trade:</p>
                <p className="text-sm">Buy $185 Call @ $2.50</p>
                <p className="text-sm">Sell $190 Call @ $1.00</p>
                <p className="mt-2 text-sm">
                  <span className="font-semibold">Max Profit:</span> $350 per contract<br/>
                  <span className="font-semibold">Max Loss:</span> $150 per contract<br/>
                  <span className="font-semibold">Break-even:</span> $186.50
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* CTA */}
        <div className="mb-8 text-center">
          <p className="mb-4 text-lg text-muted-foreground">
            Ready to start trading with AI-powered insights?
          </p>
          <Link href="/auth/register">
            <Button size="lg">
              Create Free Account
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
