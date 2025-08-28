'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Brain, TrendingUp, AlertCircle, Zap } from 'lucide-react'

interface Insight {
  id: string
  type: 'bullish' | 'bearish' | 'neutral' | 'alert'
  title: string
  description: string
  confidence: number
  symbol?: string
}

// Mock data - in real app, this would come from API
const mockInsights: Insight[] = [
  {
    id: '1',
    type: 'bullish',
    title: 'AAPL Options Opportunity',
    description: 'Unusual call volume detected. IV rank at 15%, suggesting potential for volatility expansion.',
    confidence: 0.85,
    symbol: 'AAPL',
  },
  {
    id: '2',
    type: 'alert',
    title: 'Portfolio Risk Alert',
    description: 'Your tech sector exposure is 65%. Consider diversification to reduce concentration risk.',
    confidence: 0.92,
  },
  {
    id: '3',
    type: 'bullish',
    title: 'SPY Trend Continuation',
    description: 'ML model predicts 78% probability of upward movement in next 5 days based on current patterns.',
    confidence: 0.78,
    symbol: 'SPY',
  },
]

export function AIInsights() {
  const getIcon = (type: string) => {
    switch (type) {
      case 'bullish':
        return <TrendingUp className="h-4 w-4 text-bullish" />
      case 'bearish':
        return <TrendingUp className="h-4 w-4 text-bearish rotate-180" />
      case 'alert':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      default:
        return <Zap className="h-4 w-4 text-primary" />
    }
  }

  const getTypeBadge = (type: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      bullish: 'default',
      bearish: 'destructive',
      alert: 'secondary',
      neutral: 'outline',
    }
    
    return (
      <Badge variant={variants[type] || 'default'} className="capitalize">
        {type}
      </Badge>
    )
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="h-5 w-5" />
          <span>AI Insights</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {mockInsights.map((insight) => (
            <div
              key={insight.id}
              className="p-3 rounded-lg border bg-muted/20 hover:bg-muted/40 transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getIcon(insight.type)}
                  <h4 className="font-medium text-sm">{insight.title}</h4>
                </div>
                {getTypeBadge(insight.type)}
              </div>
              
              <p className="text-xs text-muted-foreground mb-2">
                {insight.description}
              </p>
              
              <div className="flex items-center justify-between">
                {insight.symbol && (
                  <Badge variant="outline" className="text-xs">
                    {insight.symbol}
                  </Badge>
                )}
                <div className="flex items-center space-x-1">
                  <span className="text-xs text-muted-foreground">Confidence:</span>
                  <span className="text-xs font-medium">
                    {Math.round(insight.confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 text-center">
          <a
            href="/insights"
            className="text-xs text-primary hover:underline"
          >
            View all insights →
          </a>
        </div>
      </CardContent>
    </Card>
  )
}
