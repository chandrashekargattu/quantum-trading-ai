'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle2,
  DollarSign,
  Target,
  Brain,
  Shield,
  Zap,
  BookOpen,
  Activity,
  BarChart3,
  PieChart,
  Rocket
} from 'lucide-react';
import { Line, Bar } from 'recharts';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface RecoveryPlan {
  phases: Array<{
    name: string;
    duration: string;
    goals: string[];
    capital_allocation?: Record<string, number>;
    daily_limits?: Record<string, number>;
  }>;
  strategies: Array<{
    id: string;
    name: string;
    type: string;
    expected_stats: {
      monthly_return: string;
      win_rate: string;
      max_drawdown?: string;
    };
  }>;
  rules: Array<{
    category: string;
    rule: string;
    implementation: string;
  }>;
  milestones: Array<{
    target: string;
    amount: number;
    estimated_time: string;
    reward: string;
  }>;
}

interface PortfolioAnalysis {
  total_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions: Array<{
    symbol: string;
    pnl: number;
    recommendation: string;
    confidence: number;
  }>;
  recovery_strategies: Array<{
    name: string;
    description: string;
    priority: string;
    potential_monthly_income?: number;
  }>;
  recommendations: string[];
}

export default function ZerodhaRecoveryDashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const [portfolio, setPortfolio] = useState<PortfolioAnalysis | null>(null);
  const [recoveryPlan, setRecoveryPlan] = useState<RecoveryPlan | null>(null);
  const [activeStrategy, setActiveStrategy] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Mock data for demonstration
  useEffect(() => {
    // In real implementation, fetch from API
    setPortfolio({
      total_pnl: -250000,
      unrealized_pnl: -50000,
      realized_pnl: -200000,
      positions: [
        { symbol: 'RELIANCE', pnl: -15000, recommendation: 'HOLD_FOR_RECOVERY', confidence: 0.75 },
        { symbol: 'TCS', pnl: -8000, recommendation: 'CUT_LOSS', confidence: 0.85 },
        { symbol: 'INFY', pnl: 5000, recommendation: 'BOOK_PARTIAL_PROFIT', confidence: 0.8 }
      ],
      recovery_strategies: [
        {
          name: 'NIFTY/BANKNIFTY Credit Spreads',
          description: 'Weekly credit spreads with 80%+ win rate',
          priority: 'HIGH',
          potential_monthly_income: 50000
        },
        {
          name: 'Covered Call Strategy',
          description: 'Generate monthly income by selling OTM calls',
          priority: 'MEDIUM',
          potential_monthly_income: 20000
        }
      ],
      recommendations: [
        'Switch to paper trading for skill development',
        'Focus on index ETFs for next 3 months',
        'Implement strict 2% daily loss limit'
      ]
    });

    setRecoveryPlan({
      phases: [
        {
          name: 'Stabilization',
          duration: '30 days',
          goals: ['Stop further losses', 'Build discipline', 'Learn risk management'],
          capital_allocation: { paper_trading: 0.5, small_real_trades: 0.3, education: 0.2 }
        },
        {
          name: 'Systematic Recovery',
          duration: '60 days',
          goals: ['Consistent small profits', 'Build confidence', 'Refine strategy']
        },
        {
          name: 'Growth & Scaling',
          duration: 'Ongoing',
          goals: ['Scale profitable strategies', 'Diversify approaches', 'Build wealth']
        }
      ],
      strategies: [
        {
          id: 'index_option_selling',
          name: 'NIFTY/BANKNIFTY Option Selling',
          type: 'income_generation',
          expected_stats: {
            monthly_return: '3-5%',
            win_rate: '80%',
            max_drawdown: '5%'
          }
        },
        {
          id: 'momentum_breakout',
          name: 'High Momentum Stock Trading',
          type: 'capital_growth',
          expected_stats: {
            monthly_return: '8-12%',
            win_rate: '55%',
            max_drawdown: '8%'
          }
        },
        {
          id: 'ai_ml_signals',
          name: 'Quantum AI Trading Signals',
          type: 'systematic',
          expected_stats: {
            monthly_return: '12-18%',
            win_rate: '65%'
          }
        }
      ],
      rules: [
        {
          category: 'Risk Management',
          rule: 'Never risk more than 2% on a single trade',
          implementation: 'Automated position sizing'
        },
        {
          category: 'Daily Limits',
          rule: 'Stop trading after 2% daily loss',
          implementation: 'Auto-lockout feature'
        }
      ],
      milestones: [
        {
          target: 'Recover 10% of losses',
          amount: 25000,
          estimated_time: '30 days',
          reward: 'Unlock advanced charting features'
        },
        {
          target: 'Recover 25% of losses',
          amount: 62500,
          estimated_time: '75 days',
          reward: 'Access to premium strategies'
        },
        {
          target: 'Recover 50% of losses',
          amount: 125000,
          estimated_time: '150 days',
          reward: 'Reduced brokerage rates'
        },
        {
          target: 'Recover 100% of losses',
          amount: 250000,
          estimated_time: '300 days',
          reward: 'Elite trader status'
        }
      ]
    });
  }, []);

  const connectZerodha = () => {
    // In real implementation, redirect to Zerodha OAuth
    setLoading(true);
    setTimeout(() => {
      setIsConnected(true);
      setLoading(false);
    }, 2000);
  };

  const startStrategy = (strategyId: string) => {
    setActiveStrategy(strategyId);
    // In real implementation, call API to start strategy
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(amount);
  };

  if (!isConnected) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 to-black p-4">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold">Connect Your Zerodha Account</CardTitle>
            <CardDescription>
              Start your journey to recover losses and build wealth
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertTitle>Secure Connection</AlertTitle>
              <AlertDescription>
                We use OAuth 2.0 for secure authentication. Your credentials are never stored.
              </AlertDescription>
            </Alert>
            <Button 
              onClick={connectZerodha}
              className="w-full"
              size="lg"
              disabled={loading}
            >
              {loading ? 'Connecting...' : 'Connect Zerodha Account'}
            </Button>
            <p className="text-sm text-muted-foreground text-center">
              By connecting, you agree to our terms and authorize read-only access to your trading data.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-white">Recovery Dashboard</h1>
            <p className="text-gray-400 mt-2">Your personalized path to profitable trading</p>
          </div>
          <Badge variant="outline" className="text-green-400 border-green-400">
            <Activity className="w-4 h-4 mr-2" />
            Connected to Zerodha
          </Badge>
        </div>

        {/* Portfolio Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-red-950/50 border-red-900">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
              <TrendingDown className="h-4 w-4 text-red-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-400">
                {formatCurrency(portfolio?.total_pnl || 0)}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                Time to turn this around! 💪
              </p>
            </CardContent>
          </Card>

          <Card className="bg-yellow-950/50 border-yellow-900">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Recovery Progress</CardTitle>
              <Target className="h-4 w-4 text-yellow-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-400">0%</div>
              <Progress value={0} className="mt-2" />
              <p className="text-xs text-gray-400 mt-1">
                Journey of 1000 miles starts with one step
              </p>
            </CardContent>
          </Card>

          <Card className="bg-green-950/50 border-green-900">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Projected Monthly</CardTitle>
              <TrendingUp className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">
                {formatCurrency(70000)}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                With recommended strategies
              </p>
            </CardContent>
          </Card>
        </div>

        {/* AI Recommendations Alert */}
        <Alert className="bg-blue-950/50 border-blue-900">
          <Brain className="h-4 w-4" />
          <AlertTitle>AI Analysis Complete</AlertTitle>
          <AlertDescription>
            Based on your trading history, we&apos;ve identified patterns and created a personalized recovery plan.
            Your main issues: Revenge trading, No stop losses, and Overtrading during volatility.
          </AlertDescription>
        </Alert>

        {/* Main Content Tabs */}
        <Tabs defaultValue="recovery-plan" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="recovery-plan">Recovery Plan</TabsTrigger>
            <TabsTrigger value="strategies">Strategies</TabsTrigger>
            <TabsTrigger value="positions">Positions</TabsTrigger>
            <TabsTrigger value="education">Education</TabsTrigger>
          </TabsList>

          <TabsContent value="recovery-plan" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Your 3-Phase Recovery Journey</CardTitle>
                <CardDescription>
                  Systematic approach to recover losses and build sustainable profits
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {recoveryPlan?.phases.map((phase, index) => (
                    <motion.div
                      key={phase.name}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex gap-4"
                    >
                      <div className="flex-shrink-0">
                        <div className={cn(
                          "w-12 h-12 rounded-full flex items-center justify-center",
                          index === 0 ? "bg-blue-500" : "bg-gray-700"
                        )}>
                          {index + 1}
                        </div>
                      </div>
                      <div className="flex-grow">
                        <h3 className="font-semibold text-lg">{phase.name}</h3>
                        <p className="text-sm text-gray-400">{phase.duration}</p>
                        <ul className="mt-2 space-y-1">
                          {phase.goals.map((goal) => (
                            <li key={goal} className="text-sm flex items-center gap-2">
                              <CheckCircle2 className="w-4 h-4 text-green-400" />
                              {goal}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  ))}
                </div>

                <div className="mt-8">
                  <h4 className="font-semibold mb-4">Recovery Milestones</h4>
                  <div className="space-y-3">
                    {recoveryPlan?.milestones.map((milestone, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Target className="w-5 h-5 text-yellow-400" />
                          <div>
                            <p className="font-medium">{milestone.target}</p>
                            <p className="text-sm text-gray-400">{milestone.estimated_time}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-semibold">{formatCurrency(milestone.amount)}</p>
                          <p className="text-xs text-green-400">{milestone.reward}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="strategies" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recoveryPlan?.strategies.map((strategy) => (
                <Card key={strategy.id} className="hover:border-blue-500 transition-colors">
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="text-lg">{strategy.name}</CardTitle>
                        <Badge variant="outline" className="mt-1">
                          {strategy.type}
                        </Badge>
                      </div>
                      {strategy.id === 'ai_ml_signals' && (
                        <Brain className="w-6 h-6 text-blue-400" />
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">Monthly Return</p>
                          <p className="font-semibold text-green-400">
                            {strategy.expected_stats.monthly_return}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Win Rate</p>
                          <p className="font-semibold text-blue-400">
                            {strategy.expected_stats.win_rate}
                          </p>
                        </div>
                      </div>
                      {strategy.expected_stats.max_drawdown && (
                        <div>
                          <p className="text-gray-400 text-sm">Max Drawdown</p>
                          <p className="font-semibold text-yellow-400">
                            {strategy.expected_stats.max_drawdown}
                          </p>
                        </div>
                      )}
                      <Button
                        onClick={() => startStrategy(strategy.id)}
                        disabled={activeStrategy === strategy.id}
                        className="w-full"
                        variant={activeStrategy === strategy.id ? "secondary" : "default"}
                      >
                        {activeStrategy === strategy.id ? (
                          <>
                            <Zap className="w-4 h-4 mr-2" />
                            Strategy Active
                          </>
                        ) : (
                          'Start Strategy'
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}

              {/* Special AI Strategy Card */}
              <Card className="border-gradient bg-gradient-to-br from-blue-950/50 to-purple-950/50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Rocket className="w-5 h-5" />
                    Quantum AI Advantage
                  </CardTitle>
                  <CardDescription>
                    Our proprietary AI system analyzing millions of data points
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span>Multi-timeframe analysis</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span>Sentiment analysis from 50+ sources</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span>Real-time risk assessment</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span>Automatic position sizing</span>
                    </div>
                  </div>
                  <Button className="w-full mt-4" variant="secondary">
                    <Brain className="w-4 h-4 mr-2" />
                    AI is analyzing markets 24/7
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="positions" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Current Positions Analysis</CardTitle>
                <CardDescription>
                  AI-powered recommendations for each position
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {portfolio?.positions.map((position) => (
                    <div
                      key={position.symbol}
                      className="flex items-center justify-between p-4 bg-gray-800 rounded-lg"
                    >
                      <div className="flex-grow">
                        <h4 className="font-semibold">{position.symbol}</h4>
                        <p className={cn(
                          "text-sm",
                          position.pnl < 0 ? "text-red-400" : "text-green-400"
                        )}>
                          P&L: {formatCurrency(position.pnl)}
                        </p>
                      </div>
                      <div className="text-right">
                        <Badge
                          variant={position.recommendation === 'CUT_LOSS' ? 'destructive' : 'default'}
                        >
                          {position.recommendation}
                        </Badge>
                        <p className="text-sm text-gray-400 mt-1">
                          Confidence: {(position.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <Alert className="mt-6">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Important</AlertTitle>
                  <AlertDescription>
                    These are AI suggestions. Always use your judgment and follow risk management rules.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="education" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Your Learning Path</CardTitle>
                <CardDescription>
                  Master these concepts to become a profitable trader
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-red-950/50 border border-red-900 rounded-lg">
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                      <div>
                        <h4 className="font-semibold">Critical: Risk Management</h4>
                        <p className="text-sm text-gray-400">Complete this first - 2 hour course</p>
                      </div>
                    </div>
                    <Button size="sm" className="mt-3">Start Course</Button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-800 rounded-lg">
                      <BookOpen className="w-5 h-5 text-blue-400 mb-2" />
                      <h4 className="font-medium">Trading Psychology</h4>
                      <p className="text-sm text-gray-400 mt-1">
                        Control emotions, trade better
                      </p>
                    </div>
                    <div className="p-4 bg-gray-800 rounded-lg">
                      <BarChart3 className="w-5 h-5 text-green-400 mb-2" />
                      <h4 className="font-medium">Technical Analysis</h4>
                      <p className="text-sm text-gray-400 mt-1">
                        Chart patterns & indicators
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Trading Rules</CardTitle>
                <CardDescription>
                  These rules will be automatically enforced
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {recoveryPlan?.rules.map((rule, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div className="flex-grow">
                        <p className="font-medium">{rule.rule}</p>
                        <p className="text-sm text-gray-400">{rule.implementation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Quick Actions */}
        <Card className="bg-gradient-to-r from-blue-950/50 to-purple-950/50">
          <CardHeader>
            <CardTitle>Ready to Start Your Recovery?</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              <Button size="lg" className="flex-1">
                <Zap className="w-4 h-4 mr-2" />
                Start Paper Trading
              </Button>
              <Button size="lg" variant="secondary" className="flex-1">
                <Brain className="w-4 h-4 mr-2" />
                Activate AI Signals
              </Button>
              <Button size="lg" variant="outline" className="flex-1">
                <BookOpen className="w-4 h-4 mr-2" />
                Begin Education
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
