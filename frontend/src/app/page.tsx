import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowRight, BarChart3, Brain, Zap, Shield } from 'lucide-react'

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 text-white">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="container relative z-10 mx-auto px-4 py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              AI-Powered Options Trading
              <span className="block text-yellow-300"> Reimagined</span>
            </h1>
            <p className="mb-8 text-lg text-gray-100 sm:text-xl">
              Leverage advanced machine learning algorithms to analyze market patterns, 
              predict price movements, and execute sophisticated options strategies with confidence.
            </p>
            <div className="flex flex-col gap-4 sm:flex-row sm:justify-center">
              <Link href="/auth/register">
                <Button size="lg" className="w-full sm:w-auto bg-white text-gray-900 hover:bg-gray-100">
                  Start Trading Now
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/demo">
                <Button size="lg" variant="outline" className="w-full sm:w-auto border-white text-white hover:bg-white hover:text-gray-900">
                  View Live Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute inset-0 -z-10 opacity-20">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff20_1px,transparent_1px),linear-gradient(to_bottom,#ffffff20_1px,transparent_1px)] bg-[size:14px_24px]" />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto px-4">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-bold text-gray-900 dark:text-white">Cutting-Edge Trading Features</h2>
            <p className="mx-auto max-w-2xl text-gray-600 dark:text-gray-400">
              Everything you need to trade options like a professional, powered by AI
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900">
                <Brain className="h-6 w-6 text-blue-600 dark:text-blue-300" />
              </div>
              <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">AI Price Predictions</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                LSTM neural networks analyze historical data to forecast price movements with high accuracy
              </p>
            </div>

            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900">
                <BarChart3 className="h-6 w-6 text-green-600 dark:text-green-300" />
              </div>
              <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">Real-Time Analytics</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Live options chain analysis with Greeks calculation and volatility surface visualization
              </p>
            </div>

            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900">
                <Zap className="h-6 w-6 text-purple-600 dark:text-purple-300" />
              </div>
              <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">Automated Strategies</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Deploy sophisticated options strategies like Iron Condors and Butterflies automatically
              </p>
            </div>

            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-6 shadow-sm hover:shadow-lg transition-shadow">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-red-100 dark:bg-red-900">
                <Shield className="h-6 w-6 text-red-600 dark:text-red-300" />
              </div>
              <h3 className="mb-2 text-xl font-semibold text-gray-900 dark:text-white">Risk Management</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Advanced portfolio risk metrics including VaR, stress testing, and position sizing
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="border-y border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 py-16">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">10M+</div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">Options Analyzed Daily</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">95%</div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">Prediction Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">50ms</div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">Execution Speed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-600 dark:text-red-400">24/7</div>
              <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">Market Monitoring</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto px-4 text-center">
          <h2 className="mb-4 text-3xl font-bold text-gray-900 dark:text-white">Ready to Transform Your Trading?</h2>
          <p className="mb-8 text-lg text-gray-600 dark:text-gray-400">
            Join thousands of traders using AI to gain an edge in the options market
          </p>
          <Link href="/auth/register">
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white">
              Get Started Free
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </section>
    </main>
  )
}