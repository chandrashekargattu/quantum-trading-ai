import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowRight, BarChart3, Brain, Zap, Shield } from 'lucide-react'

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary/20 via-background to-background">
        <div className="container relative z-10 mx-auto px-4 py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              AI-Powered Options Trading
              <span className="text-primary"> Reimagined</span>
            </h1>
            <p className="mb-8 text-lg text-muted-foreground sm:text-xl">
              Leverage advanced machine learning algorithms to analyze market patterns, 
              predict price movements, and execute sophisticated options strategies with confidence.
            </p>
            <div className="flex flex-col gap-4 sm:flex-row sm:justify-center">
              <Link href="/auth/register">
                <Button size="lg" className="w-full sm:w-auto">
                  Start Trading Now
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/demo">
                <Button size="lg" variant="outline" className="w-full sm:w-auto">
                  View Live Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:14px_24px]" />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-bold">Cutting-Edge Trading Features</h2>
            <p className="mx-auto max-w-2xl text-muted-foreground">
              Everything you need to trade options like a professional, powered by AI
            </p>
          </div>
          
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            <FeatureCard
              icon={<Brain className="h-10 w-10" />}
              title="AI Price Predictions"
              description="LSTM neural networks analyze historical data to forecast price movements with high accuracy"
            />
            <FeatureCard
              icon={<BarChart3 className="h-10 w-10" />}
              title="Real-Time Analytics"
              description="Live options chain analysis with Greeks calculation and volatility surface visualization"
            />
            <FeatureCard
              icon={<Zap className="h-10 w-10" />}
              title="Automated Strategies"
              description="Deploy sophisticated options strategies like Iron Condors and Butterflies automatically"
            />
            <FeatureCard
              icon={<Shield className="h-10 w-10" />}
              title="Risk Management"
              description="Advanced portfolio risk metrics including VaR, stress testing, and position sizing"
            />
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-muted/50 py-20">
        <div className="container mx-auto px-4">
          <div className="grid gap-8 text-center md:grid-cols-4">
            <StatCard number="10M+" label="Options Analyzed Daily" />
            <StatCard number="95%" label="Prediction Accuracy" />
            <StatCard number="50ms" label="Execution Speed" />
            <StatCard number="24/7" label="Market Monitoring" />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-4xl rounded-2xl bg-primary p-12 text-center text-primary-foreground">
            <h2 className="mb-4 text-3xl font-bold">Ready to Transform Your Trading?</h2>
            <p className="mb-8 text-lg opacity-90">
              Join thousands of traders using AI to make smarter options trading decisions
            </p>
            <Link href="/auth/register">
              <Button size="lg" variant="secondary">
                Get Started Free
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </main>
  )
}

function FeatureCard({ icon, title, description }: { 
  icon: React.ReactNode
  title: string
  description: string 
}) {
  return (
    <div className="group rounded-lg border bg-card p-6 transition-all hover:shadow-lg">
      <div className="mb-4 text-primary">{icon}</div>
      <h3 className="mb-2 text-xl font-semibold">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  )
}

function StatCard({ number, label }: { number: string; label: string }) {
  return (
    <div>
      <div className="text-4xl font-bold text-primary">{number}</div>
      <div className="mt-2 text-sm text-muted-foreground">{label}</div>
    </div>
  )
}

