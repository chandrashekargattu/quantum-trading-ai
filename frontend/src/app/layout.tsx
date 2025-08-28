import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { ThemeToggle } from '@/components/theme-toggle'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Quantum Trading AI - Advanced Options Trading Platform',
  description: 'AI-powered options trading platform with real-time market analysis, ML predictions, and automated trading strategies',
  keywords: 'options trading, AI trading, machine learning, stock market, algorithmic trading',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className={inter.className}>
        <Providers>
          <ThemeToggle />
          {children}
        </Providers>
      </body>
    </html>
  )
}
