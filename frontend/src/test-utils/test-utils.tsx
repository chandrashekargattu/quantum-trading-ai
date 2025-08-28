import React, { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'

// Create a custom render function that includes providers
const createTestQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
        staleTime: 0,
      },
    },
  })
}

interface AllTheProvidersProps {
  children: React.ReactNode
}

const AllTheProviders = ({ children }: AllTheProvidersProps) => {
  const queryClient = createTestQueryClient()

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options })

// Re-export everything
export * from '@testing-library/react'

// Override render method
export { customRender as render }

// Mock data generators
export const mockUser = {
  id: '123e4567-e89b-12d3-a456-426614174000',
  email: 'test@example.com',
  username: 'testuser',
  full_name: 'Test User',
  is_active: true,
  is_verified: true,
  account_type: 'paper',
  subscription_tier: 'free',
  created_at: '2024-01-01T00:00:00Z',
}

export const mockPortfolio = {
  id: '123e4567-e89b-12d3-a456-426614174001',
  name: 'Test Portfolio',
  total_value: 100000,
  cash_balance: 50000,
  buying_power: 50000,
  total_return: 5000,
  total_return_percent: 5,
  daily_return: 200,
  daily_return_percent: 0.2,
  is_active: true,
  is_default: true,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}

export const mockStock = {
  id: '123e4567-e89b-12d3-a456-426614174002',
  symbol: 'AAPL',
  name: 'Apple Inc.',
  current_price: 150.0,
  previous_close: 148.0,
  change_amount: 2.0,
  change_percent: 1.35,
  volume: 1000000,
  market_cap: 2500000000000,
  pe_ratio: 25.5,
  is_optionable: true,
  last_updated: '2024-01-01T00:00:00Z',
}

export const mockTrade = {
  id: '123e4567-e89b-12d3-a456-426614174003',
  trade_id: 'T20240101120000',
  symbol: 'AAPL',
  asset_type: 'stock',
  side: 'buy',
  quantity: 10,
  price: 150.0,
  total_amount: 1500.0,
  commission: 0,
  fees: 0,
  status: 'filled',
  is_paper: true,
  created_at: '2024-01-01T12:00:00Z',
}

// Test utilities
export const waitForLoadingToFinish = () =>
  waitFor(() => {
    const loadingElements = screen.queryAllByText(/loading/i)
    expect(loadingElements).toHaveLength(0)
  })

// Mock API responses
export const mockApiResponses = {
  success: <T,>(data: T) => ({
    json: async () => data,
    ok: true,
    status: 200,
  }),
  error: (status: number, message: string) => ({
    json: async () => ({ detail: message }),
    ok: false,
    status,
  }),
}

// Add necessary imports at the top
import { screen, waitFor } from '@testing-library/react'
