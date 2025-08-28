import { test, expect } from '@playwright/test'
import { loginUser, createTestUser } from './helpers/auth'
import { mockMarketData } from './helpers/mock-data'

test.describe('Trading E2E Tests', () => {
  let userEmail: string
  let userPassword: string

  test.beforeAll(async () => {
    // Create a test user for all trading tests
    const testUser = await createTestUser()
    userEmail = testUser.email
    userPassword = testUser.password
  })

  test.beforeEach(async ({ page }) => {
    // Login before each test
    await loginUser(page, userEmail, userPassword)
    await page.goto('/trading')
    
    // Mock market data API
    await page.route('**/api/v1/market/stocks/*', async route => {
      const symbol = route.request().url().split('/').pop()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockMarketData[symbol] || mockMarketData.AAPL)
      })
    })
  })

  test('should search and select a stock symbol', async ({ page }) => {
    // Search for stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    
    // Wait for stock data to load
    await expect(page.locator('[data-testid="stock-info"]')).toBeVisible()
    await expect(page.locator('text=Apple Inc.')).toBeVisible()
    await expect(page.locator('[data-testid="stock-price"]')).toContainText('$150')
    
    // Check real-time updates
    await expect(page.locator('[data-testid="price-change"]')).toContainText('+2.50')
    await expect(page.locator('[data-testid="price-change-percent"]')).toContainText('+1.69%')
  })

  test('should place a market buy order', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Fill order form
    await page.selectOption('[data-testid="order-type"]', 'MARKET')
    await page.selectOption('[data-testid="order-side"]', 'BUY')
    await page.fill('[data-testid="order-quantity"]', '100')
    
    // Mock order placement API
    await page.route('**/api/v1/orders', async route => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'order-123',
            symbol: 'AAPL',
            type: 'MARKET',
            side: 'BUY',
            quantity: 100,
            status: 'PENDING',
            createdAt: new Date().toISOString()
          })
        })
      }
    })
    
    // Place order
    await page.click('[data-testid="place-order-btn"]')
    
    // Should show confirmation
    await expect(page.locator('text=Order placed successfully')).toBeVisible()
    
    // Order should appear in open orders
    await expect(page.locator('[data-testid="open-orders"]')).toContainText('AAPL')
    await expect(page.locator('[data-testid="open-orders"]')).toContainText('BUY 100')
  })

  test('should place a limit sell order', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'GOOGL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Fill order form
    await page.selectOption('[data-testid="order-type"]', 'LIMIT')
    await page.selectOption('[data-testid="order-side"]', 'SELL')
    await page.fill('[data-testid="order-quantity"]', '50')
    await page.fill('[data-testid="order-price"]', '2600')
    
    // Place order
    await page.click('[data-testid="place-order-btn"]')
    
    // Verify order details
    await expect(page.locator('[data-testid="open-orders"]')).toContainText('SELL 50 @ $2600')
  })

  test('should place a stop-loss order', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'MSFT')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Fill stop-loss order
    await page.selectOption('[data-testid="order-type"]', 'STOP_LIMIT')
    await page.selectOption('[data-testid="order-side"]', 'SELL')
    await page.fill('[data-testid="order-quantity"]', '75')
    await page.fill('[data-testid="stop-price"]', '295')
    await page.fill('[data-testid="order-price"]', '294')
    
    // Place order
    await page.click('[data-testid="place-order-btn"]')
    
    // Verify stop order
    await expect(page.locator('[data-testid="open-orders"]')).toContainText('STOP_LIMIT')
    await expect(page.locator('[data-testid="open-orders"]')).toContainText('Stop: $295')
  })

  test('should cancel an open order', async ({ page }) => {
    // Place an order first
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    await page.selectOption('[data-testid="order-type"]', 'LIMIT')
    await page.fill('[data-testid="order-quantity"]', '100')
    await page.fill('[data-testid="order-price"]', '149')
    await page.click('[data-testid="place-order-btn"]')
    
    // Wait for order to appear
    await page.waitForSelector('[data-testid="order-row-order-123"]')
    
    // Mock cancel API
    await page.route('**/api/v1/orders/*/cancel', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Order cancelled' })
      })
    })
    
    // Cancel order
    await page.click('[data-testid="cancel-order-order-123"]')
    
    // Confirm cancellation
    await page.click('text=Yes, Cancel Order')
    
    // Order should be removed from open orders
    await expect(page.locator('[data-testid="order-row-order-123"]')).not.toBeVisible()
    await expect(page.locator('text=Order cancelled successfully')).toBeVisible()
  })

  test('should display order book depth', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    
    // Mock order book API
    await page.route('**/api/v1/market/orderbook/*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          symbol: 'AAPL',
          bids: [
            { price: 149.95, quantity: 1000 },
            { price: 149.90, quantity: 2000 },
            { price: 149.85, quantity: 1500 }
          ],
          asks: [
            { price: 150.00, quantity: 800 },
            { price: 150.05, quantity: 1200 },
            { price: 150.10, quantity: 900 }
          ]
        })
      })
    })
    
    // Show order book
    await page.click('[data-testid="toggle-orderbook"]')
    
    // Verify order book display
    await expect(page.locator('[data-testid="orderbook"]')).toBeVisible()
    await expect(page.locator('[data-testid="bid-149.95"]')).toContainText('1,000')
    await expect(page.locator('[data-testid="ask-150.00"]')).toContainText('800')
    
    // Check spread
    await expect(page.locator('[data-testid="spread"]')).toContainText('$0.05')
  })

  test('should handle quick order buttons', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Use quick buy button
    await page.click('[data-testid="quick-buy-100"]')
    
    // Should populate order form
    await expect(page.locator('[data-testid="order-side"]')).toHaveValue('BUY')
    await expect(page.locator('[data-testid="order-quantity"]')).toHaveValue('100')
    
    // Use quick sell button
    await page.click('[data-testid="quick-sell-100"]')
    
    // Should update order form
    await expect(page.locator('[data-testid="order-side"]')).toHaveValue('SELL')
  })

  test('should display positions and P&L', async ({ page }) => {
    // Mock positions API
    await page.route('**/api/v1/portfolios/*/positions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'pos-1',
            symbol: 'AAPL',
            quantity: 100,
            avgPrice: 145,
            currentPrice: 150,
            unrealizedPnL: 500,
            unrealizedPnLPercent: 3.45
          },
          {
            id: 'pos-2',
            symbol: 'GOOGL',
            quantity: 50,
            avgPrice: 2600,
            currentPrice: 2550,
            unrealizedPnL: -2500,
            unrealizedPnLPercent: -1.92
          }
        ])
      })
    })
    
    // Navigate to positions tab
    await page.click('[data-testid="positions-tab"]')
    
    // Verify positions display
    await expect(page.locator('[data-testid="position-AAPL"]')).toBeVisible()
    await expect(page.locator('[data-testid="position-AAPL"]')).toContainText('100 shares')
    await expect(page.locator('[data-testid="position-AAPL"]')).toContainText('+$500.00')
    await expect(page.locator('[data-testid="position-AAPL"] .profit')).toBeVisible()
    
    await expect(page.locator('[data-testid="position-GOOGL"]')).toContainText('-$2,500.00')
    await expect(page.locator('[data-testid="position-GOOGL"] .loss')).toBeVisible()
  })

  test('should close a position', async ({ page }) => {
    // Navigate to positions
    await page.click('[data-testid="positions-tab"]')
    await page.waitForSelector('[data-testid="position-AAPL"]')
    
    // Mock close position API
    await page.route('**/api/v1/portfolios/*/positions/*/close', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Position closed' })
      })
    })
    
    // Close position
    await page.click('[data-testid="close-position-AAPL"]')
    
    // Confirm
    await page.click('text=Yes, Close Position')
    
    // Position should be removed
    await expect(page.locator('[data-testid="position-AAPL"]')).not.toBeVisible()
    await expect(page.locator('text=Position closed successfully')).toBeVisible()
  })

  test('should handle real-time price updates', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    const initialPrice = await page.locator('[data-testid="stock-price"]').textContent()
    
    // Simulate WebSocket price update
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('ws-price-update', {
        detail: {
          symbol: 'AAPL',
          price: 151.50,
          change: 4.00,
          changePercent: 2.71
        }
      }))
    })
    
    // Price should update
    await expect(page.locator('[data-testid="stock-price"]')).not.toContainText(initialPrice!)
    await expect(page.locator('[data-testid="stock-price"]')).toContainText('$151.50')
    await expect(page.locator('[data-testid="price-change"]')).toContainText('+4.00')
  })

  test('should handle order execution notifications', async ({ page }) => {
    // Place an order
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    await page.selectOption('[data-testid="order-type"]', 'LIMIT')
    await page.fill('[data-testid="order-quantity"]', '100')
    await page.fill('[data-testid="order-price"]', '150')
    await page.click('[data-testid="place-order-btn"]')
    
    // Simulate order fill via WebSocket
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('ws-order-update', {
        detail: {
          orderId: 'order-123',
          status: 'FILLED',
          filledQuantity: 100,
          avgFillPrice: 150
        }
      }))
    })
    
    // Should show notification
    await expect(page.locator('text=Order Filled')).toBeVisible()
    await expect(page.locator('text=100 shares of AAPL at $150')).toBeVisible()
    
    // Order should move to order history
    await page.click('[data-testid="order-history-tab"]')
    await expect(page.locator('[data-testid="order-history"]')).toContainText('FILLED')
  })

  test('should validate order form inputs', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Try to place order without quantity
    await page.click('[data-testid="place-order-btn"]')
    await expect(page.locator('text=Quantity is required')).toBeVisible()
    
    // Invalid quantity
    await page.fill('[data-testid="order-quantity"]', '0')
    await page.click('[data-testid="place-order-btn"]')
    await expect(page.locator('text=Quantity must be greater than 0')).toBeVisible()
    
    // Limit order without price
    await page.selectOption('[data-testid="order-type"]', 'LIMIT')
    await page.fill('[data-testid="order-quantity"]', '100')
    await page.click('[data-testid="place-order-btn"]')
    await expect(page.locator('text=Price is required for limit orders')).toBeVisible()
  })

  test('should handle insufficient funds error', async ({ page }) => {
    // Select stock
    await page.fill('[data-testid="symbol-search"]', 'AAPL')
    await page.press('[data-testid="symbol-search"]', 'Enter')
    await page.waitForSelector('[data-testid="stock-info"]')
    
    // Try to place large order
    await page.fill('[data-testid="order-quantity"]', '10000')
    
    // Mock insufficient funds error
    await page.route('**/api/v1/orders', async route => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ 
          detail: 'Insufficient funds. Required: $1,500,000, Available: $50,000' 
        })
      })
    })
    
    await page.click('[data-testid="place-order-btn"]')
    
    // Should show error
    await expect(page.locator('text=Insufficient funds')).toBeVisible()
  })
})
