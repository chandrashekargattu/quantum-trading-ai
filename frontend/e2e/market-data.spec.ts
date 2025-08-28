import { test, expect } from '@playwright/test'
import { loginUser, createTestUser } from './helpers/auth'

test.describe('Market Data E2E Tests', () => {
  let userEmail: string
  let userPassword: string

  test.beforeAll(async () => {
    const testUser = await createTestUser()
    userEmail = testUser.email
    userPassword = testUser.password
  })

  test.beforeEach(async ({ page }) => {
    await loginUser(page, userEmail, userPassword)
    await page.goto('/market')
    
    // Mock market status API
    await page.route('**/api/v1/market/status', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          marketOpen: true,
          nextOpen: '2024-01-02T09:30:00-05:00',
          nextClose: '2024-01-02T16:00:00-05:00'
        })
      })
    })
  })

  test('should display live market quotes', async ({ page }) => {
    // Mock market quotes API
    await page.route('**/api/v1/market/quotes/batch', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          AAPL: {
            symbol: 'AAPL',
            price: 195.50,
            change: 2.50,
            changePercent: 1.29,
            volume: 45678900,
            high: 196.80,
            low: 193.20,
            previousClose: 193.00
          },
          GOOGL: {
            symbol: 'GOOGL',
            price: 145.30,
            change: -1.20,
            changePercent: -0.82,
            volume: 23456789,
            high: 147.50,
            low: 144.80,
            previousClose: 146.50
          }
        })
      })
    })
    
    // Should display market quotes
    await expect(page.locator('[data-testid="quote-AAPL"]')).toBeVisible()
    await expect(page.locator('[data-testid="quote-AAPL-price"]')).toContainText('$195.50')
    await expect(page.locator('[data-testid="quote-AAPL-change"]')).toContainText('+$2.50')
    await expect(page.locator('[data-testid="quote-AAPL-percent"]')).toContainText('+1.29%')
    
    // Check negative change styling
    await expect(page.locator('[data-testid="quote-GOOGL-change"]')).toHaveClass(/text-red/)
  })

  test('should search for stocks', async ({ page }) => {
    // Mock search API
    await page.route('**/api/v1/market/search*', async route => {
      if (route.request().url().includes('query=APP')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { symbol: 'AAPL', name: 'Apple Inc.', type: 'stock', exchange: 'NASDAQ' },
            { symbol: 'APP', name: 'AppLovin Corp.', type: 'stock', exchange: 'NASDAQ' },
            { symbol: 'APPS', name: 'Digital Turbine Inc.', type: 'stock', exchange: 'NASDAQ' }
          ])
        })
      }
    })
    
    // Type in search
    await page.fill('[data-testid="stock-search"]', 'APP')
    
    // Should show suggestions
    await expect(page.locator('[data-testid="search-suggestion-AAPL"]')).toBeVisible()
    await expect(page.locator('[data-testid="search-suggestion-APP"]')).toBeVisible()
    await expect(page.locator('[data-testid="search-suggestion-APPS"]')).toBeVisible()
    
    // Click on Apple
    await page.click('[data-testid="search-suggestion-AAPL"]')
    
    // Should navigate to stock detail
    await expect(page).toHaveURL(/.*\/market\/AAPL/)
  })

  test('should display stock details', async ({ page }) => {
    // Navigate to stock detail
    await page.goto('/market/AAPL')
    
    // Mock stock detail API
    await page.route('**/api/v1/market/stocks/AAPL', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          symbol: 'AAPL',
          name: 'Apple Inc.',
          price: 195.50,
          change: 2.50,
          changePercent: 1.29,
          volume: 45678900,
          marketCap: 3045000000000,
          pe: 32.5,
          eps: 6.01,
          beta: 1.25,
          week52High: 199.62,
          week52Low: 164.08,
          avgVolume: 54321000,
          dividend: 0.96,
          dividendYield: 0.49
        })
      })
    })
    
    // Check company info
    await expect(page.locator('[data-testid="company-name"]')).toContainText('Apple Inc.')
    await expect(page.locator('[data-testid="stock-price"]')).toContainText('$195.50')
    
    // Check key stats
    await expect(page.locator('[data-testid="market-cap"]')).toContainText('$3.05T')
    await expect(page.locator('[data-testid="pe-ratio"]')).toContainText('32.5')
    await expect(page.locator('[data-testid="52-week-range"]')).toContainText('$164.08 - $199.62')
  })

  test('should display interactive price chart', async ({ page }) => {
    await page.goto('/market/AAPL')
    
    // Mock historical data API
    await page.route('**/api/v1/market/stocks/AAPL/history*', async route => {
      const url = route.request().url()
      const period = url.includes('period=1D') ? '1D' : '1M'
      
      let data = []
      if (period === '1D') {
        // Intraday data
        for (let i = 0; i < 78; i++) { // 9:30 AM to 4:00 PM every 5 minutes
          const time = new Date()
          time.setHours(9, 30, 0, 0)
          time.setMinutes(time.getMinutes() + i * 5)
          data.push({
            timestamp: time.toISOString(),
            open: 193 + Math.random() * 4,
            high: 194 + Math.random() * 4,
            low: 192 + Math.random() * 4,
            close: 193 + Math.random() * 4,
            volume: Math.floor(Math.random() * 1000000)
          })
        }
      } else {
        // Daily data for 1 month
        for (let i = 30; i >= 0; i--) {
          const date = new Date()
          date.setDate(date.getDate() - i)
          data.push({
            timestamp: date.toISOString(),
            open: 190 + Math.random() * 10,
            high: 192 + Math.random() * 10,
            low: 188 + Math.random() * 10,
            close: 190 + Math.random() * 10,
            volume: Math.floor(40000000 + Math.random() * 20000000)
          })
        }
      }
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(data)
      })
    })
    
    // Chart should be visible
    await expect(page.locator('[data-testid="price-chart"]')).toBeVisible()
    
    // Change time period
    await page.click('[data-testid="chart-period-1M"]')
    await expect(page.locator('[data-testid="chart-period-1M"]')).toHaveClass(/active/)
    
    // Change chart type
    await page.click('[data-testid="chart-type-menu"]')
    await page.click('[data-testid="chart-type-candlestick"]')
    
    // Add indicator
    await page.click('[data-testid="add-indicator-btn"]')
    await page.click('[data-testid="indicator-sma"]')
    await expect(page.locator('[data-testid="indicator-sma-line"]')).toBeVisible()
  })

  test('should display options chain', async ({ page }) => {
    await page.goto('/market/AAPL/options')
    
    // Mock options chain API
    await page.route('**/api/v1/options/chain/AAPL*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          symbol: 'AAPL',
          expirations: ['2024-01-19', '2024-01-26', '2024-02-16'],
          strikes: [185, 190, 195, 200, 205],
          chain: {
            '2024-01-19': {
              '195': {
                call: {
                  strike: 195,
                  expiration: '2024-01-19',
                  bid: 3.20,
                  ask: 3.25,
                  last: 3.22,
                  volume: 1234,
                  openInterest: 5678,
                  impliedVolatility: 0.25,
                  delta: 0.55,
                  gamma: 0.02,
                  theta: -0.08,
                  vega: 0.15
                },
                put: {
                  strike: 195,
                  expiration: '2024-01-19',
                  bid: 2.80,
                  ask: 2.85,
                  last: 2.82,
                  volume: 987,
                  openInterest: 4321,
                  impliedVolatility: 0.24,
                  delta: -0.45,
                  gamma: 0.02,
                  theta: -0.07,
                  vega: 0.14
                }
              }
            }
          }
        })
      })
    })
    
    // Options chain should load
    await expect(page.locator('[data-testid="options-chain"]')).toBeVisible()
    
    // Select expiration
    await page.selectOption('[data-testid="expiration-select"]', '2024-01-19')
    
    // Check option display
    await expect(page.locator('[data-testid="call-195"]')).toBeVisible()
    await expect(page.locator('[data-testid="call-195-bid"]')).toContainText('$3.20')
    await expect(page.locator('[data-testid="call-195-ask"]')).toContainText('$3.25')
    await expect(page.locator('[data-testid="call-195-iv"]')).toContainText('25.0%')
    
    // Click to view Greeks
    await page.click('[data-testid="call-195-greeks-btn"]')
    await expect(page.locator('[data-testid="greeks-modal"]')).toBeVisible()
    await expect(page.locator('[data-testid="delta-value"]')).toContainText('0.55')
  })

  test('should stream real-time prices via WebSocket', async ({ page }) => {
    await page.goto('/market/watchlist')
    
    // Add stocks to watchlist
    await page.click('[data-testid="add-to-watchlist-btn"]')
    await page.fill('[data-testid="add-symbol-input"]', 'AAPL')
    await page.click('[data-testid="add-symbol-submit"]')
    
    // Mock WebSocket connection
    await page.evaluate(() => {
      // Override WebSocket
      (window as any).WebSocket = class MockWebSocket {
        onopen: any
        onmessage: any
        onclose: any
        onerror: any
        
        constructor(url: string) {
          setTimeout(() => {
            if (this.onopen) this.onopen({ type: 'open' })
            
            // Simulate price updates
            let price = 195.50
            setInterval(() => {
              price += (Math.random() - 0.5) * 0.5
              if (this.onmessage) {
                this.onmessage({
                  type: 'message',
                  data: JSON.stringify({
                    type: 'quote',
                    data: {
                      symbol: 'AAPL',
                      price: price,
                      bid: price - 0.01,
                      ask: price + 0.01,
                      volume: Math.floor(Math.random() * 1000000)
                    }
                  })
                })
              }
            }, 1000)
          }, 100)
        }
        
        send(data: string) {
          const msg = JSON.parse(data)
          if (msg.type === 'subscribe') {
            console.log('Subscribed to:', msg.symbols)
          }
        }
        
        close() {
          if (this.onclose) this.onclose({ type: 'close' })
        }
      }
    })
    
    // Check real-time updates
    await expect(page.locator('[data-testid="realtime-indicator"]')).toBeVisible()
    
    // Price should update
    const initialPrice = await page.locator('[data-testid="AAPL-price"]').textContent()
    await page.waitForTimeout(2000)
    const updatedPrice = await page.locator('[data-testid="AAPL-price"]').textContent()
    expect(initialPrice).not.toBe(updatedPrice)
  })

  test('should display market movers', async ({ page }) => {
    // Mock market movers API
    await page.route('**/api/v1/market/movers', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          gainers: [
            { symbol: 'NVDA', name: 'NVIDIA Corp', change: 12.50, changePercent: 2.85, price: 450.00 },
            { symbol: 'AMD', name: 'AMD', change: 5.20, changePercent: 2.45, price: 217.80 },
            { symbol: 'TSLA', name: 'Tesla Inc', change: 8.90, changePercent: 2.10, price: 432.50 }
          ],
          losers: [
            { symbol: 'META', name: 'Meta Platforms', change: -8.30, changePercent: -2.20, price: 369.20 },
            { symbol: 'AMZN', name: 'Amazon', change: -3.50, changePercent: -2.00, price: 171.50 },
            { symbol: 'NFLX', name: 'Netflix', change: -7.80, changePercent: -1.85, price: 413.70 }
          ],
          mostActive: [
            { symbol: 'AAPL', name: 'Apple Inc', volume: 85000000, price: 195.50 },
            { symbol: 'SPY', name: 'SPDR S&P 500', volume: 78000000, price: 475.30 },
            { symbol: 'TSLA', name: 'Tesla Inc', volume: 65000000, price: 432.50 }
          ]
        })
      })
    })
    
    await page.goto('/market/movers')
    
    // Check gainers
    await expect(page.locator('[data-testid="gainers-section"]')).toBeVisible()
    await expect(page.locator('[data-testid="gainer-NVDA"]')).toBeVisible()
    await expect(page.locator('[data-testid="gainer-NVDA-percent"]')).toContainText('+2.85%')
    
    // Check losers
    await expect(page.locator('[data-testid="losers-section"]')).toBeVisible()
    await expect(page.locator('[data-testid="loser-META"]')).toBeVisible()
    await expect(page.locator('[data-testid="loser-META-percent"]')).toContainText('-2.20%')
    
    // Check most active
    await page.click('[data-testid="tab-most-active"]')
    await expect(page.locator('[data-testid="active-AAPL"]')).toBeVisible()
    await expect(page.locator('[data-testid="active-AAPL-volume"]')).toContainText('85M')
  })

  test('should display market indicators', async ({ page }) => {
    // Mock market indicators API
    await page.route('**/api/v1/market/indicators', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          vix: { value: 15.32, change: -0.45, changePercent: -2.85 },
          dxy: { value: 103.45, change: 0.23, changePercent: 0.22 },
          yields: {
            '2Y': { value: 4.85, change: 0.02, changePercent: 0.41 },
            '10Y': { value: 4.32, change: -0.03, changePercent: -0.69 },
            '30Y': { value: 4.45, change: -0.05, changePercent: -1.11 }
          },
          commodities: {
            gold: { value: 2045.30, change: 12.50, changePercent: 0.61 },
            oil: { value: 75.85, change: -1.20, changePercent: -1.56 }
          }
        })
      })
    })
    
    await page.goto('/market/indicators')
    
    // Check VIX
    await expect(page.locator('[data-testid="vix-value"]')).toContainText('15.32')
    await expect(page.locator('[data-testid="vix-change"]')).toContainText('-2.85%')
    
    // Check yields
    await expect(page.locator('[data-testid="yield-10Y"]')).toContainText('4.32%')
    
    // Check commodities
    await expect(page.locator('[data-testid="gold-price"]')).toContainText('$2,045.30')
    await expect(page.locator('[data-testid="oil-price"]')).toContainText('$75.85')
  })

  test('should handle market data errors gracefully', async ({ page }) => {
    // Mock error response
    await page.route('**/api/v1/market/quotes/INVALID', async route => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Symbol not found'
        })
      })
    })
    
    // Search for invalid symbol
    await page.fill('[data-testid="stock-search"]', 'INVALID')
    await page.keyboard.press('Enter')
    
    // Should show error message
    await expect(page.locator('text=Symbol not found')).toBeVisible()
  })

  test('should export market data', async ({ page }) => {
    await page.goto('/market/AAPL')
    
    // Mock export API
    await page.route('**/api/v1/market/stocks/AAPL/export*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'text/csv',
        body: 'Date,Open,High,Low,Close,Volume\n2024-01-01,193.00,196.80,192.50,195.50,45678900'
      })
    })
    
    // Click export
    const downloadPromise = page.waitForEvent('download')
    await page.click('[data-testid="export-data-btn"]')
    await page.click('[data-testid="export-csv"]')
    
    const download = await downloadPromise
    expect(download.suggestedFilename()).toContain('AAPL')
    expect(download.suggestedFilename()).toContain('.csv')
  })
})
