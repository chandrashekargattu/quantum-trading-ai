import { test, expect } from '@playwright/test'
import { loginUser, createTestUser } from './helpers/auth'
import { mockPortfolio, mockPositions } from './helpers/mock-data'

test.describe('Portfolio Management E2E Tests', () => {
  let userEmail: string
  let userPassword: string

  test.beforeAll(async () => {
    const testUser = await createTestUser()
    userEmail = testUser.email
    userPassword = testUser.password
  })

  test.beforeEach(async ({ page }) => {
    await loginUser(page, userEmail, userPassword)
    await page.goto('/portfolio')
    
    // Mock portfolio API
    await page.route('**/api/v1/portfolios', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([mockPortfolio])
        })
      }
    })
  })

  test('should create a new portfolio', async ({ page }) => {
    // Mock create portfolio API
    await page.route('**/api/v1/portfolios', async route => {
      if (route.request().method() === 'POST') {
        const data = await route.request().postDataJSON()
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'portfolio-new',
            name: data.name,
            initialCapital: data.initialCapital,
            currentValue: data.initialCapital,
            cashBalance: data.initialCapital,
            totalReturn: 0,
            totalReturnPercent: 0,
            createdAt: new Date().toISOString()
          })
        })
      }
    })
    
    // Click create button
    await page.click('[data-testid="create-portfolio-btn"]')
    
    // Fill form
    await page.fill('input[name="portfolioName"]', 'Growth Portfolio')
    await page.fill('input[name="initialCapital"]', '50000')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Should show success message
    await expect(page.locator('text=Portfolio created successfully')).toBeVisible()
    
    // New portfolio should appear in list
    await expect(page.locator('text=Growth Portfolio')).toBeVisible()
  })

  test('should display portfolio overview', async ({ page }) => {
    // Mock positions API
    await page.route('**/api/v1/portfolios/*/positions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPositions)
      })
    })
    
    // Check portfolio summary
    await expect(page.locator('[data-testid="total-value"]')).toContainText('$125,000')
    await expect(page.locator('[data-testid="total-return"]')).toContainText('+$25,000')
    await expect(page.locator('[data-testid="return-percent"]')).toContainText('+25.00%')
    await expect(page.locator('[data-testid="cash-balance"]')).toContainText('$50,000')
    
    // Check positions
    await expect(page.locator('[data-testid="position-AAPL"]')).toBeVisible()
    await expect(page.locator('[data-testid="position-GOOGL"]')).toBeVisible()
  })

  test('should add funds to portfolio', async ({ page }) => {
    // Mock add funds API
    await page.route('**/api/v1/portfolios/*/deposit', async route => {
      const data = await route.request().postDataJSON()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'trans-1',
          type: 'DEPOSIT',
          amount: data.amount,
          timestamp: new Date().toISOString()
        })
      })
    })
    
    // Click add funds
    await page.click('[data-testid="add-funds-btn"]')
    
    // Enter amount
    await page.fill('input[name="amount"]', '10000')
    await page.click('button[type="submit"]')
    
    // Should show success
    await expect(page.locator('text=Funds added successfully')).toBeVisible()
    
    // Cash balance should update
    await expect(page.locator('[data-testid="cash-balance"]')).toContainText('$60,000')
  })

  test('should withdraw funds from portfolio', async ({ page }) => {
    // Mock withdraw API
    await page.route('**/api/v1/portfolios/*/withdraw', async route => {
      const data = await route.request().postDataJSON()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'trans-2',
          type: 'WITHDRAWAL',
          amount: data.amount,
          timestamp: new Date().toISOString()
        })
      })
    })
    
    // Click withdraw funds
    await page.click('[data-testid="withdraw-funds-btn"]')
    
    // Enter amount
    await page.fill('input[name="amount"]', '5000')
    await page.click('button[type="submit"]')
    
    // Should validate available balance
    await expect(page.locator('text=Withdrawal processed successfully')).toBeVisible()
  })

  test('should display position details', async ({ page }) => {
    // Mock positions
    await page.route('**/api/v1/portfolios/*/positions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPositions)
      })
    })
    
    // Click on position
    await page.click('[data-testid="position-AAPL"]')
    
    // Should show position details modal
    await expect(page.locator('[data-testid="position-details-modal"]')).toBeVisible()
    await expect(page.locator('[data-testid="position-symbol"]')).toContainText('AAPL')
    await expect(page.locator('[data-testid="position-quantity"]')).toContainText('100 shares')
    await expect(page.locator('[data-testid="position-avg-cost"]')).toContainText('$145.00')
    await expect(page.locator('[data-testid="position-current-price"]')).toContainText('$150.00')
    await expect(page.locator('[data-testid="position-pnl"]')).toContainText('+$500.00')
  })

  test('should display performance chart', async ({ page }) => {
    // Mock performance API
    await page.route('**/api/v1/portfolios/*/performance', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          chartData: {
            timestamps: ['2024-01-01', '2024-01-02', '2024-01-03'],
            values: [100000, 102000, 105000]
          },
          totalReturn: 5000,
          sharpeRatio: 1.5,
          maxDrawdown: -2000
        })
      })
    })
    
    // Switch to performance tab
    await page.click('[data-testid="performance-tab"]')
    
    // Check chart is visible
    await expect(page.locator('[data-testid="performance-chart"]')).toBeVisible()
    
    // Check time period selector
    await page.click('[data-testid="period-1M"]')
    await expect(page.locator('[data-testid="period-1M"]')).toHaveClass(/active/)
    
    // Check metrics
    await expect(page.locator('[data-testid="sharpe-ratio"]')).toContainText('1.50')
    await expect(page.locator('[data-testid="max-drawdown"]')).toContainText('-$2,000')
  })

  test('should display transaction history', async ({ page }) => {
    // Mock transactions API
    await page.route('**/api/v1/portfolios/*/transactions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'tx-1',
            type: 'BUY',
            symbol: 'AAPL',
            quantity: 100,
            price: 145,
            amount: 14500,
            timestamp: new Date().toISOString()
          },
          {
            id: 'tx-2',
            type: 'DEPOSIT',
            amount: 10000,
            timestamp: new Date().toISOString()
          }
        ])
      })
    })
    
    // Switch to transactions tab
    await page.click('[data-testid="transactions-tab"]')
    
    // Check transactions display
    await expect(page.locator('[data-testid="transaction-tx-1"]')).toBeVisible()
    await expect(page.locator('[data-testid="transaction-tx-1"]')).toContainText('BUY')
    await expect(page.locator('[data-testid="transaction-tx-1"]')).toContainText('AAPL')
    await expect(page.locator('[data-testid="transaction-tx-1"]')).toContainText('$14,500')
    
    await expect(page.locator('[data-testid="transaction-tx-2"]')).toContainText('DEPOSIT')
    await expect(page.locator('[data-testid="transaction-tx-2"]')).toContainText('$10,000')
  })

  test('should delete portfolio with confirmation', async ({ page }) => {
    // Create a test portfolio first
    await page.route('**/api/v1/portfolios', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            mockPortfolio,
            { ...mockPortfolio, id: 'portfolio-test', name: 'Test Portfolio' }
          ])
        })
      }
    })
    
    // Mock delete API
    await page.route('**/api/v1/portfolios/portfolio-test', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 204
        })
      }
    })
    
    // Select test portfolio
    await page.selectOption('[data-testid="portfolio-selector"]', 'portfolio-test')
    
    // Click delete
    await page.click('[data-testid="delete-portfolio-btn"]')
    
    // Confirm dialog
    await expect(page.locator('text=Are you sure you want to delete this portfolio?')).toBeVisible()
    await page.click('text=Yes, Delete')
    
    // Should show success
    await expect(page.locator('text=Portfolio deleted successfully')).toBeVisible()
    
    // Portfolio should be removed from list
    await expect(page.locator('option[value="portfolio-test"]')).not.toBeVisible()
  })

  test('should handle portfolio switching', async ({ page }) => {
    // Mock multiple portfolios
    const portfolio2 = {
      ...mockPortfolio,
      id: 'portfolio-2',
      name: 'Retirement Portfolio',
      currentValue: 500000
    }
    
    await page.route('**/api/v1/portfolios', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([mockPortfolio, portfolio2])
      })
    })
    
    // Switch portfolio
    await page.selectOption('[data-testid="portfolio-selector"]', 'portfolio-2')
    
    // Should update display
    await expect(page.locator('[data-testid="portfolio-name"]')).toContainText('Retirement Portfolio')
    await expect(page.locator('[data-testid="total-value"]')).toContainText('$500,000')
  })

  test('should calculate and display allocation chart', async ({ page }) => {
    // Mock positions with allocation data
    await page.route('**/api/v1/portfolios/*/positions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPositions)
      })
    })
    
    // Navigate to allocation tab
    await page.click('[data-testid="allocation-tab"]')
    
    // Check pie chart is visible
    await expect(page.locator('[data-testid="allocation-chart"]')).toBeVisible()
    
    // Check allocation percentages
    await expect(page.locator('[data-testid="allocation-AAPL"]')).toContainText('37%') // 15k/40.5k
    await expect(page.locator('[data-testid="allocation-GOOGL"]')).toContainText('63%') // 25.5k/40.5k
    
    // Check sector allocation
    await page.click('[data-testid="view-sectors"]')
    await expect(page.locator('[data-testid="sector-technology"]')).toContainText('100%')
  })

  test('should handle real-time portfolio updates', async ({ page }) => {
    // Initial load
    await page.route('**/api/v1/portfolios/*/positions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPositions)
      })
    })
    
    const initialValue = await page.locator('[data-testid="total-value"]').textContent()
    
    // Simulate WebSocket portfolio update
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('ws-portfolio-update', {
        detail: {
          portfolioId: 'portfolio-1',
          currentValue: 130000,
          dayChange: 5000,
          dayChangePercent: 4
        }
      }))
    })
    
    // Values should update
    await expect(page.locator('[data-testid="total-value"]')).not.toContainText(initialValue!)
    await expect(page.locator('[data-testid="total-value"]')).toContainText('$130,000')
    await expect(page.locator('[data-testid="day-change"]')).toContainText('+$5,000')
  })
})
