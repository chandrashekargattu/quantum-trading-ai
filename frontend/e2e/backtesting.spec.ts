import { test, expect } from '@playwright/test'
import { loginUser, createTestUser } from './helpers/auth'
import { mockBacktestConfig, mockBacktestResult } from './helpers/mock-data'

test.describe('Backtesting E2E Tests', () => {
  let userEmail: string
  let userPassword: string

  test.beforeAll(async () => {
    const testUser = await createTestUser()
    userEmail = testUser.email
    userPassword = testUser.password
  })

  test.beforeEach(async ({ page }) => {
    await loginUser(page, userEmail, userPassword)
    await page.goto('/backtesting')
    
    // Mock backtest APIs
    await page.route('**/api/v1/backtest/configs', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([mockBacktestConfig])
        })
      }
    })
    
    await page.route('**/api/v1/backtest/results', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([mockBacktestResult])
      })
    })
  })

  test('should create a backtest configuration', async ({ page }) => {
    // Mock create config API
    await page.route('**/api/v1/backtest/configs', async route => {
      if (route.request().method() === 'POST') {
        const data = await route.request().postDataJSON()
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'config-new',
            ...data,
            createdAt: new Date().toISOString()
          })
        })
      }
    })
    
    // Click create configuration
    await page.click('[data-testid="create-config-btn"]')
    
    // Fill configuration form
    await page.fill('input[name="configName"]', 'RSI Mean Reversion')
    await page.selectOption('select[name="strategy"]', 'rsi')
    
    // Add symbols
    await page.fill('input[name="symbols"]', 'AAPL,GOOGL,MSFT')
    
    // Set date range
    await page.fill('input[name="startDate"]', '2023-01-01')
    await page.fill('input[name="endDate"]', '2023-12-31')
    
    // Set capital and position sizing
    await page.fill('input[name="initialCapital"]', '50000')
    await page.fill('input[name="positionSize"]', '5000')
    await page.fill('input[name="maxPositions"]', '5')
    
    // Set strategy parameters
    await page.fill('input[name="rsiPeriod"]', '14')
    await page.fill('input[name="oversoldLevel"]', '30')
    await page.fill('input[name="overboughtLevel"]', '70')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Should show success
    await expect(page.locator('text=Configuration created successfully')).toBeVisible()
    
    // New config should appear in list
    await expect(page.locator('text=RSI Mean Reversion')).toBeVisible()
  })

  test('should run a backtest', async ({ page }) => {
    // Select configuration
    await page.selectOption('[data-testid="config-selector"]', 'config-1')
    
    // Mock run backtest API
    await page.route('**/api/v1/backtest/run/*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          backtestId: 'backtest-123'
        })
      })
    })
    
    // Mock status polling
    let pollCount = 0
    await page.route('**/api/v1/backtest/status/*', async route => {
      pollCount++
      const progress = Math.min(pollCount * 20, 100)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'backtest-123',
          status: progress === 100 ? 'COMPLETED' : 'RUNNING',
          progress,
          message: `Processing... ${progress}%`,
          currentDate: '2023-06-15'
        })
      })
    })
    
    // Click run button
    await page.click('[data-testid="run-backtest-btn"]')
    
    // Should show progress
    await expect(page.locator('[data-testid="backtest-progress"]')).toBeVisible()
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible()
    
    // Wait for completion
    await expect(page.locator('text=Backtest completed')).toBeVisible({ timeout: 10000 })
  })

  test('should display backtest results', async ({ page }) => {
    // Click on result
    await page.click('[data-testid="result-result-1"]')
    
    // Should show result details
    await expect(page.locator('[data-testid="result-details"]')).toBeVisible()
    
    // Check performance metrics
    await expect(page.locator('[data-testid="total-return"]')).toContainText('+$15,000')
    await expect(page.locator('[data-testid="return-percent"]')).toContainText('+15.00%')
    await expect(page.locator('[data-testid="sharpe-ratio"]')).toContainText('1.80')
    await expect(page.locator('[data-testid="max-drawdown"]')).toContainText('-5.00%')
    await expect(page.locator('[data-testid="win-rate"]')).toContainText('65.33%')
    
    // Check trade statistics
    await expect(page.locator('[data-testid="total-trades"]')).toContainText('150')
    await expect(page.locator('[data-testid="winning-trades"]')).toContainText('98')
    await expect(page.locator('[data-testid="losing-trades"]')).toContainText('52')
    await expect(page.locator('[data-testid="profit-factor"]')).toContainText('2.20')
  })

  test('should display equity curve chart', async ({ page }) => {
    // Mock detailed result with equity data
    await page.route('**/api/v1/backtest/results/*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...mockBacktestResult,
          equity: [100000, 102000, 98000, 105000, 110000, 115000],
          timestamps: [
            '2023-01-01', '2023-02-01', '2023-03-01',
            '2023-04-01', '2023-05-01', '2023-06-01'
          ]
        })
      })
    })
    
    // View result
    await page.click('[data-testid="result-result-1"]')
    
    // Navigate to charts tab
    await page.click('[data-testid="charts-tab"]')
    
    // Check equity curve
    await expect(page.locator('[data-testid="equity-curve-chart"]')).toBeVisible()
    
    // Check drawdown chart
    await expect(page.locator('[data-testid="drawdown-chart"]')).toBeVisible()
    
    // Check returns distribution
    await expect(page.locator('[data-testid="returns-histogram"]')).toBeVisible()
  })

  test('should display trade history', async ({ page }) => {
    // Mock trades data
    await page.route('**/api/v1/backtest/results/*/trades', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 'trade-1',
            symbol: 'AAPL',
            entryDate: '2023-01-15',
            exitDate: '2023-01-20',
            side: 'LONG',
            quantity: 100,
            entryPrice: 145,
            exitPrice: 150,
            pnl: 500,
            pnlPercent: 3.45,
            duration: 5
          },
          {
            id: 'trade-2',
            symbol: 'GOOGL',
            entryDate: '2023-02-01',
            exitDate: '2023-02-05',
            side: 'SHORT',
            quantity: 50,
            entryPrice: 2600,
            exitPrice: 2550,
            pnl: 2500,
            pnlPercent: 1.92,
            duration: 4
          }
        ])
      })
    })
    
    // View result
    await page.click('[data-testid="result-result-1"]')
    
    // Navigate to trades tab
    await page.click('[data-testid="trades-tab"]')
    
    // Check trades display
    await expect(page.locator('[data-testid="trade-trade-1"]')).toBeVisible()
    await expect(page.locator('[data-testid="trade-trade-1"]')).toContainText('AAPL')
    await expect(page.locator('[data-testid="trade-trade-1"]')).toContainText('LONG')
    await expect(page.locator('[data-testid="trade-trade-1"]')).toContainText('+$500')
    
    // Filter trades
    await page.selectOption('[data-testid="trade-filter"]', 'winners')
    await expect(page.locator('[data-testid="trade-trade-1"]')).toBeVisible()
    
    // Sort trades
    await page.click('[data-testid="sort-by-pnl"]')
    await expect(page.locator('[data-testid="trades-list"] tr').first()).toContainText('$2,500')
  })

  test('should compare multiple backtest results', async ({ page }) => {
    // Mock multiple results
    const result2 = {
      ...mockBacktestResult,
      id: 'result-2',
      configId: 'config-2',
      totalReturn: -5000,
      totalReturnPercent: -5,
      sharpeRatio: -0.5
    }
    
    await page.route('**/api/v1/backtest/results', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([mockBacktestResult, result2])
      })
    })
    
    // Enter comparison mode
    await page.click('[data-testid="compare-btn"]')
    
    // Select results to compare
    await page.check('[data-testid="compare-result-1"]')
    await page.check('[data-testid="compare-result-2"]')
    
    // View comparison
    await page.click('[data-testid="view-comparison-btn"]')
    
    // Should show comparison view
    await expect(page.locator('[data-testid="comparison-view"]')).toBeVisible()
    
    // Check comparison metrics
    await expect(page.locator('[data-testid="compare-return-result-1"]')).toContainText('+15.00%')
    await expect(page.locator('[data-testid="compare-return-result-2"]')).toContainText('-5.00%')
    
    // Check comparison chart
    await expect(page.locator('[data-testid="comparison-chart"]')).toBeVisible()
  })

  test('should optimize strategy parameters', async ({ page }) => {
    // Select configuration
    await page.selectOption('[data-testid="config-selector"]', 'config-1')
    
    // Click optimize button
    await page.click('[data-testid="optimize-btn"]')
    
    // Set parameter ranges
    await page.fill('input[name="fastPeriod.min"]', '5')
    await page.fill('input[name="fastPeriod.max"]', '20')
    await page.fill('input[name="fastPeriod.step"]', '5')
    
    await page.fill('input[name="slowPeriod.min"]', '20')
    await page.fill('input[name="slowPeriod.max"]', '50')
    await page.fill('input[name="slowPeriod.step"]', '10')
    
    // Mock optimization API
    await page.route('**/api/v1/backtest/optimize/*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          optimizationId: 'opt-123'
        })
      })
    })
    
    // Mock optimization results
    await page.route('**/api/v1/backtest/optimize/*/results', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'COMPLETED',
          results: [
            { parameters: { fastPeriod: 10, slowPeriod: 30 }, sharpeRatio: 1.8, totalReturn: 15000 },
            { parameters: { fastPeriod: 15, slowPeriod: 40 }, sharpeRatio: 1.5, totalReturn: 12000 },
            { parameters: { fastPeriod: 5, slowPeriod: 20 }, sharpeRatio: 1.2, totalReturn: 10000 }
          ]
        })
      })
    })
    
    // Start optimization
    await page.click('[data-testid="start-optimization-btn"]')
    
    // Should show results
    await expect(page.locator('[data-testid="optimization-results"]')).toBeVisible({ timeout: 10000 })
    
    // Best parameters should be highlighted
    await expect(page.locator('[data-testid="best-params"]')).toContainText('Fast: 10, Slow: 30')
    await expect(page.locator('[data-testid="best-sharpe"]')).toContainText('1.80')
  })

  test('should export backtest results', async ({ page }) => {
    // View result
    await page.click('[data-testid="result-result-1"]')
    
    // Mock export API
    await page.route('**/api/v1/backtest/results/*/export', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'text/csv',
        body: 'Date,Symbol,Side,Entry,Exit,PnL\n2023-01-15,AAPL,LONG,145,150,500'
      })
    })
    
    // Click export
    const downloadPromise = page.waitForEvent('download')
    await page.click('[data-testid="export-results-btn"]')
    await page.click('[data-testid="export-csv"]')
    
    const download = await downloadPromise
    expect(download.suggestedFilename()).toContain('backtest-results')
    expect(download.suggestedFilename()).toContain('.csv')
  })

  test('should delete backtest configuration', async ({ page }) => {
    // Mock delete API
    await page.route('**/api/v1/backtest/configs/config-1', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 204
        })
      }
    })
    
    // Select config
    await page.selectOption('[data-testid="config-selector"]', 'config-1')
    
    // Click delete
    await page.click('[data-testid="delete-config-btn"]')
    
    // Confirm
    await page.click('text=Yes, Delete')
    
    // Should be removed
    await expect(page.locator('option[value="config-1"]')).not.toBeVisible()
    await expect(page.locator('text=Configuration deleted successfully')).toBeVisible()
  })

  test('should handle backtest errors', async ({ page }) => {
    // Select configuration
    await page.selectOption('[data-testid="config-selector"]', 'config-1')
    
    // Mock error response
    await page.route('**/api/v1/backtest/run/*', async route => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Insufficient historical data for the selected period'
        })
      })
    })
    
    // Try to run backtest
    await page.click('[data-testid="run-backtest-btn"]')
    
    // Should show error
    await expect(page.locator('text=Insufficient historical data')).toBeVisible()
  })

  test('should save and load backtest templates', async ({ page }) => {
    // Create configuration
    await page.click('[data-testid="create-config-btn"]')
    
    // Fill form
    await page.fill('input[name="configName"]', 'Template Strategy')
    await page.selectOption('select[name="strategy"]', 'ma-crossover')
    await page.fill('input[name="symbols"]', 'SPY,QQQ')
    
    // Save as template
    await page.check('input[name="saveAsTemplate"]')
    await page.fill('input[name="templateName"]', 'MA Template')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Later, load template
    await page.click('[data-testid="create-config-btn"]')
    await page.click('[data-testid="load-template-btn"]')
    
    // Select template
    await page.click('[data-testid="template-MA Template"]')
    
    // Form should be pre-filled
    await expect(page.locator('select[name="strategy"]')).toHaveValue('ma-crossover')
    await expect(page.locator('input[name="symbols"]')).toHaveValue('SPY,QQQ')
  })
})
