import { test, expect } from '@playwright/test'
import { loginUser, createTestUser } from './helpers/auth'
import { mockAlerts, mockMarketData } from './helpers/mock-data'

test.describe('Alerts E2E Tests', () => {
  let userEmail: string
  let userPassword: string

  test.beforeAll(async () => {
    const testUser = await createTestUser()
    userEmail = testUser.email
    userPassword = testUser.password
  })

  test.beforeEach(async ({ page }) => {
    await loginUser(page, userEmail, userPassword)
    await page.goto('/alerts')
    
    // Mock alerts API
    await page.route('**/api/v1/alerts', async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockAlerts)
        })
      }
    })
  })

  test('should create a price alert', async ({ page }) => {
    // Mock create alert API
    await page.route('**/api/v1/alerts', async route => {
      if (route.request().method() === 'POST') {
        const data = await route.request().postDataJSON()
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'alert-new',
            ...data,
            enabled: true,
            triggered: false,
            createdAt: new Date().toISOString()
          })
        })
      }
    })
    
    // Click create alert button
    await page.click('[data-testid="create-alert-btn"]')
    
    // Fill alert form
    await page.fill('input[name="symbol"]', 'TSLA')
    await page.selectOption('select[name="type"]', 'PRICE')
    await page.selectOption('select[name="condition"]', 'ABOVE')
    await page.fill('input[name="value"]', '800')
    await page.fill('textarea[name="message"]', 'TSLA breakout alert')
    
    // Configure notifications
    await page.check('input[name="sendEmail"]')
    await page.check('input[name="sendPush"]')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Should show success
    await expect(page.locator('text=Alert created successfully')).toBeVisible()
    
    // New alert should appear in list
    await expect(page.locator('[data-testid="alert-TSLA-PRICE-ABOVE-800"]')).toBeVisible()
  })

  test('should create a volume alert', async ({ page }) => {
    // Click create alert
    await page.click('[data-testid="create-alert-btn"]')
    
    // Fill form for volume alert
    await page.fill('input[name="symbol"]', 'AAPL')
    await page.selectOption('select[name="type"]', 'VOLUME')
    await page.selectOption('select[name="condition"]', 'ABOVE')
    await page.fill('input[name="value"]', '100000000')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Should appear in list
    await expect(page.locator('text=AAPL - VOLUME ABOVE 100M')).toBeVisible()
  })

  test('should display and manage active alerts', async ({ page }) => {
    // Check alerts are displayed
    await expect(page.locator('[data-testid="alert-alert-1"]')).toBeVisible()
    await expect(page.locator('[data-testid="alert-alert-2"]')).toBeVisible()
    
    // Check alert details
    const alert1 = page.locator('[data-testid="alert-alert-1"]')
    await expect(alert1).toContainText('AAPL')
    await expect(alert1).toContainText('PRICE ABOVE $160')
    await expect(alert1).toContainText('Active')
    
    // Toggle alert
    await page.click('[data-testid="toggle-alert-alert-1"]')
    
    // Mock update API
    await page.route('**/api/v1/alerts/alert-1', async route => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...mockAlerts[0],
            enabled: false
          })
        })
      }
    })
    
    await expect(alert1).toContainText('Disabled')
  })

  test('should edit alert value', async ({ page }) => {
    // Click edit on first alert
    await page.click('[data-testid="edit-alert-alert-1"]')
    
    // Edit form should appear
    await expect(page.locator('[data-testid="edit-alert-form"]')).toBeVisible()
    
    // Change value
    await page.fill('input[name="value"]', '165')
    
    // Mock update API
    await page.route('**/api/v1/alerts/alert-1', async route => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            ...mockAlerts[0],
            value: 165
          })
        })
      }
    })
    
    // Save
    await page.click('button[text="Save"]')
    
    // Should update display
    await expect(page.locator('[data-testid="alert-alert-1"]')).toContainText('$165')
  })

  test('should delete alert with confirmation', async ({ page }) => {
    // Mock delete API
    await page.route('**/api/v1/alerts/alert-2', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 204
        })
      }
    })
    
    // Click delete
    await page.click('[data-testid="delete-alert-alert-2"]')
    
    // Confirm dialog
    await expect(page.locator('text=Are you sure you want to delete this alert?')).toBeVisible()
    await page.click('text=Yes, Delete')
    
    // Alert should be removed
    await expect(page.locator('[data-testid="alert-alert-2"]')).not.toBeVisible()
    await expect(page.locator('text=Alert deleted successfully')).toBeVisible()
  })

  test('should display triggered alerts', async ({ page }) => {
    // Mock triggered alerts
    const triggeredAlert = {
      ...mockAlerts[0],
      triggered: true,
      triggeredAt: new Date().toISOString(),
      triggeredValue: 161.5
    }
    
    await page.route('**/api/v1/alerts/triggered', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([triggeredAlert])
      })
    })
    
    // Navigate to triggered tab
    await page.click('[data-testid="triggered-alerts-tab"]')
    
    // Check triggered alert display
    await expect(page.locator('[data-testid="triggered-alert-alert-1"]')).toBeVisible()
    await expect(page.locator('[data-testid="triggered-alert-alert-1"]')).toContainText('Triggered at $161.50')
    await expect(page.locator('[data-testid="triggered-alert-alert-1"]')).toContainText(
      new Date(triggeredAlert.triggeredAt).toLocaleString()
    )
  })

  test('should receive real-time alert notifications', async ({ page }) => {
    // Simulate WebSocket alert trigger
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('ws-alert-triggered', {
        detail: {
          alertId: 'alert-1',
          symbol: 'AAPL',
          type: 'PRICE',
          condition: 'ABOVE',
          targetValue: 160,
          currentValue: 161.5,
          message: 'AAPL price alert triggered'
        }
      }))
    })
    
    // Should show notification
    await expect(page.locator('[data-testid="notification-toast"]')).toBeVisible()
    await expect(page.locator('[data-testid="notification-toast"]')).toContainText('AAPL price alert triggered')
    
    // Notification count should update
    await expect(page.locator('[data-testid="notification-badge"]')).toContainText('1')
  })

  test('should create technical indicator alert', async ({ page }) => {
    // Click create alert
    await page.click('[data-testid="create-alert-btn"]')
    
    // Select technical alert type
    await page.selectOption('select[name="type"]', 'TECHNICAL')
    
    // Additional options should appear
    await expect(page.locator('select[name="indicator"]')).toBeVisible()
    
    // Fill technical alert
    await page.fill('input[name="symbol"]', 'MSFT')
    await page.selectOption('select[name="indicator"]', 'RSI')
    await page.selectOption('select[name="condition"]', 'BELOW')
    await page.fill('input[name="value"]', '30')
    await page.fill('textarea[name="message"]', 'MSFT oversold')
    
    // Submit
    await page.click('button[type="submit"]')
    
    // Should show in list
    await expect(page.locator('text=MSFT - RSI BELOW 30')).toBeVisible()
  })

  test('should filter alerts by symbol', async ({ page }) => {
    // Enter filter
    await page.fill('[data-testid="alert-filter"]', 'AAPL')
    
    // Should only show AAPL alerts
    await expect(page.locator('[data-testid="alert-alert-1"]')).toBeVisible()
    await expect(page.locator('[data-testid="alert-alert-2"]')).not.toBeVisible()
    
    // Clear filter
    await page.fill('[data-testid="alert-filter"]', '')
    
    // Should show all alerts again
    await expect(page.locator('[data-testid="alert-alert-2"]')).toBeVisible()
  })

  test('should test alert before saving', async ({ page }) => {
    // Create alert
    await page.click('[data-testid="create-alert-btn"]')
    
    // Fill form
    await page.fill('input[name="symbol"]', 'GOOGL')
    await page.selectOption('select[name="type"]', 'PRICE')
    await page.selectOption('select[name="condition"]', 'BELOW')
    await page.fill('input[name="value"]', '2500')
    
    // Mock test alert API
    await page.route('**/api/v1/alerts/test', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          wouldTrigger: true,
          currentValue: 2480,
          message: 'Alert would trigger immediately'
        })
      })
    })
    
    // Click test button
    await page.click('[data-testid="test-alert-btn"]')
    
    // Should show test result
    await expect(page.locator('text=Alert would trigger immediately')).toBeVisible()
    await expect(page.locator('text=Current value: $2,480')).toBeVisible()
  })

  test('should bulk manage alerts', async ({ page }) => {
    // Enter selection mode
    await page.click('[data-testid="bulk-select-btn"]')
    
    // Select multiple alerts
    await page.check('[data-testid="select-alert-alert-1"]')
    await page.check('[data-testid="select-alert-alert-2"]')
    
    // Bulk actions should appear
    await expect(page.locator('[data-testid="bulk-actions"]')).toBeVisible()
    
    // Mock bulk delete API
    await page.route('**/api/v1/alerts/bulk-delete', async route => {
      await route.fulfill({
        status: 204
      })
    })
    
    // Bulk delete
    await page.click('[data-testid="bulk-delete-btn"]')
    await page.click('text=Yes, Delete All')
    
    // Alerts should be removed
    await expect(page.locator('[data-testid="alert-alert-1"]')).not.toBeVisible()
    await expect(page.locator('[data-testid="alert-alert-2"]')).not.toBeVisible()
  })

  test('should display alert statistics', async ({ page }) => {
    // Mock stats API
    await page.route('**/api/v1/alerts/stats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          totalAlerts: 25,
          activeAlerts: 15,
          triggeredToday: 3,
          triggeredThisWeek: 8,
          triggeredThisMonth: 20,
          byType: {
            PRICE: 12,
            VOLUME: 5,
            TECHNICAL: 8
          }
        })
      })
    })
    
    // Navigate to stats
    await page.click('[data-testid="alert-stats-btn"]')
    
    // Check stats display
    await expect(page.locator('[data-testid="total-alerts"]')).toContainText('25')
    await expect(page.locator('[data-testid="active-alerts"]')).toContainText('15')
    await expect(page.locator('[data-testid="triggered-today"]')).toContainText('3')
    
    // Check chart
    await expect(page.locator('[data-testid="alerts-by-type-chart"]')).toBeVisible()
  })
})
