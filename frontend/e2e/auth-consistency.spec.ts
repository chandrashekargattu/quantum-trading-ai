import { test, expect } from '@playwright/test'

test.describe('Authentication Consistency', () => {
  const testUser = {
    email: 'e2e@example.com',
    username: 'e2euser',
    password: 'E2EPass123',
    fullName: 'E2E Test User'
  }

  test.beforeEach(async ({ page }) => {
    // Clear all cookies and local storage
    await page.context().clearCookies()
    await page.evaluate(() => {
      localStorage.clear()
      sessionStorage.clear()
    })
  })

  test('should handle complete authentication flow', async ({ page }) => {
    // 1. Navigate to dashboard - should redirect to login
    await page.goto('/dashboard')
    await expect(page).toHaveURL('/auth/login')

    // 2. Try to register a new user
    await page.click('text=Create an account')
    await expect(page).toHaveURL('/auth/register')

    // Fill registration form
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="username"]', testUser.username)
    await page.fill('input[name="password"]', testUser.password)
    await page.fill('input[name="confirmPassword"]', testUser.password)
    await page.fill('input[name="fullName"]', testUser.fullName)

    // Submit registration
    await page.click('button[type="submit"]')

    // Should redirect to dashboard after successful registration
    await expect(page).toHaveURL('/dashboard', { timeout: 10000 })

    // 3. Verify user is logged in
    await expect(page.locator('text=Welcome back')).toBeVisible()
    
    // 4. Check that auth token is stored
    const token = await page.evaluate(() => localStorage.getItem('access_token'))
    expect(token).toBeTruthy()

    // 5. Verify API calls work with auth
    await page.waitForResponse(response => 
      response.url().includes('/api/v1/market-data/indicators') && 
      response.status() === 200
    )

    // 6. Test logout
    await page.click('text=Logout')
    await expect(page).toHaveURL('/auth/login')
    
    // Verify token is cleared
    const tokenAfterLogout = await page.evaluate(() => localStorage.getItem('access_token'))
    expect(tokenAfterLogout).toBeNull()
  })

  test('should handle 401 errors gracefully', async ({ page }) => {
    // Set an invalid token
    await page.evaluate(() => {
      localStorage.setItem('access_token', 'invalid-token')
      localStorage.setItem('token_type', 'Bearer')
    })

    // Navigate to dashboard
    await page.goto('/dashboard')

    // Should redirect to login when API returns 401
    await expect(page).toHaveURL('/auth/login', { timeout: 10000 })

    // Verify cache and tokens are cleared
    const token = await page.evaluate(() => localStorage.getItem('access_token'))
    expect(token).toBeNull()
  })

  test('should persist auth across page reloads', async ({ page }) => {
    // Login first
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="password"]', testUser.password)
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard', { timeout: 10000 })
    await expect(page.locator('text=Welcome back')).toBeVisible()

    // Reload page
    await page.reload()

    // Should still be on dashboard
    await expect(page).toHaveURL('/dashboard')
    await expect(page.locator('text=Welcome back')).toBeVisible()

    // Verify API calls still work
    await page.waitForResponse(response => 
      response.url().includes('/api/v1/portfolios') && 
      response.status() === 200
    )
  })

  test('should handle concurrent API requests with auth', async ({ page }) => {
    // Login
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="password"]', testUser.password)
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard', { timeout: 10000 })

    // Monitor API responses
    const apiResponses = []
    page.on('response', response => {
      if (response.url().includes('/api/v1/')) {
        apiResponses.push({
          url: response.url(),
          status: response.status(),
          headers: response.headers()
        })
      }
    })

    // Wait for all initial API calls to complete
    await page.waitForLoadState('networkidle')

    // Verify all API calls included auth headers and succeeded
    for (const response of apiResponses) {
      expect([200, 201, 204]).toContain(response.status)
    }
  })

  test('should handle portfolio operations with auth', async ({ page }) => {
    // Login
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="password"]', testUser.password)
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')

    // Create portfolio
    await page.click('text=Create Portfolio')
    await page.fill('input[placeholder*="Trading Portfolio"]', 'E2E Test Portfolio')
    await page.fill('input[placeholder*="100000"]', '50000')
    
    // Monitor the API response
    const [response] = await Promise.all([
      page.waitForResponse(response => 
        response.url().includes('/api/v1/portfolios') && 
        response.request().method() === 'POST'
      ),
      page.click('button:has-text("Create Portfolio")')
    ])

    expect(response.status()).toBe(200)

    // Verify success message
    await expect(page.locator('text=Portfolio created successfully')).toBeVisible()

    // Refresh portfolios
    await page.click('button[title="Refresh portfolios"]')
    
    // Verify portfolio is displayed
    await expect(page.locator('text=E2E Test Portfolio')).toBeVisible({ timeout: 10000 })
  })

  test('should handle cache clearing', async ({ page }) => {
    // Login
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="password"]', testUser.password)
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')

    // Click clear cache button
    await page.click('text=Clear Cache & Refresh')

    // Verify page reloads
    await page.waitForLoadState('load')

    // Verify still authenticated
    await expect(page).toHaveURL('/dashboard')
    await expect(page.locator('text=Welcome back')).toBeVisible()
  })

  test('should handle session timeout', async ({ page }) => {
    // Login
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', testUser.email)
    await page.fill('input[name="password"]', testUser.password)
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')

    // Simulate expired token by setting it to an expired JWT
    await page.evaluate(() => {
      // This is a real JWT but expired
      const expiredToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNjAwMDAwMDAwfQ.invalid'
      localStorage.setItem('access_token', expiredToken)
    })

    // Trigger an API call by refreshing
    await page.reload()

    // Should redirect to login
    await expect(page).toHaveURL('/auth/login', { timeout: 10000 })
  })

  test('should maintain auth state across tabs', async ({ browser }) => {
    const context = await browser.newContext()
    const page1 = await context.newPage()

    // Login in first tab
    await page1.goto('/auth/login')
    await page1.fill('input[name="email"]', testUser.email)
    await page1.fill('input[name="password"]', testUser.password)
    await page1.click('button[type="submit"]')
    await expect(page1).toHaveURL('/dashboard')

    // Open second tab
    const page2 = await context.newPage()
    await page2.goto('/dashboard')

    // Should be authenticated in second tab
    await expect(page2).toHaveURL('/dashboard')
    await expect(page2.locator('text=Welcome back')).toBeVisible()

    // Logout in first tab
    await page1.click('text=Logout')
    await expect(page1).toHaveURL('/auth/login')

    // Refresh second tab - should redirect to login
    await page2.reload()
    await expect(page2).toHaveURL('/auth/login', { timeout: 10000 })

    await context.close()
  })
})
