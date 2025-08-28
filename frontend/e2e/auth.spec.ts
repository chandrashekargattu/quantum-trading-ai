import { test, expect } from '@playwright/test'

// Helper to generate unique email
const generateUniqueEmail = () => {
  const timestamp = Date.now()
  return `test.user.${timestamp}@example.com`
}

test.describe('Authentication E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should complete full registration and login flow', async ({ page }) => {
    const email = generateUniqueEmail()
    const password = 'SecurePass123!'
    
    // Navigate to registration
    await page.click('text=Get Started')
    await page.waitForURL('**/auth/register')
    
    // Fill registration form
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="username"]', 'testuser')
    await page.fill('input[name="fullName"]', 'Test User')
    await page.fill('input[name="password"]', password)
    await page.fill('input[name="confirmPassword"]', password)
    
    // Accept terms
    await page.check('input[type="checkbox"]')
    
    // Submit registration
    await page.click('button[type="submit"]')
    
    // Should redirect to dashboard
    await page.waitForURL('**/dashboard')
    await expect(page).toHaveURL(/.*dashboard/)
    
    // Verify user is logged in
    await expect(page.locator('text=Welcome, Test User')).toBeVisible()
    
    // Logout
    await page.click('[data-testid="user-menu"]')
    await page.click('text=Logout')
    
    // Should redirect to home
    await page.waitForURL('**/')
    
    // Now test login with same credentials
    await page.click('text=Sign In')
    await page.waitForURL('**/auth/login')
    
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="password"]', password)
    await page.click('button[type="submit"]')
    
    // Should be logged in again
    await page.waitForURL('**/dashboard')
    await expect(page.locator('text=Welcome, Test User')).toBeVisible()
  })

  test('should handle invalid login credentials', async ({ page }) => {
    await page.goto('/auth/login')
    
    await page.fill('input[name="email"]', 'nonexistent@example.com')
    await page.fill('input[name="password"]', 'wrongpassword')
    await page.click('button[type="submit"]')
    
    // Should show error message
    await expect(page.locator('text=Invalid credentials')).toBeVisible()
    
    // Should stay on login page
    await expect(page).toHaveURL(/.*auth\/login/)
  })

  test('should validate registration form', async ({ page }) => {
    await page.goto('/auth/register')
    
    // Try to submit empty form
    await page.click('button[type="submit"]')
    
    // Should show validation errors
    await expect(page.locator('text=Email is required')).toBeVisible()
    await expect(page.locator('text=Username is required')).toBeVisible()
    await expect(page.locator('text=Password is required')).toBeVisible()
    
    // Test password requirements
    await page.fill('input[name="password"]', 'weak')
    await expect(page.locator('text=At least 8 characters')).toBeVisible()
    await expect(page.locator('text=One uppercase letter')).toBeVisible()
    await expect(page.locator('text=One number')).toBeVisible()
    
    // Test password mismatch
    await page.fill('input[name="password"]', 'SecurePass123!')
    await page.fill('input[name="confirmPassword"]', 'DifferentPass123!')
    await page.click('button[type="submit"]')
    
    await expect(page.locator('text=Passwords do not match')).toBeVisible()
  })

  test('should protect authenticated routes', async ({ page }) => {
    // Try to access dashboard without login
    await page.goto('/dashboard')
    
    // Should redirect to login
    await page.waitForURL('**/auth/login')
    await expect(page).toHaveURL(/.*auth\/login/)
  })

  test('should handle session expiry', async ({ page, context }) => {
    const email = generateUniqueEmail()
    const password = 'SecurePass123!'
    
    // Register and login
    await page.goto('/auth/register')
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="username"]', 'sessiontest')
    await page.fill('input[name="fullName"]', 'Session Test')
    await page.fill('input[name="password"]', password)
    await page.fill('input[name="confirmPassword"]', password)
    await page.check('input[type="checkbox"]')
    await page.click('button[type="submit"]')
    
    await page.waitForURL('**/dashboard')
    
    // Clear session storage to simulate expiry
    await context.clearCookies()
    await page.evaluate(() => {
      localStorage.clear()
      sessionStorage.clear()
    })
    
    // Try to navigate to protected page
    await page.goto('/portfolio')
    
    // Should redirect to login
    await page.waitForURL('**/auth/login')
  })

  test('should handle forgot password flow', async ({ page }) => {
    await page.goto('/auth/login')
    
    await page.click('text=Forgot password?')
    await page.waitForURL('**/auth/forgot-password')
    
    // Enter email
    await page.fill('input[name="email"]', 'test@example.com')
    await page.click('button[type="submit"]')
    
    // Should show success message
    await expect(page.locator('text=If an account exists with this email')).toBeVisible()
  })

  test('should handle OAuth login', async ({ page }) => {
    await page.goto('/auth/login')
    
    // Test Google OAuth button
    const googleButton = page.locator('button:has-text("Continue with Google")')
    await expect(googleButton).toBeVisible()
    
    // Test GitHub OAuth button
    const githubButton = page.locator('button:has-text("Continue with GitHub")')
    await expect(githubButton).toBeVisible()
    
    // Note: Actual OAuth flow would require mocking OAuth providers
  })

  test('should remember user preference', async ({ page }) => {
    const email = generateUniqueEmail()
    const password = 'SecurePass123!'
    
    // Register
    await page.goto('/auth/register')
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="username"]', 'remembertest')
    await page.fill('input[name="fullName"]', 'Remember Test')
    await page.fill('input[name="password"]', password)
    await page.fill('input[name="confirmPassword"]', password)
    await page.check('input[type="checkbox"]')
    await page.click('button[type="submit"]')
    
    await page.waitForURL('**/dashboard')
    
    // Set a preference
    await page.click('[data-testid="settings-menu"]')
    await page.click('[data-testid="theme-toggle"]')
    
    // Logout
    await page.click('[data-testid="user-menu"]')
    await page.click('text=Logout')
    
    // Login again
    await page.goto('/auth/login')
    await page.fill('input[name="email"]', email)
    await page.fill('input[name="password"]', password)
    await page.check('input[name="remember"]')
    await page.click('button[type="submit"]')
    
    await page.waitForURL('**/dashboard')
    
    // Check theme preference is retained
    const body = page.locator('body')
    await expect(body).toHaveAttribute('data-theme', 'dark')
  })

  test('should handle concurrent login attempts', async ({ browser }) => {
    const email = generateUniqueEmail()
    const password = 'SecurePass123!'
    
    // First register the user
    const context1 = await browser.newContext()
    const page1 = await context1.newPage()
    
    await page1.goto('/auth/register')
    await page1.fill('input[name="email"]', email)
    await page1.fill('input[name="username"]', 'concurrenttest')
    await page1.fill('input[name="fullName"]', 'Concurrent Test')
    await page1.fill('input[name="password"]', password)
    await page1.fill('input[name="confirmPassword"]', password)
    await page1.check('input[type="checkbox"]')
    await page1.click('button[type="submit"]')
    await page1.waitForURL('**/dashboard')
    
    // Logout
    await page1.click('[data-testid="user-menu"]')
    await page1.click('text=Logout')
    
    // Now try concurrent logins
    const context2 = await browser.newContext()
    const page2 = await context2.newPage()
    
    // Both pages attempt login
    await Promise.all([
      (async () => {
        await page1.goto('/auth/login')
        await page1.fill('input[name="email"]', email)
        await page1.fill('input[name="password"]', password)
        await page1.click('button[type="submit"]')
      })(),
      (async () => {
        await page2.goto('/auth/login')
        await page2.fill('input[name="email"]', email)
        await page2.fill('input[name="password"]', password)
        await page2.click('button[type="submit"]')
      })()
    ])
    
    // Both should succeed
    await expect(page1).toHaveURL(/.*dashboard/)
    await expect(page2).toHaveURL(/.*dashboard/)
    
    await context1.close()
    await context2.close()
  })
})
