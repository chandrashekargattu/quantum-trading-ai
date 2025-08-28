import { Page } from '@playwright/test'

export interface TestUser {
  email: string
  password: string
  username: string
  fullName: string
}

// Generate unique test user data
export const generateTestUser = (): TestUser => {
  const timestamp = Date.now()
  return {
    email: `test.user.${timestamp}@example.com`,
    password: 'SecureTestPass123!',
    username: `testuser${timestamp}`,
    fullName: `Test User ${timestamp}`
  }
}

// Create a test user via API
export const createTestUser = async (): Promise<TestUser> => {
  const user = generateTestUser()
  
  // In a real scenario, this would call your backend API
  // For now, we'll just return the user data
  // You can implement actual API calls here if needed
  
  return user
}

// Login helper function
export const loginUser = async (page: Page, email: string, password: string) => {
  await page.goto('/auth/login')
  await page.fill('input[name="email"]', email)
  await page.fill('input[name="password"]', password)
  await page.click('button[type="submit"]')
  await page.waitForURL('**/dashboard')
}

// Logout helper function
export const logoutUser = async (page: Page) => {
  await page.click('[data-testid="user-menu"]')
  await page.click('text=Logout')
  await page.waitForURL('**/')
}

// Register helper function
export const registerUser = async (page: Page, user: TestUser) => {
  await page.goto('/auth/register')
  await page.fill('input[name="email"]', user.email)
  await page.fill('input[name="username"]', user.username)
  await page.fill('input[name="fullName"]', user.fullName)
  await page.fill('input[name="password"]', user.password)
  await page.fill('input[name="confirmPassword"]', user.password)
  await page.check('input[type="checkbox"]')
  await page.click('button[type="submit"]')
  await page.waitForURL('**/dashboard')
}

// Check if user is logged in
export const isLoggedIn = async (page: Page): Promise<boolean> => {
  try {
    await page.waitForSelector('[data-testid="user-menu"]', { timeout: 1000 })
    return true
  } catch {
    return false
  }
}

// Wait for authentication to complete
export const waitForAuth = async (page: Page) => {
  await page.waitForLoadState('networkidle')
  await page.waitForSelector('[data-testid="user-menu"]')
}
