import { clearCache } from './api-cache'

interface AuthInterceptorConfig {
  onAuthError?: () => void
  excludePaths?: string[]
}

class AuthInterceptor {
  private config: AuthInterceptorConfig
  private isRefreshing = false
  private refreshPromise: Promise<boolean> | null = null

  constructor(config: AuthInterceptorConfig = {}) {
    this.config = config
  }

  /**
   * Get auth headers for requests
   */
  getAuthHeaders(): Record<string, string> {
    const token = localStorage.getItem('access_token')
    const tokenType = localStorage.getItem('token_type') || 'Bearer'
    
    if (!token) {
      return {}
    }

    return {
      'Authorization': `${tokenType} ${token}`
    }
  }

  /**
   * Check if request should include auth
   */
  shouldIncludeAuth(url: string): boolean {
    if (!this.config.excludePaths) return true
    
    return !this.config.excludePaths.some(path => url.includes(path))
  }

  /**
   * Intercept fetch requests
   */
  async fetch(url: string, options: RequestInit = {}): Promise<Response> {
    // Add auth headers if needed
    if (this.shouldIncludeAuth(url)) {
      options.headers = {
        ...options.headers,
        ...this.getAuthHeaders()
      }
    }

    const response = await fetch(url, options)

    // Handle 401 errors
    if (response.status === 401 && this.shouldIncludeAuth(url)) {
      console.warn('Auth error detected, clearing cache and auth state')
      
      // Clear all cache
      clearCache()
      
      // Clear auth tokens
      localStorage.removeItem('access_token')
      localStorage.removeItem('token_type')
      
      // Trigger auth error callback
      if (this.config.onAuthError) {
        this.config.onAuthError()
      }

      // Return the 401 response
      return response
    }

    return response
  }

  /**
   * Create a wrapped version of the interceptor for easy use
   */
  createFetch() {
    return this.fetch.bind(this)
  }
}

// Create default interceptor
export const authInterceptor = new AuthInterceptor({
  onAuthError: () => {
    // Redirect to login
    if (typeof window !== 'undefined' && window.location.pathname !== '/auth/login') {
      window.location.href = '/auth/login'
    }
  },
  excludePaths: ['/auth/login', '/auth/register']
})

// Export wrapped fetch
export const authenticatedFetch = authInterceptor.createFetch()
