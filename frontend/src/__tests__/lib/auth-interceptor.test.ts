import { authInterceptor, AuthInterceptor } from '@/lib/auth-interceptor'
import { clearCache } from '@/lib/api-cache'

// Mock dependencies
jest.mock('@/lib/api-cache', () => ({
  clearCache: jest.fn()
}))

// Mock fetch
global.fetch = jest.fn()

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn()
}
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true
})

// Mock window.location
delete (window as any).location
window.location = { 
  href: '',
  pathname: '/dashboard',
  reload: jest.fn()
} as any

describe('AuthInterceptor', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockLocalStorage.getItem.mockReturnValue(null)
    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'test' })
    })
  })

  describe('getAuthHeaders', () => {
    it('should return empty object when no token', () => {
      const interceptor = new AuthInterceptor()
      const headers = interceptor.getAuthHeaders()
      expect(headers).toEqual({})
    })

    it('should return auth headers when token exists', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        if (key === 'token_type') return 'Bearer'
        return null
      })

      const interceptor = new AuthInterceptor()
      const headers = interceptor.getAuthHeaders()
      expect(headers).toEqual({
        'Authorization': 'Bearer test-token'
      })
    })

    it('should use default Bearer type when token_type not set', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        return null
      })

      const interceptor = new AuthInterceptor()
      const headers = interceptor.getAuthHeaders()
      expect(headers).toEqual({
        'Authorization': 'Bearer test-token'
      })
    })
  })

  describe('shouldIncludeAuth', () => {
    it('should return true when no exclude paths', () => {
      const interceptor = new AuthInterceptor()
      expect(interceptor.shouldIncludeAuth('/api/test')).toBe(true)
    })

    it('should return false for excluded paths', () => {
      const interceptor = new AuthInterceptor({
        excludePaths: ['/auth/login', '/auth/register']
      })
      expect(interceptor.shouldIncludeAuth('/auth/login')).toBe(false)
      expect(interceptor.shouldIncludeAuth('/api/auth/login')).toBe(false)
    })

    it('should return true for non-excluded paths', () => {
      const interceptor = new AuthInterceptor({
        excludePaths: ['/auth/login']
      })
      expect(interceptor.shouldIncludeAuth('/api/users')).toBe(true)
    })
  })

  describe('fetch', () => {
    it('should add auth headers to requests', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        if (key === 'token_type') return 'Bearer'
        return null
      })

      const interceptor = new AuthInterceptor()
      await interceptor.fetch('/api/test', { method: 'GET' })

      expect(global.fetch).toHaveBeenCalledWith('/api/test', {
        method: 'GET',
        headers: {
          'Authorization': 'Bearer test-token'
        }
      })
    })

    it('should not add auth headers to excluded paths', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        return null
      })

      const interceptor = new AuthInterceptor({
        excludePaths: ['/auth/login']
      })
      await interceptor.fetch('/auth/login', { method: 'POST' })

      expect(global.fetch).toHaveBeenCalledWith('/auth/login', {
        method: 'POST'
      })
    })

    it('should handle 401 responses', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized'
      })

      const onAuthError = jest.fn()
      const interceptor = new AuthInterceptor({ onAuthError })
      
      const response = await interceptor.fetch('/api/test')

      expect(response.status).toBe(401)
      expect(clearCache).toHaveBeenCalled()
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('access_token')
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('token_type')
      expect(onAuthError).toHaveBeenCalled()
    })

    it('should not handle 401 for excluded paths', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 401
      })

      const onAuthError = jest.fn()
      const interceptor = new AuthInterceptor({ 
        onAuthError,
        excludePaths: ['/auth/login']
      })
      
      await interceptor.fetch('/auth/login')

      expect(clearCache).not.toHaveBeenCalled()
      expect(mockLocalStorage.removeItem).not.toHaveBeenCalled()
      expect(onAuthError).not.toHaveBeenCalled()
    })

    it('should preserve existing headers', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'access_token') return 'test-token'
        return null
      })

      const interceptor = new AuthInterceptor()
      await interceptor.fetch('/api/test', {
        headers: {
          'Content-Type': 'application/json'
        }
      })

      expect(global.fetch).toHaveBeenCalledWith('/api/test', {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-token'
        }
      })
    })
  })

  describe('default interceptor', () => {
    it('should redirect to login on 401', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 401
      })

      window.location.pathname = '/dashboard'
      await authInterceptor.fetch('/api/test')

      expect(window.location.href).toBe('/auth/login')
    })

    it('should not redirect if already on login page', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 401
      })

      window.location.pathname = '/auth/login'
      window.location.href = ''
      await authInterceptor.fetch('/api/test')

      expect(window.location.href).toBe('')
    })
  })
})
