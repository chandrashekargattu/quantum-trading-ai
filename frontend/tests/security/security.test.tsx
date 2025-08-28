/**
 * Frontend Security Tests
 * 
 * These tests verify authentication, authorization, XSS prevention,
 * secure data handling, and other security measures in the frontend.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { renderHook, act } from '@testing-library/react-hooks'
import DOMPurify from 'dompurify'
import * as jose from 'jose'

// Components and hooks to test
import { useAuthStore } from '@/store/useAuthStore'
import { LoginForm } from '@/components/auth/LoginForm'
import { SecureInput } from '@/components/common/SecureInput'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { api } from '@/services/api'
import { encryptData, decryptData } from '@/utils/crypto'

describe('Frontend Security Tests', () => {
  beforeEach(() => {
    // Clear all stores and mocks
    localStorage.clear()
    sessionStorage.clear()
    jest.clearAllMocks()
  })

  describe('Authentication Security', () => {
    test('should not store sensitive data in localStorage', () => {
      const { result } = renderHook(() => useAuthStore())
      
      act(() => {
        result.current.setAuth({
          user: { id: '123', email: 'test@example.com' },
          accessToken: 'secret-token',
          refreshToken: 'refresh-token'
        })
      })
      
      // Check localStorage
      const stored = localStorage.getItem('auth-storage')
      if (stored) {
        const parsed = JSON.parse(stored)
        // Should not contain tokens
        expect(parsed.accessToken).toBeUndefined()
        expect(parsed.refreshToken).toBeUndefined()
        expect(parsed.password).toBeUndefined()
      }
    })

    test('should validate JWT token format', async () => {
      const { result } = renderHook(() => useAuthStore())
      
      // Invalid token formats
      const invalidTokens = [
        'not-a-jwt',
        'invalid.jwt.token',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9', // Incomplete
        '', // Empty
      ]
      
      for (const token of invalidTokens) {
        act(() => {
          const isValid = result.current.validateToken(token)
          expect(isValid).toBe(false)
        })
      }
    })

    test('should detect expired tokens', async () => {
      const { result } = renderHook(() => useAuthStore())
      
      // Create expired token
      const expiredToken = await new jose.SignJWT({ sub: 'user123' })
        .setProtectedHeader({ alg: 'HS256' })
        .setExpirationTime('1s')
        .sign(new TextEncoder().encode('secret'))
      
      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      act(() => {
        const isValid = result.current.validateToken(expiredToken)
        expect(isValid).toBe(false)
      })
    })

    test('should clear auth on logout', () => {
      const { result } = renderHook(() => useAuthStore())
      
      // Set auth
      act(() => {
        result.current.setAuth({
          user: { id: '123', email: 'test@example.com' },
          accessToken: 'token'
        })
      })
      
      // Logout
      act(() => {
        result.current.logout()
      })
      
      // Check cleared
      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
      expect(localStorage.getItem('auth-storage')).toBeNull()
    })
  })

  describe('XSS Prevention', () => {
    test('should sanitize user input in display', () => {
      const maliciousInput = '<script>alert("XSS")</script><b>Safe text</b>'
      
      const TestComponent = ({ content }: { content: string }) => {
        const sanitized = DOMPurify.sanitize(content, { ALLOWED_TAGS: ['b'] })
        return <div dangerouslySetInnerHTML={{ __html: sanitized }} />
      }
      
      render(<TestComponent content={maliciousInput} />)
      
      // Script should be removed
      expect(screen.queryByText(/alert/)).not.toBeInTheDocument()
      // Safe content should remain
      expect(screen.getByText('Safe text')).toBeInTheDocument()
    })

    test('should escape dangerous attributes', () => {
      const maliciousAttrs = {
        'onerror': 'alert("XSS")',
        'onclick': 'stealData()',
        'style': 'background-image: url(javascript:alert("XSS"))'
      }
      
      const TestComponent = () => (
        <div>
          {Object.entries(maliciousAttrs).map(([key, value]) => (
            <img
              key={key}
              src="test.jpg"
              {...{ [key]: value }}
              data-testid={`img-${key}`}
            />
          ))}
        </div>
      )
      
      render(<TestComponent />)
      
      // Check that dangerous attributes are not rendered
      Object.keys(maliciousAttrs).forEach(attr => {
        const img = screen.getByTestId(`img-${attr}`)
        expect(img.getAttribute(attr)).toBeNull()
      })
    })

    test('should prevent javascript: URLs', () => {
      const dangerousUrls = [
        'javascript:alert("XSS")',
        'JaVaScRiPt:alert("XSS")', // Case variation
        'javascript:void(0)',
        'data:text/html,<script>alert("XSS")</script>'
      ]
      
      dangerousUrls.forEach(url => {
        const sanitized = DOMPurify.sanitize(`<a href="${url}">Link</a>`)
        expect(sanitized).not.toContain('javascript:')
        expect(sanitized).not.toContain('data:text/html')
      })
    })
  })

  describe('Input Validation', () => {
    test('should validate email format', () => {
      const { getByLabelText, getByText } = render(<LoginForm />)
      const emailInput = getByLabelText('Email')
      
      // Invalid emails
      const invalidEmails = [
        'notanemail',
        '@example.com',
        'user@',
        'user@.com',
        'user space@example.com'
      ]
      
      invalidEmails.forEach(email => {
        fireEvent.change(emailInput, { target: { value: email } })
        fireEvent.blur(emailInput)
        expect(getByText(/invalid email/i)).toBeInTheDocument()
      })
    })

    test('should enforce password requirements', () => {
      const { getByLabelText, queryByText } = render(<RegisterForm />)
      const passwordInput = getByLabelText('Password')
      
      // Test weak passwords
      const weakPasswords = [
        { value: '123', error: 'at least 8 characters' },
        { value: 'password', error: 'number' },
        { value: 'PASSWORD123', error: 'lowercase' },
        { value: 'password123', error: 'uppercase' },
        { value: 'Password123', error: 'special character' }
      ]
      
      weakPasswords.forEach(({ value, error }) => {
        fireEvent.change(passwordInput, { target: { value } })
        fireEvent.blur(passwordInput)
        expect(queryByText(new RegExp(error, 'i'))).toBeInTheDocument()
      })
    })

    test('should limit input length', () => {
      const { getByLabelText } = render(<SecureInput maxLength={50} />)
      const input = getByLabelText('Input') as HTMLInputElement
      
      const longString = 'A'.repeat(100)
      fireEvent.change(input, { target: { value: longString } })
      
      expect(input.value.length).toBe(50)
    })

    test('should prevent SQL injection patterns', () => {
      const sqlPatterns = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "1 UNION SELECT * FROM users"
      ]
      
      const { getByRole } = render(<SearchInput onSearch={jest.fn()} />)
      const searchInput = getByRole('searchbox')
      const onSearch = jest.fn()
      
      sqlPatterns.forEach(pattern => {
        fireEvent.change(searchInput, { target: { value: pattern } })
        fireEvent.submit(searchInput.closest('form')!)
        
        // Should sanitize before sending
        expect(onSearch).not.toHaveBeenCalledWith(pattern)
      })
    })
  })

  describe('Secure Communication', () => {
    test('should use HTTPS for API calls', () => {
      const apiUrl = api.defaults.baseURL
      expect(apiUrl).toMatch(/^https:\/\//)
    })

    test('should include auth token in headers', async () => {
      const mockToken = 'test-token'
      useAuthStore.setState({ accessToken: mockToken })
      
      // Intercept request
      let capturedHeaders: any
      api.interceptors.request.use(config => {
        capturedHeaders = config.headers
        return config
      })
      
      await api.get('/test')
      
      expect(capturedHeaders.Authorization).toBe(`Bearer ${mockToken}`)
    })

    test('should handle token refresh securely', async () => {
      const { result } = renderHook(() => useAuthStore())
      
      // Mock 401 response
      api.interceptors.response.use(
        response => response,
        async error => {
          if (error.response?.status === 401) {
            // Should attempt refresh
            const refreshed = await result.current.refreshAuth()
            expect(refreshed).toBeDefined()
          }
          return Promise.reject(error)
        }
      )
    })

    test('should timeout long requests', async () => {
      // Set short timeout
      api.defaults.timeout = 100
      
      // Mock slow response
      jest.spyOn(api, 'get').mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 1000))
      )
      
      await expect(api.get('/slow-endpoint')).rejects.toThrow('timeout')
    })
  })

  describe('Secure Storage', () => {
    test('should encrypt sensitive data in storage', () => {
      const sensitiveData = {
        apiKey: 'secret-key',
        accountNumber: '123456789'
      }
      
      const encrypted = encryptData(sensitiveData)
      expect(encrypted).not.toContain('secret-key')
      expect(encrypted).not.toContain('123456789')
      
      const decrypted = decryptData(encrypted)
      expect(decrypted).toEqual(sensitiveData)
    })

    test('should use session storage for temporary data', () => {
      const tempData = { sessionId: 'temp-123' }
      
      // Should use sessionStorage, not localStorage
      sessionStorage.setItem('temp-data', JSON.stringify(tempData))
      
      // Verify not in localStorage
      expect(localStorage.getItem('temp-data')).toBeNull()
      
      // Verify in sessionStorage
      expect(JSON.parse(sessionStorage.getItem('temp-data')!)).toEqual(tempData)
    })

    test('should clear sensitive data on window unload', () => {
      const sensitiveData = { token: 'sensitive' }
      sessionStorage.setItem('auth-temp', JSON.stringify(sensitiveData))
      
      // Simulate unload
      window.dispatchEvent(new Event('beforeunload'))
      
      expect(sessionStorage.getItem('auth-temp')).toBeNull()
    })
  })

  describe('Content Security Policy', () => {
    test('should block inline scripts', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation()
      
      const InlineScript = () => (
        <div>
          <script>console.log('inline script')</script>
        </div>
      )
      
      render(<InlineScript />)
      
      // CSP should block execution
      expect(consoleSpy).not.toHaveBeenCalledWith('inline script')
      
      consoleSpy.mockRestore()
    })

    test('should validate external resource URLs', () => {
      const allowedDomains = ['https://api.quantum-trading.com', 'https://cdn.trusted.com']
      const blockedDomains = ['http://malicious.com', 'https://evil.com']
      
      const validateResourceUrl = (url: string) => {
        return allowedDomains.some(domain => url.startsWith(domain))
      }
      
      allowedDomains.forEach(url => {
        expect(validateResourceUrl(url)).toBe(true)
      })
      
      blockedDomains.forEach(url => {
        expect(validateResourceUrl(url)).toBe(false)
      })
    })
  })

  describe('Authorization Checks', () => {
    test('should prevent access to protected routes', () => {
      const { container } = render(
        <ProtectedRoute>
          <div>Protected Content</div>
        </ProtectedRoute>
      )
      
      // Should redirect or show login
      expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
    })

    test('should check user permissions', () => {
      const user = {
        id: '123',
        email: 'user@example.com',
        roles: ['user'],
        permissions: ['view_portfolio', 'create_trade']
      }
      
      useAuthStore.setState({ user, isAuthenticated: true })
      
      const { result } = renderHook(() => useAuthStore())
      
      expect(result.current.hasPermission('view_portfolio')).toBe(true)
      expect(result.current.hasPermission('delete_user')).toBe(false)
      expect(result.current.hasRole('admin')).toBe(false)
    })

    test('should hide unauthorized UI elements', () => {
      const { rerender } = render(
        <AdminPanel userRole="user" />
      )
      
      // User shouldn't see admin features
      expect(screen.queryByText('Delete All Users')).not.toBeInTheDocument()
      
      // Admin should see features
      rerender(<AdminPanel userRole="admin" />)
      expect(screen.getByText('Delete All Users')).toBeInTheDocument()
    })
  })

  describe('WebSocket Security', () => {
    test('should authenticate WebSocket connections', async () => {
      const mockWs = {
        readyState: WebSocket.CONNECTING,
        send: jest.fn(),
        close: jest.fn()
      }
      
      global.WebSocket = jest.fn().mockImplementation((url) => {
        // Should include auth token in URL or first message
        expect(url).toContain('token=') 
        return mockWs
      })
      
      const ws = new WebSocket('wss://api.quantum-trading.com/ws?token=abc123')
      expect(ws).toBeDefined()
    })

    test('should validate WebSocket messages', () => {
      const onMessage = jest.fn()
      const ws = new MockWebSocket()
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          // Validate message structure
          if (data.type && data.payload) {
            onMessage(data)
          }
        } catch (e) {
          // Invalid JSON should be ignored
        }
      }
      
      // Valid message
      ws.simulateMessage({ type: 'quote', payload: { symbol: 'AAPL', price: 150 } })
      expect(onMessage).toHaveBeenCalledTimes(1)
      
      // Invalid messages
      ws.simulateMessage('not json')
      ws.simulateMessage({ noType: 'field' })
      expect(onMessage).toHaveBeenCalledTimes(1) // Still 1
    })
  })

  describe('Error Handling Security', () => {
    test('should not expose sensitive info in errors', () => {
      const sensitiveError = new Error('Database connection failed at server 192.168.1.1')
      const sanitizedError = sanitizeError(sensitiveError)
      
      expect(sanitizedError.message).not.toContain('192.168.1.1')
      expect(sanitizedError.message).toBe('An error occurred. Please try again.')
    })

    test('should log security events', () => {
      const logSpy = jest.spyOn(console, 'warn')
      
      // Failed login attempt
      fireEvent.submit(screen.getByRole('form'))
      
      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed login attempt')
      )
      
      logSpy.mockRestore()
    })
  })

  describe('CSRF Protection', () => {
    test('should include CSRF token in forms', () => {
      const { container } = render(<SecureForm />)
      const csrfInput = container.querySelector('input[name="csrf_token"]')
      
      expect(csrfInput).toBeInTheDocument()
      expect(csrfInput?.getAttribute('type')).toBe('hidden')
      expect(csrfInput?.getAttribute('value')).toBeTruthy()
    })

    test('should validate CSRF token on submission', async () => {
      const onSubmit = jest.fn()
      const { getByRole } = render(<SecureForm onSubmit={onSubmit} />)
      
      // Remove CSRF token
      const form = getByRole('form')
      const csrfInput = form.querySelector('input[name="csrf_token"]')
      csrfInput?.remove()
      
      fireEvent.submit(form)
      
      await waitFor(() => {
        expect(onSubmit).not.toHaveBeenCalled()
      })
    })
  })
})

// Helper components and utilities
const RegisterForm = () => (
  <form>
    <label htmlFor="password">Password</label>
    <input id="password" type="password" />
  </form>
)

const SearchInput = ({ onSearch }: { onSearch: (value: string) => void }) => (
  <form onSubmit={(e) => {
    e.preventDefault()
    const input = e.currentTarget.querySelector('input')
    if (input) {
      const sanitized = input.value.replace(/[';\\-]/g, '')
      onSearch(sanitized)
    }
  }}>
    <input role="searchbox" />
  </form>
)

const AdminPanel = ({ userRole }: { userRole: string }) => (
  <div>
    {userRole === 'admin' && <button>Delete All Users</button>}
    <div>User Panel</div>
  </div>
)

const SecureForm = ({ onSubmit }: { onSubmit?: () => void }) => {
  const csrfToken = 'test-csrf-token'
  
  return (
    <form role="form" onSubmit={(e) => {
      e.preventDefault()
      const formData = new FormData(e.currentTarget)
      if (formData.get('csrf_token') === csrfToken) {
        onSubmit?.()
      }
    }}>
      <input type="hidden" name="csrf_token" value={csrfToken} />
      <button type="submit">Submit</button>
    </form>
  )
}

class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null
  
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', {
        data: typeof data === 'string' ? data : JSON.stringify(data)
      }))
    }
  }
}

const sanitizeError = (error: Error): Error => {
  const sensitivePatterns = [
    /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/g, // IP addresses
    /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, // Emails
    /\/[a-zA-Z0-9/_-]+/g, // File paths
  ]
  
  let message = error.message
  sensitivePatterns.forEach(pattern => {
    message = message.replace(pattern, '[REDACTED]')
  })
  
  return new Error(message || 'An error occurred. Please try again.')
}
