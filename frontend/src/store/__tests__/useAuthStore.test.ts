import { renderHook, act } from '@testing-library/react'
import { useAuthStore } from '../useAuthStore'

// Mock fetch
global.fetch = jest.fn()

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
global.localStorage = localStorageMock as any

describe('useAuthStore', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    localStorageMock.clear()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useAuthStore())
    
    expect(result.current.user).toBeNull()
    expect(result.current.isAuthenticated).toBe(false)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('loads user from localStorage on initialization', () => {
    const mockUser = {
      id: '1',
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User',
    }
    
    localStorageMock.getItem.mockReturnValue(JSON.stringify(mockUser))
    
    const { result } = renderHook(() => useAuthStore())
    
    expect(localStorageMock.getItem).toHaveBeenCalledWith('user')
    expect(result.current.user).toEqual(mockUser)
    expect(result.current.isAuthenticated).toBe(true)
  })

  it('handles successful login', async () => {
    const mockResponse = {
      access_token: 'test-token',
      token_type: 'bearer',
    }
    
    const mockUser = {
      id: '1',
      email: 'test@example.com',
      first_name: 'Test',
      last_name: 'User',
    }
    
    ;(fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockUser,
      })
    
    const { result } = renderHook(() => useAuthStore())
    
    await act(async () => {
      await result.current.login('test@example.com', 'password123')
    })
    
    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        username: 'test@example.com',
        password: 'password123',
      }),
    })
    
    expect(localStorageMock.setItem).toHaveBeenCalledWith('access_token', 'test-token')
    expect(localStorageMock.setItem).toHaveBeenCalledWith('token_type', 'bearer')
    expect(localStorageMock.setItem).toHaveBeenCalledWith('user', JSON.stringify({
      id: '1',
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User',
    }))
    
    expect(result.current.isAuthenticated).toBe(true)
    expect(result.current.user).toEqual({
      id: '1',
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User',
    })
    expect(result.current.error).toBeNull()
  })

  it('handles login failure', async () => {
    const errorMessage = 'Invalid credentials'
    
    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: errorMessage }),
    })
    
    const { result } = renderHook(() => useAuthStore())
    
    await act(async () => {
      try {
        await result.current.login('test@example.com', 'wrongpassword')
      } catch (error) {
        // Expected error
      }
    })
    
    expect(result.current.isAuthenticated).toBe(false)
    expect(result.current.user).toBeNull()
    expect(result.current.error).toBe(errorMessage)
    expect(localStorageMock.setItem).not.toHaveBeenCalled()
  })

  it('handles network error during login', async () => {
    ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))
    
    const { result } = renderHook(() => useAuthStore())
    
    await act(async () => {
      try {
        await result.current.login('test@example.com', 'password')
      } catch (error) {
        // Expected error
      }
    })
    
    expect(result.current.error).toBe('Network error')
    expect(result.current.isAuthenticated).toBe(false)
  })

  it('handles logout', () => {
    const { result } = renderHook(() => useAuthStore())
    
    // Set initial authenticated state
    act(() => {
      result.current.user = {
        id: '1',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
      }
      result.current.isAuthenticated = true
    })
    
    act(() => {
      result.current.logout()
    })
    
    expect(localStorageMock.removeItem).toHaveBeenCalledWith('access_token')
    expect(localStorageMock.removeItem).toHaveBeenCalledWith('token_type')
    expect(localStorageMock.removeItem).toHaveBeenCalledWith('user')
    expect(result.current.user).toBeNull()
    expect(result.current.isAuthenticated).toBe(false)
  })

  it('clears error', () => {
    const { result } = renderHook(() => useAuthStore())
    
    // Set initial error
    act(() => {
      result.current.error = 'Some error'
    })
    
    act(() => {
      result.current.clearError()
    })
    
    expect(result.current.error).toBeNull()
  })

  it('handles loading state during login', async () => {
    let resolvePromise: (value: any) => void
    const promise = new Promise((resolve) => {
      resolvePromise = resolve
    })
    
    ;(fetch as jest.Mock).mockReturnValueOnce(promise)
    
    const { result } = renderHook(() => useAuthStore())
    
    // Start login
    const loginPromise = act(async () => {
      return result.current.login('test@example.com', 'password')
    })
    
    // Check loading state
    expect(result.current.isLoading).toBe(true)
    
    // Resolve the promise
    act(() => {
      resolvePromise!({
        ok: true,
        json: async () => ({ access_token: 'token', token_type: 'bearer' }),
      })
    })
    
    await loginPromise
    
    expect(result.current.isLoading).toBe(false)
  })

  it('validates required fields', async () => {
    const { result } = renderHook(() => useAuthStore())
    
    await act(async () => {
      try {
        await result.current.login('', 'password')
      } catch (error: any) {
        expect(error.message).toContain('Email is required')
      }
    })
    
    await act(async () => {
      try {
        await result.current.login('test@example.com', '')
      } catch (error: any) {
        expect(error.message).toContain('Password is required')
      }
    })
  })
})
