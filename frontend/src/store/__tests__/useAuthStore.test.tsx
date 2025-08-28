import { renderHook, act, waitFor } from '@testing-library/react'
import { useAuthStore } from '../useAuthStore'
import { authService } from '@/services/api/auth'
import { mockUser } from '@/test-utils/test-utils'

// Mock the auth service
jest.mock('@/services/api/auth')

describe('useAuthStore', () => {
  beforeEach(() => {
    // Reset store state
    const { result } = renderHook(() => useAuthStore())
    act(() => {
      result.current.logout()
    })
    
    // Clear mocks
    jest.clearAllMocks()
    
    // Clear localStorage
    localStorage.clear()
  })

  it('initializes with default state', () => {
    const { result } = renderHook(() => useAuthStore())

    expect(result.current.user).toBeNull()
    expect(result.current.isAuthenticated).toBe(false)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  describe('login', () => {
    it('successfully logs in user', async () => {
      const mockToken = { 
        access_token: 'test-token', 
        refresh_token: 'refresh-token',
        token_type: 'bearer'
      }
      
      ;(authService.login as jest.Mock).mockResolvedValue(mockToken)
      ;(authService.getCurrentUser as jest.Mock).mockResolvedValue(mockUser)

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        await result.current.login('testuser', 'password123')
      })

      expect(authService.login).toHaveBeenCalledWith({
        username: 'testuser',
        password: 'password123',
      })
      expect(authService.getCurrentUser).toHaveBeenCalled()
      expect(result.current.user).toEqual(mockUser)
      expect(result.current.isAuthenticated).toBe(true)
      expect(result.current.isLoading).toBe(false)
      expect(result.current.error).toBeNull()
    })

    it('handles login failure', async () => {
      const errorMessage = 'Invalid credentials'
      ;(authService.login as jest.Mock).mockRejectedValue({
        response: { data: { detail: errorMessage } }
      })

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        try {
          await result.current.login('testuser', 'wrongpassword')
        } catch (error) {
          // Expected error
        }
      })

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
      expect(result.current.isLoading).toBe(false)
      expect(result.current.error).toBe(errorMessage)
    })

    it('sets loading state during login', async () => {
      ;(authService.login as jest.Mock).mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 100))
      )

      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.login('testuser', 'password123')
      })

      expect(result.current.isLoading).toBe(true)

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false)
      })
    })
  })

  describe('logout', () => {
    it('successfully logs out user', async () => {
      // Set up authenticated state
      const { result } = renderHook(() => useAuthStore())
      act(() => {
        result.current.user = mockUser
        result.current.isAuthenticated = true
      })

      ;(authService.logout as jest.Mock).mockResolvedValue(undefined)

      await act(async () => {
        await result.current.logout()
      })

      expect(authService.logout).toHaveBeenCalled()
      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
      expect(result.current.error).toBeNull()
    })

    it('clears state even if logout API fails', async () => {
      const { result } = renderHook(() => useAuthStore())
      act(() => {
        result.current.user = mockUser
        result.current.isAuthenticated = true
      })

      ;(authService.logout as jest.Mock).mockRejectedValue(new Error('API Error'))

      await act(async () => {
        await result.current.logout()
      })

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })
  })

  describe('register', () => {
    it('successfully registers user', async () => {
      const registrationData = {
        email: 'new@example.com',
        username: 'newuser',
        password: 'Password123!',
        full_name: 'New User',
      }

      ;(authService.register as jest.Mock).mockResolvedValue(mockUser)

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        await result.current.register(registrationData)
      })

      expect(authService.register).toHaveBeenCalledWith(registrationData)
      expect(result.current.user).toEqual(mockUser)
      expect(result.current.isAuthenticated).toBe(false) // Not authenticated after registration
      expect(result.current.error).toBeNull()
    })

    it('handles registration failure', async () => {
      const errorMessage = 'Email already exists'
      ;(authService.register as jest.Mock).mockRejectedValue({
        response: { data: { detail: errorMessage } }
      })

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        try {
          await result.current.register({
            email: 'existing@example.com',
            username: 'existinguser',
            password: 'Password123!',
          })
        } catch (error) {
          // Expected error
        }
      })

      expect(result.current.error).toBe(errorMessage)
      expect(result.current.user).toBeNull()
    })
  })

  describe('fetchUser', () => {
    it('fetches user when token exists', async () => {
      localStorage.setItem('auth_token', 'test-token')
      ;(authService.getCurrentUser as jest.Mock).mockResolvedValue(mockUser)

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        await result.current.fetchUser()
      })

      expect(authService.getCurrentUser).toHaveBeenCalled()
      expect(result.current.user).toEqual(mockUser)
      expect(result.current.isAuthenticated).toBe(true)
    })

    it('does not fetch user when no token exists', async () => {
      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        await result.current.fetchUser()
      })

      expect(authService.getCurrentUser).not.toHaveBeenCalled()
      expect(result.current.isAuthenticated).toBe(false)
    })

    it('handles fetch user failure gracefully', async () => {
      localStorage.setItem('auth_token', 'invalid-token')
      ;(authService.getCurrentUser as jest.Mock).mockRejectedValue(new Error('Unauthorized'))

      const { result } = renderHook(() => useAuthStore())

      await act(async () => {
        await result.current.fetchUser()
      })

      expect(result.current.user).toBeNull()
      expect(result.current.isAuthenticated).toBe(false)
    })
  })

  describe('clearError', () => {
    it('clears error state', () => {
      const { result } = renderHook(() => useAuthStore())
      
      act(() => {
        result.current.error = 'Some error'
      })

      expect(result.current.error).toBe('Some error')

      act(() => {
        result.current.clearError()
      })

      expect(result.current.error).toBeNull()
    })
  })

  describe('persistence', () => {
    it('persists user and authentication state', () => {
      const { result } = renderHook(() => useAuthStore())
      
      act(() => {
        result.current.user = mockUser
        result.current.isAuthenticated = true
      })

      // Simulate page reload by creating new hook instance
      const { result: newResult } = renderHook(() => useAuthStore())

      expect(newResult.current.user).toEqual(mockUser)
      expect(newResult.current.isAuthenticated).toBe(true)
    })

    it('does not persist loading and error states', () => {
      const { result } = renderHook(() => useAuthStore())
      
      act(() => {
        result.current.isLoading = true
        result.current.error = 'Some error'
      })

      // Simulate page reload
      const { result: newResult } = renderHook(() => useAuthStore())

      expect(newResult.current.isLoading).toBe(false)
      expect(newResult.current.error).toBeNull()
    })
  })
})
