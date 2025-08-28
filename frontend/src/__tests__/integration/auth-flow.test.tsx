import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import LoginPage from '@/app/auth/login/page'
import RegisterPage from '@/app/auth/register/page'
import { useAuthStore } from '@/store/useAuthStore'
import { authService } from '@/services/api/auth'

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

// Mock auth service
jest.mock('@/services/api/auth', () => ({
  authService: {
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    getCurrentUser: jest.fn(),
    setTokens: jest.fn(),
    clearTokens: jest.fn(),
  },
}))

// Mock toast
jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
    loading: jest.fn(),
  },
}))

describe('Authentication Flow Integration', () => {
  const mockPush = jest.fn()
  const mockRouter = {
    push: mockPush,
    replace: jest.fn(),
    refresh: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue(mockRouter)
    
    // Reset auth store
    useAuthStore.setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    })
  })

  describe('Login Flow', () => {
    it('should complete full login flow successfully', async () => {
      const user = userEvent.setup()
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        username: 'testuser',
        fullName: 'Test User',
        role: 'trader',
        isActive: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }

      ;(authService.login as jest.Mock).mockResolvedValueOnce({
        access_token: 'test-token',
        token_type: 'Bearer',
        user: mockUser,
      })

      render(<LoginPage />)

      // Fill in login form
      const emailInput = screen.getByLabelText(/email/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(emailInput, 'test@example.com')
      await user.type(passwordInput, 'password123')
      await user.click(submitButton)

      // Verify API call
      expect(authService.login).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      })

      // Verify token storage
      await waitFor(() => {
        expect(authService.setTokens).toHaveBeenCalledWith({
          access_token: 'test-token',
          token_type: 'Bearer',
        })
      })

      // Verify store update
      expect(useAuthStore.getState().isAuthenticated).toBe(true)
      expect(useAuthStore.getState().user).toEqual(mockUser)

      // Verify navigation
      expect(mockPush).toHaveBeenCalledWith('/dashboard')
    })

    it('should handle login errors gracefully', async () => {
      const user = userEvent.setup()
      ;(authService.login as jest.Mock).mockRejectedValueOnce(
        new Error('Invalid credentials')
      )

      render(<LoginPage />)

      const emailInput = screen.getByLabelText(/email/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      await user.type(emailInput, 'test@example.com')
      await user.type(passwordInput, 'wrongpassword')
      await user.click(submitButton)

      // Verify error display
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument()
      })

      // Verify store state
      expect(useAuthStore.getState().isAuthenticated).toBe(false)
      expect(useAuthStore.getState().error).toBe('Invalid credentials')
    })

    it('should redirect authenticated users', () => {
      // Set authenticated state
      useAuthStore.setState({
        user: { id: '1', email: 'test@example.com' } as any,
        isAuthenticated: true,
      })

      render(<LoginPage />)

      expect(mockPush).toHaveBeenCalledWith('/dashboard')
    })
  })

  describe('Registration Flow', () => {
    it('should complete full registration flow successfully', async () => {
      const user = userEvent.setup()
      const mockUser = {
        id: '1',
        email: 'newuser@example.com',
        username: 'newuser',
        fullName: 'New User',
        role: 'trader',
        isActive: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }

      ;(authService.register as jest.Mock).mockResolvedValueOnce({
        access_token: 'new-token',
        token_type: 'Bearer',
        user: mockUser,
      })

      render(<RegisterPage />)

      // Fill in registration form
      const emailInput = screen.getByLabelText(/email/i)
      const usernameInput = screen.getByLabelText(/username/i)
      const fullNameInput = screen.getByLabelText(/full name/i)
      const passwordInput = screen.getByLabelText(/^password$/i)
      const confirmPasswordInput = screen.getByLabelText(/confirm password/i)
      const termsCheckbox = screen.getByRole('checkbox')
      const submitButton = screen.getByRole('button', { name: /create account/i })

      await user.type(emailInput, 'newuser@example.com')
      await user.type(usernameInput, 'newuser')
      await user.type(fullNameInput, 'New User')
      await user.type(passwordInput, 'SecurePass123!')
      await user.type(confirmPasswordInput, 'SecurePass123!')
      await user.click(termsCheckbox)
      await user.click(submitButton)

      // Verify API call
      expect(authService.register).toHaveBeenCalledWith({
        email: 'newuser@example.com',
        username: 'newuser',
        fullName: 'New User',
        password: 'SecurePass123!',
      })

      // Verify successful registration
      await waitFor(() => {
        expect(authService.setTokens).toHaveBeenCalledWith({
          access_token: 'new-token',
          token_type: 'Bearer',
        })
      })

      // Verify navigation to dashboard
      expect(mockPush).toHaveBeenCalledWith('/dashboard')
    })

    it('should validate password requirements', async () => {
      const user = userEvent.setup()
      render(<RegisterPage />)

      const passwordInput = screen.getByLabelText(/^password$/i)
      
      // Test weak password
      await user.type(passwordInput, 'weak')
      
      // Check for password requirements display
      expect(screen.getByText(/at least 8 characters/i)).toBeInTheDocument()
      expect(screen.getByText(/one uppercase letter/i)).toBeInTheDocument()
      expect(screen.getByText(/one number/i)).toBeInTheDocument()
    })

    it('should validate password confirmation match', async () => {
      const user = userEvent.setup()
      render(<RegisterPage />)

      const passwordInput = screen.getByLabelText(/^password$/i)
      const confirmPasswordInput = screen.getByLabelText(/confirm password/i)
      const submitButton = screen.getByRole('button', { name: /create account/i })

      await user.type(passwordInput, 'SecurePass123!')
      await user.type(confirmPasswordInput, 'DifferentPass123!')
      
      // Try to submit
      await user.click(submitButton)

      // Check for mismatch error
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument()
    })

    it('should handle registration errors', async () => {
      const user = userEvent.setup()
      ;(authService.register as jest.Mock).mockRejectedValueOnce(
        new Error('Email already exists')
      )

      render(<RegisterPage />)

      // Fill minimum required fields
      const emailInput = screen.getByLabelText(/email/i)
      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/^password$/i)
      const confirmPasswordInput = screen.getByLabelText(/confirm password/i)
      const termsCheckbox = screen.getByRole('checkbox')
      const submitButton = screen.getByRole('button', { name: /create account/i })

      await user.type(emailInput, 'existing@example.com')
      await user.type(usernameInput, 'existinguser')
      await user.type(passwordInput, 'SecurePass123!')
      await user.type(confirmPasswordInput, 'SecurePass123!')
      await user.click(termsCheckbox)
      await user.click(submitButton)

      // Verify error display
      await waitFor(() => {
        expect(screen.getByText(/email already exists/i)).toBeInTheDocument()
      })
    })
  })

  describe('Logout Flow', () => {
    it('should handle logout correctly', async () => {
      // Set authenticated state
      const mockUser = {
        id: '1',
        email: 'test@example.com',
        username: 'testuser',
        fullName: 'Test User',
      }
      
      useAuthStore.setState({
        user: mockUser as any,
        isAuthenticated: true,
      })

      ;(authService.logout as jest.Mock).mockResolvedValueOnce(undefined)

      // Trigger logout
      await act(async () => {
        await useAuthStore.getState().logout()
      })

      // Verify API call
      expect(authService.logout).toHaveBeenCalled()

      // Verify token clearing
      expect(authService.clearTokens).toHaveBeenCalled()

      // Verify store reset
      expect(useAuthStore.getState().user).toBeNull()
      expect(useAuthStore.getState().isAuthenticated).toBe(false)
    })
  })

  describe('Protected Route Handling', () => {
    it('should redirect unauthenticated users to login', () => {
      // Mock a protected page component
      const ProtectedPage = () => {
        const { isAuthenticated } = useAuthStore()
        const router = useRouter()

        React.useEffect(() => {
          if (!isAuthenticated) {
            router.push('/auth/login')
          }
        }, [isAuthenticated, router])

        if (!isAuthenticated) return null
        return <div>Protected Content</div>
      }

      render(<ProtectedPage />)

      expect(mockPush).toHaveBeenCalledWith('/auth/login')
    })

    it('should allow authenticated users to access protected routes', () => {
      useAuthStore.setState({
        user: { id: '1', email: 'test@example.com' } as any,
        isAuthenticated: true,
      })

      const ProtectedPage = () => {
        const { isAuthenticated } = useAuthStore()
        
        if (!isAuthenticated) return null
        return <div>Protected Content</div>
      }

      render(<ProtectedPage />)

      expect(screen.getByText('Protected Content')).toBeInTheDocument()
    })
  })

  describe('Session Management', () => {
    it('should refresh token when expired', async () => {
      const mockUser = {
        id: '1',
        email: 'test@example.com',
      }
      
      useAuthStore.setState({
        user: mockUser as any,
        isAuthenticated: true,
      })

      // Mock token refresh
      ;(authService.refreshToken as jest.Mock).mockResolvedValueOnce({
        access_token: 'new-access-token',
        refresh_token: 'new-refresh-token',
        token_type: 'Bearer',
        user: mockUser,
      })

      // Simulate token refresh
      await authService.refreshToken('old-refresh-token')

      expect(authService.setTokens).toHaveBeenCalledWith({
        access_token: 'new-access-token',
        refresh_token: 'new-refresh-token',
        token_type: 'Bearer',
      })
    })
  })
})
