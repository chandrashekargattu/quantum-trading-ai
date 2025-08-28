import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import LoginPage from '../page'

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

// Mock fetch
global.fetch = jest.fn()

describe('LoginPage', () => {
  const mockPush = jest.fn()
  const user = userEvent.setup()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
    localStorage.clear()
  })

  afterEach(() => {
    jest.resetAllMocks()
  })

  it('renders login form correctly', () => {
    render(<LoginPage />)
    
    expect(screen.getByText('Sign in to your account')).toBeInTheDocument()
    expect(screen.getByLabelText('Email Address')).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByText(/don't have an account/i)).toBeInTheDocument()
  })

  it('displays branding panel on large screens', () => {
    render(<LoginPage />)
    
    expect(screen.getByText('Welcome Back to Smarter Trading')).toBeInTheDocument()
    expect(screen.getByText('Quantum Trading AI')).toBeInTheDocument()
  })

  it('shows validation errors for empty fields', async () => {
    render(<LoginPage />)
    
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.click(submitButton)
    
    // HTML5 validation should prevent submission
    expect(fetch).not.toHaveBeenCalled()
  })

  it('toggles password visibility', async () => {
    render(<LoginPage />)
    
    const passwordInput = screen.getByLabelText('Password') as HTMLInputElement
    const toggleButton = screen.getByRole('button', { name: '' }) // Eye icon button
    
    expect(passwordInput.type).toBe('password')
    
    await user.click(toggleButton)
    expect(passwordInput.type).toBe('text')
    
    await user.click(toggleButton)
    expect(passwordInput.type).toBe('password')
  })

  it('handles successful login', async () => {
    const mockResponse = {
      access_token: 'test-token',
      token_type: 'bearer',
    }
    
    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    })
    
    render(<LoginPage />)
    
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(emailInput, 'test@example.com')
    await user.type(passwordInput, 'password123')
    await user.click(submitButton)
    
    await waitFor(() => {
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
    })
    
    await waitFor(() => {
      expect(localStorage.getItem('access_token')).toBe('test-token')
      expect(localStorage.getItem('token_type')).toBe('bearer')
      expect(mockPush).toHaveBeenCalledWith('/dashboard')
    })
  })

  it('handles login failure with error message', async () => {
    const errorMessage = 'Invalid credentials'
    
    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: errorMessage }),
    })
    
    render(<LoginPage />)
    
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(emailInput, 'test@example.com')
    await user.type(passwordInput, 'wrongpassword')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument()
    })
    
    expect(localStorage.getItem('access_token')).toBeNull()
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('handles network error', async () => {
    ;(fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'))
    
    render(<LoginPage />)
    
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(emailInput, 'test@example.com')
    await user.type(passwordInput, 'password123')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument()
    })
  })

  it('disables form during submission', async () => {
    ;(fetch as jest.Mock).mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 1000))
    )
    
    render(<LoginPage />)
    
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })
    
    await user.type(emailInput, 'test@example.com')
    await user.type(passwordInput, 'password123')
    await user.click(submitButton)
    
    expect(emailInput).toBeDisabled()
    expect(passwordInput).toBeDisabled()
    expect(submitButton).toBeDisabled()
    expect(screen.getByText('Signing in...')).toBeInTheDocument()
  })

  it('links to registration page', () => {
    render(<LoginPage />)
    
    const signUpLink = screen.getByRole('link', { name: /sign up for free/i })
    expect(signUpLink).toHaveAttribute('href', '/auth/register')
  })

  it('links to forgot password page', () => {
    render(<LoginPage />)
    
    const forgotPasswordLink = screen.getByRole('link', { name: /forgot password/i })
    expect(forgotPasswordLink).toHaveAttribute('href', '/auth/forgot-password')
  })

  it('renders social login buttons', () => {
    render(<LoginPage />)
    
    expect(screen.getByRole('button', { name: /google/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /github/i })).toBeInTheDocument()
  })
})