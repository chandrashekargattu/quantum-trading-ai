import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import RegisterPage from '../page'

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

// Mock fetch
global.fetch = jest.fn()

// Mock alert
global.alert = jest.fn()

describe('RegisterPage', () => {
  const mockPush = jest.fn()
  const user = userEvent.setup()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  afterEach(() => {
    jest.resetAllMocks()
  })

  it('renders registration form correctly', () => {
    render(<RegisterPage />)
    
    expect(screen.getByText('Create your account')).toBeInTheDocument()
    expect(screen.getByLabelText('First Name')).toBeInTheDocument()
    expect(screen.getByLabelText('Last Name')).toBeInTheDocument()
    expect(screen.getByLabelText('Email Address')).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByLabelText('Confirm Password')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument()
  })

  it('displays branding panel with features', () => {
    render(<RegisterPage />)
    
    expect(screen.getByText('Start Your Journey to Smarter Trading')).toBeInTheDocument()
    expect(screen.getByText('Real-time market analysis with AI predictions')).toBeInTheDocument()
    expect(screen.getByText('Automated trading strategies that work 24/7')).toBeInTheDocument()
    expect(screen.getByText('Advanced risk management to protect your capital')).toBeInTheDocument()
  })

  it('validates password match', async () => {
    render(<RegisterPage />)
    
    const passwordInput = screen.getByLabelText('Password')
    const confirmPasswordInput = screen.getByLabelText('Confirm Password')
    const submitButton = screen.getByRole('button', { name: /create account/i })
    
    await user.type(passwordInput, 'password123')
    await user.type(confirmPasswordInput, 'password456')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText('Passwords do not match')).toBeInTheDocument()
    })
    
    expect(fetch).not.toHaveBeenCalled()
  })

  it('validates password length', async () => {
    render(<RegisterPage />)
    
    const passwordInput = screen.getByLabelText('Password')
    const confirmPasswordInput = screen.getByLabelText('Confirm Password')
    const submitButton = screen.getByRole('button', { name: /create account/i })
    
    await user.type(passwordInput, 'short')
    await user.type(confirmPasswordInput, 'short')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText('Password must be at least 8 characters long')).toBeInTheDocument()
    })
    
    expect(fetch).not.toHaveBeenCalled()
  })

  it('toggles password visibility for both fields', async () => {
    render(<RegisterPage />)
    
    const passwordInput = screen.getByLabelText('Password') as HTMLInputElement
    const confirmPasswordInput = screen.getByLabelText('Confirm Password') as HTMLInputElement
    const toggleButtons = screen.getAllByRole('button', { name: '' }).filter(btn => 
      btn.querySelector('svg')
    )
    
    expect(passwordInput.type).toBe('password')
    expect(confirmPasswordInput.type).toBe('password')
    
    // Toggle password field
    await user.click(toggleButtons[0])
    expect(passwordInput.type).toBe('text')
    
    // Toggle confirm password field
    await user.click(toggleButtons[1])
    expect(confirmPasswordInput.type).toBe('text')
    
    // Toggle back
    await user.click(toggleButtons[0])
    await user.click(toggleButtons[1])
    expect(passwordInput.type).toBe('password')
    expect(confirmPasswordInput.type).toBe('password')
  })

  it('handles successful registration', async () => {
    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: 'User created successfully' }),
    })
    
    render(<RegisterPage />)
    
    const firstNameInput = screen.getByLabelText('First Name')
    const lastNameInput = screen.getByLabelText('Last Name')
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const confirmPasswordInput = screen.getByLabelText('Confirm Password')
    const submitButton = screen.getByRole('button', { name: /create account/i })
    
    await user.type(firstNameInput, 'John')
    await user.type(lastNameInput, 'Doe')
    await user.type(emailInput, 'john@example.com')
    await user.type(passwordInput, 'password123')
    await user.type(confirmPasswordInput, 'password123')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('http://localhost:8000/api/v1/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          first_name: 'John',
          last_name: 'Doe',
          email: 'john@example.com',
          password: 'password123',
        }),
      })
    })
    
    await waitFor(() => {
      expect(alert).toHaveBeenCalledWith('Success! Your account has been created. Please login.')
      expect(mockPush).toHaveBeenCalledWith('/auth/login')
    })
  })

  it('handles registration failure', async () => {
    const errorMessage = 'Email already exists'
    
    ;(fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: errorMessage }),
    })
    
    render(<RegisterPage />)
    
    const firstNameInput = screen.getByLabelText('First Name')
    const lastNameInput = screen.getByLabelText('Last Name')
    const emailInput = screen.getByLabelText('Email Address')
    const passwordInput = screen.getByLabelText('Password')
    const confirmPasswordInput = screen.getByLabelText('Confirm Password')
    const submitButton = screen.getByRole('button', { name: /create account/i })
    
    await user.type(firstNameInput, 'John')
    await user.type(lastNameInput, 'Doe')
    await user.type(emailInput, 'existing@example.com')
    await user.type(passwordInput, 'password123')
    await user.type(confirmPasswordInput, 'password123')
    await user.click(submitButton)
    
    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument()
    })
    
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('disables form during submission', async () => {
    ;(fetch as jest.Mock).mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 1000))
    )
    
    render(<RegisterPage />)
    
    const submitButton = screen.getByRole('button', { name: /create account/i })
    const inputs = [
      screen.getByLabelText('First Name'),
      screen.getByLabelText('Last Name'),
      screen.getByLabelText('Email Address'),
      screen.getByLabelText('Password'),
      screen.getByLabelText('Confirm Password'),
    ]
    
    // Fill form
    await user.type(inputs[0], 'John')
    await user.type(inputs[1], 'Doe')
    await user.type(inputs[2], 'john@example.com')
    await user.type(inputs[3], 'password123')
    await user.type(inputs[4], 'password123')
    
    await user.click(submitButton)
    
    inputs.forEach(input => {
      expect(input).toBeDisabled()
    })
    expect(submitButton).toBeDisabled()
    expect(screen.getByText('Creating Account...')).toBeInTheDocument()
  })

  it('displays icons in input fields', () => {
    render(<RegisterPage />)
    
    // Check for user icons (first name and last name)
    const userIcons = screen.getAllByTestId('user-icon')
    expect(userIcons).toHaveLength(2)
    
    // Check for mail icon
    expect(screen.getByTestId('mail-icon')).toBeInTheDocument()
    
    // Check for lock icons (password fields)
    const lockIcons = screen.getAllByTestId('lock-icon')
    expect(lockIcons).toHaveLength(2)
  })

  it('links to login page', () => {
    render(<RegisterPage />)
    
    const signInLink = screen.getByRole('link', { name: /sign in/i })
    expect(signInLink).toHaveAttribute('href', '/auth/login')
  })

  it('links to terms and privacy policy', () => {
    render(<RegisterPage />)
    
    const termsLink = screen.getByRole('link', { name: /terms of service/i })
    const privacyLink = screen.getByRole('link', { name: /privacy policy/i })
    
    expect(termsLink).toHaveAttribute('href', '/terms')
    expect(privacyLink).toHaveAttribute('href', '/privacy')
  })

  it('renders social signup buttons', () => {
    render(<RegisterPage />)
    
    expect(screen.getByRole('button', { name: /google/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /github/i })).toBeInTheDocument()
  })

  it('validates email format', async () => {
    render(<RegisterPage />)
    
    const emailInput = screen.getByLabelText('Email Address') as HTMLInputElement
    
    await user.type(emailInput, 'invalid-email')
    
    // HTML5 email validation
    expect(emailInput.validity.valid).toBe(false)
  })
})
