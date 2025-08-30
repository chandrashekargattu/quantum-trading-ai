const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterCredentials {
  email: string
  password: string
  username: string
  fullName?: string
}

export interface User {
  id: string
  email: string
  username: string
  fullName?: string
  role: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}

export interface AuthResponse {
  access_token: string
  refresh_token?: string
  token_type: string
  expires_in?: number
  user: User
}

export interface PasswordResetRequest {
  email: string
}

export interface PasswordResetConfirm {
  token: string
  newPassword: string
}

export interface PasswordChange {
  currentPassword: string
  newPassword: string
}

class AuthService {
  private baseUrl = `${API_BASE_URL}/api/v1/auth`

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    // Backend expects form data with username field
    const formData = new URLSearchParams()
    formData.append('username', credentials.email) // Backend uses username field for email
    formData.append('password', credentials.password)
    
    const response = await fetch(`${this.baseUrl}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString()
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Login failed')
    }
    
    const data = await response.json()
    
    // The backend doesn't return user data, so we need to fetch it
    if (data.access_token) {
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('token_type', data.token_type || 'bearer')
      if (data.refresh_token) {
        localStorage.setItem('refresh_token', data.refresh_token)
      }
      
      // Fetch user data
      const userResponse = await fetch(`${API_BASE_URL}/api/v1/users/me`, {
        headers: {
          'Authorization': `${data.token_type || 'bearer'} ${data.access_token}`
        }
      })
      
      if (userResponse.ok) {
        const user = await userResponse.json()
        return { ...data, user }
      }
    }
    
    return data
  }

  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    // Map frontend fields to backend expected fields
    const registrationData = {
      email: credentials.email,
      username: credentials.username || credentials.email.split('@')[0], // Use email prefix as username if not provided
      password: credentials.password,
      full_name: credentials.fullName || ''
    }
    
    const response = await fetch(`${this.baseUrl}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(registrationData)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Registration failed')
    }
    
    const user = await response.json()
    
    // After successful registration, login to get tokens
    const loginResponse = await this.login({
      email: credentials.email,
      password: credentials.password
    })
    
    return loginResponse
  }

  async logout(refreshToken?: string): Promise<void> {
    const body = refreshToken ? { refresh_token: refreshToken } : {}
    
    await fetch(`${this.baseUrl}/logout`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  }

  async refreshToken(refreshToken: string): Promise<AuthResponse> {
    const response = await fetch(`${this.baseUrl}/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken })
    })
    
    if (!response.ok) {
      throw new Error('Token refresh failed')
    }
    
    return response.json()
  }

  async getCurrentUser(): Promise<User> {
    const response = await fetch(`${this.baseUrl}/me`, {
      headers: {
        'Authorization': `Bearer ${this.getAccessToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to fetch current user')
    }
    
    return response.json()
  }

  async requestPasswordReset(data: PasswordResetRequest): Promise<void> {
    const response = await fetch(`${this.baseUrl}/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Password reset request failed')
    }
  }

  async confirmPasswordReset(data: PasswordResetConfirm): Promise<void> {
    const response = await fetch(`${this.baseUrl}/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Password reset failed')
    }
  }

  async changePassword(data: PasswordChange): Promise<void> {
    const response = await fetch(`${this.baseUrl}/change-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAccessToken()}`
      },
      body: JSON.stringify(data)
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Password change failed')
    }
  }

  async verifyEmail(token: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/verify-email/${token}`, {
      method: 'POST'
    })
    
    if (!response.ok) {
      throw new Error('Email verification failed')
    }
  }

  async resendVerificationEmail(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/resend-verification`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.getAccessToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to resend verification email')
    }
  }

  // Token management
  setTokens(tokens: { access_token: string; refresh_token?: string; token_type: string }) {
    localStorage.setItem('access_token', tokens.access_token)
    localStorage.setItem('token_type', tokens.token_type)
    if (tokens.refresh_token) {
      localStorage.setItem('refresh_token', tokens.refresh_token)
    }
  }

  getAccessToken(): string | null {
    return localStorage.getItem('access_token')
  }

  getRefreshToken(): string | null {
    return localStorage.getItem('refresh_token')
  }

  getTokenType(): string {
    return localStorage.getItem('token_type') || 'Bearer'
  }

  clearTokens() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    localStorage.removeItem('token_type')
  }

  isAuthenticated(): boolean {
    return !!this.getAccessToken()
  }

  // Request interceptor for adding auth headers
  getAuthHeaders(): Record<string, string> {
    const token = this.getAccessToken()
    const tokenType = this.getTokenType()
    
    if (token) {
      return {
        'Authorization': `${tokenType} ${token}`
      }
    }
    
    return {}
  }
}

export const authService = new AuthService()