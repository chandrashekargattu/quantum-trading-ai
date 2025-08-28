import { apiClient } from './client'

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  email: string
  username: string
  password: string
  full_name?: string
}

export interface AuthResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface User {
  id: string
  email: string
  username: string
  full_name?: string
  is_active: boolean
  is_verified: boolean
  account_type: string
  subscription_tier: string
  created_at: string
}

export const authService = {
  async login(data: LoginRequest): Promise<AuthResponse> {
    const formData = new FormData()
    formData.append('username', data.username)
    formData.append('password', data.password)

    const response = await apiClient.post<AuthResponse>('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })

    // Store tokens
    apiClient.setToken(response.access_token)
    if (typeof window !== 'undefined') {
      localStorage.setItem('refresh_token', response.refresh_token)
    }

    return response
  },

  async register(data: RegisterRequest): Promise<User> {
    return apiClient.post<User>('/auth/register', data)
  },

  async refresh(): Promise<AuthResponse> {
    const refreshToken = typeof window !== 'undefined' 
      ? localStorage.getItem('refresh_token') 
      : null

    if (!refreshToken) {
      throw new Error('No refresh token available')
    }

    const response = await apiClient.post<AuthResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    })

    // Update tokens
    apiClient.setToken(response.access_token)
    if (typeof window !== 'undefined') {
      localStorage.setItem('refresh_token', response.refresh_token)
    }

    return response
  },

  async logout(): Promise<void> {
    try {
      const token = typeof window !== 'undefined' 
        ? localStorage.getItem('auth_token') 
        : null
      
      if (token) {
        await apiClient.post('/auth/logout', { token })
      }
    } finally {
      // Clear tokens regardless of API response
      apiClient.clearToken()
      if (typeof window !== 'undefined') {
        localStorage.removeItem('refresh_token')
      }
    }
  },

  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    await apiClient.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>('/users/me')
  },
}
