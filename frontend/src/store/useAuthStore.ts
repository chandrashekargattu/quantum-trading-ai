import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  username: string
  full_name: string | null
  account_type: string
  subscription_tier: string
  is_active: boolean
  is_verified: boolean
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  clearError: () => void
  fetchUser: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        // Validation
        if (!email) {
          throw new Error('Email is required')
        }
        if (!password) {
          throw new Error('Password is required')
        }

        set({ isLoading: true, error: null })

        try {
          // Login request
          const loginResponse = await fetch('http://localhost:8000/api/v1/auth/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
              username: email,
              password: password,
            }),
          })

          if (!loginResponse.ok) {
            const error = await loginResponse.json()
            throw new Error(error.detail || 'Login failed')
          }

          const { access_token, token_type } = await loginResponse.json()
          
          // Store tokens
          localStorage.setItem('access_token', access_token)
          localStorage.setItem('token_type', token_type)

          // Get user info
          const userResponse = await fetch('http://localhost:8000/api/v1/auth/me', {
            headers: {
              'Authorization': `${token_type} ${access_token}`,
            },
          })

          if (!userResponse.ok) {
            throw new Error('Failed to get user info')
          }

          const userData = await userResponse.json()
          
          // Use user data directly from API
          const user: User = userData

          // Store user in localStorage
          localStorage.setItem('user', JSON.stringify(user))

          set({
            user,
            isAuthenticated: true,
            isLoading: false,
            error: null,
          })
        } catch (error: any) {
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            error: error.message || 'An error occurred',
          })
          throw error
        }
      },

      logout: () => {
        // Clear tokens and user data
        localStorage.removeItem('access_token')
        localStorage.removeItem('token_type')
        localStorage.removeItem('user')
        
        set({
          user: null,
          isAuthenticated: false,
          error: null,
        })
      },

      clearError: () => {
        set({ error: null })
      },

      fetchUser: async () => {
        const token = localStorage.getItem('access_token')
        const tokenType = localStorage.getItem('token_type')

        if (!token || !tokenType) {
          set({ isAuthenticated: false, user: null })
          return
        }

        set({ isLoading: true })

        try {
          const response = await fetch('http://localhost:8000/api/v1/auth/me', {
            headers: {
              'Authorization': `${tokenType} ${token}`,
            },
          })

          if (!response.ok) {
            throw new Error('Failed to fetch user')
          }

          const userData = await response.json()
          const user: User = userData

          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (error) {
          localStorage.removeItem('access_token')
          localStorage.removeItem('token_type')
          localStorage.removeItem('user')
          
          set({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          })
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)