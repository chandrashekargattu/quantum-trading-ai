import axios from 'axios'
import { apiClient } from '../client'
import { toast } from 'react-hot-toast'

jest.mock('axios')
jest.mock('react-hot-toast')

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
}
Object.defineProperty(window, 'localStorage', { value: localStorageMock })

// Mock window.location
delete (window as any).location
window.location = { href: '' } as any

describe('ApiClient', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    localStorageMock.clear()
    window.location.href = ''
  })

  describe('token management', () => {
    it('sets token and stores in localStorage', () => {
      const token = 'test-token-123'
      apiClient.setToken(token)

      expect(localStorageMock.setItem).toHaveBeenCalledWith('auth_token', token)
    })

    it('clears token and removes from localStorage', () => {
      apiClient.clearToken()

      expect(localStorageMock.removeItem).toHaveBeenCalledWith('auth_token')
    })

    it('loads token from localStorage on initialization', () => {
      const token = 'stored-token-123'
      localStorageMock.getItem.mockReturnValue(token)

      apiClient.loadToken()

      expect(localStorageMock.getItem).toHaveBeenCalledWith('auth_token')
    })
  })

  describe('request interceptors', () => {
    it('adds authorization header when token is set', async () => {
      const token = 'test-token-123'
      const mockCreate = axios.create as jest.Mock
      const mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
        get: jest.fn().mockResolvedValue({ data: {} }),
      }
      mockCreate.mockReturnValue(mockInstance)

      // Get the request interceptor
      const requestInterceptor = mockInstance.interceptors.request.use.mock.calls[0][0]
      
      apiClient.setToken(token)
      
      const config = { headers: {} }
      const modifiedConfig = requestInterceptor(config)

      expect(modifiedConfig.headers.Authorization).toBe(`Bearer ${token}`)
    })
  })

  describe('response interceptors', () => {
    it('handles 401 errors by clearing token and redirecting', async () => {
      const mockCreate = axios.create as jest.Mock
      const mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
      }
      mockCreate.mockReturnValue(mockInstance)

      // Get the error interceptor
      const errorInterceptor = mockInstance.interceptors.response.use.mock.calls[0][1]

      const error = {
        response: { status: 401 },
        config: {},
      }

      await expect(errorInterceptor(error)).rejects.toEqual(error)
      expect(window.location.href).toBe('/auth/login')
    })

    it('shows toast error for non-silent requests', async () => {
      const mockCreate = axios.create as jest.Mock
      const mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
      }
      mockCreate.mockReturnValue(mockInstance)

      const errorInterceptor = mockInstance.interceptors.response.use.mock.calls[0][1]

      const error = {
        response: {
          status: 400,
          data: { detail: 'Bad request error' },
        },
        config: { silent: false },
      }

      await expect(errorInterceptor(error)).rejects.toEqual(error)
      expect(toast.error).toHaveBeenCalledWith('Bad request error')
    })

    it('does not show toast for silent requests', async () => {
      const mockCreate = axios.create as jest.Mock
      const mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
      }
      mockCreate.mockReturnValue(mockInstance)

      const errorInterceptor = mockInstance.interceptors.response.use.mock.calls[0][1]

      const error = {
        response: {
          status: 400,
          data: { detail: 'Bad request error' },
        },
        config: { silent: true },
      }

      await expect(errorInterceptor(error)).rejects.toEqual(error)
      expect(toast.error).not.toHaveBeenCalled()
    })

    it('handles network errors', async () => {
      const mockCreate = axios.create as jest.Mock
      const mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
      }
      mockCreate.mockReturnValue(mockInstance)

      const errorInterceptor = mockInstance.interceptors.response.use.mock.calls[0][1]

      const error = {
        request: {},
        config: {},
      }

      await expect(errorInterceptor(error)).rejects.toEqual(error)
      expect(toast.error).toHaveBeenCalledWith('Network error. Please check your connection.')
    })
  })

  describe('HTTP methods', () => {
    let mockInstance: any

    beforeEach(() => {
      const mockCreate = axios.create as jest.Mock
      mockInstance = {
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
        get: jest.fn().mockResolvedValue({ data: { result: 'get' } }),
        post: jest.fn().mockResolvedValue({ data: { result: 'post' } }),
        put: jest.fn().mockResolvedValue({ data: { result: 'put' } }),
        patch: jest.fn().mockResolvedValue({ data: { result: 'patch' } }),
        delete: jest.fn().mockResolvedValue({ data: { result: 'delete' } }),
      }
      mockCreate.mockReturnValue(mockInstance)
    })

    it('performs GET request', async () => {
      const result = await apiClient.get('/test')
      
      expect(mockInstance.get).toHaveBeenCalledWith('/test', undefined)
      expect(result).toEqual({ result: 'get' })
    })

    it('performs POST request', async () => {
      const data = { foo: 'bar' }
      const result = await apiClient.post('/test', data)
      
      expect(mockInstance.post).toHaveBeenCalledWith('/test', data, undefined)
      expect(result).toEqual({ result: 'post' })
    })

    it('performs PUT request', async () => {
      const data = { foo: 'bar' }
      const result = await apiClient.put('/test', data)
      
      expect(mockInstance.put).toHaveBeenCalledWith('/test', data, undefined)
      expect(result).toEqual({ result: 'put' })
    })

    it('performs PATCH request', async () => {
      const data = { foo: 'bar' }
      const result = await apiClient.patch('/test', data)
      
      expect(mockInstance.patch).toHaveBeenCalledWith('/test', data, undefined)
      expect(result).toEqual({ result: 'patch' })
    })

    it('performs DELETE request', async () => {
      const result = await apiClient.delete('/test')
      
      expect(mockInstance.delete).toHaveBeenCalledWith('/test', undefined)
      expect(result).toEqual({ result: 'delete' })
    })
  })
})
