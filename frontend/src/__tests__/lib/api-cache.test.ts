import { cachedFetch, prefetch, clearCache, getCacheStats } from '@/lib/api-cache'
import { authenticatedFetch } from '@/lib/auth-interceptor'

// Mock auth interceptor
jest.mock('@/lib/auth-interceptor', () => ({
  authenticatedFetch: jest.fn()
}))

const mockAuthenticatedFetch = authenticatedFetch as jest.MockedFunction<typeof authenticatedFetch>

describe('API Cache', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    clearCache() // Clear cache before each test
    
    // Default successful response
    mockAuthenticatedFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'test' }),
      text: async () => 'test'
    } as Response)
  })

  describe('cachedFetch', () => {
    it('should fetch and cache successful responses', async () => {
      const url = 'http://api.test/data'
      const result = await cachedFetch(url)

      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)
      expect(mockAuthenticatedFetch).toHaveBeenCalledWith(url, undefined)
      expect(result).toEqual({ data: 'test' })
    })

    it('should return cached data on subsequent calls', async () => {
      const url = 'http://api.test/data'
      
      // First call
      const result1 = await cachedFetch(url)
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)

      // Second call should use cache
      const result2 = await cachedFetch(url)
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1) // Not called again
      expect(result2).toEqual(result1)
    })

    it('should respect TTL', async () => {
      const url = 'http://api.test/data'
      const ttl = 100 // 100ms

      // First call with short TTL
      await cachedFetch(url, undefined, { ttl })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)

      // Wait for cache to expire
      await new Promise(resolve => setTimeout(resolve, 150))

      // Should fetch again
      await cachedFetch(url, undefined, { ttl })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(2)
    })

    it('should force refresh when requested', async () => {
      const url = 'http://api.test/data'
      
      // First call
      await cachedFetch(url)
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)

      // Force refresh
      await cachedFetch(url, undefined, { forceRefresh: true })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(2)
    })

    it('should deduplicate concurrent requests', async () => {
      const url = 'http://api.test/data'
      
      // Make multiple concurrent requests
      const promises = [
        cachedFetch(url),
        cachedFetch(url),
        cachedFetch(url)
      ]

      const results = await Promise.all(promises)

      // Should only fetch once
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)
      expect(results[0]).toEqual(results[1])
      expect(results[1]).toEqual(results[2])
    })

    it('should not deduplicate when disabled', async () => {
      const url = 'http://api.test/data'
      
      // Make multiple concurrent requests with dedupe disabled
      const promises = [
        cachedFetch(url, undefined, { dedupe: false }),
        cachedFetch(url, undefined, { dedupe: false })
      ]

      await Promise.all(promises)

      // Should fetch multiple times
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(2)
    })

    it('should handle 401 errors', async () => {
      mockAuthenticatedFetch.mockResolvedValue({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized'
      } as Response)

      const url = 'http://api.test/data'
      
      await expect(cachedFetch(url)).rejects.toThrow('Authentication required')
      
      // Should not cache 401 errors
      const stats = getCacheStats()
      expect(stats.size).toBe(0)
    })

    it('should handle other HTTP errors', async () => {
      mockAuthenticatedFetch.mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Server error'
      } as Response)

      const url = 'http://api.test/data'
      
      await expect(cachedFetch(url)).rejects.toThrow('HTTP error! status: 500: Server error')
    })

    it('should cache based on URL and method', async () => {
      const url = 'http://api.test/data'
      
      // GET request
      await cachedFetch(url)
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)

      // POST request to same URL should not use cache
      await cachedFetch(url, { method: 'POST' })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(2)
    })

    it('should cache based on request body', async () => {
      const url = 'http://api.test/data'
      
      // POST with body 1
      await cachedFetch(url, {
        method: 'POST',
        body: JSON.stringify({ id: 1 })
      })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)

      // POST with body 2 should not use cache
      await cachedFetch(url, {
        method: 'POST',
        body: JSON.stringify({ id: 2 })
      })
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(2)
    })
  })

  describe('prefetch', () => {
    it('should fetch data without throwing errors', async () => {
      const url = 'http://api.test/data'
      
      // Should not throw
      expect(() => prefetch(url)).not.toThrow()
      
      // Wait for prefetch to complete
      await new Promise(resolve => setTimeout(resolve, 50))
      
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)
    })

    it('should silently handle errors', async () => {
      mockAuthenticatedFetch.mockRejectedValue(new Error('Network error'))
      
      const url = 'http://api.test/data'
      
      // Should not throw
      expect(() => prefetch(url)).not.toThrow()
    })

    it('should cache prefetched data', async () => {
      const url = 'http://api.test/data'
      
      prefetch(url)
      
      // Wait for prefetch to complete
      await new Promise(resolve => setTimeout(resolve, 50))
      
      // Subsequent fetch should use cache
      await cachedFetch(url)
      expect(mockAuthenticatedFetch).toHaveBeenCalledTimes(1)
    })
  })

  describe('clearCache', () => {
    it('should clear all cached entries', async () => {
      // Cache some data
      await cachedFetch('http://api.test/data1')
      await cachedFetch('http://api.test/data2')
      
      let stats = getCacheStats()
      expect(stats.size).toBe(2)

      // Clear cache
      clearCache()
      
      stats = getCacheStats()
      expect(stats.size).toBe(0)
    })

    it('should clear cache by pattern', async () => {
      // Cache some data
      await cachedFetch('http://api.test/users/1')
      await cachedFetch('http://api.test/users/2')
      await cachedFetch('http://api.test/posts/1')
      
      let stats = getCacheStats()
      expect(stats.size).toBe(3)

      // Clear only user cache
      clearCache('users')
      
      stats = getCacheStats()
      expect(stats.size).toBe(1) // Only posts remain
    })
  })

  describe('getCacheStats', () => {
    it('should return cache statistics', async () => {
      let stats = getCacheStats()
      expect(stats.size).toBe(0)

      await cachedFetch('http://api.test/data1')
      await cachedFetch('http://api.test/data2')
      
      stats = getCacheStats()
      expect(stats.size).toBe(2)
      expect(stats.hits).toBe(2) // Approximation
    })
  })

  describe('edge cases', () => {
    it('should handle network errors', async () => {
      mockAuthenticatedFetch.mockRejectedValue(new Error('Network error'))
      
      await expect(cachedFetch('http://api.test/data')).rejects.toThrow('Network error')
    })

    it('should handle JSON parse errors', async () => {
      mockAuthenticatedFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => {
          throw new Error('Invalid JSON')
        }
      } as Response)
      
      await expect(cachedFetch('http://api.test/data')).rejects.toThrow('Invalid JSON')
    })

    it('should handle cache size limits', async () => {
      // Cache should evict oldest entries when full
      const maxSize = 500 // From implementation
      
      // Fill cache
      for (let i = 0; i < maxSize + 10; i++) {
        await cachedFetch(`http://api.test/data${i}`)
      }
      
      const stats = getCacheStats()
      expect(stats.size).toBeLessThanOrEqual(maxSize)
    })
  })
})
