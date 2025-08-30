import { authenticatedFetch } from './auth-interceptor'

// Simple but effective cache implementation
interface CacheItem<T> {
  data: T
  timestamp: number
  ttl: number
}

class SimpleCache {
  private cache = new Map<string, CacheItem<any>>()
  private maxSize = 500

  get<T>(key: string): T | null {
    const item = this.cache.get(key)
    if (!item) return null

    // Check if expired
    if (Date.now() - item.timestamp > item.ttl) {
      this.cache.delete(key)
      return null
    }

    return item.data
  }

  set<T>(key: string, data: T, ttl: number = 5 * 60 * 1000) {
    // Evict oldest items if cache is full
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value
      this.cache.delete(oldestKey)
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    })
  }

  delete(key: string) {
    return this.cache.delete(key)
  }

  clear() {
    this.cache.clear()
  }

  get size() {
    return this.cache.size
  }

  keys() {
    return Array.from(this.cache.keys())
  }
}

const cache = new SimpleCache()

// Request deduplication map
const pendingRequests = new Map<string, Promise<any>>()

export interface CacheOptions {
  ttl?: number // Time to live in milliseconds
  forceRefresh?: boolean
  dedupe?: boolean // Deduplicate concurrent requests
}

export async function cachedFetch<T>(
  url: string,
  options?: RequestInit,
  cacheOptions: CacheOptions = {}
): Promise<T> {
  const {
    ttl = 1000 * 60 * 5, // 5 minutes default
    forceRefresh = false,
    dedupe = true
  } = cacheOptions

  // Create cache key from URL and relevant options
  const cacheKey = `${options?.method || 'GET'}:${url}:${JSON.stringify(options?.body || '')}`

  // Check if we should deduplicate this request
  if (dedupe && pendingRequests.has(cacheKey)) {
    return pendingRequests.get(cacheKey)!
  }

  // Check cache first unless force refresh
  if (!forceRefresh) {
    const cached = cache.get(cacheKey)
    if (cached) {
      return cached
    }
  }

  // Create the fetch promise
  const fetchPromise = authenticatedFetch(url, options)
    .then(async (response) => {
      if (!response.ok) {
        // If 401, the auth interceptor will handle it
        if (response.status === 401) {
          cache.delete(cacheKey)
          // Don't cache 401 errors
          throw new Error(`Authentication required`)
        }
        
        // For other errors, include status
        const errorText = await response.text().catch(() => '')
        throw new Error(`HTTP error! status: ${response.status}${errorText ? `: ${errorText}` : ''}`)
      }
      const data = await response.json()
      
      // Cache successful responses
      cache.set(cacheKey, data, { ttl })
      
      return data
    })
    .catch((error) => {
      // Remove from pending requests on error
      pendingRequests.delete(cacheKey)
      throw error
    })

  // Store in pending requests for deduplication
  if (dedupe) {
    pendingRequests.set(cacheKey, fetchPromise)
  }

  return fetchPromise
}

// Prefetch function for critical data
export function prefetch(url: string, options?: RequestInit) {
  cachedFetch(url, options, { dedupe: true }).catch(() => {
    // Silently handle prefetch errors
  })
}

// Clear cache for specific pattern or all
export function clearCache(pattern?: string) {
  if (pattern) {
    const keys = Array.from(cache.keys())
    keys.forEach(key => {
      if (key.includes(pattern)) {
        cache.delete(key)
      }
    })
  } else {
    cache.clear()
  }
}

// Get cache stats
export function getCacheStats() {
  return {
    size: cache.size,
    calculatedSize: cache.calculatedSize,
    hits: cache.size, // Approximate
  }
}
