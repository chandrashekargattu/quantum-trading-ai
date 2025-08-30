interface PerformanceMetric {
  name: string
  startTime: number
  endTime?: number
  duration?: number
}

class PerformanceMonitor {
  private metrics: Map<string, PerformanceMetric> = new Map()
  private enabled = process.env.NODE_ENV === 'development'

  start(name: string) {
    if (!this.enabled) return
    
    this.metrics.set(name, {
      name,
      startTime: performance.now()
    })
  }

  end(name: string) {
    if (!this.enabled) return
    
    const metric = this.metrics.get(name)
    if (!metric) return

    metric.endTime = performance.now()
    metric.duration = metric.endTime - metric.startTime

    // Log if duration is significant
    if (metric.duration > 100) {
      console.warn(`[Performance] ${name} took ${metric.duration.toFixed(2)}ms`)
    } else {
      console.log(`[Performance] ${name} took ${metric.duration.toFixed(2)}ms`)
    }
  }

  measure(name: string, fn: () => void | Promise<void>) {
    this.start(name)
    const result = fn()
    
    if (result instanceof Promise) {
      return result.finally(() => this.end(name))
    } else {
      this.end(name)
      return result
    }
  }

  getMetrics() {
    return Array.from(this.metrics.values())
      .filter(m => m.duration !== undefined)
      .sort((a, b) => (b.duration || 0) - (a.duration || 0))
  }

  logSummary() {
    if (!this.enabled) return

    const metrics = this.getMetrics()
    const total = metrics.reduce((sum, m) => sum + (m.duration || 0), 0)

    console.group('[Performance Summary]')
    console.table(metrics.map(m => ({
      name: m.name,
      duration: `${m.duration?.toFixed(2)}ms`,
      percentage: `${((m.duration || 0) / total * 100).toFixed(1)}%`
    })))
    console.log(`Total: ${total.toFixed(2)}ms`)
    console.groupEnd()
  }

  clear() {
    this.metrics.clear()
  }

  // React component profiler integration
  onRender(id: string, phase: string, actualDuration: number) {
    if (!this.enabled) return

    const key = `${id}-${phase}`
    this.metrics.set(key, {
      name: key,
      startTime: 0,
      endTime: actualDuration,
      duration: actualDuration
    })

    if (actualDuration > 16) { // More than one frame
      console.warn(`[Render Performance] ${id} (${phase}) took ${actualDuration.toFixed(2)}ms`)
    }
  }
}

export const perfMonitor = new PerformanceMonitor()

// Web Vitals monitoring
export function reportWebVitals(metric: any) {
  if (metric.label === 'web-vital') {
    console.log('[Web Vital]', metric.name, metric.value)
    
    // Send to analytics if needed
    if (window.gtag) {
      window.gtag('event', metric.name, {
        value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
        metric_id: metric.id,
        metric_value: metric.value,
        metric_delta: metric.delta,
      })
    }
  }
}

// Utility to measure API call performance
export async function measureApiCall<T>(
  name: string,
  apiCall: () => Promise<T>
): Promise<T> {
  const start = performance.now()
  
  try {
    const result = await apiCall()
    const duration = performance.now() - start
    
    if (duration > 1000) {
      console.error(`[API Performance] ${name} took ${duration.toFixed(0)}ms - Too slow!`)
    } else if (duration > 500) {
      console.warn(`[API Performance] ${name} took ${duration.toFixed(0)}ms`)
    } else {
      console.log(`[API Performance] ${name} took ${duration.toFixed(0)}ms`)
    }
    
    return result
  } catch (error) {
    const duration = performance.now() - start
    console.error(`[API Performance] ${name} failed after ${duration.toFixed(0)}ms`, error)
    throw error
  }
}

// Intersection Observer for lazy loading
export function createLazyLoadObserver(
  callback: (entries: IntersectionObserverEntry[]) => void,
  options?: IntersectionObserverInit
) {
  return new IntersectionObserver(callback, {
    rootMargin: '50px', // Start loading 50px before visible
    threshold: 0.01,
    ...options
  })
}
