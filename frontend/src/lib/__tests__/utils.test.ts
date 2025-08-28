import {
  cn,
  formatCurrency,
  formatPercentage,
  formatNumber,
  formatDate,
  calculatePriceChange,
  debounce,
  throttle,
} from '../utils'

describe('utils', () => {
  describe('cn (classname merger)', () => {
    it('merges class names correctly', () => {
      expect(cn('foo', 'bar')).toBe('foo bar')
      expect(cn('foo', { bar: true, baz: false })).toBe('foo bar')
      expect(cn('foo', undefined, 'bar')).toBe('foo bar')
    })

    it('handles tailwind class conflicts', () => {
      expect(cn('text-sm', 'text-lg')).toBe('text-lg')
      expect(cn('p-4', 'p-2')).toBe('p-2')
    })
  })

  describe('formatCurrency', () => {
    it('formats USD currency correctly', () => {
      expect(formatCurrency(1234.56)).toBe('$1,234.56')
      expect(formatCurrency(1000000)).toBe('$1,000,000.00')
      expect(formatCurrency(0)).toBe('$0.00')
      expect(formatCurrency(-500)).toBe('-$500.00')
    })

    it('handles different currencies', () => {
      expect(formatCurrency(1234.56, 'EUR')).toMatch(/€|EUR/)
      expect(formatCurrency(1234.56, 'GBP')).toMatch(/£|GBP/)
    })
  })

  describe('formatPercentage', () => {
    it('formats percentages correctly', () => {
      expect(formatPercentage(0.1234)).toBe('12.34%')
      expect(formatPercentage(0.5)).toBe('50.00%')
      expect(formatPercentage(1)).toBe('100.00%')
      expect(formatPercentage(-0.25)).toBe('-25.00%')
    })

    it('respects decimal places', () => {
      expect(formatPercentage(0.12345, 3)).toBe('12.345%')
      expect(formatPercentage(0.1, 0)).toBe('10%')
    })
  })

  describe('formatNumber', () => {
    it('formats numbers with proper decimals', () => {
      expect(formatNumber(1234.5678)).toBe('1,234.57')
      expect(formatNumber(1000000)).toBe('1,000,000.00')
      expect(formatNumber(0.123456, 4)).toBe('0.1235')
    })

    it('handles edge cases', () => {
      expect(formatNumber(0)).toBe('0.00')
      expect(formatNumber(-1234.56)).toBe('-1,234.56')
      expect(formatNumber(NaN)).toBe('NaN')
    })
  })

  describe('formatDate', () => {
    const testDate = new Date('2024-01-15T14:30:00')

    it('formats short date correctly', () => {
      const formatted = formatDate(testDate, 'short')
      expect(formatted).toMatch(/Jan 15, 2024/)
    })

    it('formats long date correctly', () => {
      const formatted = formatDate(testDate, 'long')
      expect(formatted).toMatch(/January 15, 2024/)
      expect(formatted).toMatch(/2:30 PM|14:30/)
    })

    it('handles string dates', () => {
      const formatted = formatDate('2024-01-15T14:30:00', 'short')
      expect(formatted).toMatch(/Jan 15, 2024/)
    })
  })

  describe('calculatePriceChange', () => {
    it('calculates positive price changes', () => {
      const result = calculatePriceChange(150, 100)
      expect(result.amount).toBe(50)
      expect(result.percentage).toBe(50)
      expect(result.direction).toBe('up')
    })

    it('calculates negative price changes', () => {
      const result = calculatePriceChange(80, 100)
      expect(result.amount).toBe(-20)
      expect(result.percentage).toBe(-20)
      expect(result.direction).toBe('down')
    })

    it('handles unchanged prices', () => {
      const result = calculatePriceChange(100, 100)
      expect(result.amount).toBe(0)
      expect(result.percentage).toBe(0)
      expect(result.direction).toBe('unchanged')
    })

    it('handles zero previous price', () => {
      const result = calculatePriceChange(100, 0)
      expect(result.amount).toBe(100)
      expect(result.percentage).toBe(0)
      expect(result.direction).toBe('up')
    })
  })

  describe('debounce', () => {
    beforeEach(() => {
      jest.useFakeTimers()
    })

    afterEach(() => {
      jest.useRealTimers()
    })

    it('delays function execution', () => {
      const mockFn = jest.fn()
      const debouncedFn = debounce(mockFn, 300)

      debouncedFn('test')
      expect(mockFn).not.toHaveBeenCalled()

      jest.advanceTimersByTime(299)
      expect(mockFn).not.toHaveBeenCalled()

      jest.advanceTimersByTime(1)
      expect(mockFn).toHaveBeenCalledWith('test')
      expect(mockFn).toHaveBeenCalledTimes(1)
    })

    it('cancels previous calls', () => {
      const mockFn = jest.fn()
      const debouncedFn = debounce(mockFn, 300)

      debouncedFn('first')
      jest.advanceTimersByTime(100)
      debouncedFn('second')
      jest.advanceTimersByTime(100)
      debouncedFn('third')
      jest.advanceTimersByTime(300)

      expect(mockFn).toHaveBeenCalledTimes(1)
      expect(mockFn).toHaveBeenCalledWith('third')
    })
  })

  describe('throttle', () => {
    beforeEach(() => {
      jest.useFakeTimers()
    })

    afterEach(() => {
      jest.useRealTimers()
    })

    it('limits function execution frequency', () => {
      const mockFn = jest.fn()
      const throttledFn = throttle(mockFn, 300)

      throttledFn('first')
      expect(mockFn).toHaveBeenCalledWith('first')
      expect(mockFn).toHaveBeenCalledTimes(1)

      throttledFn('second')
      expect(mockFn).toHaveBeenCalledTimes(1) // Still 1

      jest.advanceTimersByTime(300)
      throttledFn('third')
      expect(mockFn).toHaveBeenCalledWith('third')
      expect(mockFn).toHaveBeenCalledTimes(2)
    })

    it('ignores calls during throttle period', () => {
      const mockFn = jest.fn()
      const throttledFn = throttle(mockFn, 300)

      throttledFn('first')
      
      // Multiple calls during throttle period
      for (let i = 0; i < 10; i++) {
        throttledFn(`call ${i}`)
      }
      
      expect(mockFn).toHaveBeenCalledTimes(1)
      expect(mockFn).toHaveBeenCalledWith('first')

      jest.advanceTimersByTime(300)
      throttledFn('after throttle')
      expect(mockFn).toHaveBeenCalledTimes(2)
      expect(mockFn).toHaveBeenCalledWith('after throttle')
    })
  })
})
