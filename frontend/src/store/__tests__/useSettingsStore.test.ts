import { renderHook, act } from '@testing-library/react'
import { useSettingsStore } from '../useSettingsStore'

describe('useSettingsStore', () => {
  beforeEach(() => {
    // Clear localStorage and reset store
    localStorage.clear()
    useSettingsStore.setState({
      chartSettings: {
        theme: 'dark',
        showVolume: true,
        showMA: true,
        maPeriods: [20, 50, 200],
        showBollingerBands: false,
        showRSI: false,
        showMACD: false,
        candleType: 'candles',
        timeframe: '1h'
      },
      tradingSettings: {
        defaultOrderType: 'LIMIT',
        defaultTimeInForce: 'DAY',
        confirmOrders: true,
        quickOrderSizes: [100, 500, 1000, 5000],
        slippageTolerance: 0.1,
        showOrderBook: true,
        orderBookDepth: 10
      },
      notificationSettings: {
        enablePriceAlerts: true,
        enableTradeAlerts: true,
        enableNewsAlerts: true,
        enableRiskAlerts: true,
        soundEnabled: true,
        desktopNotifications: true,
        emailNotifications: false
      },
      displaySettings: {
        theme: 'system',
        compactMode: false,
        showProfitInPercent: true,
        timezone: 'America/New_York',
        dateFormat: 'MM/DD/YYYY',
        numberFormat: 'comma',
        currency: 'USD'
      },
      riskSettings: {
        maxPositionSize: 10000,
        maxDailyLoss: 1000,
        maxOpenPositions: 10,
        enableStopLoss: true,
        defaultStopLossPercent: 2,
        enableTakeProfit: false,
        defaultTakeProfitPercent: 5,
        marginCallLevel: 25
      },
      favoriteSymbols: [],
      workspaceLayout: 'default',
      pinnedWidgets: []
    })
  })

  describe('Chart Settings', () => {
    it('should update chart settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateChartSettings({
          theme: 'light',
          showVolume: false,
          timeframe: '5m'
        })
      })

      expect(result.current.chartSettings.theme).toBe('light')
      expect(result.current.chartSettings.showVolume).toBe(false)
      expect(result.current.chartSettings.timeframe).toBe('5m')
      expect(result.current.chartSettings.showMA).toBe(true) // Unchanged
    })

    it('should update MA periods', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateChartSettings({
          maPeriods: [10, 20, 50]
        })
      })

      expect(result.current.chartSettings.maPeriods).toEqual([10, 20, 50])
    })

    it('should toggle indicators', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateChartSettings({
          showBollingerBands: true,
          showRSI: true,
          showMACD: true
        })
      })

      expect(result.current.chartSettings.showBollingerBands).toBe(true)
      expect(result.current.chartSettings.showRSI).toBe(true)
      expect(result.current.chartSettings.showMACD).toBe(true)
    })
  })

  describe('Trading Settings', () => {
    it('should update trading settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateTradingSettings({
          defaultOrderType: 'MARKET',
          confirmOrders: false,
          slippageTolerance: 0.5
        })
      })

      expect(result.current.tradingSettings.defaultOrderType).toBe('MARKET')
      expect(result.current.tradingSettings.confirmOrders).toBe(false)
      expect(result.current.tradingSettings.slippageTolerance).toBe(0.5)
    })

    it('should update quick order sizes', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateTradingSettings({
          quickOrderSizes: [50, 100, 250, 500]
        })
      })

      expect(result.current.tradingSettings.quickOrderSizes).toEqual([50, 100, 250, 500])
    })

    it('should update order book settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateTradingSettings({
          showOrderBook: false,
          orderBookDepth: 20
        })
      })

      expect(result.current.tradingSettings.showOrderBook).toBe(false)
      expect(result.current.tradingSettings.orderBookDepth).toBe(20)
    })
  })

  describe('Notification Settings', () => {
    it('should update notification settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateNotificationSettings({
          soundEnabled: false,
          emailNotifications: true,
          desktopNotifications: false
        })
      })

      expect(result.current.notificationSettings.soundEnabled).toBe(false)
      expect(result.current.notificationSettings.emailNotifications).toBe(true)
      expect(result.current.notificationSettings.desktopNotifications).toBe(false)
    })

    it('should toggle alert types', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateNotificationSettings({
          enablePriceAlerts: false,
          enableTradeAlerts: false,
          enableNewsAlerts: false,
          enableRiskAlerts: false
        })
      })

      expect(result.current.notificationSettings.enablePriceAlerts).toBe(false)
      expect(result.current.notificationSettings.enableTradeAlerts).toBe(false)
      expect(result.current.notificationSettings.enableNewsAlerts).toBe(false)
      expect(result.current.notificationSettings.enableRiskAlerts).toBe(false)
    })
  })

  describe('Display Settings', () => {
    it('should update display settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateDisplaySettings({
          theme: 'dark',
          compactMode: true,
          showProfitInPercent: false
        })
      })

      expect(result.current.displaySettings.theme).toBe('dark')
      expect(result.current.displaySettings.compactMode).toBe(true)
      expect(result.current.displaySettings.showProfitInPercent).toBe(false)
    })

    it('should update regional settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateDisplaySettings({
          timezone: 'Europe/London',
          dateFormat: 'DD/MM/YYYY',
          numberFormat: 'space',
          currency: 'EUR'
        })
      })

      expect(result.current.displaySettings.timezone).toBe('Europe/London')
      expect(result.current.displaySettings.dateFormat).toBe('DD/MM/YYYY')
      expect(result.current.displaySettings.numberFormat).toBe('space')
      expect(result.current.displaySettings.currency).toBe('EUR')
    })
  })

  describe('Risk Settings', () => {
    it('should update risk settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateRiskSettings({
          maxPositionSize: 25000,
          maxDailyLoss: 2500,
          maxOpenPositions: 5
        })
      })

      expect(result.current.riskSettings.maxPositionSize).toBe(25000)
      expect(result.current.riskSettings.maxDailyLoss).toBe(2500)
      expect(result.current.riskSettings.maxOpenPositions).toBe(5)
    })

    it('should update stop loss and take profit settings', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateRiskSettings({
          enableStopLoss: false,
          defaultStopLossPercent: 3,
          enableTakeProfit: true,
          defaultTakeProfitPercent: 10
        })
      })

      expect(result.current.riskSettings.enableStopLoss).toBe(false)
      expect(result.current.riskSettings.defaultStopLossPercent).toBe(3)
      expect(result.current.riskSettings.enableTakeProfit).toBe(true)
      expect(result.current.riskSettings.defaultTakeProfitPercent).toBe(10)
    })
  })

  describe('Favorite Symbols', () => {
    it('should add favorite symbol', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.addFavoriteSymbol('AAPL')
      })

      expect(result.current.favoriteSymbols).toContain('AAPL')
    })

    it('should not add duplicate favorite symbols', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.addFavoriteSymbol('AAPL')
        result.current.addFavoriteSymbol('AAPL')
      })

      expect(result.current.favoriteSymbols).toHaveLength(1)
    })

    it('should remove favorite symbol', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.addFavoriteSymbol('AAPL')
        result.current.addFavoriteSymbol('GOOGL')
        result.current.removeFavoriteSymbol('AAPL')
      })

      expect(result.current.favoriteSymbols).toEqual(['GOOGL'])
    })
  })

  describe('Workspace Settings', () => {
    it('should set workspace layout', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.setWorkspaceLayout('trading')
      })

      expect(result.current.workspaceLayout).toBe('trading')
    })

    it('should toggle widgets', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.toggleWidget('portfolio-summary')
      })

      expect(result.current.pinnedWidgets).toContain('portfolio-summary')

      act(() => {
        result.current.toggleWidget('portfolio-summary')
      })

      expect(result.current.pinnedWidgets).not.toContain('portfolio-summary')
    })

    it('should manage multiple pinned widgets', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.toggleWidget('portfolio-summary')
        result.current.toggleWidget('market-overview')
        result.current.toggleWidget('risk-metrics')
      })

      expect(result.current.pinnedWidgets).toHaveLength(3)
      expect(result.current.pinnedWidgets).toEqual([
        'portfolio-summary',
        'market-overview',
        'risk-metrics'
      ])
    })
  })

  describe('Settings Management', () => {
    it('should reset all settings to defaults', () => {
      const { result } = renderHook(() => useSettingsStore())

      // Change some settings
      act(() => {
        result.current.updateChartSettings({ theme: 'light' })
        result.current.addFavoriteSymbol('AAPL')
        result.current.setWorkspaceLayout('trading')
      })

      // Reset
      act(() => {
        result.current.resetSettings()
      })

      expect(result.current.chartSettings.theme).toBe('dark')
      expect(result.current.favoriteSymbols).toHaveLength(0)
      expect(result.current.workspaceLayout).toBe('default')
    })

    it('should export settings as JSON', () => {
      const { result } = renderHook(() => useSettingsStore())

      // Set some custom values
      act(() => {
        result.current.updateChartSettings({ theme: 'light' })
        result.current.addFavoriteSymbol('AAPL')
      })

      const exported = result.current.exportSettings()
      const parsed = JSON.parse(exported)

      expect(parsed.chartSettings.theme).toBe('light')
      expect(parsed.favoriteSymbols).toContain('AAPL')
      expect(parsed).toHaveProperty('tradingSettings')
      expect(parsed).toHaveProperty('notificationSettings')
    })

    it('should import settings from JSON', () => {
      const { result } = renderHook(() => useSettingsStore())

      const settingsToImport = {
        chartSettings: { theme: 'light', timeframe: '15m' },
        favoriteSymbols: ['AAPL', 'GOOGL'],
        workspaceLayout: 'analysis'
      }

      act(() => {
        result.current.importSettings(JSON.stringify(settingsToImport))
      })

      expect(result.current.chartSettings.theme).toBe('light')
      expect(result.current.chartSettings.timeframe).toBe('15m')
      expect(result.current.favoriteSymbols).toEqual(['AAPL', 'GOOGL'])
      expect(result.current.workspaceLayout).toBe('analysis')
    })

    it('should handle invalid import JSON', () => {
      const { result } = renderHook(() => useSettingsStore())

      expect(() => {
        act(() => {
          result.current.importSettings('invalid json')
        })
      }).toThrow('Invalid settings format')
    })

    it('should merge imported settings with defaults', () => {
      const { result } = renderHook(() => useSettingsStore())

      const partialSettings = {
        chartSettings: { theme: 'light' }
        // Missing other settings
      }

      act(() => {
        result.current.importSettings(JSON.stringify(partialSettings))
      })

      expect(result.current.chartSettings.theme).toBe('light')
      expect(result.current.chartSettings.showVolume).toBe(true) // Default preserved
      expect(result.current.tradingSettings.defaultOrderType).toBe('LIMIT') // Default preserved
    })
  })

  describe('Persistence', () => {
    it('should persist settings to localStorage', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateChartSettings({ theme: 'light' })
      })

      // Check localStorage
      const stored = localStorage.getItem('quantum-trading-settings')
      expect(stored).toBeTruthy()
      
      const parsed = JSON.parse(JSON.parse(stored!).state)
      expect(parsed.chartSettings.theme).toBe('light')
    })

    it('should restore settings from localStorage', () => {
      // Set some data in localStorage
      const storedState = {
        chartSettings: { theme: 'light' },
        favoriteSymbols: ['AAPL']
      }
      
      localStorage.setItem(
        'quantum-trading-settings',
        JSON.stringify({
          state: JSON.stringify(storedState),
          version: 0
        })
      )

      // Create new store instance
      const { result } = renderHook(() => useSettingsStore())

      expect(result.current.chartSettings.theme).toBe('light')
      expect(result.current.favoriteSymbols).toContain('AAPL')
    })
  })
})
