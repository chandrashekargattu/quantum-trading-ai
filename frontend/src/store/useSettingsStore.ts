import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ChartSettings {
  theme: 'light' | 'dark'
  showVolume: boolean
  showMA: boolean
  maPeriods: number[]
  showBollingerBands: boolean
  showRSI: boolean
  showMACD: boolean
  candleType: 'candles' | 'bars' | 'line' | 'area'
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w'
}

interface TradingSettings {
  defaultOrderType: 'LIMIT' | 'MARKET' | 'STOP_LIMIT'
  defaultTimeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY'
  confirmOrders: boolean
  quickOrderSizes: number[]
  slippageTolerance: number
  showOrderBook: boolean
  orderBookDepth: number
}

interface NotificationSettings {
  enablePriceAlerts: boolean
  enableTradeAlerts: boolean
  enableNewsAlerts: boolean
  enableRiskAlerts: boolean
  soundEnabled: boolean
  desktopNotifications: boolean
  emailNotifications: boolean
}

interface DisplaySettings {
  theme: 'light' | 'dark' | 'system'
  compactMode: boolean
  showProfitInPercent: boolean
  timezone: string
  dateFormat: string
  numberFormat: 'comma' | 'space' | 'none'
  currency: 'USD' | 'EUR' | 'GBP' | 'JPY'
}

interface RiskSettings {
  maxPositionSize: number
  maxDailyLoss: number
  maxOpenPositions: number
  enableStopLoss: boolean
  defaultStopLossPercent: number
  enableTakeProfit: boolean
  defaultTakeProfitPercent: number
  marginCallLevel: number
}

interface SettingsState {
  // Settings categories
  chartSettings: ChartSettings
  tradingSettings: TradingSettings
  notificationSettings: NotificationSettings
  displaySettings: DisplaySettings
  riskSettings: RiskSettings
  
  // Workspace
  favoriteSymbols: string[]
  workspaceLayout: 'default' | 'trading' | 'analysis' | 'monitoring'
  pinnedWidgets: string[]
  
  // Actions
  updateChartSettings: (settings: Partial<ChartSettings>) => void
  updateTradingSettings: (settings: Partial<TradingSettings>) => void
  updateNotificationSettings: (settings: Partial<NotificationSettings>) => void
  updateDisplaySettings: (settings: Partial<DisplaySettings>) => void
  updateRiskSettings: (settings: Partial<RiskSettings>) => void
  addFavoriteSymbol: (symbol: string) => void
  removeFavoriteSymbol: (symbol: string) => void
  setWorkspaceLayout: (layout: SettingsState['workspaceLayout']) => void
  toggleWidget: (widgetId: string) => void
  resetSettings: () => void
  exportSettings: () => string
  importSettings: (settingsJson: string) => void
}

const defaultSettings: Omit<SettingsState, 'updateChartSettings' | 'updateTradingSettings' | 'updateNotificationSettings' | 'updateDisplaySettings' | 'updateRiskSettings' | 'addFavoriteSymbol' | 'removeFavoriteSymbol' | 'setWorkspaceLayout' | 'toggleWidget' | 'resetSettings' | 'exportSettings' | 'importSettings'> = {
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
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      ...defaultSettings,

      updateChartSettings: (settings: Partial<ChartSettings>) => {
        set(state => ({
          chartSettings: { ...state.chartSettings, ...settings }
        }))
      },

      updateTradingSettings: (settings: Partial<TradingSettings>) => {
        set(state => ({
          tradingSettings: { ...state.tradingSettings, ...settings }
        }))
      },

      updateNotificationSettings: (settings: Partial<NotificationSettings>) => {
        set(state => ({
          notificationSettings: { ...state.notificationSettings, ...settings }
        }))
      },

      updateDisplaySettings: (settings: Partial<DisplaySettings>) => {
        set(state => ({
          displaySettings: { ...state.displaySettings, ...settings }
        }))
      },

      updateRiskSettings: (settings: Partial<RiskSettings>) => {
        set(state => ({
          riskSettings: { ...state.riskSettings, ...settings }
        }))
      },

      addFavoriteSymbol: (symbol: string) => {
        set(state => ({
          favoriteSymbols: state.favoriteSymbols.includes(symbol)
            ? state.favoriteSymbols
            : [...state.favoriteSymbols, symbol]
        }))
      },

      removeFavoriteSymbol: (symbol: string) => {
        set(state => ({
          favoriteSymbols: state.favoriteSymbols.filter(s => s !== symbol)
        }))
      },

      setWorkspaceLayout: (layout: SettingsState['workspaceLayout']) => {
        set({ workspaceLayout: layout })
      },

      toggleWidget: (widgetId: string) => {
        set(state => ({
          pinnedWidgets: state.pinnedWidgets.includes(widgetId)
            ? state.pinnedWidgets.filter(id => id !== widgetId)
            : [...state.pinnedWidgets, widgetId]
        }))
      },

      resetSettings: () => {
        set(defaultSettings)
      },

      exportSettings: () => {
        const state = get()
        const settings = {
          chartSettings: state.chartSettings,
          tradingSettings: state.tradingSettings,
          notificationSettings: state.notificationSettings,
          displaySettings: state.displaySettings,
          riskSettings: state.riskSettings,
          favoriteSymbols: state.favoriteSymbols,
          workspaceLayout: state.workspaceLayout,
          pinnedWidgets: state.pinnedWidgets
        }
        return JSON.stringify(settings, null, 2)
      },

      importSettings: (settingsJson: string) => {
        try {
          const settings = JSON.parse(settingsJson)
          set({
            chartSettings: { ...defaultSettings.chartSettings, ...settings.chartSettings },
            tradingSettings: { ...defaultSettings.tradingSettings, ...settings.tradingSettings },
            notificationSettings: { ...defaultSettings.notificationSettings, ...settings.notificationSettings },
            displaySettings: { ...defaultSettings.displaySettings, ...settings.displaySettings },
            riskSettings: { ...defaultSettings.riskSettings, ...settings.riskSettings },
            favoriteSymbols: settings.favoriteSymbols || [],
            workspaceLayout: settings.workspaceLayout || 'default',
            pinnedWidgets: settings.pinnedWidgets || []
          })
        } catch (error) {
          throw new Error('Invalid settings format')
        }
      }
    }),
    {
      name: 'quantum-trading-settings',
      partialize: (state) => ({
        chartSettings: state.chartSettings,
        tradingSettings: state.tradingSettings,
        notificationSettings: state.notificationSettings,
        displaySettings: state.displaySettings,
        riskSettings: state.riskSettings,
        favoriteSymbols: state.favoriteSymbols,
        workspaceLayout: state.workspaceLayout,
        pinnedWidgets: state.pinnedWidgets
      })
    }
  )
)
