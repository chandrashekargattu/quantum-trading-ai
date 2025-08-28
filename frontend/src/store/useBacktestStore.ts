import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { backtestService, BacktestConfig, BacktestResult, BacktestStatus } from '@/services/api/backtest'

interface BacktestState {
  // Configurations
  savedConfigs: BacktestConfig[]
  activeConfig: BacktestConfig | null
  
  // Results
  results: BacktestResult[]
  activeResult: BacktestResult | null
  
  // Running backtests
  runningBacktests: Map<string, BacktestStatus>
  
  // Configuration form
  configForm: {
    name: string
    strategy: string
    symbols: string[]
    startDate: Date
    endDate: Date
    initialCapital: number
    positionSize: number
    maxPositions: number
    commission: number
    slippage: number
    parameters: Record<string, any>
  }
  
  // Loading states
  isLoadingConfigs: boolean
  isLoadingResults: boolean
  isRunningBacktest: boolean
  
  // UI state
  selectedConfigId: string | null
  selectedResultId: string | null
  compareMode: boolean
  compareResults: string[]
  
  // Actions
  loadConfigs: () => Promise<void>
  loadResults: () => Promise<void>
  createConfig: (config: Partial<BacktestConfig>) => Promise<BacktestConfig>
  updateConfig: (configId: string, updates: Partial<BacktestConfig>) => Promise<void>
  deleteConfig: (configId: string) => Promise<void>
  selectConfig: (configId: string | null) => void
  runBacktest: (configId: string) => Promise<string>
  stopBacktest: (backtestId: string) => Promise<void>
  getBacktestStatus: (backtestId: string) => Promise<BacktestStatus>
  loadResult: (resultId: string) => Promise<void>
  deleteResult: (resultId: string) => Promise<void>
  selectResult: (resultId: string | null) => void
  toggleCompareMode: () => void
  toggleCompareResult: (resultId: string) => void
  updateConfigForm: (updates: Partial<BacktestState['configForm']>) => void
  resetConfigForm: () => void
  exportResults: (resultIds: string[]) => Promise<void>
}

const defaultConfigForm: BacktestState['configForm'] = {
  name: '',
  strategy: '',
  symbols: [],
  startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
  endDate: new Date(),
  initialCapital: 100000,
  positionSize: 10000,
  maxPositions: 10,
  commission: 0.001,
  slippage: 0.0005,
  parameters: {}
}

export const useBacktestStore = create<BacktestState>()(
  subscribeWithSelector((set, get) => ({
    savedConfigs: [],
    activeConfig: null,
    results: [],
    activeResult: null,
    runningBacktests: new Map(),
    configForm: { ...defaultConfigForm },
    isLoadingConfigs: false,
    isLoadingResults: false,
    isRunningBacktest: false,
    selectedConfigId: null,
    selectedResultId: null,
    compareMode: false,
    compareResults: [],

    loadConfigs: async () => {
      set({ isLoadingConfigs: true })
      try {
        const configs = await backtestService.getConfigs()
        set({ savedConfigs: configs, isLoadingConfigs: false })
      } catch (error) {
        set({ isLoadingConfigs: false })
        throw error
      }
    },

    loadResults: async () => {
      set({ isLoadingResults: true })
      try {
        const results = await backtestService.getResults()
        set({ results, isLoadingResults: false })
      } catch (error) {
        set({ isLoadingResults: false })
        throw error
      }
    },

    createConfig: async (configData: Partial<BacktestConfig>) => {
      const config = await backtestService.createConfig(configData)
      set(state => ({
        savedConfigs: [...state.savedConfigs, config]
      }))
      get().resetConfigForm()
      return config
    },

    updateConfig: async (configId: string, updates: Partial<BacktestConfig>) => {
      const config = await backtestService.updateConfig(configId, updates)
      set(state => ({
        savedConfigs: state.savedConfigs.map(c => c.id === configId ? config : c),
        activeConfig: state.activeConfig?.id === configId ? config : state.activeConfig
      }))
    },

    deleteConfig: async (configId: string) => {
      await backtestService.deleteConfig(configId)
      set(state => ({
        savedConfigs: state.savedConfigs.filter(c => c.id !== configId),
        activeConfig: state.activeConfig?.id === configId ? null : state.activeConfig,
        selectedConfigId: state.selectedConfigId === configId ? null : state.selectedConfigId
      }))
    },

    selectConfig: (configId: string | null) => {
      const config = configId ? get().savedConfigs.find(c => c.id === configId) : null
      set({ 
        selectedConfigId: configId,
        activeConfig: config || null
      })
      
      // Load config into form if selected
      if (config) {
        set({
          configForm: {
            name: config.name,
            strategy: config.strategy,
            symbols: config.symbols,
            startDate: new Date(config.startDate),
            endDate: new Date(config.endDate),
            initialCapital: config.initialCapital,
            positionSize: config.positionSize,
            maxPositions: config.maxPositions,
            commission: config.commission,
            slippage: config.slippage,
            parameters: config.parameters
          }
        })
      }
    },

    runBacktest: async (configId: string) => {
      set({ isRunningBacktest: true })
      try {
        const backtestId = await backtestService.runBacktest(configId)
        
        // Add to running backtests
        set(state => {
          const newRunning = new Map(state.runningBacktests)
          newRunning.set(backtestId, { 
            id: backtestId, 
            status: 'RUNNING', 
            progress: 0,
            message: 'Initializing backtest...'
          })
          return { runningBacktests: newRunning }
        })
        
        // Start polling for status
        const pollStatus = async () => {
          try {
            const status = await backtestService.getBacktestStatus(backtestId)
            
            set(state => {
              const newRunning = new Map(state.runningBacktests)
              if (status.status === 'COMPLETED' || status.status === 'FAILED') {
                newRunning.delete(backtestId)
                // Reload results if completed
                if (status.status === 'COMPLETED') {
                  get().loadResults()
                }
              } else {
                newRunning.set(backtestId, status)
              }
              return { runningBacktests: newRunning }
            })
            
            // Continue polling if still running
            if (status.status === 'RUNNING') {
              setTimeout(pollStatus, 1000)
            }
          } catch (error) {
            // Remove from running on error
            set(state => {
              const newRunning = new Map(state.runningBacktests)
              newRunning.delete(backtestId)
              return { runningBacktests: newRunning }
            })
          }
        }
        
        // Start polling
        setTimeout(pollStatus, 1000)
        
        set({ isRunningBacktest: false })
        return backtestId
      } catch (error) {
        set({ isRunningBacktest: false })
        throw error
      }
    },

    stopBacktest: async (backtestId: string) => {
      await backtestService.stopBacktest(backtestId)
      set(state => {
        const newRunning = new Map(state.runningBacktests)
        newRunning.delete(backtestId)
        return { runningBacktests: newRunning }
      })
    },

    getBacktestStatus: async (backtestId: string) => {
      const status = await backtestService.getBacktestStatus(backtestId)
      set(state => {
        const newRunning = new Map(state.runningBacktests)
        newRunning.set(backtestId, status)
        return { runningBacktests: newRunning }
      })
      return status
    },

    loadResult: async (resultId: string) => {
      set({ isLoadingResults: true })
      try {
        const result = await backtestService.getResult(resultId)
        set({ activeResult: result, isLoadingResults: false })
      } catch (error) {
        set({ isLoadingResults: false })
        throw error
      }
    },

    deleteResult: async (resultId: string) => {
      await backtestService.deleteResult(resultId)
      set(state => ({
        results: state.results.filter(r => r.id !== resultId),
        activeResult: state.activeResult?.id === resultId ? null : state.activeResult,
        selectedResultId: state.selectedResultId === resultId ? null : state.selectedResultId,
        compareResults: state.compareResults.filter(id => id !== resultId)
      }))
    },

    selectResult: (resultId: string | null) => {
      set({ selectedResultId: resultId })
      if (resultId) {
        get().loadResult(resultId)
      } else {
        set({ activeResult: null })
      }
    },

    toggleCompareMode: () => {
      set(state => ({
        compareMode: !state.compareMode,
        compareResults: state.compareMode ? [] : []
      }))
    },

    toggleCompareResult: (resultId: string) => {
      set(state => ({
        compareResults: state.compareResults.includes(resultId)
          ? state.compareResults.filter(id => id !== resultId)
          : [...state.compareResults, resultId].slice(0, 4) // Max 4 comparisons
      }))
    },

    updateConfigForm: (updates: Partial<BacktestState['configForm']>) => {
      set(state => ({
        configForm: { ...state.configForm, ...updates }
      }))
    },

    resetConfigForm: () => {
      set({ configForm: { ...defaultConfigForm } })
    },

    exportResults: async (resultIds: string[]) => {
      await backtestService.exportResults(resultIds)
    }
  }))
)
