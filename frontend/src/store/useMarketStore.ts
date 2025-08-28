import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { marketService, Stock, Option } from '@/services/api/market'

interface MarketState {
  // Watchlist
  watchlist: string[]
  watchlistData: Record<string, Stock>
  
  // Current selections
  selectedSymbol: string | null
  selectedStock: Stock | null
  selectedExpiration: string | null
  
  // Market data
  optionChain: {
    calls: Option[]
    puts: Option[]
    expirations: string[]
    strikes: number[]
  } | null
  
  // Loading states
  isLoadingStock: boolean
  isLoadingOptions: boolean
  
  // Actions
  addToWatchlist: (symbol: string) => void
  removeFromWatchlist: (symbol: string) => void
  selectSymbol: (symbol: string) => Promise<void>
  loadOptionChain: (symbol: string) => Promise<void>
  updateStockData: (symbol: string, data: Partial<Stock>) => void
  clearSelection: () => void
}

export const useMarketStore = create<MarketState>()(
  subscribeWithSelector((set, get) => ({
    watchlist: [],
    watchlistData: {},
    selectedSymbol: null,
    selectedStock: null,
    selectedExpiration: null,
    optionChain: null,
    isLoadingStock: false,
    isLoadingOptions: false,

    addToWatchlist: (symbol: string) => {
      const { watchlist } = get()
      if (!watchlist.includes(symbol)) {
        set({ watchlist: [...watchlist, symbol] })
      }
    },

    removeFromWatchlist: (symbol: string) => {
      const { watchlist, watchlistData } = get()
      const newWatchlist = watchlist.filter(s => s !== symbol)
      const newWatchlistData = { ...watchlistData }
      delete newWatchlistData[symbol]
      set({ 
        watchlist: newWatchlist,
        watchlistData: newWatchlistData 
      })
    },

    selectSymbol: async (symbol: string) => {
      set({ 
        selectedSymbol: symbol, 
        isLoadingStock: true,
        selectedStock: null,
        optionChain: null 
      })
      
      try {
        const stock = await marketService.getStock(symbol)
        set({ 
          selectedStock: stock, 
          isLoadingStock: false 
        })
        
        // Update watchlist data if symbol is in watchlist
        const { watchlist, watchlistData } = get()
        if (watchlist.includes(symbol)) {
          set({
            watchlistData: {
              ...watchlistData,
              [symbol]: stock
            }
          })
        }
      } catch (error) {
        set({ isLoadingStock: false })
        throw error
      }
    },

    loadOptionChain: async (symbol: string) => {
      set({ isLoadingOptions: true })
      
      try {
        const chain = await marketService.getOptionChain(symbol)
        set({ 
          optionChain: chain,
          isLoadingOptions: false 
        })
      } catch (error) {
        set({ isLoadingOptions: false })
        throw error
      }
    },

    updateStockData: (symbol: string, data: Partial<Stock>) => {
      const { selectedStock, watchlistData } = get()
      
      // Update selected stock if it matches
      if (selectedStock?.symbol === symbol) {
        set({
          selectedStock: { ...selectedStock, ...data }
        })
      }
      
      // Update watchlist data
      if (watchlistData[symbol]) {
        set({
          watchlistData: {
            ...watchlistData,
            [symbol]: { ...watchlistData[symbol], ...data }
          }
        })
      }
    },

    clearSelection: () => {
      set({
        selectedSymbol: null,
        selectedStock: null,
        selectedExpiration: null,
        optionChain: null
      })
    },
  }))
)
