import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { portfolioService } from '@/services/api/portfolio-optimized'
import type { Portfolio, Position, Performance } from '@/services/api/portfolio'

interface PortfolioState {
  // Portfolio data
  portfolios: Portfolio[]
  activePortfolio: Portfolio | null
  positions: Position[]
  performance: Performance | null
  
  // Loading states
  isLoadingPortfolios: boolean
  isLoadingPositions: boolean
  isLoadingPerformance: boolean
  
  // UI state
  selectedPositionId: string | null
  
  // Actions
  loadPortfolios: () => Promise<void>
  selectPortfolio: (portfolioId: string) => Promise<void>
  createPortfolio: (name: string, initialCapital: number) => Promise<Portfolio>
  deletePortfolio: (portfolioId: string) => Promise<void>
  loadPositions: (portfolioId: string) => Promise<void>
  loadPerformance: (portfolioId: string, period: string) => Promise<void>
  updatePosition: (positionId: string, data: Partial<Position>) => void
  closePosition: (positionId: string) => Promise<void>
  selectPosition: (positionId: string | null) => void
  refreshPortfolioData: () => Promise<void>
}

export const usePortfolioStore = create<PortfolioState>()(
  subscribeWithSelector((set, get) => ({
    portfolios: [],
    activePortfolio: null,
    positions: [],
    performance: null,
    isLoadingPortfolios: false,
    isLoadingPositions: false,
    isLoadingPerformance: false,
    selectedPositionId: null,

    loadPortfolios: async () => {
      set({ isLoadingPortfolios: true })
      try {
        const portfolios = await portfolioService.getPortfolios()
        set({ portfolios, isLoadingPortfolios: false })
        
        // Auto-select first portfolio if none selected
        const { activePortfolio } = get()
        if (!activePortfolio && portfolios.length > 0) {
          await get().selectPortfolio(portfolios[0].id)
        }
      } catch (error) {
        set({ isLoadingPortfolios: false })
        throw error
      }
    },

    selectPortfolio: async (portfolioId: string) => {
      const { portfolios } = get()
      const portfolio = portfolios.find(p => p.id === portfolioId)
      
      if (!portfolio) {
        throw new Error('Portfolio not found')
      }
      
      set({ activePortfolio: portfolio })
      
      // Load positions and performance for the selected portfolio
      await Promise.all([
        get().loadPositions(portfolioId),
        get().loadPerformance(portfolioId, '1M')
      ])
    },

    createPortfolio: async (name: string, initialCapital: number) => {
      const portfolio = await portfolioService.createPortfolio({ name, initialCapital })
      set(state => ({
        portfolios: [...state.portfolios, portfolio]
      }))
      await get().selectPortfolio(portfolio.id)
      return portfolio
    },

    deletePortfolio: async (portfolioId: string) => {
      await portfolioService.deletePortfolio(portfolioId)
      set(state => ({
        portfolios: state.portfolios.filter(p => p.id !== portfolioId),
        activePortfolio: state.activePortfolio?.id === portfolioId ? null : state.activePortfolio,
        positions: state.activePortfolio?.id === portfolioId ? [] : state.positions,
        performance: state.activePortfolio?.id === portfolioId ? null : state.performance
      }))
      
      // Select another portfolio if available
      const { portfolios } = get()
      if (portfolios.length > 0 && !get().activePortfolio) {
        await get().selectPortfolio(portfolios[0].id)
      }
    },

    loadPositions: async (portfolioId: string) => {
      set({ isLoadingPositions: true })
      try {
        const positions = await portfolioService.getPositions(portfolioId)
        set({ positions, isLoadingPositions: false })
      } catch (error) {
        set({ isLoadingPositions: false })
        throw error
      }
    },

    loadPerformance: async (portfolioId: string, period: string) => {
      set({ isLoadingPerformance: true })
      try {
        const performance = await portfolioService.getPerformance(portfolioId, period)
        set({ performance, isLoadingPerformance: false })
      } catch (error) {
        set({ isLoadingPerformance: false })
        throw error
      }
    },

    updatePosition: (positionId: string, data: Partial<Position>) => {
      set(state => ({
        positions: state.positions.map(pos =>
          pos.id === positionId ? { ...pos, ...data } : pos
        )
      }))
    },

    closePosition: async (positionId: string) => {
      const { activePortfolio } = get()
      if (!activePortfolio) throw new Error('No active portfolio')
      
      await portfolioService.closePosition(activePortfolio.id, positionId)
      
      // Reload positions and update portfolio value
      await get().loadPositions(activePortfolio.id)
      await get().refreshPortfolioData()
    },

    selectPosition: (positionId: string | null) => {
      set({ selectedPositionId: positionId })
    },

    refreshPortfolioData: async () => {
      const { activePortfolio } = get()
      if (!activePortfolio) return
      
      // Reload portfolio data
      const portfolios = await portfolioService.getPortfolios()
      const updatedPortfolio = portfolios.find(p => p.id === activePortfolio.id)
      
      if (updatedPortfolio) {
        set(state => ({
          portfolios,
          activePortfolio: updatedPortfolio
        }))
      }
    }
  }))
)
