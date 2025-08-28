import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { alertService, Alert, AlertType, AlertCondition } from '@/services/api/alerts'

interface AlertState {
  // Alerts
  alerts: Alert[]
  activeAlerts: Alert[]
  triggeredAlerts: Alert[]
  
  // Notifications
  notifications: Notification[]
  unreadCount: number
  
  // Alert creation
  alertForm: {
    symbol: string
    type: AlertType
    condition: AlertCondition
    value: number
    message?: string
    sendEmail: boolean
    sendPush: boolean
  }
  
  // Loading states
  isLoadingAlerts: boolean
  isCreatingAlert: boolean
  
  // UI state
  selectedAlertId: string | null
  showNotifications: boolean
  
  // Actions
  loadAlerts: () => Promise<void>
  loadActiveAlerts: () => Promise<void>
  loadTriggeredAlerts: (limit?: number) => Promise<void>
  createAlert: (alert: Partial<Alert>) => Promise<Alert>
  updateAlert: (alertId: string, updates: Partial<Alert>) => Promise<void>
  deleteAlert: (alertId: string) => Promise<void>
  toggleAlert: (alertId: string, enabled: boolean) => Promise<void>
  markAlertAsRead: (alertId: string) => void
  markAllAlertsAsRead: () => void
  updateAlertForm: (updates: Partial<AlertState['alertForm']>) => void
  resetAlertForm: () => void
  selectAlert: (alertId: string | null) => void
  toggleNotifications: () => void
  addNotification: (notification: Notification) => void
  removeNotification: (notificationId: string) => void
}

interface Notification {
  id: string
  alertId: string
  title: string
  message: string
  timestamp: Date
  read: boolean
  type: 'price' | 'volume' | 'technical' | 'news' | 'risk'
}

const defaultAlertForm: AlertState['alertForm'] = {
  symbol: '',
  type: 'PRICE',
  condition: 'ABOVE',
  value: 0,
  message: '',
  sendEmail: true,
  sendPush: true
}

export const useAlertStore = create<AlertState>()(
  subscribeWithSelector((set, get) => ({
    alerts: [],
    activeAlerts: [],
    triggeredAlerts: [],
    notifications: [],
    unreadCount: 0,
    alertForm: { ...defaultAlertForm },
    isLoadingAlerts: false,
    isCreatingAlert: false,
    selectedAlertId: null,
    showNotifications: false,

    loadAlerts: async () => {
      set({ isLoadingAlerts: true })
      try {
        const alerts = await alertService.getAlerts()
        const activeAlerts = alerts.filter(a => a.enabled && !a.triggered)
        const triggeredAlerts = alerts.filter(a => a.triggered)
        
        set({ 
          alerts, 
          activeAlerts,
          triggeredAlerts: triggeredAlerts.slice(0, 50), // Keep recent 50
          isLoadingAlerts: false 
        })
      } catch (error) {
        set({ isLoadingAlerts: false })
        throw error
      }
    },

    loadActiveAlerts: async () => {
      set({ isLoadingAlerts: true })
      try {
        const alerts = await alertService.getActiveAlerts()
        set({ activeAlerts: alerts, isLoadingAlerts: false })
      } catch (error) {
        set({ isLoadingAlerts: false })
        throw error
      }
    },

    loadTriggeredAlerts: async (limit = 50) => {
      set({ isLoadingAlerts: true })
      try {
        const alerts = await alertService.getTriggeredAlerts(limit)
        set({ triggeredAlerts: alerts, isLoadingAlerts: false })
      } catch (error) {
        set({ isLoadingAlerts: false })
        throw error
      }
    },

    createAlert: async (alertData: Partial<Alert>) => {
      set({ isCreatingAlert: true })
      try {
        const alert = await alertService.createAlert(alertData)
        
        set(state => ({
          alerts: [alert, ...state.alerts],
          activeAlerts: alert.enabled ? [alert, ...state.activeAlerts] : state.activeAlerts,
          isCreatingAlert: false
        }))
        
        // Reset form after successful creation
        get().resetAlertForm()
        
        return alert
      } catch (error) {
        set({ isCreatingAlert: false })
        throw error
      }
    },

    updateAlert: async (alertId: string, updates: Partial<Alert>) => {
      const alert = await alertService.updateAlert(alertId, updates)
      
      set(state => ({
        alerts: state.alerts.map(a => a.id === alertId ? alert : a),
        activeAlerts: state.activeAlerts.filter(a => a.id !== alertId)
          .concat(alert.enabled && !alert.triggered ? [alert] : []),
        triggeredAlerts: state.triggeredAlerts.map(a => a.id === alertId ? alert : a)
      }))
    },

    deleteAlert: async (alertId: string) => {
      await alertService.deleteAlert(alertId)
      
      set(state => ({
        alerts: state.alerts.filter(a => a.id !== alertId),
        activeAlerts: state.activeAlerts.filter(a => a.id !== alertId),
        triggeredAlerts: state.triggeredAlerts.filter(a => a.id !== alertId),
        notifications: state.notifications.filter(n => n.alertId !== alertId)
      }))
    },

    toggleAlert: async (alertId: string, enabled: boolean) => {
      await get().updateAlert(alertId, { enabled })
    },

    markAlertAsRead: (alertId: string) => {
      set(state => {
        const notifications = state.notifications.map(n =>
          n.alertId === alertId ? { ...n, read: true } : n
        )
        const unreadCount = notifications.filter(n => !n.read).length
        
        return { notifications, unreadCount }
      })
    },

    markAllAlertsAsRead: () => {
      set(state => ({
        notifications: state.notifications.map(n => ({ ...n, read: true })),
        unreadCount: 0
      }))
    },

    updateAlertForm: (updates: Partial<AlertState['alertForm']>) => {
      set(state => ({
        alertForm: { ...state.alertForm, ...updates }
      }))
    },

    resetAlertForm: () => {
      set({ alertForm: { ...defaultAlertForm } })
    },

    selectAlert: (alertId: string | null) => {
      set({ selectedAlertId: alertId })
    },

    toggleNotifications: () => {
      set(state => ({ showNotifications: !state.showNotifications }))
    },

    addNotification: (notification: Notification) => {
      set(state => {
        const notifications = [notification, ...state.notifications].slice(0, 100) // Keep last 100
        const unreadCount = notifications.filter(n => !n.read).length
        
        return { notifications, unreadCount }
      })
      
      // Auto-hide notifications panel if it was closed
      if (!get().showNotifications) {
        // Show unread indicator
        set({ showNotifications: false })
      }
    },

    removeNotification: (notificationId: string) => {
      set(state => {
        const notifications = state.notifications.filter(n => n.id !== notificationId)
        const unreadCount = notifications.filter(n => !n.read).length
        
        return { notifications, unreadCount }
      })
    }
  }))
)
