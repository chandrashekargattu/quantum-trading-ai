import { renderHook, act, waitFor } from '@testing-library/react'
import { useAlertStore } from '../useAlertStore'
import { alertService } from '@/services/api/alerts'

// Mock the alert service
jest.mock('@/services/api/alerts', () => ({
  alertService: {
    getAlerts: jest.fn(),
    getActiveAlerts: jest.fn(),
    getTriggeredAlerts: jest.fn(),
    createAlert: jest.fn(),
    updateAlert: jest.fn(),
    deleteAlert: jest.fn(),
  },
  Alert: {},
  AlertType: {},
  AlertCondition: {},
}))

describe('useAlertStore', () => {
  const mockAlerts = [
    {
      id: 'alert-1',
      symbol: 'AAPL',
      type: 'PRICE',
      condition: 'ABOVE',
      value: 155,
      enabled: true,
      triggered: false,
      message: 'AAPL above $155',
      createdAt: new Date().toISOString()
    },
    {
      id: 'alert-2',
      symbol: 'GOOGL',
      type: 'VOLUME',
      condition: 'ABOVE',
      value: 1000000,
      enabled: true,
      triggered: true,
      message: 'High volume on GOOGL',
      createdAt: new Date().toISOString(),
      triggeredAt: new Date().toISOString()
    },
    {
      id: 'alert-3',
      symbol: 'MSFT',
      type: 'PRICE',
      condition: 'BELOW',
      value: 300,
      enabled: false,
      triggered: false,
      message: 'MSFT below $300',
      createdAt: new Date().toISOString()
    }
  ]

  const mockNotification = {
    id: 'notif-1',
    alertId: 'alert-1',
    title: 'Price Alert',
    message: 'AAPL has reached $155',
    timestamp: new Date(),
    read: false,
    type: 'price' as const
  }

  beforeEach(() => {
    jest.clearAllMocks()
    // Reset store state
    useAlertStore.setState({
      alerts: [],
      activeAlerts: [],
      triggeredAlerts: [],
      notifications: [],
      unreadCount: 0,
      alertForm: {
        symbol: '',
        type: 'PRICE',
        condition: 'ABOVE',
        value: 0,
        message: '',
        sendEmail: true,
        sendPush: true
      },
      isLoadingAlerts: false,
      isCreatingAlert: false,
      selectedAlertId: null,
      showNotifications: false,
    })
  })

  describe('Alert Loading', () => {
    it('should load all alerts and categorize them', async () => {
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      const { result } = renderHook(() => useAlertStore())

      await act(async () => {
        await result.current.loadAlerts()
      })

      expect(result.current.alerts).toEqual(mockAlerts)
      expect(result.current.activeAlerts).toHaveLength(1) // alert-1 only
      expect(result.current.triggeredAlerts).toHaveLength(1) // alert-2 only
      expect(result.current.isLoadingAlerts).toBe(false)
    })

    it('should load only active alerts', async () => {
      const activeOnly = mockAlerts.filter(a => a.enabled && !a.triggered)
      ;(alertService.getActiveAlerts as jest.Mock).mockResolvedValueOnce(activeOnly)
      const { result } = renderHook(() => useAlertStore())

      await act(async () => {
        await result.current.loadActiveAlerts()
      })

      expect(result.current.activeAlerts).toEqual(activeOnly)
      expect(alertService.getActiveAlerts).toHaveBeenCalled()
    })

    it('should load triggered alerts with limit', async () => {
      const triggered = mockAlerts.filter(a => a.triggered)
      ;(alertService.getTriggeredAlerts as jest.Mock).mockResolvedValueOnce(triggered)
      const { result } = renderHook(() => useAlertStore())

      await act(async () => {
        await result.current.loadTriggeredAlerts(25)
      })

      expect(result.current.triggeredAlerts).toEqual(triggered)
      expect(alertService.getTriggeredAlerts).toHaveBeenCalledWith(25)
    })
  })

  describe('Alert Creation', () => {
    it('should create new alert', async () => {
      const newAlert = {
        id: 'alert-4',
        symbol: 'TSLA',
        type: 'PRICE',
        condition: 'ABOVE',
        value: 700,
        enabled: true,
        triggered: false,
        message: 'TSLA breakout',
        createdAt: new Date().toISOString()
      }
      
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce(newAlert)
      const { result } = renderHook(() => useAlertStore())

      // Set form data
      act(() => {
        result.current.updateAlertForm({
          symbol: 'TSLA',
          value: 700,
          message: 'TSLA breakout'
        })
      })

      await act(async () => {
        const created = await result.current.createAlert(newAlert)
        expect(created).toEqual(newAlert)
      })

      expect(result.current.alerts).toContainEqual(newAlert)
      expect(result.current.activeAlerts).toContainEqual(newAlert)
      expect(result.current.alertForm.symbol).toBe('') // Should reset
      expect(result.current.isCreatingAlert).toBe(false)
    })

    it('should not add disabled alerts to active list', async () => {
      const disabledAlert = { ...mockAlerts[0], enabled: false }
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce(disabledAlert)
      const { result } = renderHook(() => useAlertStore())

      await act(async () => {
        await result.current.createAlert(disabledAlert)
      })

      expect(result.current.alerts).toContainEqual(disabledAlert)
      expect(result.current.activeAlerts).not.toContainEqual(disabledAlert)
    })
  })

  describe('Alert Updates', () => {
    it('should update alert', async () => {
      const updatedAlert = { ...mockAlerts[0], value: 160 }
      ;(alertService.updateAlert as jest.Mock).mockResolvedValueOnce(updatedAlert)
      const { result } = renderHook(() => useAlertStore())
      
      // Set initial state
      act(() => {
        useAlertStore.setState({ alerts: mockAlerts })
      })

      await act(async () => {
        await result.current.updateAlert('alert-1', { value: 160 })
      })

      const alert = result.current.alerts.find(a => a.id === 'alert-1')
      expect(alert?.value).toBe(160)
      expect(alertService.updateAlert).toHaveBeenCalledWith('alert-1', { value: 160 })
    })

    it('should toggle alert enabled state', async () => {
      const toggledAlert = { ...mockAlerts[0], enabled: false }
      ;(alertService.updateAlert as jest.Mock).mockResolvedValueOnce(toggledAlert)
      const { result } = renderHook(() => useAlertStore())
      
      // Set initial state
      act(() => {
        useAlertStore.setState({ 
          alerts: mockAlerts,
          activeAlerts: [mockAlerts[0]]
        })
      })

      await act(async () => {
        await result.current.toggleAlert('alert-1', false)
      })

      expect(result.current.activeAlerts).not.toContainEqual(expect.objectContaining({ id: 'alert-1' }))
    })

    it('should handle triggered alert updates', async () => {
      const triggeredAlert = { ...mockAlerts[0], triggered: true }
      ;(alertService.updateAlert as jest.Mock).mockResolvedValueOnce(triggeredAlert)
      const { result } = renderHook(() => useAlertStore())
      
      // Set initial state
      act(() => {
        useAlertStore.setState({ 
          alerts: mockAlerts,
          activeAlerts: [mockAlerts[0]]
        })
      })

      await act(async () => {
        await result.current.updateAlert('alert-1', { triggered: true })
      })

      expect(result.current.activeAlerts).not.toContainEqual(expect.objectContaining({ id: 'alert-1' }))
    })
  })

  describe('Alert Deletion', () => {
    it('should delete alert', async () => {
      ;(alertService.deleteAlert as jest.Mock).mockResolvedValueOnce(undefined)
      const { result } = renderHook(() => useAlertStore())
      
      // Set initial state
      act(() => {
        useAlertStore.setState({ 
          alerts: mockAlerts,
          activeAlerts: [mockAlerts[0]],
          notifications: [mockNotification]
        })
      })

      await act(async () => {
        await result.current.deleteAlert('alert-1')
      })

      expect(result.current.alerts).not.toContainEqual(expect.objectContaining({ id: 'alert-1' }))
      expect(result.current.activeAlerts).toHaveLength(0)
      expect(result.current.notifications).toHaveLength(0)
      expect(alertService.deleteAlert).toHaveBeenCalledWith('alert-1')
    })
  })

  describe('Alert Form Management', () => {
    it('should update alert form', () => {
      const { result } = renderHook(() => useAlertStore())

      act(() => {
        result.current.updateAlertForm({
          symbol: 'AAPL',
          value: 155,
          message: 'Target reached'
        })
      })

      expect(result.current.alertForm.symbol).toBe('AAPL')
      expect(result.current.alertForm.value).toBe(155)
      expect(result.current.alertForm.message).toBe('Target reached')
    })

    it('should reset alert form', () => {
      const { result } = renderHook(() => useAlertStore())

      // Set some values
      act(() => {
        result.current.updateAlertForm({
          symbol: 'AAPL',
          value: 155,
          message: 'Test'
        })
      })

      act(() => {
        result.current.resetAlertForm()
      })

      expect(result.current.alertForm).toEqual({
        symbol: '',
        type: 'PRICE',
        condition: 'ABOVE',
        value: 0,
        message: '',
        sendEmail: true,
        sendPush: true
      })
    })
  })

  describe('Notification Management', () => {
    it('should add notification', () => {
      const { result } = renderHook(() => useAlertStore())

      act(() => {
        result.current.addNotification(mockNotification)
      })

      expect(result.current.notifications).toContainEqual(mockNotification)
      expect(result.current.unreadCount).toBe(1)
    })

    it('should mark alert as read', () => {
      const { result } = renderHook(() => useAlertStore())
      
      // Add notification
      act(() => {
        result.current.addNotification(mockNotification)
      })

      act(() => {
        result.current.markAlertAsRead('alert-1')
      })

      expect(result.current.notifications[0].read).toBe(true)
      expect(result.current.unreadCount).toBe(0)
    })

    it('should mark all alerts as read', () => {
      const { result } = renderHook(() => useAlertStore())
      
      // Add multiple notifications
      act(() => {
        result.current.addNotification(mockNotification)
        result.current.addNotification({ ...mockNotification, id: 'notif-2', alertId: 'alert-2' })
      })

      act(() => {
        result.current.markAllAlertsAsRead()
      })

      expect(result.current.notifications.every(n => n.read)).toBe(true)
      expect(result.current.unreadCount).toBe(0)
    })

    it('should remove notification', () => {
      const { result } = renderHook(() => useAlertStore())
      
      // Add notification
      act(() => {
        result.current.addNotification(mockNotification)
      })

      act(() => {
        result.current.removeNotification('notif-1')
      })

      expect(result.current.notifications).toHaveLength(0)
      expect(result.current.unreadCount).toBe(0)
    })

    it('should limit notifications to 100', () => {
      const { result } = renderHook(() => useAlertStore())
      
      // Add 105 notifications
      act(() => {
        for (let i = 0; i < 105; i++) {
          result.current.addNotification({
            ...mockNotification,
            id: `notif-${i}`,
            timestamp: new Date(Date.now() - i * 1000)
          })
        }
      })

      expect(result.current.notifications).toHaveLength(100)
      // Should keep the most recent ones
      expect(result.current.notifications[0].id).toBe('notif-104')
    })
  })

  describe('UI State Management', () => {
    it('should select and deselect alert', () => {
      const { result } = renderHook(() => useAlertStore())

      act(() => {
        result.current.selectAlert('alert-1')
      })
      expect(result.current.selectedAlertId).toBe('alert-1')

      act(() => {
        result.current.selectAlert(null)
      })
      expect(result.current.selectedAlertId).toBeNull()
    })

    it('should toggle notifications panel', () => {
      const { result } = renderHook(() => useAlertStore())

      expect(result.current.showNotifications).toBe(false)

      act(() => {
        result.current.toggleNotifications()
      })
      expect(result.current.showNotifications).toBe(true)

      act(() => {
        result.current.toggleNotifications()
      })
      expect(result.current.showNotifications).toBe(false)
    })
  })

  describe('Error Handling', () => {
    it('should handle alert creation errors', async () => {
      const error = new Error('Invalid alert parameters')
      ;(alertService.createAlert as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useAlertStore())

      await expect(
        act(async () => {
          await result.current.createAlert({})
        })
      ).rejects.toThrow('Invalid alert parameters')

      expect(result.current.isCreatingAlert).toBe(false)
      expect(result.current.alerts).toHaveLength(0)
    })

    it('should handle alert loading errors', async () => {
      const error = new Error('Network error')
      ;(alertService.getAlerts as jest.Mock).mockRejectedValueOnce(error)
      const { result } = renderHook(() => useAlertStore())

      await expect(
        act(async () => {
          await result.current.loadAlerts()
        })
      ).rejects.toThrow('Network error')

      expect(result.current.isLoadingAlerts).toBe(false)
    })
  })

  describe('Complex Workflows', () => {
    it('should handle complete alert lifecycle', async () => {
      const newAlert = {
        id: 'alert-new',
        symbol: 'AAPL',
        type: 'PRICE',
        condition: 'ABOVE',
        value: 160,
        enabled: true,
        triggered: false,
        message: 'New high',
        createdAt: new Date().toISOString()
      }
      
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce(newAlert)
      ;(alertService.updateAlert as jest.Mock).mockImplementation((id, updates) => 
        Promise.resolve({ ...newAlert, ...updates })
      )
      
      const { result } = renderHook(() => useAlertStore())

      // Create alert
      await act(async () => {
        await result.current.createAlert(newAlert)
      })
      expect(result.current.activeAlerts).toHaveLength(1)

      // Add notification when triggered
      act(() => {
        result.current.addNotification({
          id: 'notif-new',
          alertId: 'alert-new',
          title: 'Price Alert',
          message: 'AAPL reached $160',
          timestamp: new Date(),
          read: false,
          type: 'price'
        })
      })
      expect(result.current.unreadCount).toBe(1)

      // Update alert as triggered
      await act(async () => {
        await result.current.updateAlert('alert-new', { triggered: true })
      })
      expect(result.current.activeAlerts).toHaveLength(0)

      // Mark notification as read
      act(() => {
        result.current.markAlertAsRead('alert-new')
      })
      expect(result.current.unreadCount).toBe(0)
    })

    it('should maintain notification order and unread count', () => {
      const { result } = renderHook(() => useAlertStore())
      
      // Add notifications with different read states
      act(() => {
        result.current.addNotification({ ...mockNotification, id: 'n1', read: false })
        result.current.addNotification({ ...mockNotification, id: 'n2', read: true })
        result.current.addNotification({ ...mockNotification, id: 'n3', read: false })
      })

      expect(result.current.unreadCount).toBe(2)
      expect(result.current.notifications[0].id).toBe('n3') // Most recent first

      // Mark one as read
      act(() => {
        result.current.markAlertAsRead('alert-1')
      })

      expect(result.current.unreadCount).toBe(0) // All have same alertId
    })
  })
})
