import React from 'react'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useAlertStore } from '@/store/useAlertStore'
import { useMarketStore } from '@/store/useMarketStore'
import { alertService } from '@/services/api/alerts'
import { marketService } from '@/services/api/market-optimized'
import toast from 'react-hot-toast'

// Mock services
jest.mock('@/services/api/alerts')
jest.mock('@/services/api/market-optimized')
jest.mock('react-hot-toast')

// Alert Management Component
const AlertManagement = () => {
  const {
    alerts,
    activeAlerts,
    triggeredAlerts,
    notifications,
    unreadCount,
    alertForm,
    isCreatingAlert,
    loadAlerts,
    createAlert,
    updateAlert,
    deleteAlert,
    toggleAlert,
    updateAlertForm,
    resetAlertForm,
    markAlertAsRead,
    markAllAlertsAsRead,
  } = useAlertStore()
  
  const { selectedSymbol, selectSymbol } = useMarketStore()
  
  const [showCreateForm, setShowCreateForm] = React.useState(false)
  const [editingAlert, setEditingAlert] = React.useState<string | null>(null)
  const [activeTab, setActiveTab] = React.useState<'active' | 'triggered' | 'all'>('active')
  
  React.useEffect(() => {
    loadAlerts()
  }, [loadAlerts])
  
  const handleCreateAlert = async () => {
    try {
      await createAlert({
        ...alertForm,
        symbol: selectedSymbol || alertForm.symbol,
      })
      setShowCreateForm(false)
      resetAlertForm()
      toast.success('Alert created successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to create alert')
    }
  }
  
  const handleUpdateAlert = async (alertId: string, updates: any) => {
    try {
      await updateAlert(alertId, updates)
      setEditingAlert(null)
      toast.success('Alert updated successfully')
    } catch (error: any) {
      toast.error(error.message || 'Failed to update alert')
    }
  }
  
  const handleDeleteAlert = async (alertId: string) => {
    if (window.confirm('Are you sure you want to delete this alert?')) {
      try {
        await deleteAlert(alertId)
        toast.success('Alert deleted successfully')
      } catch (error: any) {
        toast.error(error.message || 'Failed to delete alert')
      }
    }
  }
  
  const displayAlerts = activeTab === 'active' 
    ? activeAlerts 
    : activeTab === 'triggered' 
    ? triggeredAlerts 
    : alerts
  
  return (
    <div>
      <h1>Alert Management</h1>
      
      {/* Notifications Panel */}
      <div data-testid="notifications-panel">
        <h3>
          Notifications 
          {unreadCount > 0 && <span className="badge">{unreadCount}</span>}
        </h3>
        {notifications.length === 0 ? (
          <p>No notifications</p>
        ) : (
          <>
            <button onClick={() => markAllAlertsAsRead()}>Mark all as read</button>
            <ul>
              {notifications.slice(0, 5).map(notification => (
                <li
                  key={notification.id}
                  data-testid={`notification-${notification.id}`}
                  className={notification.read ? 'read' : 'unread'}
                  onClick={() => markAlertAsRead(notification.alertId)}
                >
                  <strong>{notification.title}</strong>
                  <p>{notification.message}</p>
                  <time>{new Date(notification.timestamp).toLocaleString()}</time>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      
      {/* Create Alert Button */}
      <button 
        onClick={() => setShowCreateForm(true)}
        data-testid="create-alert-btn"
      >
        Create New Alert
      </button>
      
      {/* Create Alert Form */}
      {showCreateForm && (
        <div data-testid="create-alert-form">
          <h2>Create Alert</h2>
          
          <input
            type="text"
            placeholder="Symbol"
            value={alertForm.symbol}
            onChange={(e) => updateAlertForm({ symbol: e.target.value.toUpperCase() })}
          />
          
          <select
            value={alertForm.type}
            onChange={(e) => updateAlertForm({ type: e.target.value as any })}
          >
            <option value="PRICE">Price</option>
            <option value="VOLUME">Volume</option>
            <option value="TECHNICAL">Technical</option>
            <option value="NEWS">News</option>
            <option value="RISK">Risk</option>
          </select>
          
          <select
            value={alertForm.condition}
            onChange={(e) => updateAlertForm({ condition: e.target.value as any })}
          >
            <option value="ABOVE">Above</option>
            <option value="BELOW">Below</option>
            <option value="CROSSES_ABOVE">Crosses Above</option>
            <option value="CROSSES_BELOW">Crosses Below</option>
          </select>
          
          <input
            type="number"
            placeholder="Value"
            value={alertForm.value}
            onChange={(e) => updateAlertForm({ value: parseFloat(e.target.value) })}
          />
          
          <textarea
            placeholder="Message (optional)"
            value={alertForm.message}
            onChange={(e) => updateAlertForm({ message: e.target.value })}
          />
          
          <label>
            <input
              type="checkbox"
              checked={alertForm.sendEmail}
              onChange={(e) => updateAlertForm({ sendEmail: e.target.checked })}
            />
            Send Email
          </label>
          
          <label>
            <input
              type="checkbox"
              checked={alertForm.sendPush}
              onChange={(e) => updateAlertForm({ sendPush: e.target.checked })}
            />
            Send Push Notification
          </label>
          
          <button 
            onClick={handleCreateAlert}
            disabled={isCreatingAlert || !alertForm.symbol || !alertForm.value}
          >
            Create Alert
          </button>
          
          <button onClick={() => {
            setShowCreateForm(false)
            resetAlertForm()
          }}>
            Cancel
          </button>
        </div>
      )}
      
      {/* Alert Tabs */}
      <div data-testid="alert-tabs">
        <button
          className={activeTab === 'active' ? 'active' : ''}
          onClick={() => setActiveTab('active')}
        >
          Active ({activeAlerts.length})
        </button>
        <button
          className={activeTab === 'triggered' ? 'active' : ''}
          onClick={() => setActiveTab('triggered')}
        >
          Triggered ({triggeredAlerts.length})
        </button>
        <button
          className={activeTab === 'all' ? 'active' : ''}
          onClick={() => setActiveTab('all')}
        >
          All ({alerts.length})
        </button>
      </div>
      
      {/* Alerts List */}
      <div data-testid="alerts-list">
        {displayAlerts.length === 0 ? (
          <p>No alerts to display</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Type</th>
                <th>Condition</th>
                <th>Value</th>
                <th>Current</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayAlerts.map(alert => (
                <tr key={alert.id} data-testid={`alert-row-${alert.id}`}>
                  <td>{alert.symbol}</td>
                  <td>{alert.type}</td>
                  <td>{alert.condition}</td>
                  <td>{alert.value}</td>
                  <td>{alert.currentValue || '-'}</td>
                  <td>
                    <span className={`status ${alert.triggered ? 'triggered' : 'active'}`}>
                      {alert.triggered ? 'Triggered' : alert.enabled ? 'Active' : 'Disabled'}
                    </span>
                  </td>
                  <td>
                    {editingAlert === alert.id ? (
                      <>
                        <input
                          type="number"
                          defaultValue={alert.value}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              const input = e.target as HTMLInputElement
                              handleUpdateAlert(alert.id, { value: parseFloat(input.value) })
                            }
                          }}
                        />
                        <button onClick={() => setEditingAlert(null)}>Cancel</button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => toggleAlert(alert.id, !alert.enabled)}
                          data-testid={`toggle-alert-${alert.id}`}
                        >
                          {alert.enabled ? 'Disable' : 'Enable'}
                        </button>
                        <button
                          onClick={() => setEditingAlert(alert.id)}
                          data-testid={`edit-alert-${alert.id}`}
                          disabled={alert.triggered}
                        >
                          Edit
                        </button>
                        <button
                          onClick={() => handleDeleteAlert(alert.id)}
                          data-testid={`delete-alert-${alert.id}`}
                        >
                          Delete
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      {/* Alert Statistics */}
      <div data-testid="alert-stats">
        <h3>Alert Statistics</h3>
        <div>
          <span>Total Alerts: {alerts.length}</span>
          <span>Active: {activeAlerts.length}</span>
          <span>Triggered Today: {
            triggeredAlerts.filter(a => 
              new Date(a.triggeredAt!).toDateString() === new Date().toDateString()
            ).length
          }</span>
        </div>
      </div>
    </div>
  )
}

describe('Alert Management Integration', () => {
  const mockAlerts = [
    {
      id: 'alert-1',
      userId: 'user-1',
      symbol: 'AAPL',
      type: 'PRICE' as const,
      condition: 'ABOVE' as const,
      value: 160,
      currentValue: 155,
      enabled: true,
      triggered: false,
      sendEmail: true,
      sendPush: true,
      sendSMS: false,
      priority: 'HIGH' as const,
      message: 'AAPL breakout alert',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: 'alert-2',
      userId: 'user-1',
      symbol: 'GOOGL',
      type: 'VOLUME' as const,
      condition: 'ABOVE' as const,
      value: 2000000,
      currentValue: 1500000,
      enabled: true,
      triggered: true,
      triggeredAt: new Date().toISOString(),
      triggeredValue: 2100000,
      sendEmail: true,
      sendPush: false,
      sendSMS: false,
      priority: 'MEDIUM' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: 'alert-3',
      userId: 'user-1',
      symbol: 'MSFT',
      type: 'PRICE' as const,
      condition: 'BELOW' as const,
      value: 300,
      enabled: false,
      triggered: false,
      sendEmail: false,
      sendPush: true,
      sendSMS: false,
      priority: 'LOW' as const,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ]
  
  const mockNotifications = [
    {
      id: 'notif-1',
      alertId: 'alert-2',
      title: 'Volume Alert Triggered',
      message: 'GOOGL volume exceeded 2M shares',
      timestamp: new Date(),
      read: false,
      type: 'volume' as const,
    },
  ]
  
  beforeEach(() => {
    jest.clearAllMocks()
    window.confirm = jest.fn(() => true)
    
    // Reset store
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
        sendPush: true,
      },
      isLoadingAlerts: false,
      isCreatingAlert: false,
    })
    
    useMarketStore.setState({
      selectedSymbol: null,
    })
  })
  
  describe('Alert Creation', () => {
    it('should create new price alert', async () => {
      const user = userEvent.setup()
      const newAlert = {
        id: 'alert-4',
        userId: 'user-1',
        symbol: 'TSLA',
        type: 'PRICE' as const,
        condition: 'ABOVE' as const,
        value: 700,
        enabled: true,
        triggered: false,
        sendEmail: true,
        sendPush: true,
        sendSMS: false,
        priority: 'MEDIUM' as const,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce(newAlert)
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('create-alert-btn')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('create-alert-btn'))
      
      const form = screen.getByTestId('create-alert-form')
      const symbolInput = within(form).getByPlaceholderText('Symbol')
      const valueInput = within(form).getByPlaceholderText('Value')
      const messageInput = within(form).getByPlaceholderText('Message (optional)')
      const createButton = within(form).getByText('Create Alert')
      
      await user.type(symbolInput, 'TSLA')
      await user.clear(valueInput)
      await user.type(valueInput, '700')
      await user.type(messageInput, 'Tesla breakout alert')
      await user.click(createButton)
      
      expect(alertService.createAlert).toHaveBeenCalledWith({
        symbol: 'TSLA',
        type: 'PRICE',
        condition: 'ABOVE',
        value: 700,
        message: 'Tesla breakout alert',
        sendEmail: true,
        sendPush: true,
      })
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Alert created successfully')
      })
    })
    
    it('should create volume alert', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce({} as any)
      
      render(<AlertManagement />)
      
      await user.click(screen.getByTestId('create-alert-btn'))
      
      const form = screen.getByTestId('create-alert-form')
      const symbolInput = within(form).getByPlaceholderText('Symbol')
      const typeSelect = within(form).getByRole('combobox', { name: '' }).parentElement!.querySelector('select')!
      const valueInput = within(form).getByPlaceholderText('Value')
      const createButton = within(form).getByText('Create Alert')
      
      await user.type(symbolInput, 'AAPL')
      await user.selectOptions(typeSelect, 'VOLUME')
      await user.clear(valueInput)
      await user.type(valueInput, '10000000')
      await user.click(createButton)
      
      expect(alertService.createAlert).toHaveBeenCalledWith(
        expect.objectContaining({
          symbol: 'AAPL',
          type: 'VOLUME',
          value: 10000000,
        })
      )
    })
    
    it('should validate required fields', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<AlertManagement />)
      
      await user.click(screen.getByTestId('create-alert-btn'))
      
      const form = screen.getByTestId('create-alert-form')
      const createButton = within(form).getByText('Create Alert')
      
      // Button should be disabled without required fields
      expect(createButton).toBeDisabled()
      
      // Fill symbol only
      const symbolInput = within(form).getByPlaceholderText('Symbol')
      await user.type(symbolInput, 'AAPL')
      
      // Still disabled without value
      expect(createButton).toBeDisabled()
      
      // Fill value
      const valueInput = within(form).getByPlaceholderText('Value')
      await user.type(valueInput, '160')
      
      // Now enabled
      expect(createButton).not.toBeDisabled()
    })
  })
  
  describe('Alert Management', () => {
    it('should display alerts in correct tabs', async () => {
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('alert-tabs')).toBeInTheDocument()
      })
      
      // Check active tab
      const tabs = screen.getByTestId('alert-tabs')
      expect(within(tabs).getByText('Active (1)')).toBeInTheDocument()
      expect(within(tabs).getByText('Triggered (1)')).toBeInTheDocument()
      expect(within(tabs).getByText('All (3)')).toBeInTheDocument()
      
      // Active tab should be selected by default
      expect(screen.getByTestId('alert-row-alert-1')).toBeInTheDocument()
      expect(screen.queryByTestId('alert-row-alert-2')).not.toBeInTheDocument()
      
      // Switch to triggered tab
      await userEvent.click(within(tabs).getByText('Triggered (1)'))
      
      expect(screen.queryByTestId('alert-row-alert-1')).not.toBeInTheDocument()
      expect(screen.getByTestId('alert-row-alert-2')).toBeInTheDocument()
      
      // Switch to all tab
      await userEvent.click(within(tabs).getByText('All (3)'))
      
      expect(screen.getByTestId('alert-row-alert-1')).toBeInTheDocument()
      expect(screen.getByTestId('alert-row-alert-2')).toBeInTheDocument()
      expect(screen.getByTestId('alert-row-alert-3')).toBeInTheDocument()
    })
    
    it('should toggle alert enabled state', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.updateAlert as jest.Mock).mockResolvedValueOnce({
        ...mockAlerts[0],
        enabled: false,
      })
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('toggle-alert-alert-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('toggle-alert-alert-1'))
      
      expect(alertService.updateAlert).toHaveBeenCalledWith('alert-1', { enabled: false })
    })
    
    it('should edit alert value', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.updateAlert as jest.Mock).mockResolvedValueOnce({
        ...mockAlerts[0],
        value: 165,
      })
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('edit-alert-alert-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('edit-alert-alert-1'))
      
      // Find the input in the same row
      const alertRow = screen.getByTestId('alert-row-alert-1')
      const input = within(alertRow).getByRole('spinbutton')
      
      await user.clear(input)
      await user.type(input, '165')
      await user.keyboard('{Enter}')
      
      expect(alertService.updateAlert).toHaveBeenCalledWith('alert-1', { value: 165 })
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Alert updated successfully')
      })
    })
    
    it('should delete alert with confirmation', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.deleteAlert as jest.Mock).mockResolvedValueOnce(undefined)
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('delete-alert-alert-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('delete-alert-alert-1'))
      
      expect(window.confirm).toHaveBeenCalledWith('Are you sure you want to delete this alert?')
      expect(alertService.deleteAlert).toHaveBeenCalledWith('alert-1')
      
      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith('Alert deleted successfully')
      })
    })
    
    it('should not delete if cancelled', async () => {
      const user = userEvent.setup()
      window.confirm = jest.fn(() => false)
      
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('delete-alert-alert-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('delete-alert-alert-1'))
      
      expect(alertService.deleteAlert).not.toHaveBeenCalled()
    })
  })
  
  describe('Notifications', () => {
    it('should display notifications with unread count', async () => {
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      // Add notifications to store
      useAlertStore.setState({
        notifications: mockNotifications,
        unreadCount: 1,
      })
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        const panel = screen.getByTestId('notifications-panel')
        expect(within(panel).getByText('1')).toHaveClass('badge')
        expect(screen.getByTestId('notification-notif-1')).toBeInTheDocument()
        expect(screen.getByTestId('notification-notif-1')).toHaveClass('unread')
      })
    })
    
    it('should mark notification as read', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      useAlertStore.setState({
        notifications: mockNotifications,
        unreadCount: 1,
      })
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('notification-notif-1')).toBeInTheDocument()
      })
      
      await user.click(screen.getByTestId('notification-notif-1'))
      
      // Notification should be marked as read
      expect(useAlertStore.getState().unreadCount).toBe(0)
    })
    
    it('should mark all notifications as read', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      const multipleNotifications = [
        ...mockNotifications,
        {
          id: 'notif-2',
          alertId: 'alert-1',
          title: 'Price Alert',
          message: 'AAPL reached $161',
          timestamp: new Date(),
          read: false,
          type: 'price' as const,
        },
      ]
      
      useAlertStore.setState({
        notifications: multipleNotifications,
        unreadCount: 2,
      })
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByText('Mark all as read')).toBeInTheDocument()
      })
      
      await user.click(screen.getByText('Mark all as read'))
      
      expect(useAlertStore.getState().unreadCount).toBe(0)
    })
  })
  
  describe('Alert Statistics', () => {
    it('should display alert statistics', async () => {
      const todayAlert = {
        ...mockAlerts[1],
        triggeredAt: new Date().toISOString(),
      }
      
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce([
        ...mockAlerts.slice(0, 1),
        todayAlert,
        ...mockAlerts.slice(2),
      ])
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        const stats = screen.getByTestId('alert-stats')
        expect(within(stats).getByText('Total Alerts: 3')).toBeInTheDocument()
        expect(within(stats).getByText('Active: 1')).toBeInTheDocument()
        expect(within(stats).getByText('Triggered Today: 1')).toBeInTheDocument()
      })
    })
  })
  
  describe('Real-time Updates', () => {
    it('should handle alert trigger notification', async () => {
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      
      render(<AlertManagement />)
      
      await waitFor(() => {
        expect(screen.getByTestId('alert-row-alert-1')).toBeInTheDocument()
      })
      
      // Simulate alert trigger
      act(() => {
        useAlertStore.getState().addNotification({
          id: 'notif-new',
          alertId: 'alert-1',
          title: 'Price Alert Triggered',
          message: 'AAPL reached $161',
          timestamp: new Date(),
          read: false,
          type: 'price',
        })
        
        // Update alert state
        const updatedAlerts = mockAlerts.map(a => 
          a.id === 'alert-1' 
            ? { ...a, triggered: true, triggeredAt: new Date().toISOString() }
            : a
        )
        useAlertStore.setState({
          alerts: updatedAlerts,
          activeAlerts: updatedAlerts.filter(a => a.enabled && !a.triggered),
          triggeredAlerts: updatedAlerts.filter(a => a.triggered),
        })
      })
      
      // Alert should move to triggered tab
      expect(screen.queryByTestId('alert-row-alert-1')).not.toBeInTheDocument()
      
      // Check triggered tab
      const tabs = screen.getByTestId('alert-tabs')
      await userEvent.click(within(tabs).getByText(/Triggered/))
      
      expect(screen.getByTestId('alert-row-alert-1')).toBeInTheDocument()
    })
  })
  
  describe('Integration with Market Data', () => {
    it('should use selected symbol for alert creation', async () => {
      const user = userEvent.setup()
      ;(alertService.getAlerts as jest.Mock).mockResolvedValueOnce(mockAlerts)
      ;(alertService.createAlert as jest.Mock).mockResolvedValueOnce({} as any)
      
      // Set selected symbol in market store
      useMarketStore.setState({ selectedSymbol: 'NVDA' })
      
      render(<AlertManagement />)
      
      await user.click(screen.getByTestId('create-alert-btn'))
      
      const form = screen.getByTestId('create-alert-form')
      const symbolInput = within(form).getByPlaceholderText('Symbol')
      const valueInput = within(form).getByPlaceholderText('Value')
      const createButton = within(form).getByText('Create Alert')
      
      // Symbol should be pre-filled
      expect(symbolInput).toHaveValue('NVDA')
      
      await user.type(valueInput, '500')
      await user.click(createButton)
      
      expect(alertService.createAlert).toHaveBeenCalledWith(
        expect.objectContaining({
          symbol: 'NVDA',
        })
      )
    })
  })
})
