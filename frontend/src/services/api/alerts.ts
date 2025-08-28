export type AlertType = 'PRICE' | 'VOLUME' | 'TECHNICAL' | 'NEWS' | 'RISK' | 'PORTFOLIO'
export type AlertCondition = 'ABOVE' | 'BELOW' | 'CROSSES_ABOVE' | 'CROSSES_BELOW' | 'BETWEEN' | 'OUTSIDE'
export type AlertPriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface Alert {
  id: string
  userId: string
  name?: string
  symbol?: string
  type: AlertType
  condition: AlertCondition
  value: number
  value2?: number // For BETWEEN and OUTSIDE conditions
  currentValue?: number
  message?: string
  priority: AlertPriority
  enabled: boolean
  triggered: boolean
  triggeredAt?: string
  triggeredValue?: number
  sendEmail: boolean
  sendPush: boolean
  sendSMS: boolean
  metadata?: Record<string, any>
  createdAt: string
  updatedAt: string
}

export interface AlertTrigger {
  id: string
  alertId: string
  alert: Alert
  triggeredValue: number
  message: string
  timestamp: string
  acknowledged: boolean
  acknowledgedAt?: string
}

export interface AlertTemplate {
  id: string
  name: string
  description: string
  type: AlertType
  condition: AlertCondition
  defaultMessage: string
  parameters: Array<{
    name: string
    type: 'number' | 'string' | 'boolean'
    required: boolean
    defaultValue?: any
  }>
}

export interface AlertStats {
  totalAlerts: number
  activeAlerts: number
  triggeredToday: number
  triggeredThisWeek: number
  triggeredThisMonth: number
  byType: Record<AlertType, number>
  byPriority: Record<AlertPriority, number>
}

class AlertService {
  async getAlerts(): Promise<Alert[]> {
    const response = await fetch('/api/v1/alerts')
    if (!response.ok) throw new Error('Failed to fetch alerts')
    return response.json()
  }

  async getActiveAlerts(): Promise<Alert[]> {
    const response = await fetch('/api/v1/alerts/active')
    if (!response.ok) throw new Error('Failed to fetch active alerts')
    return response.json()
  }

  async getTriggeredAlerts(limit = 50): Promise<Alert[]> {
    const response = await fetch(`/api/v1/alerts/triggered?limit=${limit}`)
    if (!response.ok) throw new Error('Failed to fetch triggered alerts')
    return response.json()
  }

  async getAlert(id: string): Promise<Alert> {
    const response = await fetch(`/api/v1/alerts/${id}`)
    if (!response.ok) throw new Error('Failed to fetch alert')
    return response.json()
  }

  async createAlert(alert: Partial<Alert>): Promise<Alert> {
    const response = await fetch('/api/v1/alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alert)
    })
    if (!response.ok) throw new Error('Failed to create alert')
    return response.json()
  }

  async updateAlert(id: string, updates: Partial<Alert>): Promise<Alert> {
    const response = await fetch(`/api/v1/alerts/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    })
    if (!response.ok) throw new Error('Failed to update alert')
    return response.json()
  }

  async deleteAlert(id: string): Promise<void> {
    const response = await fetch(`/api/v1/alerts/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete alert')
  }

  async toggleAlert(id: string, enabled: boolean): Promise<Alert> {
    return this.updateAlert(id, { enabled })
  }

  async testAlert(id: string): Promise<void> {
    const response = await fetch(`/api/v1/alerts/${id}/test`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to test alert')
  }

  async getAlertTriggers(alertId?: string, limit = 100): Promise<AlertTrigger[]> {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (alertId) params.append('alertId', alertId)
    
    const response = await fetch(`/api/v1/alerts/triggers?${params}`)
    if (!response.ok) throw new Error('Failed to fetch alert triggers')
    return response.json()
  }

  async acknowledgeAlertTrigger(triggerId: string): Promise<void> {
    const response = await fetch(`/api/v1/alerts/triggers/${triggerId}/acknowledge`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to acknowledge alert trigger')
  }

  async getAlertTemplates(): Promise<AlertTemplate[]> {
    const response = await fetch('/api/v1/alerts/templates')
    if (!response.ok) throw new Error('Failed to fetch alert templates')
    return response.json()
  }

  async createAlertFromTemplate(
    templateId: string,
    parameters: Record<string, any>
  ): Promise<Alert> {
    const response = await fetch(`/api/v1/alerts/templates/${templateId}/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parameters)
    })
    if (!response.ok) throw new Error('Failed to create alert from template')
    return response.json()
  }

  async getAlertStats(): Promise<AlertStats> {
    const response = await fetch('/api/v1/alerts/stats')
    if (!response.ok) throw new Error('Failed to fetch alert statistics')
    return response.json()
  }

  async bulkUpdateAlerts(
    alertIds: string[],
    updates: Partial<Alert>
  ): Promise<Alert[]> {
    const response = await fetch('/api/v1/alerts/bulk-update', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alertIds, updates })
    })
    if (!response.ok) throw new Error('Failed to bulk update alerts')
    return response.json()
  }

  async bulkDeleteAlerts(alertIds: string[]): Promise<void> {
    const response = await fetch('/api/v1/alerts/bulk-delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alertIds })
    })
    if (!response.ok) throw new Error('Failed to bulk delete alerts')
  }

  // WebSocket connection for real-time alert notifications
  connectAlertNotifications(
    onAlert: (trigger: AlertTrigger) => void,
    onError?: (error: Error) => void
  ): () => void {
    const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/alerts`)
    
    ws.onmessage = (event) => {
      try {
        const trigger = JSON.parse(event.data)
        onAlert(trigger)
      } catch (error) {
        onError?.(new Error('Failed to parse alert notification'))
      }
    }
    
    ws.onerror = () => {
      onError?.(new Error('WebSocket connection error'))
    }
    
    return () => ws.close()
  }
}

export const alertService = new AlertService()
