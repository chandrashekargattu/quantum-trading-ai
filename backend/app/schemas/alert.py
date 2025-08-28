"""Alert-related schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class AlertCreate(BaseModel):
    """Alert creation schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    alert_type: str = Field(..., pattern="^(price|technical|volume|options|news)$")
    symbol: Optional[str] = None
    asset_type: Optional[str] = Field("stock", pattern="^(stock|option|index)$")
    condition_type: str = Field(..., pattern="^(above|below|crosses_above|crosses_below|equals|change_percent)$")
    condition_value: float
    condition_field: str
    conditions: Optional[List[Dict[str, Any]]] = None
    notification_channels: Optional[List[str]] = Field(default=["email"])
    webhook_url: Optional[str] = None
    is_one_time: bool = False
    cooldown_minutes: Optional[int] = Field(60, ge=1)
    expires_at: Optional[datetime] = None


class AlertUpdate(BaseModel):
    """Alert update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    condition_type: Optional[str] = None
    condition_value: Optional[float] = None
    condition_field: Optional[str] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    notification_channels: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    is_active: Optional[bool] = None
    is_one_time: Optional[bool] = None
    cooldown_minutes: Optional[int] = Field(None, ge=1)
    expires_at: Optional[datetime] = None


class AlertResponse(BaseModel):
    """Alert response schema."""
    id: UUID
    name: str
    description: Optional[str] = None
    alert_type: str
    symbol: Optional[str] = None
    asset_type: str
    condition_type: str
    condition_value: float
    condition_field: str
    conditions: Optional[List[Dict[str, Any]]] = None
    notification_channels: List[str]
    webhook_url: Optional[str] = None
    is_active: bool
    is_one_time: bool
    triggered_count: int
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AlertHistoryResponse(BaseModel):
    """Alert history response schema."""
    id: UUID
    alert_id: UUID
    triggered_at: datetime
    trigger_value: Optional[float] = None
    condition_met: str
    market_data: Optional[Dict[str, Any]] = None
    notifications_sent: Optional[List[str]] = None
    notification_errors: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class NotificationCreate(BaseModel):
    """Notification creation schema."""
    title: str
    message: str
    notification_type: Optional[str] = Field("alert", pattern="^(alert|trade|system|news)$")
    priority: Optional[str] = Field("normal", pattern="^(low|normal|high|urgent)$")
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[UUID] = None
    action_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class NotificationResponse(BaseModel):
    """Notification response schema."""
    id: UUID
    title: str
    message: str
    notification_type: str
    priority: str
    is_read: bool
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[UUID] = None
    action_url: Optional[str] = None
    created_at: datetime
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
