"""Alert model for user notifications and trading alerts."""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class Alert(Base):
    """Alert model for price, technical, and custom alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Alert identification
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Alert type and target
    alert_type = Column(String, nullable=False)  # price, technical, volume, options, news
    symbol = Column(String, index=True)
    asset_type = Column(String, default="stock")  # stock, option, index
    
    # Condition
    condition_type = Column(String, nullable=False)  # above, below, crosses_above, crosses_below, equals, change_percent
    condition_value = Column(Float)
    condition_field = Column(String)  # price, volume, rsi, macd, etc.
    
    # Additional conditions (for complex alerts)
    conditions = Column(JSON)  # List of additional conditions
    
    # Notification settings
    notification_channels = Column(JSON, default=list)  # email, sms, push, webhook
    webhook_url = Column(String)
    
    # Alert status
    is_active = Column(Boolean, default=True)
    is_one_time = Column(Boolean, default=False)  # Disable after triggering
    triggered_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)
    
    # Cooldown period
    cooldown_minutes = Column(Integer, default=60)  # Prevent spam
    
    # Expiration
    expires_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    alert_history = relationship("AlertHistory", back_populates="alert", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_alert_user_active', 'user_id', 'is_active'),
        Index('idx_alert_symbol', 'symbol'),
    )
    
    def __repr__(self):
        return f"<Alert(name={self.name}, symbol={self.symbol}, type={self.alert_type})>"


class AlertHistory(Base):
    """History of triggered alerts."""
    
    __tablename__ = "alert_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(UUID(as_uuid=True), ForeignKey("alerts.id"), nullable=False)
    
    # Trigger details
    triggered_at = Column(DateTime, default=datetime.utcnow)
    trigger_value = Column(Float)  # The value that triggered the alert
    condition_met = Column(String)  # Description of condition met
    
    # Market context
    market_data = Column(JSON)  # Snapshot of relevant market data
    
    # Notification status
    notifications_sent = Column(JSON)  # Which channels were notified
    notification_errors = Column(JSON)  # Any errors in sending
    
    # Relationships
    alert = relationship("Alert", back_populates="alert_history")
    
    def __repr__(self):
        return f"<AlertHistory(alert_id={self.alert_id}, triggered_at={self.triggered_at})>"


class Strategy(Base):
    """Trading strategies created by users."""
    
    __tablename__ = "strategies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Strategy identification
    name = Column(String, nullable=False)
    description = Column(String)
    strategy_type = Column(String)  # trend_following, mean_reversion, arbitrage, options_spread
    
    # Configuration
    config = Column(JSON, nullable=False)  # Strategy-specific configuration
    
    # Backtesting results
    backtest_results = Column(JSON)  # Performance metrics from backtesting
    
    # Status
    is_active = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)  # Share with community
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)
    win_rate = Column(Float)
    avg_return = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Risk parameters
    max_position_size = Column(Float)
    stop_loss_percent = Column(Float)
    take_profit_percent = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_executed = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="strategies")
    
    def __repr__(self):
        return f"<Strategy(name={self.name}, type={self.strategy_type}, active={self.is_active})>"


class Notification(Base):
    """User notifications and messages."""
    
    __tablename__ = "notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Notification details
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    notification_type = Column(String)  # alert, trade, system, news
    
    # Priority and status
    priority = Column(String, default="normal")  # low, normal, high, urgent
    is_read = Column(Boolean, default=False)
    
    # Related entities
    related_entity_type = Column(String)  # trade, alert, position
    related_entity_id = Column(UUID(as_uuid=True))
    
    # Action
    action_url = Column(String)  # Link to relevant page
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_user_unread', 'user_id', 'is_read'),
        Index('idx_notification_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Notification(title={self.title}, type={self.notification_type}, read={self.is_read})>"
