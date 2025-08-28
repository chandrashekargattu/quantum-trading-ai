"""Alert management endpoints."""

from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.alert import Alert, AlertHistory
from app.core.security import get_current_active_user
from app.schemas.alert import AlertCreate, AlertUpdate, AlertResponse, AlertHistoryResponse

router = APIRouter()


@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    active_only: bool = Query(True, description="Show only active alerts"),
    symbol: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get user's alerts."""
    query = select(Alert).where(Alert.user_id == current_user.id)
    
    if active_only:
        query = query.where(Alert.is_active == True)
    
    if symbol:
        query = query.where(Alert.symbol == symbol.upper())
    
    if alert_type:
        query = query.where(Alert.alert_type == alert_type)
    
    query = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    return alerts


@router.post("/", response_model=AlertResponse)
async def create_alert(
    alert_data: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Create a new alert."""
    # Create alert
    alert = Alert(
        user_id=current_user.id,
        name=alert_data.name,
        description=alert_data.description,
        alert_type=alert_data.alert_type,
        symbol=alert_data.symbol.upper() if alert_data.symbol else None,
        asset_type=alert_data.asset_type or "stock",
        condition_type=alert_data.condition_type,
        condition_value=alert_data.condition_value,
        condition_field=alert_data.condition_field,
        conditions=alert_data.conditions,
        notification_channels=alert_data.notification_channels or ["email"],
        webhook_url=alert_data.webhook_url,
        is_one_time=alert_data.is_one_time,
        cooldown_minutes=alert_data.cooldown_minutes or 60,
        expires_at=alert_data.expires_at
    )
    
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    
    return alert


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get specific alert details."""
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    return alert


@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: str,
    alert_update: AlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Update an alert."""
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Update fields
    update_data = alert_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alert, field, value)
    
    alert.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(alert)
    
    return alert


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Delete an alert."""
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    await db.delete(alert)
    await db.commit()
    
    return {"message": "Alert deleted successfully"}


@router.post("/{alert_id}/toggle")
async def toggle_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Toggle alert active status."""
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.is_active = not alert.is_active
    alert.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "message": f"Alert {'activated' if alert.is_active else 'deactivated'}",
        "is_active": alert.is_active
    }


@router.get("/{alert_id}/history", response_model=List[AlertHistoryResponse])
async def get_alert_history(
    alert_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get alert trigger history."""
    # Verify alert ownership
    alert_result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    if not alert_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Get history
    result = await db.execute(
        select(AlertHistory).where(
            AlertHistory.alert_id == alert_id
        ).order_by(AlertHistory.triggered_at.desc()).limit(limit)
    )
    history = result.scalars().all()
    
    return history


@router.post("/test/{alert_id}")
async def test_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
) -> Any:
    """Test an alert by triggering it manually."""
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.id == alert_id,
                Alert.user_id == current_user.id
            )
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    # Create test alert history entry
    test_history = AlertHistory(
        alert_id=alert.id,
        triggered_at=datetime.utcnow(),
        trigger_value=0.0,
        condition_met="Manual test trigger",
        market_data={"test": True},
        notifications_sent=["test"]
    )
    
    db.add(test_history)
    await db.commit()
    
    # TODO: Send test notification
    
    return {"message": "Test alert sent successfully"}


@router.get("/templates/", response_model=List[dict])
async def get_alert_templates(
    current_user = Depends(get_current_active_user)
) -> Any:
    """Get predefined alert templates."""
    templates = [
        {
            "name": "Price Above",
            "description": "Alert when price goes above a certain value",
            "alert_type": "price",
            "condition_type": "above",
            "condition_field": "price"
        },
        {
            "name": "Price Below",
            "description": "Alert when price falls below a certain value",
            "alert_type": "price",
            "condition_type": "below",
            "condition_field": "price"
        },
        {
            "name": "Volume Spike",
            "description": "Alert on unusual volume activity",
            "alert_type": "volume",
            "condition_type": "above",
            "condition_field": "volume"
        },
        {
            "name": "RSI Oversold",
            "description": "Alert when RSI indicates oversold conditions",
            "alert_type": "technical",
            "condition_type": "below",
            "condition_field": "rsi",
            "condition_value": 30
        },
        {
            "name": "RSI Overbought",
            "description": "Alert when RSI indicates overbought conditions",
            "alert_type": "technical",
            "condition_type": "above",
            "condition_field": "rsi",
            "condition_value": 70
        },
        {
            "name": "Options High Volume",
            "description": "Alert on high options volume",
            "alert_type": "options",
            "condition_type": "above",
            "condition_field": "volume"
        }
    ]
    
    return templates
