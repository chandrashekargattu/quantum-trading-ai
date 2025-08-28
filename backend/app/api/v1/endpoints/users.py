"""User management endpoints."""

from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.db.database import get_db
from app.models.user import User
from app.core.security import get_current_active_user, get_current_superuser
from app.schemas.auth import UserResponse, UserUpdate
from app.schemas.base import MessageResponse

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get current user information."""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Update current user information."""
    update_data = user_update.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user


@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Get all users (admin only)."""
    result = await db.execute(
        select(User).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Get specific user by ID (admin only)."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.post("/{user_id}/activate", response_model=MessageResponse)
async def activate_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Activate a user account (admin only)."""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = True
    user.is_verified = True
    await db.commit()
    
    return {"message": "User activated successfully"}


@router.post("/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Deactivate a user account (admin only)."""
    if str(current_user.id) == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = False
    await db.commit()
    
    return {"message": "User deactivated successfully"}


@router.get("/me/preferences", response_model=dict)
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get user preferences."""
    return current_user.preferences or {}


@router.put("/me/preferences", response_model=dict)
async def update_user_preferences(
    preferences: dict,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Update user preferences."""
    current_user.preferences = preferences
    await db.commit()
    
    return preferences


@router.get("/me/api-key", response_model=dict)
async def get_api_key(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get user's API key."""
    return {
        "api_key": current_user.api_key,
        "created_at": current_user.api_key_created
    }


@router.post("/me/api-key/regenerate", response_model=dict)
async def regenerate_api_key(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Regenerate user's API key."""
    from app.core.security import generate_api_key
    from datetime import datetime
    
    current_user.api_key = generate_api_key()
    current_user.api_key_created = datetime.utcnow()
    
    await db.commit()
    
    return {
        "api_key": current_user.api_key,
        "created_at": current_user.api_key_created
    }
