"""Authentication endpoints."""

from typing import Any, Optional
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.user import User
from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.config import settings
from app.schemas.user import UserCreate, UserResponse
from app.schemas.auth import Token, TokenPayload
from app.core.security import get_current_user

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """OAuth2 compatible token login, get an access token for future requests."""
    
    # Security check: Reject if credentials are in URL
    if request.url.query:
        query_params = str(request.url.query).lower()
        if 'password' in query_params or 'username' in query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security Error: Credentials must not be sent in URL parameters. Use POST body only."
            )
    
    # Get user by username/email
    result = await db.execute(
        select(User).where(
            (User.username == form_data.username) | (User.email == form_data.username)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Register a new user."""
    
    # Security check: Reject if credentials are in URL
    if request.url.query:
        query_params = str(request.url.query).lower()
        if 'password' in query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security Error: Passwords must not be sent in URL parameters. Use POST body only."
            )
    
    # Check if user already exists
    result = await db.execute(
        select(User).where(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user
    user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        account_type="paper",  # Default to paper trading
        account_balance=100000.0,  # Default paper trading balance
        buying_power=100000.0
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


@router.post("/logout")
async def logout(response: Response) -> Any:
    """Logout user (client should remove token)."""
    # In a stateless JWT system, logout is handled client-side
    # This endpoint is provided for completeness
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_user)
) -> Any:
    """Get current user."""
    return current_user


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
) -> Any:
    """Refresh access token."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(current_user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Change user password."""
    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    current_user.hashed_password = get_password_hash(new_password)
    await db.commit()
    
    return {"message": "Password updated successfully"}


@router.post("/reset-password-request")
async def reset_password_request(
    email: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Request password reset."""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if user:
        # In a real application, send password reset email here
        # For now, just return success
        pass
    
    # Always return success to prevent email enumeration
    return {
        "message": "If the email exists, a password reset link has been sent"
    }


@router.post("/reset-password")
async def reset_password(
    token: str,
    new_password: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """Reset password with token."""
    # In a real application, verify the reset token
    # For now, this is a placeholder
    
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # This would verify token and update password
    # Placeholder implementation
    
    return {"message": "Password reset successfully"}


@router.delete("/delete-account")
async def delete_account(
    password: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Delete user account."""
    if not verify_password(password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    # Soft delete - just mark as inactive
    current_user.is_active = False
    await db.commit()
    
    return {"message": "Account deleted successfully"}