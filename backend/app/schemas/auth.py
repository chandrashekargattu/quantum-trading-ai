"""Authentication schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from uuid import UUID


class Token(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class TokenData(BaseModel):
    """Token payload data."""
    sub: Optional[str] = None
    exp: Optional[datetime] = None


class TokenPayload(BaseModel):
    """Token payload for JWT."""
    sub: str  # User ID
    exp: Optional[datetime] = None


class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can contain _ and -)')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserResponse(BaseModel):
    """User response schema."""
    id: UUID
    email: EmailStr
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    account_type: str
    subscription_tier: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    notification_settings: Optional[dict] = None
    preferences: Optional[dict] = None
