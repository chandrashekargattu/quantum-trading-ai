"""User schemas for request/response validation."""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User response schema."""
    id: UUID
    is_verified: bool = False
    account_type: str = "paper"
    account_balance: float = 100000.0
    buying_power: float = 100000.0
    subscription_tier: str = "free"
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserInDB(UserResponse):
    """User in database schema with hashed password."""
    hashed_password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class TokenData(BaseModel):
    """Token data schema."""
    sub: Optional[str] = None
    exp: Optional[datetime] = None
