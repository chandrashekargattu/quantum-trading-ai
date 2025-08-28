"""Base schemas for common response types."""

from datetime import datetime
from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel, Field
from uuid import UUID

# Generic type for paginated responses
T = TypeVar('T')


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int
    
    @staticmethod
    def create(items: List[T], total: int, page: int, page_size: int) -> "PaginatedResponse[T]":
        return PaginatedResponse[T](
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=(total + page_size - 1) // page_size
        )


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps."""
    created_at: datetime
    updated_at: datetime


class UUIDMixin(BaseModel):
    """Mixin for models with UUID."""
    id: UUID
    
    class Config:
        from_attributes = True
