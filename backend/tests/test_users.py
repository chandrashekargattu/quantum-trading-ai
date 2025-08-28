"""Tests for user management endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User


class TestUsers:
    """Test user management endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, client: TestClient, test_user: User, auth_headers: dict):
        """Test getting current user info."""
        response = client.get(
            f"{settings.API_V1_STR}/users/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
        assert "hashed_password" not in data
    
    @pytest.mark.asyncio
    async def test_get_current_user_unauthorized(self, client: TestClient):
        """Test getting current user without auth."""
        response = client.get(f"{settings.API_V1_STR}/users/me")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_update_current_user(self, client: TestClient, auth_headers: dict):
        """Test updating current user."""
        response = client.put(
            f"{settings.API_V1_STR}/users/me",
            json={
                "full_name": "Updated Name",
                "email": "updated@example.com"
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"
        assert data["email"] == "updated@example.com"
    
    @pytest.mark.asyncio
    async def test_get_user_preferences(self, client: TestClient, auth_headers: dict):
        """Test getting user preferences."""
        response = client.get(
            f"{settings.API_V1_STR}/users/me/preferences",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
    
    @pytest.mark.asyncio
    async def test_update_user_preferences(self, client: TestClient, auth_headers: dict):
        """Test updating user preferences."""
        preferences = {
            "theme": "dark",
            "notifications": {
                "email": True,
                "sms": False
            },
            "default_chart": "candlestick"
        }
        
        response = client.put(
            f"{settings.API_V1_STR}/users/me/preferences",
            json=preferences,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["theme"] == "dark"
        assert data["notifications"]["email"] is True
    
    @pytest.mark.asyncio
    async def test_get_api_key(self, client: TestClient, auth_headers: dict):
        """Test getting API key."""
        response = client.get(
            f"{settings.API_V1_STR}/users/me/api-key",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert "created_at" in data
    
    @pytest.mark.asyncio
    async def test_regenerate_api_key(self, client: TestClient, auth_headers: dict):
        """Test regenerating API key."""
        # Get initial API key
        initial_response = client.get(
            f"{settings.API_V1_STR}/users/me/api-key",
            headers=auth_headers
        )
        initial_key = initial_response.json().get("api_key")
        
        # Regenerate
        response = client.post(
            f"{settings.API_V1_STR}/users/me/api-key/regenerate",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["api_key"] != initial_key  # Should be different
        assert len(data["api_key"]) > 20  # Should be a reasonable length
