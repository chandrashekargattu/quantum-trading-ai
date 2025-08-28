"""Tests for authentication endpoints."""
import pytest
from datetime import datetime, timedelta
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.core.security import get_password_hash, verify_password, create_access_token
from app.schemas.auth import Token, UserCreate, UserResponse


class TestAuthEndpoints:
    """Test cases for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_register_new_user(self, client: AsyncClient):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "SecurePass123!",
            "full_name": "John Doe"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert data["full_name"] == user_data["full_name"]
        assert "id" in data
        assert "password" not in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user: User):
        """Test registration with existing email."""
        user_data = {
            "email": test_user.email,
            "username": "anotheruser",
            "password": "AnotherPass123!",
            "full_name": "Jane Smith"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email format."""
        user_data = {
            "email": "invalid-email",
            "username": "testuser",
            "password": "SecurePass123!",
            "full_name": "John Doe"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password."""
        user_data = {
            "email": "user@example.com",
            "username": "weakpass",
            "password": "weak",
            "full_name": "John Doe"
        }
        
        response = await client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_login_valid_credentials(self, client: AsyncClient, test_user: User):
        """Test login with valid credentials."""
        login_data = {
            "username": test_user.email,
            "password": "testpassword123"  # This should match the test user fixture
        }
        
        response = await client.post(
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_email(self, client: AsyncClient):
        """Test login with non-existent email."""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "somepassword"
        }
        
        response = await client.post(
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "incorrect" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, client: AsyncClient, test_user: User):
        """Test login with wrong password."""
        login_data = {
            "username": test_user.email,
            "password": "wrongpassword"
        }
        
        response = await client.post(
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "incorrect" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, client: AsyncClient, db: AsyncSession):
        """Test login with inactive user account."""
        # Create an inactive user
        user = User(
            email="inactive@example.com",
            username="inactive",
            hashed_password=get_password_hash("password123"),
            full_name="Inactive User",
            is_active=False
        )
        db.add(user)
        await db.commit()
        
        login_data = {
            "username": "inactive@example.com",
            "password": "password123"
        }
        
        response = await client.post(
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "inactive" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_current_user(self, client: AsyncClient, test_user: User, auth_headers: dict):
        """Test getting current user info."""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == test_user.email
        assert data["id"] == str(test_user.id)
        assert "password" not in data

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth(self, client: AsyncClient):
        """Test getting current user without authentication."""
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, client: AsyncClient):
        """Test getting current user with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = await client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, client: AsyncClient, test_user: User):
        """Test getting current user with expired token."""
        # Create an expired token
        expired_token = create_access_token(
            subject=str(test_user.id),
            expires_delta=timedelta(minutes=-1)  # Expired 1 minute ago
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_refresh_token(self, client: AsyncClient, test_user: User, auth_headers: dict):
        """Test refreshing access token."""
        # First login to get a refresh token
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": test_user.email, "password": "testpassword123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Now test refresh
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        # Verify the new token is valid by decoding it
        from jose import jwt
        from app.core.config import settings
        
        payload = jwt.decode(data["access_token"], settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["type"] == "access"
        assert "sub" in payload

    @pytest.mark.asyncio
    async def test_logout(self, client: AsyncClient, auth_headers: dict):
        """Test user logout."""
        # Extract token from auth headers
        token = auth_headers["Authorization"].split()[1]
        response = await client.post(
            "/api/v1/auth/logout",
            json={"token": token},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Successfully logged out"
        
        # Verify the token is invalidated (this would require token blacklisting implementation)
        # For now, we just check the logout endpoint responds correctly

    @pytest.mark.asyncio
    async def test_change_password(self, client: AsyncClient, test_user: User, auth_headers: dict):
        """Test changing user password."""
        password_data = {
            "current_password": "testpassword123",
            "new_password": "NewSecurePass456!"
        }
        
        response = await client.post(
            "/api/v1/auth/change-password",
            json=password_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Password changed successfully"
        
        # Verify new password works
        login_data = {
            "username": test_user.email,
            "password": "NewSecurePass456!"
        }
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert login_response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, client: AsyncClient, auth_headers: dict):
        """Test changing password with wrong current password."""
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "NewSecurePass456!"
        }
        
        response = await client.post(
            "/api/v1/auth/change-password",
            json=password_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "incorrect" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_password_reset_request(self, client: AsyncClient, test_user: User):
        """Test requesting password reset."""
        reset_data = {"email": test_user.email}
        
        response = await client.post("/api/v1/auth/forgot-password", json=reset_data)
        
        assert response.status_code == status.HTTP_200_OK
        assert "password reset link has been sent" in response.json()["message"].lower()

    @pytest.mark.asyncio
    async def test_password_reset_nonexistent_email(self, client: AsyncClient):
        """Test password reset for non-existent email."""
        reset_data = {"email": "nonexistent@example.com"}
        
        response = await client.post("/api/v1/auth/forgot-password", json=reset_data)
        
        # Should return success to prevent email enumeration
        assert response.status_code == status.HTTP_200_OK
        assert "password reset link has been sent" in response.json()["message"].lower()

    @pytest.mark.asyncio
    async def test_password_hash_security(self):
        """Test password hashing security."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)
        
        # Hash should not contain the original password
        assert password not in hashed
        # Hash should be different each time (due to salt)
        assert hashed != get_password_hash(password)
        # But should still verify correctly
        assert verify_password(password, hashed)
        assert not verify_password("WrongPassword", hashed)