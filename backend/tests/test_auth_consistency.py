"""Test authentication consistency across the platform."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from jose import jwt

from app.core.config import settings
from app.core.security import create_access_token
from app.models.user import User
from app.models.portfolio import Portfolio


class TestAuthenticationConsistency:
    """Test suite for authentication consistency."""

    @pytest.fixture
    async def test_user(self, db: AsyncSession):
        """Create a test user."""
        user = User(
            email="auth_test@example.com",
            username="authtest",
            hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
            full_name="Auth Test User",
            is_active=True,
            is_verified=True
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    @pytest.fixture
    def auth_headers(self, test_user: User):
        """Generate auth headers for test user."""
        token = create_access_token(data={"sub": test_user.email})
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def expired_token_headers(self, test_user: User):
        """Generate expired auth headers."""
        token = create_access_token(
            data={"sub": test_user.email},
            expires_delta=timedelta(minutes=-10)  # Already expired
        )
        return {"Authorization": f"Bearer {token}"}

    async def test_auth_required_endpoints_without_token(self, client: AsyncClient):
        """Test that protected endpoints return 401 without token."""
        endpoints = [
            "/api/v1/portfolios/",
            "/api/v1/market-data/indicators",
            "/api/v1/trades/",
            "/api/v1/users/me",
            "/api/v1/alerts/",
            "/api/v1/strategies/",
        ]

        for endpoint in endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
            assert "detail" in response.json()

    async def test_auth_required_endpoints_with_invalid_token(self, client: AsyncClient):
        """Test that protected endpoints return 401 with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        
        response = await client.get("/api/v1/portfolios/", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    async def test_auth_required_endpoints_with_expired_token(
        self, client: AsyncClient, expired_token_headers: dict
    ):
        """Test that protected endpoints return 401 with expired token."""
        response = await client.get("/api/v1/portfolios/", headers=expired_token_headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    async def test_valid_auth_flow(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test complete authentication flow."""
        # 1. Login
        login_data = {
            "username": test_user.email,
            "password": "secret"
        }
        response = await client.post(
            "/api/v1/auth/login",
            data=login_data
        )
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert token_data["token_type"] == "bearer"

        # 2. Use token to access protected endpoint
        headers = {"Authorization": f"{token_data['token_type']} {token_data['access_token']}"}
        response = await client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200
        user_data = response.json()
        assert user_data["email"] == test_user.email

        # 3. Access multiple protected endpoints
        response = await client.get("/api/v1/portfolios/", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

        response = await client.get("/api/v1/market-data/indicators", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_auth_consistency_across_methods(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test that auth works consistently across different HTTP methods."""
        # GET
        response = await client.get("/api/v1/portfolios/", headers=auth_headers)
        assert response.status_code == 200

        # POST
        portfolio_data = {
            "name": "Test Portfolio",
            "initial_cash": 100000
        }
        response = await client.post(
            "/api/v1/portfolios/",
            json=portfolio_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        portfolio = response.json()
        portfolio_id = portfolio["id"]

        # PUT
        update_data = {"name": "Updated Portfolio"}
        response = await client.put(
            f"/api/v1/portfolios/{portfolio_id}",
            json=update_data,
            headers=auth_headers
        )
        assert response.status_code == 200

        # DELETE
        response = await client.delete(
            f"/api/v1/portfolios/{portfolio_id}",
            headers=auth_headers
        )
        assert response.status_code == 200

    async def test_token_type_variations(self, client: AsyncClient, test_user: User):
        """Test different token type formats."""
        token = create_access_token(data={"sub": test_user.email})
        
        # Test with "Bearer" (standard)
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200

        # Test with "bearer" (lowercase)
        headers = {"Authorization": f"bearer {token}"}
        response = await client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401  # Should fail with incorrect case

        # Test without token type
        headers = {"Authorization": token}
        response = await client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401

    async def test_auth_persistence_across_requests(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test that auth state persists correctly across multiple requests."""
        # Make multiple requests in sequence
        for i in range(5):
            response = await client.get("/api/v1/portfolios/", headers=auth_headers)
            assert response.status_code == 200, f"Request {i} failed"

        # Make concurrent-like requests
        responses = []
        for i in range(5):
            response = await client.get("/api/v1/market-data/indicators", headers=auth_headers)
            responses.append(response)

        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Concurrent request {i} failed"

    async def test_auth_error_messages(self, client: AsyncClient):
        """Test that auth errors return consistent messages."""
        # No token
        response = await client.get("/api/v1/portfolios/")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

        # Invalid token format
        headers = {"Authorization": "InvalidFormat"}
        response = await client.get("/api/v1/portfolios/", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

        # Invalid bearer token
        headers = {"Authorization": "Bearer invalid-jwt-token"}
        response = await client.get("/api/v1/portfolios/", headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    async def test_auth_with_inactive_user(
        self, client: AsyncClient, db: AsyncSession
    ):
        """Test that inactive users cannot authenticate."""
        # Create inactive user
        user = User(
            email="inactive@example.com",
            username="inactive",
            hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            is_active=False
        )
        db.add(user)
        await db.commit()

        # Try to login
        login_data = {
            "username": "inactive@example.com",
            "password": "secret"
        }
        response = await client.post("/api/v1/auth/login", data=login_data)
        assert response.status_code == 400
        assert response.json()["detail"] == "Inactive user"

    async def test_auth_token_claims(self, test_user: User):
        """Test that JWT tokens contain correct claims."""
        token = create_access_token(data={"sub": test_user.email})
        
        # Decode token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert payload["sub"] == test_user.email
        assert "exp" in payload  # Expiration time
        assert "iat" in payload  # Issued at time

    async def test_public_endpoints_work_without_auth(self, client: AsyncClient):
        """Test that public endpoints work without authentication."""
        public_endpoints = [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/docs",
            "/openapi.json",
            "/health"
        ]

        for endpoint in public_endpoints:
            if endpoint == "/api/v1/auth/login":
                # Login requires POST
                response = await client.post(endpoint, data={})
                assert response.status_code in [422, 400]  # Bad request, not 401
            elif endpoint == "/api/v1/auth/register":
                # Register requires POST
                response = await client.post(endpoint, json={})
                assert response.status_code == 422  # Validation error, not 401
            else:
                response = await client.get(endpoint)
                assert response.status_code != 401, f"Public endpoint {endpoint} should not require auth"
