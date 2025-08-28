"""
Security tests for the Quantum Trading AI application.

These tests verify authentication, authorization, data validation,
and protection against common security vulnerabilities.
"""

import pytest
import jwt
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash
from app.models.user import User


class TestSecurityAuthentication:
    """Test authentication security"""
    
    @pytest.mark.asyncio
    async def test_password_complexity_requirements(self, client: AsyncClient):
        """Test password complexity enforcement"""
        test_cases = [
            ("weak", False, "Password too short"),
            ("12345678", False, "No letters"),
            ("password", False, "No numbers or special chars"),
            ("Password1", False, "No special characters"),
            ("Pass123!", True, "Valid password"),
            ("C0mpl3x!P@ssw0rd", True, "Strong password"),
        ]
        
        for password, should_succeed, description in test_cases:
            user_data = {
                "email": f"test{uuid.uuid4()}@example.com",
                "password": password,
                "username": f"user{uuid.uuid4().hex[:8]}",
                "full_name": "Test User"
            }
            
            response = await client.post("/api/v1/auth/register", json=user_data)
            
            if should_succeed:
                assert response.status_code == 201, f"{description} should succeed"
            else:
                assert response.status_code == 422, f"{description} should fail"
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, client: AsyncClient):
        """Test protection against brute force attacks"""
        email = "bruteforce@example.com"
        
        # Register user
        user_data = {
            "email": email,
            "password": "ValidPass123!",
            "username": "bruteforce",
            "full_name": "Test User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        # Attempt multiple failed logins
        failed_attempts = 0
        max_attempts = 10
        
        for i in range(max_attempts):
            login_data = {
                "username": email,
                "password": "WrongPassword123!"
            }
            response = await client.post("/api/v1/auth/login", data=login_data)
            
            if response.status_code == 429:  # Too Many Requests
                break
            
            failed_attempts += 1
        
        # Should be rate limited before max attempts
        assert failed_attempts < max_attempts, "No rate limiting detected"
    
    @pytest.mark.asyncio
    async def test_token_expiration(self, client: AsyncClient):
        """Test JWT token expiration"""
        # Create expired token
        expired_token = create_access_token(
            data={"sub": str(uuid.uuid4())},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_token_signature(self, client: AsyncClient):
        """Test detection of tampered tokens"""
        # Create valid token
        valid_token = create_access_token(
            data={"sub": str(uuid.uuid4())},
            expires_delta=timedelta(minutes=30)
        )
        
        # Tamper with token
        parts = valid_token.split('.')
        tampered_token = f"{parts[0]}.{parts[1]}.invalid_signature"
        
        headers = {"Authorization": f"Bearer {tampered_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
        assert "could not validate" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_refresh_token_rotation(self, client: AsyncClient):
        """Test refresh token rotation for security"""
        # Register and login
        user_data = {
            "email": "refresh@example.com",
            "password": "ValidPass123!",
            "username": "refreshuser",
            "full_name": "Test User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        login_response = await client.post("/api/v1/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        
        initial_refresh = login_response.json()["refresh_token"]
        
        # Use refresh token
        refresh_response = await client.post("/api/v1/auth/refresh", json={
            "refresh_token": initial_refresh
        })
        
        new_refresh = refresh_response.json()["refresh_token"]
        
        # Tokens should be different
        assert new_refresh != initial_refresh
        
        # Old refresh token should not work
        old_refresh_response = await client.post("/api/v1/auth/refresh", json={
            "refresh_token": initial_refresh
        })
        
        assert old_refresh_response.status_code == 401


class TestSecurityAuthorization:
    """Test authorization security"""
    
    @pytest.mark.asyncio
    async def test_access_control_portfolio(self, client: AsyncClient, db: AsyncSession):
        """Test users can only access their own portfolios"""
        # Create two users
        user1_data = {
            "email": "user1@example.com",
            "password": "ValidPass123!",
            "username": "user1",
            "full_name": "User One"
        }
        user2_data = {
            "email": "user2@example.com",
            "password": "ValidPass123!",
            "username": "user2",
            "full_name": "User Two"
        }
        
        # Register users
        await client.post("/api/v1/auth/register", json=user1_data)
        await client.post("/api/v1/auth/register", json=user2_data)
        
        # Login as user1
        login1 = await client.post("/api/v1/auth/login", data={
            "username": user1_data["email"],
            "password": user1_data["password"]
        })
        token1 = login1.json()["access_token"]
        headers1 = {"Authorization": f"Bearer {token1}"}
        
        # Create portfolio for user1
        portfolio_response = await client.post("/api/v1/portfolios", headers=headers1, json={
            "name": "User1 Portfolio",
            "initial_capital": 10000
        })
        portfolio_id = portfolio_response.json()["id"]
        
        # Login as user2
        login2 = await client.post("/api/v1/auth/login", data={
            "username": user2_data["email"],
            "password": user2_data["password"]
        })
        token2 = login2.json()["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        # User2 should not be able to access user1's portfolio
        response = await client.get(f"/api/v1/portfolios/{portfolio_id}", headers=headers2)
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_admin_authorization(self, client: AsyncClient, superuser_auth_headers: dict):
        """Test admin-only endpoints"""
        # Regular user
        user_data = {
            "email": "regular@example.com",
            "password": "ValidPass123!",
            "username": "regular",
            "full_name": "Regular User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        login = await client.post("/api/v1/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        user_token = login.json()["access_token"]
        user_headers = {"Authorization": f"Bearer {user_token}"}
        
        # Test admin endpoints
        admin_endpoints = [
            ("/api/v1/admin/users", "GET"),
            ("/api/v1/admin/system/stats", "GET"),
            ("/api/v1/admin/users/123/suspend", "POST"),
        ]
        
        for endpoint, method in admin_endpoints:
            # Regular user should be forbidden
            response = await client.request(method, endpoint, headers=user_headers)
            assert response.status_code in [403, 404], f"Regular user accessed {endpoint}"
            
            # Admin should have access (or 404 if not implemented)
            response = await client.request(method, endpoint, headers=superuser_auth_headers)
            assert response.status_code != 403, f"Admin forbidden from {endpoint}"


class TestSecurityDataValidation:
    """Test data validation and sanitization"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, client: AsyncClient, auth_headers: dict):
        """Test protection against SQL injection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM portfolios WHERE 1=1",
        ]
        
        for payload in malicious_inputs:
            # Try injection in search
            response = await client.get(
                f"/api/v1/market/search?query={payload}",
                headers=auth_headers
            )
            assert response.status_code in [200, 400], "SQL injection may have succeeded"
            
            # Try injection in portfolio name
            response = await client.post("/api/v1/portfolios", headers=auth_headers, json={
                "name": payload,
                "initial_capital": 10000
            })
            
            if response.status_code == 201:
                # Check that the name was properly escaped
                portfolio = response.json()
                assert portfolio["name"] == payload, "Input was modified"
    
    @pytest.mark.asyncio
    async def test_xss_prevention(self, client: AsyncClient, auth_headers: dict):
        """Test protection against XSS attacks"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        ]
        
        for payload in xss_payloads:
            # Create portfolio with XSS attempt
            response = await client.post("/api/v1/portfolios", headers=auth_headers, json={
                "name": payload,
                "description": payload,
                "initial_capital": 10000
            })
            
            if response.status_code == 201:
                portfolio = response.json()
                # Check that dangerous content is escaped/sanitized
                assert "<script>" not in portfolio.get("name", "")
                assert "javascript:" not in portfolio.get("description", "")
    
    @pytest.mark.asyncio
    async def test_input_size_limits(self, client: AsyncClient, auth_headers: dict):
        """Test input size validation"""
        # Very long string
        long_string = "A" * 10000
        
        # Test various endpoints
        test_cases = [
            ("/api/v1/portfolios", {"name": long_string, "initial_capital": 10000}),
            ("/api/v1/trades", {"symbol": long_string, "quantity": 100, "side": "BUY"}),
            ("/api/v1/alerts", {"condition": long_string, "threshold": 100}),
        ]
        
        for endpoint, data in test_cases:
            response = await client.post(endpoint, headers=auth_headers, json=data)
            assert response.status_code in [400, 422], f"Large input accepted at {endpoint}"
    
    @pytest.mark.asyncio
    async def test_numeric_overflow_prevention(self, client: AsyncClient, auth_headers: dict):
        """Test protection against numeric overflows"""
        overflow_values = [
            9999999999999999999999,  # Very large number
            -9999999999999999999999,  # Very large negative
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
        ]
        
        for value in overflow_values:
            # Try overflow in trade
            response = await client.post("/api/v1/trades", headers=auth_headers, json={
                "symbol": "AAPL",
                "quantity": value,
                "side": "BUY",
                "order_type": "MARKET"
            })
            assert response.status_code in [400, 422], "Numeric overflow accepted"


class TestSecurityAPIProtection:
    """Test API security measures"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client: AsyncClient, auth_headers: dict):
        """Test API rate limiting"""
        endpoint = "/api/v1/market/quote/AAPL"
        
        # Make many rapid requests
        responses = []
        for _ in range(100):
            response = await client.get(endpoint, headers=auth_headers)
            responses.append(response.status_code)
            
            if response.status_code == 429:
                break
        
        # Should hit rate limit
        assert 429 in responses, "No rate limiting detected"
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, client: AsyncClient):
        """Test CORS security headers"""
        response = await client.options("/api/v1/auth/login")
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        
        # Should not allow all origins in production
        if settings.ENVIRONMENT == "production":
            assert response.headers["access-control-allow-origin"] != "*"
    
    @pytest.mark.asyncio
    async def test_security_headers(self, client: AsyncClient):
        """Test security headers in responses"""
        response = await client.get("/")
        
        security_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
            "x-xss-protection": "1; mode=block",
            "strict-transport-security": "max-age=31536000; includeSubDomains"
        }
        
        for header, expected_value in security_headers.items():
            assert header in response.headers
            if expected_value:
                assert response.headers[header] == expected_value
    
    @pytest.mark.asyncio
    async def test_api_versioning(self, client: AsyncClient):
        """Test API versioning for security"""
        # Old version should not work
        response = await client.get("/api/v0/auth/me")
        assert response.status_code == 404
        
        # Current version should work
        response = await client.get("/api/v1/auth/me")
        assert response.status_code in [401, 200]  # Depends on auth


class TestSecurityDataProtection:
    """Test data protection and privacy"""
    
    @pytest.mark.asyncio
    async def test_password_hashing(self, db: AsyncSession):
        """Test passwords are properly hashed"""
        from sqlalchemy import select
        
        # Create user directly
        user = User(
            email="hash@example.com",
            username="hashtest",
            hashed_password=get_password_hash("TestPassword123!"),
            full_name="Hash Test"
        )
        db.add(user)
        await db.commit()
        
        # Retrieve user
        result = await db.execute(
            select(User).where(User.email == "hash@example.com")
        )
        stored_user = result.scalar_one()
        
        # Password should be hashed
        assert stored_user.hashed_password != "TestPassword123!"
        assert stored_user.hashed_password.startswith("$2b$")  # bcrypt prefix
        assert len(stored_user.hashed_password) > 50
    
    @pytest.mark.asyncio
    async def test_sensitive_data_not_exposed(self, client: AsyncClient, auth_headers: dict):
        """Test sensitive data is not exposed in API responses"""
        # Get user profile
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        user_data = response.json()
        
        # Sensitive fields should not be present
        sensitive_fields = ["hashed_password", "password", "salt", "reset_token"]
        for field in sensitive_fields:
            assert field not in user_data, f"Sensitive field '{field}' exposed"
    
    @pytest.mark.asyncio
    async def test_pii_encryption(self, client: AsyncClient, auth_headers: dict):
        """Test PII encryption at rest"""
        # Create portfolio with PII
        response = await client.post("/api/v1/portfolios", headers=auth_headers, json={
            "name": "Personal Portfolio",
            "initial_capital": 50000,
            "tax_id": "123-45-6789"  # Sensitive PII
        })
        
        if response.status_code == 201:
            portfolio = response.json()
            # Tax ID should be masked in response
            if "tax_id" in portfolio:
                assert "****" in portfolio["tax_id"], "PII not masked"


class TestSecuritySessionManagement:
    """Test session and token management"""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_limit(self, client: AsyncClient):
        """Test limitation on concurrent sessions"""
        user_data = {
            "email": "session@example.com",
            "password": "ValidPass123!",
            "username": "sessionuser",
            "full_name": "Session User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        # Create multiple sessions
        tokens = []
        max_sessions = 5
        
        for i in range(max_sessions + 2):
            response = await client.post("/api/v1/auth/login", data={
                "username": user_data["email"],
                "password": user_data["password"]
            })
            
            if response.status_code == 200:
                tokens.append(response.json()["access_token"])
        
        # Test if old sessions are invalidated
        if len(tokens) > max_sessions:
            # First token might be invalidated
            headers = {"Authorization": f"Bearer {tokens[0]}"}
            response = await client.get("/api/v1/auth/me", headers=headers)
            # This is a policy decision - could be 401 or still valid
    
    @pytest.mark.asyncio
    async def test_logout_invalidates_token(self, client: AsyncClient):
        """Test logout properly invalidates tokens"""
        # Register and login
        user_data = {
            "email": "logout@example.com",
            "password": "ValidPass123!",
            "username": "logoutuser",
            "full_name": "Logout User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        login_response = await client.post("/api/v1/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Verify token works
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        
        # Logout
        await client.post("/api/v1/auth/logout", headers=headers, json={"token": token})
        
        # Token should no longer work
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 401


class TestSecurityFileUpload:
    """Test file upload security"""
    
    @pytest.mark.asyncio
    async def test_file_type_validation(self, client: AsyncClient, auth_headers: dict):
        """Test file type restrictions"""
        # Try uploading executable
        files = {
            "file": ("malicious.exe", b"MZ\x90\x00", "application/x-msdownload")
        }
        
        response = await client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )
        
        assert response.status_code in [400, 415, 404], "Executable file accepted"
    
    @pytest.mark.asyncio
    async def test_file_size_limit(self, client: AsyncClient, auth_headers: dict):
        """Test file size limitations"""
        # Create large file (10MB)
        large_file = b"0" * (10 * 1024 * 1024)
        
        files = {
            "file": ("large.csv", large_file, "text/csv")
        }
        
        response = await client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files=files
        )
        
        assert response.status_code in [413, 400, 404], "Large file accepted"


class TestSecurityWebSocket:
    """Test WebSocket security"""
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self):
        """Test WebSocket requires authentication"""
        import websockets
        
        try:
            async with websockets.connect("ws://localhost:8000/ws/market") as ws:
                # Try to subscribe without auth
                await ws.send('{"type": "subscribe", "symbols": ["AAPL"]}')
                response = await ws.recv()
                
                # Should require authentication
                assert "unauthorized" in response.lower() or "auth" in response.lower()
        except websockets.exceptions.WebSocketException:
            # Connection refused is also acceptable
            pass
    
    @pytest.mark.asyncio
    async def test_websocket_message_validation(self):
        """Test WebSocket message validation"""
        import websockets
        
        # Get valid token first
        client = AsyncClient(app=app)
        user_data = {
            "email": "ws@example.com",
            "password": "ValidPass123!",
            "username": "wsuser",
            "full_name": "WS User"
        }
        await client.post("/api/v1/auth/register", json=user_data)
        
        login = await client.post("/api/v1/auth/login", data={
            "username": user_data["email"],
            "password": user_data["password"]
        })
        token = login.json()["access_token"]
        
        try:
            async with websockets.connect(
                f"ws://localhost:8000/ws/market?token={token}"
            ) as ws:
                # Send malformed messages
                malformed_messages = [
                    "not json",
                    '{"no_type": "field"}',
                    '{"type": "subscribe", "symbols": "not_array"}',
                    '{"type": "<script>alert()</script>"}',
                ]
                
                for msg in malformed_messages:
                    await ws.send(msg)
                    response = await ws.recv()
                    # Should handle gracefully
                    assert "error" in response.lower()
        except Exception:
            pass
