import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.portfolio import Portfolio
from app.models.user import User
from app.core.security import get_password_hash
import uuid
from datetime import datetime


@pytest.mark.asyncio
class TestPortfolioEndpoints:
    """Test portfolio API endpoints with trailing slash handling"""

    async def test_get_portfolios_with_trailing_slash(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/portfolios/ returns portfolios"""
        response = await client.get(
            "/api/v1/portfolios/",  # With trailing slash
            headers=test_user_token_headers
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_portfolios_redirect(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET /api/v1/portfolios redirects to /api/v1/portfolios/"""
        response = await client.get(
            "/api/v1/portfolios",  # Without trailing slash
            headers=test_user_token_headers,
            follow_redirects=False
        )
        assert response.status_code == 307  # Temporary Redirect
        assert response.headers["location"] == "/api/v1/portfolios/"

    async def test_create_portfolio_with_trailing_slash(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test POST /api/v1/portfolios/ creates a new portfolio"""
        portfolio_data = {
            "name": "Test Trading Portfolio",
            "initial_capital": 100000.0,
            "description": "Test portfolio for trading"
        }
        
        response = await client.post(
            "/api/v1/portfolios/",
            json=portfolio_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == portfolio_data["name"]
        assert data["total_value"] == portfolio_data["initial_capital"]
        assert data["cash_balance"] == portfolio_data["initial_capital"]
        assert data["buying_power"] == portfolio_data["initial_capital"]
        assert data["is_active"] is True
        assert "id" in data

    async def test_create_portfolio_validation(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test portfolio creation validation"""
        # Test with negative initial capital
        invalid_data = {
            "name": "Invalid Portfolio",
            "initial_capital": -1000.0
        }
        
        response = await client.post(
            "/api/v1/portfolios/",
            json=invalid_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 400
        assert "Initial capital must be positive" in response.json()["detail"]

        # Test with empty name
        invalid_data = {
            "name": "",
            "initial_capital": 100000.0
        }
        
        response = await client.post(
            "/api/v1/portfolios/",
            json=invalid_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 422  # Validation error

    async def test_get_portfolio_by_id(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test GET /api/v1/portfolios/{id}"""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_portfolio.id)
        assert data["name"] == test_portfolio.name

    async def test_get_portfolio_not_found(
        self, client: AsyncClient, test_user_token_headers: dict
    ):
        """Test GET portfolio with invalid ID"""
        fake_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/v1/portfolios/{fake_id}",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 404
        assert "Portfolio not found" in response.json()["detail"]

    async def test_update_portfolio(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test PATCH /api/v1/portfolios/{id}"""
        update_data = {
            "name": "Updated Portfolio Name",
            "description": "Updated description"
        }
        
        response = await client.patch(
            f"/api/v1/portfolios/{test_portfolio.id}",
            json=update_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]

    async def test_delete_portfolio(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test DELETE /api/v1/portfolios/{id}"""
        response = await client.delete(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        
        # Verify portfolio is soft deleted (is_active = False)
        get_response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=test_user_token_headers
        )
        assert get_response.status_code == 404

    async def test_get_portfolio_positions(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test GET /api/v1/portfolios/{id}/positions"""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/positions",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_add_funds_to_portfolio(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test POST /api/v1/portfolios/{id}/deposit"""
        deposit_data = {
            "amount": 50000.0
        }
        
        initial_balance = test_portfolio.cash_balance
        
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/deposit",
            json=deposit_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "DEPOSIT"
        assert data["amount"] == deposit_data["amount"]
        
        # Verify portfolio balance updated
        portfolio_response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=test_user_token_headers
        )
        portfolio_data = portfolio_response.json()
        assert portfolio_data["cash_balance"] == initial_balance + deposit_data["amount"]

    async def test_withdraw_funds_from_portfolio(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test POST /api/v1/portfolios/{id}/withdraw"""
        # First add funds
        await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/deposit",
            json={"amount": 100000.0},
            headers=test_user_token_headers
        )
        
        withdraw_data = {
            "amount": 30000.0
        }
        
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/withdraw",
            json=withdraw_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "WITHDRAWAL"
        assert data["amount"] == -withdraw_data["amount"]  # Negative for withdrawal

    async def test_withdraw_insufficient_funds(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test withdrawal with insufficient funds"""
        withdraw_data = {
            "amount": test_portfolio.cash_balance + 10000  # More than available
        }
        
        response = await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/withdraw",
            json=withdraw_data,
            headers=test_user_token_headers
        )
        
        assert response.status_code == 400
        assert "Insufficient funds" in response.json()["detail"]

    async def test_portfolio_authorization(
        self, client: AsyncClient, other_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test that users can only access their own portfolios"""
        # Try to access another user's portfolio
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}",
            headers=other_user_token_headers
        )
        
        assert response.status_code == 404  # Should not be found for other user

    async def test_portfolio_performance(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test GET /api/v1/portfolios/{id}/performance"""
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/performance?period=1d",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_return" in data
        assert "total_return_percent" in data
        assert "daily_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data

    async def test_portfolio_transactions(
        self, client: AsyncClient, test_user_token_headers: dict, test_portfolio: Portfolio
    ):
        """Test GET /api/v1/portfolios/{id}/transactions"""
        # Add some transactions first
        await client.post(
            f"/api/v1/portfolios/{test_portfolio.id}/deposit",
            json={"amount": 50000.0},
            headers=test_user_token_headers
        )
        
        response = await client.get(
            f"/api/v1/portfolios/{test_portfolio.id}/transactions?limit=10",
            headers=test_user_token_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["type"] == "DEPOSIT"


# Fixtures for testing
@pytest.fixture
async def test_portfolio(db: AsyncSession, test_user: User) -> Portfolio:
    """Create a test portfolio"""
    portfolio = Portfolio(
        id=str(uuid.uuid4()),
        user_id=test_user.id,
        name="Test Portfolio",
        description="Test portfolio for unit tests",
        portfolio_type="EQUITY",
        total_value=100000.0,
        cash_balance=100000.0,
        buying_power=100000.0,
        is_active=True,
        is_default=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return portfolio


@pytest.fixture
async def other_user_token_headers(client: AsyncClient, db: AsyncSession) -> dict:
    """Create another user for authorization tests"""
    other_user = User(
        id=str(uuid.uuid4()),
        email="other@example.com",
        username="otheruser",
        full_name="Other User",
        hashed_password=get_password_hash("otherpass123"),
        is_active=True,
        is_verified=True,
        account_type="BASIC",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(other_user)
    await db.commit()
    
    # Login to get token
    login_data = {
        "username": "other@example.com",
        "password": "otherpass123"
    }
    response = await client.post("/api/v1/auth/login", data=login_data)
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}
