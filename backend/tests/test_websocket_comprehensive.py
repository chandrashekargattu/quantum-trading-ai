"""Comprehensive WebSocket tests for real-time features."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from httpx import AsyncClient
import websockets
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.websocket import manager
from app.models.user import User


class TestWebSocketConnection:
    """Test WebSocket connection management."""
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self, client: AsyncClient, auth_headers):
        """Test WebSocket authentication with JWT token."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(
                f"/ws/market?token={token}"
            ) as websocket:
                data = websocket.receive_json()
                assert data["type"] == "connection"
                assert data["status"] == "connected"
    
    @pytest.mark.asyncio
    async def test_websocket_invalid_token(self, client: AsyncClient):
        """Test WebSocket connection with invalid token."""
        with TestClient(app) as test_client:
            with pytest.raises(websockets.exceptions.InvalidStatusCode) as exc:
                with test_client.websocket_connect("/ws/market?token=invalid"):
                    pass
            assert exc.value.status_code == 403
    
    @pytest.mark.asyncio
    async def test_multiple_connections(self, client: AsyncClient, auth_headers):
        """Test handling multiple WebSocket connections."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            # Create multiple connections
            websockets = []
            for i in range(3):
                ws = test_client.websocket_connect(f"/ws/market?token={token}")
                websockets.append(ws.__enter__())
            
            # All should be connected
            for ws in websockets:
                data = ws.receive_json()
                assert data["status"] == "connected"
            
            # Clean up
            for ws in websockets:
                ws.__exit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_connection_limit(self, client: AsyncClient, auth_headers):
        """Test WebSocket connection limits per user."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            # Create max connections
            websockets = []
            max_connections = 5
            
            for i in range(max_connections):
                ws = test_client.websocket_connect(f"/ws/market?token={token}")
                websockets.append(ws.__enter__())
            
            # Try to exceed limit
            with pytest.raises(websockets.exceptions.InvalidStatusCode) as exc:
                with test_client.websocket_connect(f"/ws/market?token={token}"):
                    pass
            assert exc.value.status_code == 429  # Too Many Requests
            
            # Clean up
            for ws in websockets:
                ws.__exit__(None, None, None)


class TestMarketDataWebSocket:
    """Test real-time market data streaming."""
    
    @pytest.mark.asyncio
    async def test_subscribe_to_symbol(self, client: AsyncClient, auth_headers):
        """Test subscribing to real-time price updates."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                # Subscribe to symbol
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["AAPL", "GOOGL"]
                })
                
                # Receive subscription confirmation
                response = ws.receive_json()
                assert response["type"] == "subscription"
                assert response["status"] == "subscribed"
                assert set(response["symbols"]) == {"AAPL", "GOOGL"}
    
    @pytest.mark.asyncio
    async def test_receive_price_updates(self, client: AsyncClient, auth_headers):
        """Test receiving real-time price updates."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                # Subscribe
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["AAPL"]
                })
                ws.receive_json()  # Skip confirmation
                
                # Simulate price update
                with patch('app.api.v1.websocket.broadcast_price_update') as mock_broadcast:
                    await manager.broadcast_price_update("AAPL", {
                        "symbol": "AAPL",
                        "price": 150.50,
                        "volume": 1000000,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Should receive update
                    update = ws.receive_json()
                    assert update["type"] == "price_update"
                    assert update["data"]["symbol"] == "AAPL"
                    assert update["data"]["price"] == 150.50
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_symbol(self, client: AsyncClient, auth_headers):
        """Test unsubscribing from symbols."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                # Subscribe first
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["AAPL", "GOOGL"]
                })
                ws.receive_json()
                
                # Unsubscribe from one
                ws.send_json({
                    "action": "unsubscribe",
                    "symbols": ["AAPL"]
                })
                
                response = ws.receive_json()
                assert response["type"] == "subscription"
                assert response["status"] == "unsubscribed"
                assert response["symbols"] == ["GOOGL"]  # Still subscribed to GOOGL


class TestOrderWebSocket:
    """Test real-time order updates."""
    
    @pytest.mark.asyncio
    async def test_order_status_updates(self, client: AsyncClient, auth_headers, test_user):
        """Test receiving order status updates."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/orders?token={token}") as ws:
                # Connection confirmed
                ws.receive_json()
                
                # Simulate order update
                order_update = {
                    "order_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "filled",
                    "filled_quantity": 100,
                    "filled_price": 150.25,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await manager.send_order_update(test_user.id, order_update)
                
                # Receive update
                update = ws.receive_json()
                assert update["type"] == "order_update"
                assert update["data"]["status"] == "filled"
    
    @pytest.mark.asyncio
    async def test_order_execution_alerts(self, client: AsyncClient, auth_headers, test_user):
        """Test real-time order execution alerts."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/orders?token={token}") as ws:
                ws.receive_json()  # Skip connection
                
                # Simulate execution alert
                alert = {
                    "type": "execution",
                    "order_id": "123e4567-e89b-12d3-a456-426614174000",
                    "message": "Order filled at better price",
                    "improvement": 0.25,
                    "saved_amount": 25.00
                }
                
                await manager.send_execution_alert(test_user.id, alert)
                
                # Receive alert
                update = ws.receive_json()
                assert update["type"] == "execution_alert"
                assert update["data"]["improvement"] == 0.25


class TestPortfolioWebSocket:
    """Test real-time portfolio updates."""
    
    @pytest.mark.asyncio
    async def test_portfolio_value_updates(self, client: AsyncClient, auth_headers, test_user):
        """Test real-time portfolio value updates."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/portfolio?token={token}") as ws:
                ws.receive_json()  # Skip connection
                
                # Portfolio update
                portfolio_update = {
                    "total_value": 125000.50,
                    "daily_pnl": 1250.25,
                    "daily_pnl_percent": 1.01,
                    "positions_count": 15,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await manager.send_portfolio_update(test_user.id, portfolio_update)
                
                update = ws.receive_json()
                assert update["type"] == "portfolio_update"
                assert update["data"]["total_value"] == 125000.50
    
    @pytest.mark.asyncio
    async def test_position_updates(self, client: AsyncClient, auth_headers, test_user):
        """Test real-time position updates."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/portfolio?token={token}") as ws:
                ws.receive_json()
                
                # Position update
                position_update = {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_cost": 145.00,
                    "current_price": 150.50,
                    "unrealized_pnl": 550.00,
                    "unrealized_pnl_percent": 3.79
                }
                
                await manager.send_position_update(test_user.id, position_update)
                
                update = ws.receive_json()
                assert update["type"] == "position_update"
                assert update["data"]["unrealized_pnl"] == 550.00


class TestAlertWebSocket:
    """Test real-time alerts and notifications."""
    
    @pytest.mark.asyncio
    async def test_price_alerts(self, client: AsyncClient, auth_headers, test_user):
        """Test real-time price alerts."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/alerts?token={token}") as ws:
                ws.receive_json()
                
                # Price alert
                alert = {
                    "alert_id": "alert123",
                    "type": "price_above",
                    "symbol": "AAPL",
                    "condition": "price > 150",
                    "current_price": 150.50,
                    "message": "AAPL price exceeded $150"
                }
                
                await manager.send_alert(test_user.id, alert)
                
                update = ws.receive_json()
                assert update["type"] == "alert"
                assert update["data"]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_risk_alerts(self, client: AsyncClient, auth_headers, test_user):
        """Test real-time risk management alerts."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/alerts?token={token}") as ws:
                ws.receive_json()
                
                # Risk alert
                risk_alert = {
                    "type": "risk_limit_breach",
                    "severity": "high",
                    "metric": "portfolio_var",
                    "current_value": 55000,
                    "limit": 50000,
                    "message": "Portfolio VaR exceeds limit",
                    "recommendations": [
                        "Reduce position sizes",
                        "Hedge with options"
                    ]
                }
                
                await manager.send_risk_alert(test_user.id, risk_alert)
                
                update = ws.receive_json()
                assert update["type"] == "risk_alert"
                assert update["data"]["severity"] == "high"


class TestWebSocketHeartbeat:
    """Test WebSocket heartbeat and reconnection."""
    
    @pytest.mark.asyncio
    async def test_heartbeat_ping_pong(self, client: AsyncClient, auth_headers):
        """Test WebSocket heartbeat mechanism."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                ws.receive_json()  # Connection
                
                # Send ping
                ws.send_json({"type": "ping"})
                
                # Should receive pong
                response = ws.receive_json()
                assert response["type"] == "pong"
                assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_connection_timeout(self, client: AsyncClient, auth_headers):
        """Test WebSocket connection timeout handling."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                ws.receive_json()
                
                # Simulate no activity for timeout period
                with patch('app.api.v1.websocket.HEARTBEAT_TIMEOUT', 1):
                    # Wait for timeout
                    await asyncio.sleep(2)
                    
                    # Connection should be closed
                    with pytest.raises(websockets.exceptions.ConnectionClosed):
                        ws.receive_json()


class TestWebSocketBroadcast:
    """Test WebSocket broadcasting functionality."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, client: AsyncClient, auth_headers):
        """Test broadcasting messages to all connected clients."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            # Create multiple connections
            websockets = []
            for i in range(3):
                ws = test_client.websocket_connect(f"/ws/market?token={token}")
                websockets.append(ws.__enter__())
                websockets[-1].receive_json()  # Skip connection message
            
            # Broadcast message
            await manager.broadcast({
                "type": "market_status",
                "status": "closed",
                "message": "Market closed for the day"
            })
            
            # All should receive
            for ws in websockets:
                msg = ws.receive_json()
                assert msg["type"] == "market_status"
                assert msg["status"] == "closed"
            
            # Clean up
            for ws in websockets:
                ws.__exit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_broadcast_to_group(self, client: AsyncClient, auth_headers):
        """Test broadcasting to specific groups."""
        token = auth_headers["Authorization"].split(" "][1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws1:
                with test_client.websocket_connect(f"/ws/orders?token={token}") as ws2:
                    ws1.receive_json()
                    ws2.receive_json()
                    
                    # Broadcast to market group only
                    await manager.broadcast_to_group("market", {
                        "type": "market_update",
                        "data": "test"
                    })
                    
                    # Only market WebSocket should receive
                    msg = ws1.receive_json()
                    assert msg["type"] == "market_update"
                    
                    # Orders WebSocket should not receive
                    with pytest.raises(asyncio.TimeoutError):
                        ws2.receive_json(timeout=0.1)


class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""
    
    @pytest.mark.asyncio
    async def test_malformed_message(self, client: AsyncClient, auth_headers):
        """Test handling of malformed messages."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                ws.receive_json()
                
                # Send malformed JSON
                ws.send_text("invalid json")
                
                # Should receive error
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "Invalid message format" in response["message"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client: AsyncClient, auth_headers):
        """Test WebSocket message rate limiting."""
        token = auth_headers["Authorization"].split(" ")[1]
        
        with TestClient(app) as test_client:
            with test_client.websocket_connect(f"/ws/market?token={token}") as ws:
                ws.receive_json()
                
                # Send many messages quickly
                for i in range(100):
                    ws.send_json({"action": "subscribe", "symbols": [f"TEST{i}"]})
                
                # Should receive rate limit error
                found_rate_limit = False
                for _ in range(100):
                    try:
                        response = ws.receive_json(timeout=0.1)
                        if response.get("type") == "error" and "rate limit" in response.get("message", "").lower():
                            found_rate_limit = True
                            break
                    except:
                        break
                
                assert found_rate_limit
