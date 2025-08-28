"""WebSocket connection management and real-time updates."""

from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import json
import asyncio
import logging
from collections import defaultdict
import jwt
from jwt.exceptions import PyJWTError

from app.core.config import settings
from app.core.security import decode_access_token
from app.models.user import User
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Connection limits
MAX_CONNECTIONS_PER_USER = 5
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_TIMEOUT = 60  # seconds
MESSAGE_RATE_LIMIT = 100  # messages per minute


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        # Store active connections by user ID
        self.active_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        # Store subscriptions
        self.subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)
        # Rate limiting
        self.message_counts: Dict[str, List[datetime]] = defaultdict(list)
        # Groups (market, orders, portfolio, alerts)
        self.groups: Dict[str, Set[WebSocket]] = defaultdict(set)
        
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        group: str = "general"
    ) -> bool:
        """Accept a new WebSocket connection."""
        # Check connection limit
        if len(self.active_connections[user_id]) >= MAX_CONNECTIONS_PER_USER:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return False
        
        await websocket.accept()
        
        # Add to active connections
        self.active_connections[user_id].append(websocket)
        self.groups[group].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "group": group,
            "connected_at": datetime.utcnow(),
            "last_heartbeat": datetime.utcnow(),
            "subscriptions": set()
        }
        
        # Send connection confirmation
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat()
            },
            websocket
        )
        
        logger.info(f"User {user_id} connected to {group} WebSocket")
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return
        
        user_id = metadata["user_id"]
        group = metadata["group"]
        
        # Remove from active connections
        if websocket in self.active_connections[user_id]:
            self.active_connections[user_id].remove(websocket)
        
        # Remove from groups
        if websocket in self.groups[group]:
            self.groups[group].remove(websocket)
        
        # Remove from subscriptions
        for symbol in metadata["subscriptions"]:
            if websocket in self.subscriptions[symbol]:
                self.subscriptions[symbol].remove(websocket)
        
        # Clean up metadata
        del self.connection_metadata[websocket]
        
        # Clean up empty collections
        if not self.active_connections[user_id]:
            del self.active_connections[user_id]
        
        logger.info(f"User {user_id} disconnected from {group} WebSocket")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def send_user_message(self, user_id: str, message: dict):
        """Send a message to all connections of a specific user."""
        if user_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected sockets
            for conn in disconnected:
                self.disconnect(conn)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for user_connections in self.active_connections.values():
            for connection in user_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected sockets
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_group(self, group: str, message: dict):
        """Broadcast a message to all clients in a specific group."""
        if group not in self.groups:
            return
        
        disconnected = []
        for connection in self.groups[group]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to group {group}: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected sockets
        for conn in disconnected:
            self.disconnect(conn)
    
    async def handle_subscription(
        self, 
        websocket: WebSocket, 
        action: str, 
        symbols: List[str]
    ):
        """Handle symbol subscription/unsubscription."""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return
        
        if action == "subscribe":
            for symbol in symbols:
                self.subscriptions[symbol].add(websocket)
                metadata["subscriptions"].add(symbol)
            
            await self.send_personal_message(
                {
                    "type": "subscription",
                    "status": "subscribed",
                    "symbols": list(metadata["subscriptions"]),
                    "timestamp": datetime.utcnow().isoformat()
                },
                websocket
            )
        
        elif action == "unsubscribe":
            for symbol in symbols:
                if websocket in self.subscriptions[symbol]:
                    self.subscriptions[symbol].remove(websocket)
                metadata["subscriptions"].discard(symbol)
            
            await self.send_personal_message(
                {
                    "type": "subscription",
                    "status": "unsubscribed",
                    "symbols": list(metadata["subscriptions"]),
                    "timestamp": datetime.utcnow().isoformat()
                },
                websocket
            )
    
    async def broadcast_price_update(self, symbol: str, price_data: dict):
        """Broadcast price update to subscribers."""
        if symbol not in self.subscriptions:
            return
        
        message = {
            "type": "price_update",
            "data": price_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection in self.subscriptions[symbol]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending price update: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected sockets
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_order_update(self, user_id: str, order_update: dict):
        """Send order update to user."""
        message = {
            "type": "order_update",
            "data": order_update,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    async def send_portfolio_update(self, user_id: str, portfolio_update: dict):
        """Send portfolio update to user."""
        message = {
            "type": "portfolio_update",
            "data": portfolio_update,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    async def send_position_update(self, user_id: str, position_update: dict):
        """Send position update to user."""
        message = {
            "type": "position_update",
            "data": position_update,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    async def send_alert(self, user_id: str, alert: dict):
        """Send alert to user."""
        message = {
            "type": "alert",
            "data": alert,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    async def send_risk_alert(self, user_id: str, risk_alert: dict):
        """Send risk alert to user."""
        message = {
            "type": "risk_alert",
            "data": risk_alert,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    async def send_execution_alert(self, user_id: str, execution_alert: dict):
        """Send execution alert to user."""
        message = {
            "type": "execution_alert",
            "data": execution_alert,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(user_id, message)
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.message_counts[user_id] = [
            timestamp for timestamp in self.message_counts[user_id]
            if timestamp > minute_ago
        ]
        
        # Check limit
        if len(self.message_counts[user_id]) >= MESSAGE_RATE_LIMIT:
            return False
        
        # Record new message
        self.message_counts[user_id].append(now)
        return True
    
    async def handle_heartbeat(self, websocket: WebSocket):
        """Handle heartbeat/ping-pong."""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            metadata["last_heartbeat"] = datetime.utcnow()
            await self.send_personal_message(
                {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                },
                websocket
            )
    
    async def check_stale_connections(self):
        """Check and remove stale connections."""
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(seconds=HEARTBEAT_TIMEOUT)
        
        stale_connections = []
        for websocket, metadata in self.connection_metadata.items():
            if metadata["last_heartbeat"] < timeout_threshold:
                stale_connections.append(websocket)
        
        for conn in stale_connections:
            logger.warning(f"Removing stale connection for user {self.connection_metadata[conn]['user_id']}")
            await conn.close()
            self.disconnect(conn)


# Global connection manager instance
manager = ConnectionManager()


async def get_current_user_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Authenticate WebSocket connection."""
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    
    try:
        # Decode token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
        
        return user
    
    except PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


async def websocket_endpoint(
    websocket: WebSocket,
    group: str,
    db: AsyncSession
):
    """Generic WebSocket endpoint handler."""
    # Get token from query params
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Authenticate
    user = await get_current_user_ws(websocket, token, db)
    if not user:
        return
    
    # Connect
    connected = await manager.connect(websocket, str(user.id), group)
    if not connected:
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Check rate limit
            if not manager.check_rate_limit(str(user.id)):
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Rate limit exceeded",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    websocket
                )
                continue
            
            # Parse message
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid message format",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    websocket
                )
                continue
            
            # Handle message types
            if message.get("type") == "ping":
                await manager.handle_heartbeat(websocket)
            
            elif message.get("action") in ["subscribe", "unsubscribe"]:
                symbols = message.get("symbols", [])
                if symbols and group == "market":
                    await manager.handle_subscription(
                        websocket,
                        message["action"],
                        symbols
                    )
            
            else:
                # Echo back unknown message types
                await manager.send_personal_message(
                    {
                        "type": "echo",
                        "data": message,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Background task to check stale connections
async def periodic_connection_check():
    """Periodically check for stale connections."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        await manager.check_stale_connections()
