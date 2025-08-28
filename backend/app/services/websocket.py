"""WebSocket service for real-time data streaming."""

import json
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import socketio
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.config import settings
from app.core.security import verify_token

logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    logger=logger,
    engineio_logger=logger if settings.DEBUG else False
)

# Create Socket.IO ASGI app
sio_app = socketio.ASGIApp(
    socketio_server=sio,
    socketio_path='socket.io'
)


class ConnectionManager:
    """Manages WebSocket connections for different channels."""
    
    def __init__(self):
        # Track active connections by channel
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Track user connections
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Track authenticated users
        self.authenticated_users: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, channel: str = "general"):
        """Accept and register a new connection."""
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        
        self.active_connections[channel].add(websocket)
        logger.info(f"New connection to channel: {channel}")
    
    async def authenticate(self, websocket: WebSocket, token: str) -> Optional[str]:
        """Authenticate a WebSocket connection."""
        user_id = verify_token(token)
        
        if user_id:
            self.authenticated_users[websocket] = user_id
            
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            
            self.user_connections[user_id].add(websocket)
            logger.info(f"User {user_id} authenticated via WebSocket")
            return user_id
        
        return None
    
    def disconnect(self, websocket: WebSocket, channel: str = "general"):
        """Remove a connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        
        # Remove from user connections if authenticated
        if websocket in self.authenticated_users:
            user_id = self.authenticated_users[websocket]
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
            del self.authenticated_users[websocket]
        
        logger.info(f"Connection disconnected from channel: {channel}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send a message to a specific user."""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
    
    async def broadcast_to_channel(self, message: dict, channel: str):
        """Broadcast a message to all connections in a channel."""
        if channel in self.active_connections:
            disconnected = set()
            
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to channel {channel}: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, channel)


# Global connection manager
manager = ConnectionManager()


# Socket.IO event handlers
@sio.event
async def connect(sid, environ, auth):
    """Handle Socket.IO connection."""
    logger.info(f"Socket.IO client connected: {sid}")
    
    # Authenticate if token provided
    if auth and 'token' in auth:
        # Verify token (implement token verification)
        pass
    
    # Join default room
    await sio.enter_room(sid, 'market_data')


@sio.event
async def disconnect(sid):
    """Handle Socket.IO disconnection."""
    logger.info(f"Socket.IO client disconnected: {sid}")


@sio.event
async def subscribe_symbol(sid, data):
    """Subscribe to real-time updates for a symbol."""
    symbol = data.get('symbol')
    if symbol:
        room = f"symbol_{symbol}"
        await sio.enter_room(sid, room)
        await sio.emit('subscribed', {'symbol': symbol, 'status': 'success'}, room=sid)
        logger.info(f"Client {sid} subscribed to {symbol}")


@sio.event
async def unsubscribe_symbol(sid, data):
    """Unsubscribe from symbol updates."""
    symbol = data.get('symbol')
    if symbol:
        room = f"symbol_{symbol}"
        await sio.leave_room(sid, room)
        await sio.emit('unsubscribed', {'symbol': symbol, 'status': 'success'}, room=sid)
        logger.info(f"Client {sid} unsubscribed from {symbol}")


@sio.event
async def subscribe_portfolio(sid, data):
    """Subscribe to portfolio updates."""
    portfolio_id = data.get('portfolio_id')
    if portfolio_id:
        room = f"portfolio_{portfolio_id}"
        await sio.enter_room(sid, room)
        logger.info(f"Client {sid} subscribed to portfolio {portfolio_id}")


# Broadcasting functions
async def broadcast_price_update(symbol: str, price_data: dict):
    """Broadcast price update for a symbol."""
    room = f"symbol_{symbol}"
    await sio.emit('price_update', {
        'symbol': symbol,
        'data': price_data
    }, room=room)


async def broadcast_market_update(update_type: str, data: dict):
    """Broadcast general market updates."""
    await sio.emit('market_update', {
        'type': update_type,
        'data': data
    }, room='market_data')


async def broadcast_trade_update(user_id: str, trade_data: dict):
    """Broadcast trade update to a user."""
    await sio.emit('trade_update', trade_data, room=f"user_{user_id}")


async def broadcast_alert(user_id: str, alert_data: dict):
    """Broadcast alert to a user."""
    await sio.emit('alert', alert_data, room=f"user_{user_id}")


# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get('type')
            
            if message_type == 'authenticate':
                token = data.get('token')
                user_id = await manager.authenticate(websocket, token)
                
                if user_id:
                    await websocket.send_json({
                        'type': 'authenticated',
                        'user_id': user_id
                    })
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Authentication failed'
                    })
            
            elif message_type == 'subscribe':
                channel = data.get('channel', 'general')
                await manager.connect(websocket, channel)
                await websocket.send_json({
                    'type': 'subscribed',
                    'channel': channel
                })
            
            elif message_type == 'ping':
                await websocket.send_json({'type': 'pong'})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
