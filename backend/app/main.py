"""Main FastAPI application for Quantum Trading AI."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from prometheus_fastapi_instrumentator import Instrumentator
import logging
import sys

from app.core.config import settings
from app.api.v1.api import api_router
from app.db.database import init_db
from app.api.v1.websocket import websocket_endpoint, periodic_connection_check
from app.ml.model_manager import ModelManager
from fastapi import WebSocket
import asyncio

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Quantum Trading AI Backend...")
    
    # Initialize database
    await init_db()
    
    # Initialize ML models
    model_manager = ModelManager()
    await model_manager.load_models()
    app.state.model_manager = model_manager
    
    # Start WebSocket connection checker
    connection_check_task = asyncio.create_task(periodic_connection_check())
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Quantum Trading AI Backend...")
    connection_check_task.cancel()
    await model_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["api.quantumtrading.ai", "localhost"]
)

# Add Prometheus metrics
# instrumentator = Instrumentator()
# instrumentator.instrument(app).expose(app)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# WebSocket endpoints
@app.websocket("/ws/market")
async def market_websocket(websocket: WebSocket):
    """Market data WebSocket endpoint."""
    from app.db.database import get_db
    async for db in get_db():
        await websocket_endpoint(websocket, "market", db)
        break

@app.websocket("/ws/orders")
async def orders_websocket(websocket: WebSocket):
    """Orders WebSocket endpoint."""
    from app.db.database import get_db
    async for db in get_db():
        await websocket_endpoint(websocket, "orders", db)
        break

@app.websocket("/ws/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """Portfolio WebSocket endpoint."""
    from app.db.database import get_db
    async for db in get_db():
        await websocket_endpoint(websocket, "portfolio", db)
        break

@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """Alerts WebSocket endpoint."""
    from app.db.database import get_db
    async for db in get_db():
        await websocket_endpoint(websocket, "alerts", db)
        break


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Quantum Trading AI API",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.VERSION
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
