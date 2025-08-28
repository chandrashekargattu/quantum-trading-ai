"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    users,
    stocks,
    options,
    trades,
    trading,  # New trading operations endpoints
    portfolios,
    alerts,
    strategies,
    market_data,
    backtest,
    quantum,
    deep_rl,
    # hft,  # Temporarily disabled - missing dependencies
    # alternative_data  # Temporarily disabled - missing dependencies
)
from app.api.v1 import zerodha

api_router = APIRouter()

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

# User management endpoints
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

# Market data endpoints
api_router.include_router(
    stocks.router,
    prefix="/stocks",
    tags=["stocks"]
)

api_router.include_router(
    options.router,
    prefix="/options",
    tags=["options"]
)

api_router.include_router(
    market_data.router,
    prefix="/market-data",
    tags=["market-data"]
)

# Trading endpoints
api_router.include_router(
    trades.router,
    prefix="/trades",
    tags=["trades"]
)

api_router.include_router(
    trading.router,
    prefix="/trading",
    tags=["trading"]
)

api_router.include_router(
    portfolios.router,
    prefix="/portfolios",
    tags=["portfolios"]
)

# Strategy and alerts endpoints
api_router.include_router(
    strategies.router,
    prefix="/strategies",
    tags=["strategies"]
)

api_router.include_router(
    alerts.router,
    prefix="/alerts",
    tags=["alerts"]
)

# Backtesting endpoints
api_router.include_router(
    backtest.router,
    prefix="/backtest",
    tags=["backtest"]
)

# Advanced features endpoints
api_router.include_router(
    quantum.router,
    prefix="/quantum",
    tags=["quantum-computing"]
)

api_router.include_router(
    deep_rl.router,
    prefix="/deep-rl",
    tags=["reinforcement-learning"]
)

# Zerodha integration endpoints
api_router.include_router(
    zerodha.router,
    prefix="/zerodha",
    tags=["zerodha-integration"]
)

# Temporarily disabled - missing dependencies
# api_router.include_router(
#     hft.router,
#     prefix="/hft",
#     tags=["high-frequency-trading"]
# )

# api_router.include_router(
#     alternative_data.router,
#     prefix="/alternative-data",
#     tags=["alternative-data"]
# )
