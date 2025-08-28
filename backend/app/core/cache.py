"""Simple cache implementation for market data."""

from typing import Any, Optional
import json
from datetime import datetime, timedelta
import asyncio

# Simple in-memory cache
_cache = {}
_cache_expiry = {}


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache if not expired."""
    if key in _cache:
        expiry = _cache_expiry.get(key)
        if expiry and datetime.now() < expiry:
            return _cache[key]
        else:
            # Remove expired entry
            _cache.pop(key, None)
            _cache_expiry.pop(key, None)
    return None


async def cache_set(key: str, value: Any, expire: int = 300) -> None:
    """Set value in cache with expiration time in seconds."""
    _cache[key] = value
    if expire > 0:
        _cache_expiry[key] = datetime.now() + timedelta(seconds=expire)


async def cache_delete(key: str) -> None:
    """Delete key from cache."""
    _cache.pop(key, None)
    _cache_expiry.pop(key, None)


async def cache_clear() -> None:
    """Clear all cache entries."""
    _cache.clear()
    _cache_expiry.clear()


# Decorator for caching function results
def cached(expire: int = 300):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            result = await cache_get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache_set(cache_key, result, expire)
            return result
        
        return wrapper
    return decorator
