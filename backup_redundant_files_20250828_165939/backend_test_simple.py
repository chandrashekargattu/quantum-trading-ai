#!/usr/bin/env python3
"""
Simple test to verify basic functionality.
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set minimal environment
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test.db'
os.environ['JWT_SECRET_KEY'] = 'test-secret'
os.environ['JWT_ALGORITHM'] = 'HS256'

print("Testing basic imports...")

try:
    from app.core.config import settings
    print("✓ Config imported successfully")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from app.core.security import get_password_hash, verify_password
    print("✓ Security functions imported successfully")
    
    # Test password hashing
    password = "test123"
    hashed = get_password_hash(password)
    verified = verify_password(password, hashed)
    assert verified, "Password verification failed"
    print("✓ Password hashing works correctly")
except Exception as e:
    print(f"✗ Security test failed: {e}")

try:
    from app.db.database import get_db
    print("✓ Database imported successfully")
except Exception as e:
    print(f"✗ Database import failed: {e}")

try:
    from app.models.user import User
    print("✓ User model imported successfully")
except Exception as e:
    print(f"✗ User model import failed: {e}")

print("\nBasic import test completed!")
print("\nTrying to run a simple async test...")

import asyncio

async def test_db_connection():
    """Test database connection."""
    try:
        from app.db.database import init_db
        await init_db()
        print("✓ Database initialization successful")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False

# Run the async test
result = asyncio.run(test_db_connection())
print(f"\nTest result: {'PASSED' if result else 'FAILED'}")
