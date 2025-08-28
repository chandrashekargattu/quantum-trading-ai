#!/usr/bin/env python3
"""Test script to verify the Quantum Trading AI backend can start."""

import sys

print("🔍 Testing Quantum Trading AI Backend...")
print("=" * 50)

# Test imports
try:
    print("1️⃣ Testing FastAPI import...")
    from fastapi import FastAPI
    print("   ✅ FastAPI imported successfully")
except Exception as e:
    print(f"   ❌ FastAPI import failed: {e}")
    sys.exit(1)

try:
    print("\n2️⃣ Testing app configuration...")
    from app.core.config import settings
    print(f"   ✅ Project: {settings.PROJECT_NAME}")
    print(f"   ✅ Version: {settings.VERSION}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    sys.exit(1)

try:
    print("\n3️⃣ Testing database connection...")
    from app.db.database import engine
    print(f"   ✅ Database URL: {settings.DATABASE_URL[:30]}...")
except Exception as e:
    print(f"   ❌ Database import failed: {e}")
    sys.exit(1)

try:
    print("\n4️⃣ Testing main app import...")
    from app.main import app
    print("   ✅ App imported successfully!")
    
    print("\n5️⃣ Testing app routes...")
    routes = [route.path for route in app.routes]
    print(f"   ✅ Found {len(routes)} routes")
    print(f"   ✅ Root route: {routes[0]}")
    
except Exception as e:
    print(f"   ❌ App import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! The server should be able to start.")
print("\n🚀 To start the server, run:")
print("   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
