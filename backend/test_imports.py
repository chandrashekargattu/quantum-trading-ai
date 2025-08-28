#!/usr/bin/env python3
"""Test imports to debug server startup"""

import sys
import traceback

print("🔍 Testing Quantum Trading AI Backend Imports")
print("=" * 50)

tests = [
    ("FastAPI", "from fastapi import FastAPI"),
    ("Config", "from app.core.config import settings"),
    ("Database", "from app.db.database import engine"),
    ("API Router", "from app.api.v1.api import api_router"),
    ("Main App", "from app.main import app"),
]

for name, import_str in tests:
    print(f"\n📦 Testing {name}...")
    try:
        exec(import_str)
        print(f"   ✅ {name} imported successfully")
    except Exception as e:
        print(f"   ❌ {name} import failed: {e}")
        print("   Traceback:")
        traceback.print_exc()
        break

print("\n" + "=" * 50)
