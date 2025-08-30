#!/usr/bin/env python3
"""
Setup and run the Quantum Trading AI Backend Server
This script handles all dependency issues and starts the server properly
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        print("✅ Success")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 Quantum Trading AI Backend Setup & Run")
    print("=" * 50)
    
    # Ensure we're in the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    print(f"📁 Working directory: {backend_dir}")
    
    # Step 1: Fix numpy compatibility
    print("\n📦 Step 1: Fixing NumPy compatibility...")
    commands = [
        ("pip uninstall numpy -y", "Uninstalling numpy"),
        ("pip install 'numpy<2.0,>=1.23.5' --no-cache-dir", "Installing compatible numpy"),
        ("pip install pandas --no-cache-dir", "Reinstalling pandas"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print("⚠️  Continuing despite error...")
    
    # Step 2: Ensure all core dependencies are installed
    print("\n📦 Step 2: Installing core dependencies...")
    core_deps = [
        "fastapi",
        "uvicorn[standard]",
        "sqlalchemy",
        "pydantic",
        "pydantic-settings",
        "httpx",
        "'python-jose[cryptography]'",
        "'passlib[bcrypt]'",
        "python-multipart",
        "aiosqlite",
        "email-validator",
        "aiohttp",
        "redis",
        "celery",
        "yfinance",
        "alpha-vantage",
        "ta",
        "websockets",
        "python-socketio",
        "scikit-learn",
        "joblib",
        "tensorflow",
        "torch",
        "prophet",
        "statsmodels",
        "gymnasium",
        "uvloop",
        "sortedcontainers",
        "numba",
    ]
    
    deps_cmd = f"pip install {' '.join(core_deps)} --quiet"
    run_command(deps_cmd, "Installing all core dependencies")
    
    # Step 3: Create a simple test
    print("\n🧪 Step 3: Testing imports...")
    test_code = '''
import sys
sys.path.insert(0, ".")
try:
    from app.main import app
    print("✅ App imports successfully!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
    
    with open("test_app.py", "w") as f:
        f.write(test_code)
    
    if not run_command("python test_app.py", "Testing app import"):
        print("❌ Import test failed!")
        return
    
    # Step 4: Start the server
    print("\n🚀 Step 4: Starting the server...")
    print("=" * 50)
    print("Backend will run on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run uvicorn directly with Python
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")

if __name__ == "__main__":
    main()
