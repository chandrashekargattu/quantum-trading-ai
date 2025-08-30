#!/usr/bin/env python3
"""Test if backend can start properly"""

import subprocess
import time
import requests
import os
import sys

def test_backend_startup():
    """Test backend startup process"""
    print("🔍 Testing backend startup...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    print(f"📁 Changed to directory: {os.getcwd()}")
    
    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = os.getcwd()
    print(f"🔧 Set PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # Start backend
    print("🚀 Starting backend...")
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to start
    print("⏳ Waiting for backend to start...")
    time.sleep(10)
    
    # Check if it's running
    try:
        response = requests.get('http://localhost:8000/docs')
        if response.status_code == 200:
            print("✅ Backend is running successfully!")
            print(f"   Status code: {response.status_code}")
            proc.terminate()
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not responding")
    
    # Print error output
    print("\n📋 Backend output:")
    stdout, stderr = proc.communicate(timeout=5)
    if stdout:
        print("STDOUT:", stdout[:500])
    if stderr:
        print("STDERR:", stderr[:500])
    
    proc.terminate()
    return False

if __name__ == "__main__":
    success = test_backend_startup()
    sys.exit(0 if success else 1)

