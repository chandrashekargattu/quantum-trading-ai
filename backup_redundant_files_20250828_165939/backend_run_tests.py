#!/usr/bin/env python3
"""
Test runner script for the backend tests.
This script sets up the environment and runs pytest.
"""

import os
import sys
import subprocess

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set environment variables
os.environ['PYTHONPATH'] = backend_dir
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test_quantum_trading.db'
os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_ALGORITHM'] = 'HS256'
os.environ['JWT_EXPIRATION_HOURS'] = '24'

# Run pytest with appropriate flags
cmd = [
    sys.executable, '-m', 'pytest',
    'tests/',
    '-v',
    '--tb=short',
    '--disable-warnings',
    '-p', 'no:cacheprovider'
]

# Add any command line arguments passed to this script
cmd.extend(sys.argv[1:])

print(f"Running tests with Python path: {backend_dir}")
print(f"Command: {' '.join(cmd)}")
print("-" * 60)

# Run the tests
result = subprocess.run(cmd, cwd=backend_dir)
sys.exit(result.returncode)
