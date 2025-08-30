#!/usr/bin/env python3
"""
Script to identify and clean up redundant files in the Quantum Trading AI project.
"""

import os
import shutil
from datetime import datetime

# Create backup directory
backup_dir = f"backup_redundant_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to remove (redundant startup scripts)
redundant_files = [
    # Backend startup scripts (keeping only one)
    "backend/run_backend.sh",
    "backend/setup_and_run.py", 
    "backend/start_backend.sh",
    "backend/start_now.py",
    "backend/start_simple_server.py",
    # Keeping: backend/start_backend_server.py
    
    # Frontend startup scripts (keeping start_platform.sh)
    "frontend/start.sh",
    "start_servers.sh",
    "start_both_servers.sh",
    # Keeping: start_platform.sh and start_recovery.sh (for Zerodha specific)
    
    # Test scripts (keeping run_all_tests.sh)
    "run_tests.sh",
    
    # Documentation duplicates
    "TEST_COVERAGE_SUMMARY.md",
    "TEST_RESULTS_SUMMARY.md", 
    "TEST_SUITE_DOCUMENTATION.md",
    "TEST_SUMMARY.md",
    "RUNNING_TESTS_GUIDE.md",
    # Keeping: FINAL_TEST_REPORT.md and TESTING.md
    
    "FINAL_SETUP_GUIDE.md",
    "START_PLATFORM.md",
    "QUICKSTART.md",
    # Keeping: START_HERE.md and INTEGRATION_GUIDE.md
    
    # Backend test summary (duplicate)
    "backend/TEST_FIX_SUMMARY.md",
    
    # Simple test runners (keeping the comprehensive one)
    "backend/test_simple.py",
    "backend/simple_test_runner.py",
    "backend/run_tests.py",
]

# Files to consolidate
files_to_keep = {
    "Startup Scripts": [
        "start_platform.sh - Main startup script for both servers",
        "start_recovery.sh - Zerodha-specific recovery startup",
        "backend/start_backend_server.py - Backend-only startup",
    ],
    "Testing": [
        "run_all_tests.sh - Comprehensive test runner",
        "FINAL_TEST_REPORT.md - Test results documentation",
        "TESTING.md - Testing guidelines",
    ],
    "Documentation": [
        "README.md - Main project documentation",
        "START_HERE.md - Quick start guide",
        "INTEGRATION_GUIDE.md - Integration instructions",
        "SECURITY_GUIDE.md - Security guidelines",
        "ZERODHA_RECOVERY_GUIDE.md - Zerodha-specific guide",
        "TRADING_GUIDE.md - Trading features guide",
    ],
    "Setup": [
        "quick_setup.py - Automated setup script",
    ]
}

def main():
    print("🧹 Cleaning up redundant files in Quantum Trading AI project")
    print("=" * 60)
    
    # Create backup directory
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"📁 Created backup directory: {backup_dir}")
    
    # Remove redundant files
    removed_count = 0
    for file_path in redundant_files:
        if os.path.exists(file_path):
            # Backup the file first
            backup_path = os.path.join(backup_dir, file_path.replace('/', '_'))
            shutil.copy2(file_path, backup_path)
            
            # Remove the file
            os.remove(file_path)
            print(f"❌ Removed: {file_path}")
            removed_count += 1
    
    print(f"\n✅ Removed {removed_count} redundant files")
    print(f"📦 Backups saved in: {backup_dir}")
    
    # Display kept files
    print("\n📋 Files kept (organized by category):")
    print("=" * 60)
    
    for category, files in files_to_keep.items():
        print(f"\n{category}:")
        for file_info in files:
            print(f"  ✅ {file_info}")
    
    # Create a consolidated startup guide
    print("\n📝 Creating consolidated documentation...")
    create_consolidated_docs()
    
    print("\n🎉 Cleanup complete!")

def create_consolidated_docs():
    """Create a single consolidated guide."""
    
    content = """# 🚀 Quantum Trading AI - Quick Reference

## Starting the Application

### Option 1: Start Everything (Recommended)
```bash
./start_platform.sh
```

### Option 2: Start Backend Only
```bash
cd backend
python start_backend_server.py
```

### Option 3: Start for Zerodha Recovery
```bash
./start_recovery.sh
```

## Running Tests
```bash
./run_all_tests.sh
```

## Setup
```bash
python quick_setup.py
```

## Access Points
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Documentation
- Getting Started: See START_HERE.md
- Integration Guide: See INTEGRATION_GUIDE.md
- Security: See SECURITY_GUIDE.md
- Zerodha Recovery: See ZERODHA_RECOVERY_GUIDE.md
"""
    
    with open("QUICK_REFERENCE.md", "w") as f:
        f.write(content)
    
    print("✅ Created QUICK_REFERENCE.md")

if __name__ == "__main__":
    main()
