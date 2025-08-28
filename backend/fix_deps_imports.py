#!/usr/bin/env python3
"""Fix all deps imports in endpoint files"""
import os
import re

def fix_deps_in_file(filepath):
    """Replace deps imports and usage in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the import
    content = re.sub(r'from app\.api import deps', 
                     'from app.db.database import get_db\nfrom app.core.security import get_current_active_user', 
                     content)
    
    # Replace deps.get_db with get_db
    content = re.sub(r'deps\.get_db', 'get_db', content)
    
    # Replace deps.get_current_active_user with get_current_active_user
    content = re.sub(r'deps\.get_current_active_user', 'get_current_active_user', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ Fixed: {filepath}")

# Fix specific files
files_to_fix = [
    "app/api/v1/endpoints/deep_rl.py",
    "app/api/v1/endpoints/alternative_data.py",
    "app/api/v1/endpoints/hft.py"
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        fix_deps_in_file(filepath)

print("\n✨ All deps imports have been fixed!")
