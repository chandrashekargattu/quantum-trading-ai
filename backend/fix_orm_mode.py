#!/usr/bin/env python3
"""Fix all orm_mode = True to from_attributes = True for Pydantic v2 compatibility"""
import os
import re

def fix_orm_mode_in_file(filepath):
    """Replace orm_mode with from_attributes in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace orm_mode = True with from_attributes = True
    new_content = re.sub(r'\borm_mode\s*=\s*True\b', 'from_attributes = True', content)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"✅ Fixed: {filepath}")
        return True
    return False

# Fix all schema files
schemas_dir = "app/schemas"
fixed_count = 0

for filename in os.listdir(schemas_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(schemas_dir, filename)
        if fix_orm_mode_in_file(filepath):
            fixed_count += 1

print(f"\n🎯 Total files fixed: {fixed_count}")
print("✨ All orm_mode = True occurrences have been replaced with from_attributes = True")
